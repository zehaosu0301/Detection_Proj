import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import logging
from itertools import cycle


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_prepare_data_from_jsonl(data_path: str) -> dict:
    """
    Loads data from a JSON Lines file and prepares it in the paired format
    required for similarity evaluation.
    """
    human_originals, human_revised = [], []
    ai_originals, ai_revised = [], []

    logging.info(f"Loading and processing data from: {data_path}")
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    question = data.get("question")
                    human_answer = data.get("human_answers", [None])[0]
                    ai_answer = data.get("chatgpt_answers", [None])[0]

                    if question and human_answer:
                        human_originals.append(question)
                        human_revised.append(human_answer)

                    if question and ai_answer:
                        ai_originals.append(question)
                        ai_revised.append(ai_answer)
                except json.JSONDecodeError:
                    logging.warning(f"Could not parse line: {line[:100]}...")
                    continue
    except FileNotFoundError:
        logging.error(f"Data file not found at: {data_path}")
        return {
            "human": {"original": [], "revised": []},
            "ai": {"original": [], "revised": []},
        }

    logging.info(
        f"Loaded {len(human_originals)} human pairs and {len(ai_originals)} AI pairs."
    )

    return {
        "human": {"original": human_originals, "revised": human_revised},
        "ai": {"original": ai_originals, "revised": ai_revised},
    }


def _get_scores_for_model(
    model: SentenceTransformer,
    data: dict,
    human_test_indices: list,
    ai_test_indices: list,
) -> tuple[list, list]:
    """Helper function: Calculates similarity scores and true labels for a single model."""
    similarities = []
    true_labels = []

    human_data = data["human"]
    ai_data = data["ai"]

    for idx in tqdm(human_test_indices, desc=f"Evaluating Human Texts (Label 0)"):
        emb1 = model.encode(human_data["original"][idx], show_progress_bar=False)
        emb2 = model.encode(human_data["revised"][idx], show_progress_bar=False)
        similarities.append(util.cos_sim(emb1, emb2).item())
        true_labels.append(0)

    for idx in tqdm(ai_test_indices, desc=f"Evaluating AI Texts (Label 1)"):
        emb1 = model.encode(ai_data["original"][idx], show_progress_bar=False)
        emb2 = model.encode(ai_data["revised"][idx], show_progress_bar=False)
        similarities.append(util.cos_sim(emb1, emb2).item())
        true_labels.append(1)

    return similarities, true_labels


def compare_base_models_performance(
    model_names: list[str],
    data_path: str = "./data/finance_data.jsonl",
    test_ratio: float = 0.5,
):
    """
    Loads, evaluates, and compares multiple base models, plotting ROC curves and printing detailed reports.
    """
    print("--- Comparing Base Model Performances ---")

    prepared_data = load_and_prepare_data_from_jsonl(data_path)
    human_total = len(prepared_data["human"]["original"])
    ai_total = len(prepared_data["ai"]["original"])

    if human_total == 0 or ai_total == 0:
        print("Not enough data to create a test set. Exiting.")
        return

    human_test_indices = list(range(int(human_total * (1 - test_ratio)), human_total))
    ai_test_indices = list(range(int(ai_total * (1 - test_ratio)), ai_total))

    all_results = {}
    true_labels = []

    for model_name in model_names:
        print(f"\n{'='*25}\n--- Evaluating Model: {model_name} ---\n{'='*25}")
        try:
            model = SentenceTransformer(model_name)

            if model.tokenizer.pad_token is None:
                logging.warning(
                    f"Model '{model_name}' is missing a pad token. Setting it to the eos_token."
                )
                model.tokenizer.pad_token = model.tokenizer.eos_token

            sims, current_labels = _get_scores_for_model(
                model, prepared_data, human_test_indices, ai_test_indices
            )

            if not true_labels:
                true_labels = current_labels

            # 1. Calculate ROC and AUC (threshold-independent)
            fpr, tpr, thresholds = roc_curve(true_labels, sims)
            auc = roc_auc_score(true_labels, sims)

            all_results[model_name] = {"fpr": fpr, "tpr": tpr, "auc": auc}

            # 2. Find the optimal threshold for classification metrics
            # We find the point on the ROC curve closest to the top-left corner (0,1)
            J = tpr - fpr
            optimal_idx = np.argmax(J)
            optimal_threshold = thresholds[optimal_idx]

            # 3. Generate predictions using the optimal threshold
            y_pred = [1 if s >= optimal_threshold else 0 for s in sims]

            # 4. Calculate and print the detailed report
            accuracy = accuracy_score(true_labels, y_pred)
            report = classification_report(
                true_labels, y_pred, target_names=["Human", "AI"]
            )
            cm = confusion_matrix(true_labels, y_pred)

            print(f"\n=== Test Results for {model_name.split('/')[-1]} ===")
            print(f"Optimal Threshold: {optimal_threshold:.4f} (Maximizes TPR-FPR)")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"AUC: {auc:.4f}\n")

            print("Classification Report:")
            print(report)

            print("Confusion Matrix:")
            print("              Predicted")
            print("           Human      AI")
            print(f"Actual Human {cm[0][0]:>5}   {cm[0][1]:>5}")
            print(f"       AI    {cm[1][0]:>5}   {cm[1][1]:>5}\n")

        except Exception as e:
            print(f"Could not evaluate model '{model_name}'. Error: {e}")

    # --- PLOTTING ---
    plt.figure(figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))
    styles = cycle(["-", "--", ":", "-."])

    for i, (name, result) in enumerate(all_results.items()):
        short_name = name.split("/")[-1]
        plt.plot(
            result["fpr"],
            result["tpr"],
            color=colors[i],
            linestyle=next(styles),
            lw=2,
            label=f'{short_name} (AUC = {result["auc"]:.4f})',
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve Comparison of Base Models", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    MODELS_TO_COMPARE = [
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "bert-base-uncased",
        "roberta-base",
        "google-t5/t5-small",
        "gpt2",
    ]
    DATA_FILE_PATH = "./data/finance.jsonl"

    compare_base_models_performance(
        model_names=MODELS_TO_COMPARE,
        data_path=DATA_FILE_PATH,
        test_ratio=1.0,
    )
