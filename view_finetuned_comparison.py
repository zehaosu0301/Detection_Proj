import torch
from sentence_transformers import SentenceTransformer
import os


def print_model_details(name: str, path: str):
    """Loads a SentenceTransformer model and prints its details."""
    print("=" * 60)
    print(f"Loading Model: '{name}'")
    print(f"Path: {path}")
    print("=" * 60)

    model = None
    try:
        # SentenceTransformer can handle both local paths and Hub model names
        model = SentenceTransformer(path)
        print("  - Status: Model loaded successfully.")
    except Exception as e:
        print(f"  - Error: Could not load model. {e}")
        return None

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,}")
    print(f"  - Model Architecture:\n{model}\n")
    return model


def compare_model_weights(model_original, model_finetuned):
    """Compares the weights of two models to check for differences."""
    print("=" * 60)
    print("Comparing Model Weights...")
    print("=" * 60)

    if model_original is None or model_finetuned is None:
        print("One or both models could not be loaded. Cannot compare weights.")
        return

    # Get the state dictionaries which contain all the model's learned weights
    state_dict_orig = model_original.state_dict()
    state_dict_finetuned = model_finetuned.state_dict()

    if len(state_dict_orig) != len(state_dict_finetuned):
        print(
            "Models have a different number of layers and cannot be compared directly."
        )
        return

    differences_found = False
    # Iterate through all the layers and compare their weights tensor by tensor
    for key in state_dict_orig:
        if key in state_dict_finetuned:
            # torch.equal performs an element-wise comparison of the two tensors
            if not torch.equal(state_dict_orig[key], state_dict_finetuned[key]):
                print(f"  - Difference found in layer: '{key}'")
                differences_found = True
                # We can stop after finding the first difference to be efficient
                break
        else:
            print(f"  - Layer '{key}' exists in original but not in fine-tuned model.")
            differences_found = True
            break

    if not differences_found:
        print("  - No differences found. The model weights are identical.")
    else:
        print(
            "\nConclusion: The model has been successfully fine-tuned, as its weights have changed."
        )


if __name__ == "__main__":
    # --- Define the paths to your models ---

    # This will be downloaded from the Hugging Face Hub
    original_model_path = "paraphrase-MiniLM-L6-v2"

    # This should be the path to your local fine-tuned model
    finetuned_model_path = "./models/paraphrase-MiniLM-L6-v2-ai-detector-incomplete"

    # --- Load and display details for each model ---
    original_model = print_model_details("Original Model", original_model_path)
    finetuned_model = print_model_details("Fine-tuned Model", finetuned_model_path)

    # --- Compare the weights to verify fine-tuning ---
    compare_model_weights(original_model, finetuned_model)
