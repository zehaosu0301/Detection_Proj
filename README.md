# Detection_Proj

Comparative Analysis of Pre-trained Transformer Models for Early Detection of AI-Generated Text in Online Media

# Enhancing AI-Generated Text Detection with DTDF and DSFF

This project presents an advanced framework for detecting AI-generated text, based on the findings of the paper, "Enhancing AI-Generated Text Detection with Dual-Text Difference Features and a Dual-Stream Fusion Framework." It introduces two core innovations: **Dual-Text Difference Features (DTDF)**, which capture the subtle semantic changes from AI-driven text revision, and a **Dual-Stream Feature Fusion (DSFF)** architecture that combines these difference features with the text's original semantic content for a more robust classification.

This repository contains the code to replicate the paper's experiments, including the implementation of the final DTDF-DSFF model and a baseline model for comparison.

## Core Concepts

- **Dual-Text Difference Features (DTDF)**: Instead of just analyzing the text itself, this method first prompts an LLM to revise the text. Then, a sophisticated CNN-LSTM network analyzes the semantic embeddings of both the original and revised versions to create a high-dimensional feature vector that represents the "footprint" of the AI's revision behavior.
- **Dual-Stream Feature Fusion (DSFF)**: This is the main model architecture proposed in the paper. It consists of two parallel CNN-LSTM streams: one processes the original text's semantic embedding, and the other processes the DTDF vector. The outputs are then intelligently fused using an attention mechanism to make a final classification, leveraging both the text's content and the AI's behavioral patterns.

## File Descriptions

This project includes several key files for experimentation:

- **`DTDF_DSFF_local.py` (Main Experiment File)**

  - **Purpose**: This is the primary script for running the final, advanced model proposed in the paper.
  - **Functionality**: It implements the complete DTDF-DSFF framework. It performs text revision, extracts advanced DTDF features using the CNN-LSTM network, and trains the dual-stream DSFF model for classification. It also includes a comparison against the baseline statistical model.
  - **Use Case**: Run this file to replicate the main results and comparisons presented in the paper.

- **`ai_local.py` (Baseline Model - Statistical Features Only)**

  - **Purpose**: This script is designed to run a baseline experiment using a simpler, feature-based approach.
  - **Functionality**: Instead of the advanced DTDF network, this model relies on a set of manually engineered **statistical features** (e.g., word overlap, sentence length changes, punctuation differences) combined with a `semantic_similarity` score. It trains a `RandomForestClassifier` on these features.
  - **Use Case**: Run this file to evaluate the performance of a traditional feature-based detection method as a benchmark against the more advanced DSFF model.

- **`ai-decter-trans.ipynb`**
  - **Purpose**: This is the main Jupyter Notebook used for the development, exploration, and step-by-step testing of the project.
  - **Functionality**: It contains the detailed code, experiments, and evolution of the project, from initial feature extraction ideas to the final implementation of the DTDF-DSFF framework.
  - **Use Case**: Use this notebook to understand the project's development history, inspect intermediate outputs, and experiment with individual components of the system.

## Setup and Installation

Follow these steps to set up the environment for running the experiments.

**1. Clone the Repository**

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

**2. Create a Python Environment**

It is highly recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install Dependencies**

Install all required libraries using pip:

```bash
pip install torch transformers sentence-transformers scikit-learn pandas numpy matplotlib seaborn tqdm openai python-dotenv nlpaug
```

**4. Download NLTK Data**

The project requires specific data packages from the Natural Language Toolkit (NLTK). The scripts will attempt to download these automatically, but you can also do it manually in a Python interpreter:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('omw-1.4')
```

**5. Set Up API Key**

The text revision process uses the OpenAI API. Create a `.env` file in the root directory of the project and add your credentials:

```
OPENAI_API_KEY="your_api_key_here"
OPENAI_BASE_URL="your_base_url_here" # Optional, if you use a proxy
```

## How to Run the Experiments

All output, including logs (`.log`) and result plots (`.png`), will be saved to a `./result/` directory, which will be created automatically.

### Running the Main Comparison Experiment

This is the recommended way to run the project. It will execute the main DTDF-DSFF model and compare it against the statistical baseline, as described in the paper.

Execute the main script from your terminal:

```bash
python DTDF_DSFF_local.py
```

This will:

1.  Load the specified dataset.
2.  Run the full training and evaluation pipeline for the advanced DTDF-DSFF model.
3.  Run the training and evaluation for the baseline statistical model.
4.  Print a final summary table comparing the performance of both methods.
5.  Generate and save comparison plots (ROC curves, confusion matrices, etc.) to the `./result/` folder.

### Running Only the Baseline Model

If you wish to only run the simpler, statistical feature-based model, execute its corresponding script:

```bash
python ai_local.py
```

## Expected Results

After running the experiments, you can find the following in the `./result/` directory:

- **`dtdf_dsff_output.log`**: A detailed log file containing all console output, including training progress, epoch-by-epoch validation scores, final performance metrics, and the summary table.
- **`roc_curves_comparison.png`**: A plot comparing the ROC curves and AUC scores of the different models.
- **`confusion_matrices_comparison.png`**: Plots of the confusion matrices for each model.
- **`classification_metrics_comparison.png`**: Bar charts comparing key metrics like Precision, Recall, and F1-Score.
- **`best_dtdf_dsff_model.pt`**: The saved weights of the best-performing DTDF-DSFF model from the training run.
