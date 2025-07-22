# Comparative Analysis of Pre-trained Transformer Models for Early Detection of AI-Generated Text in Online Media

## Core Concepts

- **Dual-Text Difference Features (DTDF)**: Instead of just analyzing the text itself, this method first prompts an LLM to revise the text. Then, a sophisticated CNN-LSTM network analyzes the semantic embeddings of both the original and revised versions to create a high-dimensional feature vector that represents the "footprint" of the AI's revision behavior.
- **Dual-Stream Feature Fusion (DSFF)**: This is the main model architecture proposed in the paper. It consists of two parallel CNN-LSTM streams: one processes the original text's semantic embedding, and the other processes the DTDF vector. The outputs are then intelligently fused using an attention mechanism to make a final classification, leveraging both the text's content and the AI's behavioral patterns.

## File Descriptions

This project includes several key files for experimentation:

- **`DTDF_DSFF_local.py` (Main Experiment File)**

  - **Purpose**: This is the primary script for running the final, advanced model proposed in the paper.
  - **Functionality**: It implements the complete DTDF-DSFF framework. It performs text revision, extracts advanced DTDF features using the CNN-LSTM network, and trains the dual-stream DSFF model for classification. It also includes a comparison against the baseline statistical model.
  - **Use Case**: Run this file to replicate the main results and comparisons presented in the paper.

- **`ai_local.py` (dual text test File)**

  - **Purpose**: It is intended to compare the semantic similarity before and after GPT3 rewriting with other features, and finally classify the final human or AI text.
  - **Functionality**: Instead of the advanced DTDF network, this model relies on a set of manually engineered **statistical features** (e.g., word overlap, sentence length changes, punctuation differences) combined with a `semantic_similarity` score. It trains a `RandomForestClassifier` on these features.
  - **Use Case**: Run this file to evaluate the performance of a traditional feature-based detection method as a benchmark against the more advanced DSFF model.

- **`ai-decter-trans.ipynb` (dual stream test file)**

  - **Purpose**: This is the main Jupyter Notebook used for the development, exploration, and step-by-step testing of the project.
  - **Functionality**: It contains the detailed code, experiments, and evolution of the project, from initial feature extraction ideas to the final implementation of the DTDF-DSFF framework.
  - **Use Case**: Use this notebook to understand the project's development history, inspect intermediate outputs, and experiment with individual components of the system.

- **`baseline.py` (baseline test file)**
  - **Purpose**: This script is designed to run a baseline experiment using a simpler, feature-based approach.

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
