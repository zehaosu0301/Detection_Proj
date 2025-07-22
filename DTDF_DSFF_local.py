"""
Complete DTDF-DSFF AI Text Detection System
This implementation combines:
1. All functionality from ai-detector-trans.ipynb
2. DTDF (Deep Text Difference Features) Network
3. DSFF (Deep Semantic Feature Fusion) Model
"""

import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier
import os
import json
import logging
import sys
import hashlib
import random
import time
import pickle
from dataclasses import dataclass, field
from tqdm.auto import tqdm

# Transformers and NLP libraries
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
)
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader

# OpenAI
import openai
from openai import AsyncOpenAI
import tiktoken

# NLTK
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# TextAttack for perturbations
try:
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas
except ImportError:
    print("Installing nlpaug...")
    os.system("pip install nlpaug")
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas

# Download NLTK data
try:
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("stopwords", quiet=True)
except:
    print("NLTK downloads completed with some warnings")


# 1. CONFIGURATION AND LOGGING
@dataclass
class DetectionConfig:
    """检测配置类"""

    # Model selection
    revision_model: str = "t5-small"
    embedding_model: str = "all-MiniLM-L6-v2"

    # API configuration
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL")
    )

    # Perturbation parameters
    perturbation_rate: float = 0.15
    perturbation_methods: List[str] = field(
        default_factory=lambda: ["synonym", "contextual"]
    )

    # Detection parameters
    similarity_threshold: float = 0.95
    use_ml_classifier: bool = True

    # Batch processing
    batch_size: int = 16
    max_length: int = 512

    def __post_init__(self):
        if self.perturbation_methods is None:
            self.perturbation_methods = ["synonym", "contextual", "backtranslation"]


def setup_logging(output_dir: str = "./result"):
    """设置日志系统"""
    log_file_path = os.path.join(output_dir, "dtdf_dsff_output.log")
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logging.info(f"Logging setup complete. Output saved to {log_file_path}")


# 2. ENHANCED TEXT PERTURBER
class EnhancedTextPerturber:
    """增强的文本扰动器"""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self._init_augmenters()

    def _init_augmenters(self):
        """初始化各种扰动器"""
        self.augmenters = {
            "synonym": naw.SynonymAug(
                aug_src="wordnet", aug_p=self.config.perturbation_rate
            ),
            "contextual": naw.ContextualWordEmbsAug(
                model_path="distilbert-base-uncased",
                action="substitute",
                aug_p=self.config.perturbation_rate,
            ),
            "random_swap": naw.RandomWordAug(
                action="swap", aug_p=self.config.perturbation_rate
            ),
            "spelling": naw.SpellingAug(aug_p=self.config.perturbation_rate),
        }

    def perturb(self, text: str, method: Optional[str] = None) -> str:
        """智能扰动策略"""
        if method is None:
            method = np.random.choice(self.config.perturbation_methods)

        try:
            if method == "backtranslation":
                return self._backtranslate(text)
            elif method in self.augmenters:
                augmented = self.augmenters[method].augment(text)
                return augmented[0] if isinstance(augmented, list) else augmented
            else:
                return self._mixed_perturbation(text)
        except Exception as e:
            print(f"Perturbation failed: {e}")
            return self._simple_perturb(text)

    def _mixed_perturbation(self, text: str) -> str:
        """混合多种扰动方法"""
        sentences = sent_tokenize(text)
        perturbed_sentences = []

        for sent in sentences:
            method = np.random.choice(list(self.augmenters.keys()))
            try:
                perturbed = self.augmenters[method].augment(sent)
                perturbed_sent = (
                    perturbed[0] if isinstance(perturbed, list) else perturbed
                )
                perturbed_sentences.append(perturbed_sent)
            except:
                perturbed_sentences.append(sent)

        return " ".join(perturbed_sentences)

    def _simple_perturb(self, text: str) -> str:
        """简单扰动作为后备"""
        words = text.split()
        num_changes = max(1, int(len(words) * self.config.perturbation_rate))
        indices = np.random.choice(
            range(len(words)), size=min(num_changes, len(words)), replace=False
        )

        for idx in indices:
            word = words[idx]
            if len(word) > 3:
                words[idx] = word[:-1] + np.random.choice(list("aeiou"))

        return " ".join(words)

    def _backtranslate(self, text: str) -> str:
        """反向翻译（需要翻译API）"""
        return text


# 3. ENHANCED LLM REVISER
class EnhancedLLMReviser:
    """增强的LLM重写器"""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = None
        self.model = None
        self.tokenizer = None
        self._init_reviser()

    def _init_reviser(self):
        """根据配置初始化重写模型或API客户端"""
        model_name = self.config.revision_model

        if model_name.startswith("t5"):
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device).eval()
            print(f"Local T5 model loaded: {model_name}")

        elif model_name == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.to(self.device).eval()
            print(f"Local GPT-2 model loaded: {model_name}")

        elif model_name.startswith("gpt-"):
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "请在DetectionConfig中提供api_key或设置OPENAI_API_KEY环境变量"
                )

            self.client = AsyncOpenAI(api_key=api_key, base_url=self.config.base_url)
            self.max_retries = 3
            self.retry_delay = 1.0
            print(f"OpenAI client initialized for model: {model_name}")
            if self.config.base_url:
                print(f"Using custom base URL: {self.config.base_url}")
        else:
            print(
                f"Warning: Unknown revision model '{model_name}'. Will use rule-based fallback."
            )

    async def revise(
        self, original_text: str, perturbed_text: str, cache: Dict[str, str]
    ) -> str:
        """使用LLM重写文本"""
        cache_key = hashlib.md5(
            f"{original_text}_{self.config.revision_model}".encode()
        ).hexdigest()

        if cache_key in cache:
            return cache[cache_key]

        try:
            model_name = self.config.revision_model
            revised = ""

            if model_name.startswith("t5") or model_name == "gpt2":
                if model_name.startswith("t5"):
                    revised = self._revise_with_t5(perturbed_text)
                else:  # gpt2
                    revised = self._revise_with_gpt2(perturbed_text)
            elif model_name.startswith("gpt-"):
                revised = await self._revise_with_api(perturbed_text)
            else:
                revised = self._rule_based_revision(perturbed_text)

            cache[cache_key] = revised
            return revised

        except Exception as e:
            print(f"Revision failed with model {self.config.revision_model}: {e}")
            fallback_revision = self._rule_based_revision(perturbed_text)
            cache[cache_key] = fallback_revision
            return fallback_revision

    def _revise_with_t5(self, text: str) -> str:
        """使用T5模型重写"""
        prompt = f"paraphrase: {text}"
        inputs = self.tokenizer.encode(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                num_beams=4,
                early_stopping=True,
                max_length=256,
                min_length=40,
                repetition_penalty=1.2,
            )

        revised = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return revised

    def _revise_with_gpt2(self, text: str) -> str:
        """使用GPT-2模型重写"""
        try:
            prompt_templates = [
                f"Paraphrase the following text while keeping the same meaning: {text}\n\nParaphrased version:",
                f"Rewrite this sentence in a different way: {text}\n\nRewritten:",
                f"Express this differently: {text}\n\nAlternative expression:",
            ]
            prompt = np.random.choice(prompt_templates)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=min(len(input_ids[0]) + 100, 1024),
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                )

            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            rewritten = ""
            for marker in [
                "Paraphrased version:",
                "Rewritten:",
                "Alternative expression:",
            ]:
                if marker in full_text:
                    rewritten = full_text.split(marker)[-1].strip()
                    break

            if not rewritten:
                rewritten = full_text[len(prompt) :].strip()

            if "\n" in rewritten:
                rewritten = rewritten.split("\n")[0]
            if "." in rewritten:
                rewritten = rewritten.split(".")[0] + "."

            return rewritten if rewritten else text

        except Exception as e:
            print(f"GPT-2 revision failed: {e}")
            return text

    async def _revise_with_api(self, text: str) -> str:
        """使用OpenAI API重写"""
        if not self.client:
            raise ValueError("OpenAI client not initialized.")

        system_prompt = """You are a professional editor. Your task is to rewrite the given text to make it more formal, standardized, and academic in style.

Key guidelines:
- Replace casual expressions with formal language
- Use precise technical terms where appropriate
- Maintain professional tone throughout
- Standardize sentence structure
- Remove colloquialisms and personal expressions
- Keep the core meaning but express it in a more refined way

The more casual or informal the original text, the more changes you should make."""

        user_prompt = f"Please rewrite this text in a formal, academic style:\n\n{text}"

        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.revision_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    top_p=0.9,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                print(f"API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    return text

        return text

    def _rule_based_revision(self, text: str) -> str:
        """基于规则的重写"""
        import re

        sentences = sent_tokenize(text)
        if len(sentences) > 2:
            sentences = self._reorder_sentences(sentences)

        revised_sentences = []
        for sent in sentences:
            synonyms = {
                "important": "significant",
                "however": "nevertheless",
                "therefore": "thus",
                "because": "since",
                "many": "numerous",
                "show": "demonstrate",
            }

            for word, syn in synonyms.items():
                sent = re.sub(r"\b" + word + r"\b", syn, sent, flags=re.IGNORECASE)

            revised_sentences.append(sent)

        return " ".join(revised_sentences)

    def _reorder_sentences(self, sentences: List[str]) -> List[str]:
        """智能句子重排"""
        if len(sentences) <= 2:
            return sentences

        first = sentences[0]
        last = sentences[-1]
        middle = sentences[1:-1]
        np.random.shuffle(middle)

        return [first] + middle + [last]


# 4. DTDF NETWORK
class DTDFNetwork(nn.Module):
    """
    DTDF (Deep Text Difference Features) Network

    Paper: Figure 3, Figure 5
    Input: V_original, V_revised (embedding vectors)
    Output: 384-dimensional DTDF feature vector

    Architecture: CNN-LSTM network for processing difference between embeddings
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        cnn_out_channels: int = 128,
        lstm_hidden_size: int = 192,
        dtdf_output_dim: int = 384,
    ):
        super(DTDFNetwork, self).__init__()

        # Input processing: combine two embedding vectors
        self.input_processor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim), nn.ReLU(), nn.Dropout(0.1)
        )

        # CNN layers for local feature extraction
        self.conv_layers = nn.Sequential(
            # First CNN layer
            nn.Conv1d(
                in_channels=1, out_channels=cnn_out_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
            nn.Dropout(0.1),
            # Second CNN layer
            nn.Conv1d(
                in_channels=cnn_out_channels,
                out_channels=cnn_out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
            nn.Dropout(0.1),
            # Third CNN layer
            nn.Conv1d(
                in_channels=cnn_out_channels,
                out_channels=cnn_out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
        )

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        # Output projection to 384-dim DTDF
        lstm_output_dim = lstm_hidden_size * 2  # bidirectional
        self.dtdf_projector = nn.Sequential(
            nn.Linear(lstm_output_dim, dtdf_output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dtdf_output_dim, dtdf_output_dim),
            nn.Tanh(),  # Normalize output
        )

        self.dtdf_output_dim = dtdf_output_dim

    def forward(
        self, v_original: torch.Tensor, v_revised: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        Args:
            v_original: Original text embeddings [batch_size, embedding_dim]
            v_revised: Revised text embeddings [batch_size, embedding_dim]
        Returns:
            dtdf_features: DTDF feature vector [batch_size, 384]
        """
        batch_size = v_original.size(0)

        # 1. Combine original and revised embeddings
        combined = torch.cat(
            [v_original, v_revised], dim=1
        )  # [batch_size, embedding_dim*2]
        processed = self.input_processor(combined)  # [batch_size, embedding_dim]

        # 2. Add channel dimension for CNN
        cnn_input = processed.unsqueeze(1)  # [batch_size, 1, embedding_dim]

        # 3. CNN feature extraction
        cnn_output = self.conv_layers(
            cnn_input
        )  # [batch_size, cnn_out_channels, embedding_dim]

        # 4. Prepare for LSTM (transpose dimensions)
        lstm_input = cnn_output.permute(
            0, 2, 1
        )  # [batch_size, seq_len, cnn_out_channels]

        # 5. LSTM processing
        lstm_output, (h_n, c_n) = self.lstm(lstm_input)

        # 6. Use final hidden state from both directions
        final_hidden = torch.cat(
            [h_n[-2], h_n[-1]], dim=1
        )  # [batch_size, lstm_hidden_size*2]

        # 7. Project to 384-dimensional DTDF features
        dtdf_features = self.dtdf_projector(final_hidden)

        return dtdf_features


# 5. DSFF MODEL (PAPER IMPLEMENTATION)
class DSFFModel(nn.Module):
    """
    DSFF (Deep Semantic Feature Fusion) Model

    Paper: Figure 6
    Architecture: Dual-path CNN-LSTM with feature fusion

    Path A: Processes original text embeddings
    Path B: Processes DTDF features
    Fusion: Combines both paths for final classification
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        dtdf_dim: int = 384,
        cnn_out_channels: int = 128,
        lstm_hidden_size: int = 192,
        fusion_hidden_size: int = 256,
        num_classes: int = 2,
    ):
        super(DSFFModel, self).__init__()

        # Path A: Original text embedding branch
        self.path_a_conv = nn.Sequential(
            nn.Conv1d(1, cnn_out_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels // 2),
            nn.Dropout(0.1),
            nn.Conv1d(
                cnn_out_channels // 2, cnn_out_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
            nn.Dropout(0.1),
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
        )

        self.path_a_lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        # Path B: DTDF feature branch
        self.path_b_conv = nn.Sequential(
            nn.Conv1d(1, cnn_out_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels // 2),
            nn.Dropout(0.1),
            nn.Conv1d(
                cnn_out_channels // 2, cnn_out_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
            nn.Dropout(0.1),
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
        )

        self.path_b_lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        # Feature fusion layers
        lstm_output_dim = lstm_hidden_size * 2  # bidirectional
        fusion_input_dim = lstm_output_dim * 2  # two paths

        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_hidden_size),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_hidden_size // 2),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden_size // 2, fusion_hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Classification head
        self.classifier = nn.Linear(fusion_hidden_size // 4, num_classes)

        # Attention mechanism for fusion
        self.attention = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_input_dim // 2, 2),  # weights for two paths
            nn.Softmax(dim=1),
        )

    def forward(
        self, original_embeddings: torch.Tensor, dtdf_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through DSFF model
        Args:
            original_embeddings: [batch_size, embedding_dim]
            dtdf_features: [batch_size, dtdf_dim]
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size = original_embeddings.size(0)

        # Path A: Process original embeddings
        path_a_input = original_embeddings.unsqueeze(
            1
        )  # [batch_size, 1, embedding_dim]
        path_a_conv_out = self.path_a_conv(
            path_a_input
        )  # [batch_size, channels, seq_len]
        path_a_lstm_input = path_a_conv_out.permute(
            0, 2, 1
        )  # [batch_size, seq_len, channels]
        _, (path_a_h_n, _) = self.path_a_lstm(path_a_lstm_input)
        path_a_output = torch.cat([path_a_h_n[-2], path_a_h_n[-1]], dim=1)

        # Path B: Process DTDF features
        path_b_input = dtdf_features.unsqueeze(1)  # [batch_size, 1, dtdf_dim]
        path_b_conv_out = self.path_b_conv(
            path_b_input
        )  # [batch_size, channels, seq_len]
        path_b_lstm_input = path_b_conv_out.permute(
            0, 2, 1
        )  # [batch_size, seq_len, channels]
        _, (path_b_h_n, _) = self.path_b_lstm(path_b_lstm_input)
        path_b_output = torch.cat([path_b_h_n[-2], path_b_h_n[-1]], dim=1)

        # Feature fusion with attention
        concatenated = torch.cat([path_a_output, path_b_output], dim=1)
        attention_weights = self.attention(concatenated)  # [batch_size, 2]

        # Apply attention weights
        weighted_path_a = path_a_output * attention_weights[:, 0:1]
        weighted_path_b = path_b_output * attention_weights[:, 1:2]
        fused_features = torch.cat([weighted_path_a, weighted_path_b], dim=1)

        # Final classification
        fusion_output = self.fusion_layers(fused_features)
        logits = self.classifier(fusion_output)

        return logits


# 6. STATISTICAL FEATURE EXTRACTOR
class MultiFeatureExtractor:
    """多维度特征提取器（保留统计特征用于对比）"""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.sentence_model = SentenceTransformer(config.embedding_model)
        self._init_linguistic_features()

        logging.info(
            f"Multi-Feature Extractor initialized with {config.embedding_model}"
        )
        total_params = sum(
            p.numel() for p in self.sentence_model.parameters() if p.requires_grad
        )
        logging.info(f"Total trainable parameters: {total_params:,}")

    def _init_linguistic_features(self):
        """初始化语言学特征提取"""
        try:
            self.stop_words = set(stopwords.words("english"))
        except:
            self.stop_words = set()

    def extract_features(self, original: str, revised: str) -> Dict[str, float]:
        """提取多维度特征"""
        features = {}

        # 1. Semantic similarity
        features["semantic_similarity"] = self._compute_semantic_similarity(
            original, revised
        )

        # 2. Lexical features
        features.update(self._extract_lexical_features(original, revised))

        # 3. Syntactic features
        features.update(self._extract_syntactic_features(original, revised))

        # 4. Stylistic features
        features.update(self._extract_stylistic_features(original, revised))

        # 5. Edit distance features
        features.update(self._extract_edit_features(original, revised))

        return features

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度"""
        emb1 = self.sentence_model.encode(text1)
        emb2 = self.sentence_model.encode(text2)

        from sklearn.metrics.pairwise import cosine_similarity

        return float(cosine_similarity([emb1], [emb2])[0, 0])

    def _extract_lexical_features(
        self, original: str, revised: str
    ) -> Dict[str, float]:
        """提取词汇级特征"""
        orig_words = set(word_tokenize(original.lower()))
        rev_words = set(word_tokenize(revised.lower()))

        overlap = len(orig_words & rev_words)
        total = len(orig_words | rev_words)

        features = {
            "word_overlap_ratio": overlap / total if total > 0 else 0,
            "vocabulary_change_ratio": (
                len(orig_words ^ rev_words) / total if total > 0 else 0
            ),
            "unique_words_ratio": (
                len(rev_words - orig_words) / len(rev_words) if rev_words else 0
            ),
        }

        # Stopword ratio change
        orig_stop = len([w for w in orig_words if w in self.stop_words])
        rev_stop = len([w for w in rev_words if w in self.stop_words])

        features["stopword_ratio_change"] = abs(
            (orig_stop / len(orig_words) if orig_words else 0)
            - (rev_stop / len(rev_words) if rev_words else 0)
        )

        return features

    def _extract_syntactic_features(
        self, original: str, revised: str
    ) -> Dict[str, float]:
        """提取句法特征"""
        orig_sents = sent_tokenize(original)
        rev_sents = sent_tokenize(revised)

        features = {
            "sentence_count_ratio": (
                len(rev_sents) / len(orig_sents) if orig_sents else 1
            ),
            "avg_sentence_length_change": (
                abs(
                    np.mean([len(word_tokenize(s)) for s in orig_sents])
                    - np.mean([len(word_tokenize(s)) for s in rev_sents])
                )
                if orig_sents and rev_sents
                else 0
            ),
        }

        # POS tag distribution changes
        try:
            orig_pos = nltk.pos_tag(word_tokenize(original))
            rev_pos = nltk.pos_tag(word_tokenize(revised))

            for pos_type in ["NN", "VB", "JJ", "RB"]:
                orig_count = sum(1 for _, pos in orig_pos if pos.startswith(pos_type))
                rev_count = sum(1 for _, pos in rev_pos if pos.startswith(pos_type))

                features[f"pos_{pos_type}_ratio_change"] = abs(
                    (orig_count / len(orig_pos) if orig_pos else 0)
                    - (rev_count / len(rev_pos) if rev_pos else 0)
                )
        except:
            pass

        return features

    def _extract_stylistic_features(
        self, original: str, revised: str
    ) -> Dict[str, float]:
        """提取风格特征"""
        features = {}

        # Punctuation usage changes
        orig_punct = sum(1 for c in original if c in ".,!?;:")
        rev_punct = sum(1 for c in revised if c in ".,!?;:")

        features["punctuation_ratio_change"] = abs(
            (orig_punct / len(original) if original else 0)
            - (rev_punct / len(revised) if revised else 0)
        )

        # Capitalization ratio changes
        orig_caps = sum(1 for c in original if c.isupper())
        rev_caps = sum(1 for c in revised if c.isupper())

        features["capitalization_ratio_change"] = abs(
            (orig_caps / len(original) if original else 0)
            - (rev_caps / len(revised) if revised else 0)
        )

        # Average word length changes
        orig_words = word_tokenize(original)
        rev_words = word_tokenize(revised)

        features["avg_word_length_change"] = (
            abs(
                np.mean([len(w) for w in orig_words])
                - np.mean([len(w) for w in rev_words])
            )
            if orig_words and rev_words
            else 0
        )

        return features

    def _extract_edit_features(self, original: str, revised: str) -> Dict[str, float]:
        """提取编辑距离相关特征"""
        from difflib import SequenceMatcher

        # Character-level similarity
        char_similarity = SequenceMatcher(None, original, revised).ratio()

        # Word-level similarity
        orig_words = word_tokenize(original)
        rev_words = word_tokenize(revised)
        word_similarity = SequenceMatcher(None, orig_words, rev_words).ratio()

        features = {
            "char_level_similarity": char_similarity,
            "word_level_similarity": word_similarity,
            "length_ratio": len(revised) / len(original) if original else 1,
        }

        return features


# 7. DTDF FEATURE EXTRACTOR
class DTDFFeatureExtractor:
    """DTDF特征提取器"""

    def __init__(self, config: DetectionConfig, device: str):
        self.device = device
        self.config = config

        # Initialize sentence embedding model
        self.sentence_model = SentenceTransformer(config.embedding_model, device=device)
        embedding_dim = self.sentence_model.get_sentence_embedding_dimension()

        # Initialize DTDF network
        self.dtdf_network = DTDFNetwork(embedding_dim=embedding_dim).to(device)
        self.dtdf_network.eval()

        logging.info(
            f"DTDF Feature Extractor initialized with embedding dim: {embedding_dim}"
        )

    def extract_dtdf_features(
        self, original_texts: List[str], revised_texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract both original embeddings and DTDF features
        Returns:
            original_embeddings: [n_samples, embedding_dim]
            dtdf_features: [n_samples, 384]
        """
        with torch.no_grad():
            # Generate text embeddings
            logging.info("Generating original text embeddings...")
            original_embeddings = self.sentence_model.encode(
                original_texts,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=True,
            )

            logging.info("Generating revised text embeddings...")
            revised_embeddings = self.sentence_model.encode(
                revised_texts,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=True,
            )

            # Extract DTDF features
            logging.info("Extracting DTDF features...")
            dtdf_features = self.dtdf_network(original_embeddings, revised_embeddings)

            return original_embeddings.cpu().numpy(), dtdf_features.cpu().numpy()


# 8. CUSTOM DATA LOADER
class CustomDataLoader:
    """数据加载器类"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logging.info(f"Created data directory: {data_dir}")

    def load_data(self, path: str) -> pd.DataFrame:
        """读取数据集"""
        if path == "finance":
            return self.load_finance_dataset()
        elif path == "sample":
            return self.create_sample_dataset()
        elif path == "test":
            return self.load_origin_dataset()

        # Handle file paths
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(self.data_dir, path)

        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path)
            elif path.endswith(".jsonl"):
                df = pd.read_json(path, lines=True)
            elif path.endswith(".json"):
                df = pd.read_json(path)
            else:
                raise ValueError(f"Unsupported file format: {path}")

            # Ensure required columns
            if "text" not in df.columns:
                raise ValueError("Missing 'text' column in dataset")
            if "label" not in df.columns:
                raise ValueError("Missing 'label' column in dataset")

            # Add ID column if missing
            if "id" not in df.columns:
                df["id"] = range(len(df))

            # Data cleaning
            df = df.dropna(subset=["text", "label"])
            df = df[df["text"].str.len() > 10]  # Filter very short texts

            logging.info(f"Loaded {len(df)} samples from {path}")
            logging.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

            return df[["id", "text", "label"]]

        except FileNotFoundError:
            logging.warning(f"File {path} not found. Creating sample dataset...")
            return self.create_sample_dataset()

    def find_finance_data_files(self) -> Dict[str, str]:
        """查找金融数据集文件"""
        logging.info("Searching for finance data files...")

        possible_paths = [
            self.data_dir,
            os.path.join(self.data_dir, "finance"),
            os.path.join(self.data_dir, "finance-dataset"),
            "./finance_data",
            ".",
            "./data",
        ]

        files_found = {}

        for path in possible_paths:
            if os.path.exists(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if "revised_human_finance" in file or "revised_human" in file:
                            files_found["revised_human"] = os.path.join(root, file)
                        elif (
                            "revised_chatgpt_finance" in file
                            or "revised_chatgpt" in file
                        ):
                            files_found["revised_chatgpt"] = os.path.join(root, file)
                        elif "finance.jsonl" in file:
                            files_found["original"] = os.path.join(root, file)

        return files_found

    def load_finance_dataset(self) -> pd.DataFrame:
        """加载金融数据集"""
        files = self.find_finance_data_files()

        logging.info("Loading finance datasets...")
        logging.info(f"Found files: {list(files.keys())}")

        all_texts = []
        all_labels = []

        # Load revised human texts
        if "revised_human" in files:
            try:
                logging.info(
                    f"Loading revised human texts from: {files['revised_human']}"
                )
                with open(files["revised_human"], "r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                item = json.loads(line)
                                for idx, text in item.items():
                                    cleaned_text = (
                                        text.strip()
                                        .replace("\\n\\n", " ")
                                        .replace("\\n", " ")
                                    )
                                    if len(cleaned_text) > 20:
                                        all_texts.append(cleaned_text)
                                        all_labels.append(0)  # 0 = human
                            except json.JSONDecodeError as e:
                                logging.warning(f"Line {line_no} parse error: {e}")
                logging.info(
                    f"Loaded {len([l for l in all_labels if l == 0])} human texts"
                )
            except Exception as e:
                logging.error(f"Error loading revised human texts: {e}")

        # Load revised ChatGPT texts
        if "revised_chatgpt" in files:
            try:
                logging.info(
                    f"Loading revised ChatGPT texts from: {files['revised_chatgpt']}"
                )
                with open(files["revised_chatgpt"], "r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                item = json.loads(line)
                                for idx, text in item.items():
                                    cleaned_text = (
                                        text.strip()
                                        .replace("\\n\\n", " ")
                                        .replace("\\n", " ")
                                    )
                                    if len(cleaned_text) > 20:
                                        all_texts.append(cleaned_text)
                                        all_labels.append(1)  # 1 = chatgpt
                            except json.JSONDecodeError as e:
                                logging.warning(f"Line {line_no} parse error: {e}")
                logging.info(
                    f"Loaded {len([l for l in all_labels if l == 1])} ChatGPT texts"
                )
            except Exception as e:
                logging.error(f"Error loading revised ChatGPT texts: {e}")

        # Create DataFrame
        df = pd.DataFrame(
            {"id": range(len(all_texts)), "text": all_texts, "label": all_labels}
        )

        # Shuffle data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df["id"] = range(len(df))

        logging.info(f"\n=== Dataset Summary ===")
        logging.info(f"Total samples: {len(df)}")
        logging.info(f"Human texts: {len(df[df['label'] == 0])}")
        logging.info(f"ChatGPT texts: {len(df[df['label'] == 1])}")

        return df[["id", "text", "label"]]

    def load_origin_dataset(self) -> pd.DataFrame:
        """Load original finance dataset from JSON Lines format"""
        files = self.find_finance_data_files()

        logging.info("Loading original finance datasets...")

        all_texts = []
        all_labels = []

        if "original" in files:
            finance_path = files["original"]
            logging.info(f"Loading from: {finance_path}")

            try:
                with open(finance_path, "r", encoding="utf-8") as f:
                    line_count = 0
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                            line_count += 1

                            # Extract human answers
                            human_answers = data.get("human_answers", [])
                            if (
                                human_answers
                                and len(human_answers) > 0
                                and human_answers[0]
                            ):
                                all_texts.append(human_answers[0])
                                all_labels.append(0)  # 0 for human

                            # Extract ChatGPT answers
                            chatgpt_answers = data.get("chatgpt_answers", [])
                            if (
                                chatgpt_answers
                                and len(chatgpt_answers) > 0
                                and chatgpt_answers[0]
                            ):
                                all_texts.append(chatgpt_answers[0])
                                all_labels.append(1)  # 1 for ChatGPT

                            if line_num % 1000 == 0:
                                logging.info(
                                    f"Processed {line_num} lines, extracted {len(all_texts)} texts"
                                )

                        except json.JSONDecodeError as e:
                            logging.error(f"Error parsing line {line_num}: {e}")
                            continue

                logging.info(f"Successfully loaded {line_count} JSON objects")
                logging.info(f"Total texts extracted: {len(all_texts)}")

            except FileNotFoundError:
                logging.error(f"File not found: {finance_path}")
                return self.create_sample_dataset()
            except Exception as e:
                logging.error(f"Error loading finance data: {e}")
                return self.create_sample_dataset()
        else:
            logging.warning("No finance data file found. Creating sample dataset...")
            return self.create_sample_dataset()

        if len(all_texts) == 0:
            logging.warning(
                "No texts extracted from finance data. Creating sample dataset..."
            )
            return self.create_sample_dataset()

        # Create DataFrame
        df = pd.DataFrame(
            {"id": range(len(all_texts)), "text": all_texts, "label": all_labels}
        )

        # Shuffle data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df["id"] = range(len(df))

        logging.info(f"\n=== Dataset Summary ===")
        logging.info(f"Total samples: {len(df)}")
        logging.info(f"Human texts: {len(df[df['label'] == 0])}")
        logging.info(f"ChatGPT texts: {len(df[df['label'] == 1])}")

        return df[["id", "text", "label"]]

    def create_sample_dataset(self) -> pd.DataFrame:
        """创建示例数据集用于测试"""
        sample_data = {
            "id": range(6),
            "text": [
                "There is most likely an error in the WSJ's data. Yahoo! Finance reports the PE on the Russell 2000 to be 15 as of 83111 and SP 500 PE to be 13 (about the same as WSJ). Good catch, though! E-mail WSJ, perhaps they will be grateful.",
                "I know this question has a lot of answers already, but I feel the answers are phrased either strongly against, or mildly for, co-signing. What it amounts down to is that this is a personal choice.",
                "I think the best investment strategy is to diversify your portfolio across different asset classes.",
                "Historical price-to-earnings (PE) ratios for small-cap and large-cap stocks can vary significantly over time and may not be directly comparable due to the different characteristics of these two categories of stocks.",
                "Co-signing a personal loan for a friend or family member can be a risky proposition. When you co-sign a loan, you are agreeing to be responsible for the loan if the borrower is unable to make the payments.",
                "The optimal approach to risk management involves careful assessment of market conditions.",
            ],
            "label": [0, 0, 0, 1, 1, 1],  # 0=human, 1=AI
        }

        df = pd.DataFrame(sample_data)
        logging.info(f"Created sample dataset with {len(df)} examples")
        return df


# 9. DTDF-DSFF DATASET AND DETECTOR
class DTDFDSFFDataset(Dataset):
    """DTDF-DSFF数据集"""

    def __init__(
        self,
        texts: List[str],
        original_embeddings: np.ndarray,
        dtdf_features: np.ndarray,
        labels: List[int],
    ):
        self.texts = texts
        self.original_embeddings = original_embeddings
        self.dtdf_features = dtdf_features
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "original_embeddings": torch.tensor(
                self.original_embeddings[idx], dtype=torch.float32
            ),
            "dtdf_features": torch.tensor(self.dtdf_features[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class AITextDetector:
    """基础AI文本检测器（用于特征提取和对比）"""

    def __init__(self, config: DetectionConfig, device: str):
        self.config = config
        self.device = device
        self.cache = {}
        self.perturber = EnhancedTextPerturber(config)
        self.reviser = EnhancedLLMReviser(config)
        self.feature_extractor = MultiFeatureExtractor(config)
        self.classifier = None
        self.scaler = StandardScaler()
        self.api_concurrency_limit = asyncio.Semaphore(10)

    async def detect_batch(
        self, texts: List[str], labels: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """批量检测（用于统计特征对比）"""
        all_features = []
        all_scores = []

        if not texts:
            return {
                "predictions": np.array([]),
                "probabilities": np.array([]),
                "features": np.array([]),
                "similarity_scores": np.array([]),
            }

        print("Processing texts...")

        # Perturbation
        perturbed_texts = [
            self.perturber.perturb(text) for text in tqdm(texts, desc="Perturbing")
        ]

        # Revision with concurrency control
        async def revise_with_limit(original_text, perturbed_text):
            async with self.api_concurrency_limit:
                return await self.reviser.revise(
                    original_text, perturbed_text, self.cache
                )

        print(f"Revising texts with {self.reviser.config.revision_model}...")
        revision_tasks = [
            revise_with_limit(orig, pert) for orig, pert in zip(texts, perturbed_texts)
        ]

        all_revised_texts = await asyncio.gather(*revision_tasks)

        # Feature extraction
        print("Extracting features...")
        for i, original_text in enumerate(texts):
            revised_text = all_revised_texts[i]
            features = self.feature_extractor.extract_features(
                original_text, revised_text
            )
            all_features.append(features)
            all_scores.append(features.get("semantic_similarity", 0.5))

        feature_matrix = pd.DataFrame(all_features).fillna(0).values

        if self.config.use_ml_classifier:
            if labels is not None and self.classifier is None:
                self._train_classifier(feature_matrix, labels)

            if self.classifier is not None:
                feature_matrix_scaled = self.scaler.transform(feature_matrix)
                predictions = self.classifier.predict(feature_matrix_scaled)
                probabilities = self.classifier.predict_proba(feature_matrix_scaled)[
                    :, 1
                ]
            else:
                predictions = (
                    np.array(all_scores) > self.config.similarity_threshold
                ).astype(int)
                probabilities = np.array(all_scores)
        else:
            predictions = (
                np.array(all_scores) > self.config.similarity_threshold
            ).astype(int)
            probabilities = np.array(all_scores)

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "features": feature_matrix,
            "similarity_scores": np.array(all_scores),
        }

    def _train_classifier(self, features: np.ndarray, labels: List[int]):
        """训练机器学习分类器"""
        print("Training classifier...")

        features_scaled = self.scaler.fit_transform(features)

        self.classifier = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )

        print(f"\nRandomForestClassifier Parameters:")
        print(self.classifier.get_params())

        self.classifier.fit(features_scaled, labels)

        # Feature importance
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        importances = self.classifier.feature_importances_

        print("\nTop 10 Most Important Features:")
        for feat, imp in sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        )[:10]:
            print(f"  {feat}: {imp:.4f}")


class DTDFDSFFDetector:
    """完整的DTDF-DSFF检测器"""

    def __init__(self, config: DetectionConfig, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        # Initialize components
        self.perturber = EnhancedTextPerturber(config)
        self.reviser = EnhancedLLMReviser(config)
        self.dtdf_extractor = DTDFFeatureExtractor(config, self.device)
        self.base_detector = AITextDetector(config, self.device)  # For comparison

        # Models and cache
        self.dsff_model = None
        self.cache = {}
        self.api_concurrency_limit = asyncio.Semaphore(10)

        logging.info(f"DTDF-DSFF Detector initialized on {self.device}")

    async def prepare_features(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare original embeddings and DTDF features
        Returns:
            original_embeddings: [n_samples, embedding_dim]
            dtdf_features: [n_samples, 384]
        """
        # 1. Perturbation
        logging.info("Applying perturbations...")
        perturbed_texts = []
        for text in tqdm(texts, desc="Perturbing texts"):
            perturbed = self.perturber.perturb(text)
            perturbed_texts.append(perturbed)

        # 2. Revision with concurrency control
        async def revise_with_limit(original_text, perturbed_text):
            async with self.api_concurrency_limit:
                return await self.reviser.revise(
                    original_text, perturbed_text, self.cache
                )

        logging.info(f"Revising texts with {self.reviser.config.revision_model}...")
        revision_tasks = [
            revise_with_limit(orig, pert) for orig, pert in zip(texts, perturbed_texts)
        ]

        revised_texts = await asyncio.gather(*revision_tasks)

        # 3. Extract DTDF features and original embeddings
        logging.info("Extracting DTDF features and original embeddings...")
        original_embeddings, dtdf_features = self.dtdf_extractor.extract_dtdf_features(
            texts, revised_texts
        )

        return original_embeddings, dtdf_features

    def prepare_datasets(
        self,
        texts: List[str],
        original_embeddings: np.ndarray,
        dtdf_features: np.ndarray,
        labels: List[int],
    ) -> Tuple[Dataset, Dataset]:
        """准备训练和验证数据集"""
        # Data split
        X_train, X_val, emb_train, emb_val, dtdf_train, dtdf_val, y_train, y_val = (
            train_test_split(
                texts,
                original_embeddings,
                dtdf_features,
                labels,
                test_size=0.2,
                random_state=42,
                stratify=labels,
            )
        )

        # Create datasets
        train_dataset = DTDFDSFFDataset(X_train, emb_train, dtdf_train, y_train)
        val_dataset = DTDFDSFFDataset(X_val, emb_val, dtdf_val, y_val)

        return train_dataset, val_dataset

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        num_epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 2e-4,
        save_path: str = "best_dtdf_dsff_model.pt",
    ):
        """训练DSFF模型"""
        # Initialize model
        if self.dsff_model is None:
            sample_embedding = train_dataset[0]["original_embeddings"]
            sample_dtdf = train_dataset[0]["dtdf_features"]

            self.dsff_model = DSFFModel(
                embedding_dim=sample_embedding.shape[0], dtdf_dim=sample_dtdf.shape[0]
            ).to(self.device)

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            self.dsff_model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )

        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 5

        for epoch in range(num_epochs):
            # Training phase
            self.dsff_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch in train_bar:
                original_emb = batch["original_embeddings"].to(self.device)
                dtdf_features = batch["dtdf_features"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                logits = self.dsff_model(original_emb, dtdf_features)
                loss = criterion(logits, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.dsff_model.parameters(), max_norm=1.0
                )

                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                train_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{100.*train_correct/train_total:.2f}%",
                    }
                )

            # Validation phase
            self.dsff_model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in tqdm(
                    val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"
                ):
                    original_emb = batch["original_embeddings"].to(self.device)
                    dtdf_features = batch["dtdf_features"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    logits = self.dsff_model(original_emb, dtdf_features)
                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100.0 * train_correct / train_total
            val_acc = 100.0 * val_correct / val_total

            logging.info(f"\nEpoch {epoch+1}:")
            logging.info(
                f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%"
            )
            logging.info(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            # Early stopping and model saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model(save_path)
                logging.info(f"  ✓ Saved best model to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logging.info(f"  Early stopping triggered after {epoch+1} epochs")
                    break

    async def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict new texts
        Returns:
            predictions: prediction labels
            probabilities: prediction probabilities
        """
        # Prepare features
        original_embeddings, dtdf_features = await self.prepare_features(texts)

        # Create dataset and loader
        dataset = DTDFDSFFDataset(
            texts, original_embeddings, dtdf_features, [0] * len(texts)  # dummy labels
        )
        loader = DataLoader(dataset, batch_size=32)

        self.dsff_model.eval()
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                original_emb = batch["original_embeddings"].to(self.device)
                dtdf_features = batch["dtdf_features"].to(self.device)

                logits = self.dsff_model(original_emb, dtdf_features)
                probs = F.softmax(logits, dim=1)

                predictions = torch.argmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(
                    probs[:, 1].cpu().numpy()
                )  # AI class probability

        return np.array(all_predictions), np.array(all_probabilities)

    def save_model(self, path: str):
        """保存模型"""
        torch.save(
            {
                "model_state_dict": self.dsff_model.state_dict(),
                "config": self.config,
                "cache": self.cache,
            },
            path,
        )

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)

        # Reinitialize model (need to know dimensions)
        # This should be adjusted based on actual use case
        self.dsff_model = DSFFModel().to(self.device)
        self.dsff_model.load_state_dict(checkpoint["model_state_dict"])
        self.dsff_model.eval()

        self.cache = checkpoint.get("cache", {})


# 10. PLOTTING AND EVALUATION FUNCTIONS
def plot_roc_curves(results: Dict, output_dir: str = "./result"):
    """绘制并保存ROC曲线"""
    plt.figure(figsize=(10, 8))
    for name, result_data in results.items():
        if "test_results" in result_data:
            fpr, tpr, _ = roc_curve(
                result_data["test_labels"], result_data["test_results"]["probabilities"]
            )
            auc = roc_auc_score(
                result_data["test_labels"], result_data["test_results"]["probabilities"]
            )
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(output_dir, "roc_curves_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_precision_recall_curves(results: Dict, output_dir: str = "./result"):
    """绘制并保存Precision-Recall曲线"""
    plt.figure(figsize=(10, 8))
    for name, result_data in results.items():
        if "test_results" in result_data:
            precision, recall, _ = precision_recall_curve(
                result_data["test_labels"], result_data["test_results"]["probabilities"]
            )
            plt.plot(recall, precision, label=f"{name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(output_dir, "precision_recall_curves_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_confusion_matrices(results: Dict, output_dir: str = "./result"):
    """绘制并保存混淆矩阵"""
    n_results = len(results)
    fig, axes = plt.subplots(1, n_results, figsize=(6 * n_results, 5), squeeze=False)

    for ax, (name, result_data) in zip(axes.flatten(), results.items()):
        if "test_results" in result_data:
            cm = confusion_matrix(
                result_data["test_labels"], result_data["test_results"]["predictions"]
            )
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                cbar=False,
                xticklabels=["Human", "AI"],
                yticklabels=["Human", "AI"],
            )
            ax.set_title(f"Confusion Matrix: {name}")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "confusion_matrices_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_classification_metrics(results: Dict, output_dir: str = "./result"):
    """绘制并保存分类指标对比"""
    metrics_data = []
    for name, result_data in results.items():
        if "test_results" in result_data:
            report = classification_report(
                result_data["test_labels"],
                result_data["test_results"]["predictions"],
                target_names=["Human", "AI"],
                output_dict=True,
            )
            ai_metrics = report["AI"]
            metrics_data.append(
                {
                    "Configuration": name,
                    "Precision": ai_metrics["precision"],
                    "Recall": ai_metrics["recall"],
                    "F1-Score": ai_metrics["f1-score"],
                }
            )

    df = pd.DataFrame(metrics_data)
    df_melted = df.melt(id_vars="Configuration", var_name="Metric", value_name="Score")

    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_melted, x="Configuration", y="Score", hue="Metric")
    plt.title('Classification Metrics for "AI" Class')
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "classification_metrics_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def find_optimal_threshold(labels: List[int], scores: np.ndarray) -> float:
    """找到最优分类阈值"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    logging.info(f"Found optimal similarity threshold: {optimal_threshold:.4f}")
    return optimal_threshold


# 11. MAIN WORKFLOW FUNCTIONS
async def run_dtdf_dsff_experiment(
    data_path: str = "sample",
    max_samples: Optional[int] = 300,
    num_epochs: int = 10,
    batch_size: int = 16,
    use_cache: bool = True,
):
    """运行完整的DTDF-DSFF实验"""

    setup_logging()
    logging.info("=== DTDF-DSFF AI Text Detection Experiment ===")

    # Set random seeds
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 1. Load data
    logging.info("Loading data...")
    loader = CustomDataLoader()
    df = loader.load_data(data_path)

    if max_samples:
        df = df.head(max_samples)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    logging.info(f"Dataset size: {len(texts)} samples")
    logging.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    # 2. Data split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=SEED, stratify=labels
    )

    logging.info(
        f"Training: {len(train_texts)} samples, Testing: {len(test_texts)} samples"
    )

    # 3. Initialize detector
    config = DetectionConfig(
        revision_model="gpt-3.5-turbo",  # Will fall back to rule-based if no API key
        embedding_model="./models/paraphrase-MiniLM-L6-v2-ai-detector-incomplete",
        perturbation_rate=0.15,
        use_ml_classifier=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = DTDFDSFFDetector(config, device)

    logging.info(f"file is : {os.getcwd()}")
    logging.info(f"file is : {os.listdir('.')}")

    # 4. Load cache
    cache_path = f"enhanced_cache_{config.revision_model}.json"
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                detector.cache = json.load(f)
            logging.info(f"Loaded cache with {len(detector.cache)} entries")
        except Exception as e:
            logging.warning(f"Could not load cache: {e}")

    # 5. Prepare training features
    logging.info("Preparing training features...")
    train_original_emb, train_dtdf_features = await detector.prepare_features(
        train_texts
    )

    # 6. Prepare datasets
    train_dataset, val_dataset = detector.prepare_datasets(
        train_texts, train_original_emb, train_dtdf_features, train_labels
    )

    # 7. Train model
    logging.info("Training DSFF model...")
    detector.train(
        train_dataset,
        val_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        save_path="best_dtdf_dsff_model.pt",
    )

    # 8. Test model
    logging.info("Testing model performance...")
    test_predictions, test_probabilities = await detector.predict(test_texts)

    # 9. Evaluate results
    accuracy = accuracy_score(test_labels, test_predictions)
    auc = roc_auc_score(test_labels, test_probabilities)
    f1 = f1_score(test_labels, test_predictions)

    logging.info(f"\n=== DTDF-DSFF Test Results ===")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"AUC: {auc:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")

    logging.info("\nClassification Report:")
    logging.info(
        f"\n{classification_report(test_labels, test_predictions, target_names=['Human', 'AI'])}"
    )

    # 10. Save cache
    if use_cache:
        with open(cache_path, "w") as f:
            json.dump(detector.cache, f)
        logging.info(f"Cache saved to {cache_path}")

    # 11. Visualization
    plt.figure(figsize=(15, 5))

    # ROC Curve
    plt.subplot(1, 3, 1)
    fpr, tpr, _ = roc_curve(test_labels, test_probabilities)
    plt.plot(fpr, tpr, label=f"DTDF-DSFF (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)

    # Confusion Matrix
    plt.subplot(1, 3, 2)
    cm = confusion_matrix(test_labels, test_predictions)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Human", "AI"],
        yticklabels=["Human", "AI"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Probability Distribution
    plt.subplot(1, 3, 3)
    human_probs = test_probabilities[np.array(test_labels) == 0]
    ai_probs = test_probabilities[np.array(test_labels) == 1]
    plt.hist(human_probs, alpha=0.5, label="Human", bins=20, density=True)
    plt.hist(ai_probs, alpha=0.5, label="AI", bins=20, density=True)
    plt.xlabel("AI Probability")
    plt.ylabel("Density")
    plt.title("Probability Distribution")
    plt.legend()

    plt.tight_layout()
    plt.savefig("./result/dtdf_dsff_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    return {
        "detector": detector,
        "test_results": {
            "predictions": test_predictions,
            "probabilities": test_probabilities,
        },
        "test_labels": test_labels,
        "accuracy": accuracy,
        "auc": auc,
        "f1": f1,
    }


async def run_comparison_experiment(
    data_path: str = "sample", max_samples: Optional[int] = 500
):

    setup_logging()
    logging.info("=== DTDF-DSFF vs Statistical Features Comparison ===")
    results = {}
    config = DetectionConfig(
        revision_model="gpt-3.5-turbo",
        embedding_model="distilbert-base-uncased",
        perturbation_rate=0.15,
        use_ml_classifier=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    shared_cache = {}
    cache_path = f"enhanced_cache_{config.revision_model}.json"
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                shared_cache = json.load(f)
            logging.info(
                f"Successfully loaded the shared cache containing {len(shared_cache)} entry"
            )
        except Exception as e:
            logging.warning(f"Failed to load shared cache.{e}")

    loader = CustomDataLoader()
    df = loader.load_data(data_path)
    if max_samples:
        df = df.head(max_samples)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # --- 3. Running the experiment: methodology I (DTDF-DSFF) ---
    logging.info("\n" + "=" * 50)
    logging.info("Methodology I being run: DTDF-DSFF model")
    logging.info("=" * 50)

    try:
        dtdf_detector = DTDFDSFFDetector(config, device)
        dtdf_detector.cache = shared_cache

        # Preparation features
        train_original_emb, train_dtdf_features = await dtdf_detector.prepare_features(
            train_texts
        )
        train_dataset, val_dataset = dtdf_detector.prepare_datasets(
            train_texts, train_original_emb, train_dtdf_features, train_labels
        )

        # train
        dtdf_detector.train(train_dataset, val_dataset, num_epochs=8, batch_size=16)

        # test (machinery etc)
        test_predictions, test_probabilities = await dtdf_detector.predict(test_texts)

        results["DTDF-DSFF"] = {
            "test_results": {
                "predictions": test_predictions,
                "probabilities": test_probabilities,
            },
            "test_labels": test_labels,
            "accuracy": accuracy_score(test_labels, test_predictions),
            "auc": roc_auc_score(test_labels, test_probabilities),
            "f1": f1_score(test_labels, test_predictions),
        }
        logging.info(f"DTDF-DSFF Method testing is complete.")

    except Exception as e:
        logging.error(f"DTDF-DSFF Experiment Failed. {e}", exc_info=True)

    # --- 4. Running the experiment: methodology II (baseline modelling based on statistical characteristics) ---
    logging.info("\n" + "=" * 50)
    logging.info(
        "Ongoing methodology II: baseline model based on statistical characteristics"
    )
    logging.info("=" * 50)

    try:
        base_detector = AITextDetector(config, device)
        base_detector.cache = shared_cache

        await base_detector.detect_batch(train_texts, train_labels)

        test_results = await base_detector.detect_batch(test_texts)

        results["Statistical Features"] = {
            "test_results": test_results,
            "test_labels": test_labels,
            "accuracy": accuracy_score(test_labels, test_results["predictions"]),
            "auc": roc_auc_score(test_labels, test_results["probabilities"]),
            "f1": f1_score(test_labels, test_results["predictions"]),
        }
        logging.info(f"Statistical feature baseline model testing is complete.")

    except Exception as e:
        logging.error(
            f"Failure of statistical characterisation experiments: {e}", exc_info=True
        )

    # --- 6. Summary and visualisation of results---
    logging.info("\n" + "=" * 50)
    logging.info("Final Comparison Summary")
    logging.info("=" * 50)

    summary_df = pd.DataFrame(
        [
            {
                "Method": name,
                "Accuracy": res["accuracy"],
                "AUC": res["auc"],
                "F1-Score": res["f1"],
            }
            for name, res in results.items()
        ]
    )
    logging.info(f"\n{summary_df.round(4).to_string(index=False)}")

    if results:
        logging.info("\nComparison visualisation chart being generated...")
        plot_roc_curves(results)
        plot_precision_recall_curves(results)
        plot_confusion_matrices(results)
        plot_classification_metrics(results)

    return results


# ==============================================================================
# 12. USAGE EXAMPLES
# ==============================================================================

if __name__ == "__main__":

    print("\nRunning comparison experiment...")
    dtdf_results = asyncio.run(
        run_dtdf_dsff_experiment(
            data_path="test", max_samples=None, num_epochs=5, batch_size=16
        )
    )

    # comparison_results = asyncio.run(
    #     run_comparison_experiment(data_path="test", max_samples=200)
    # )

    print("All experiments completed!")
