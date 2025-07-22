# Try to import, if it fails prompt to install
from __future__ import annotations
import os
import json
import hashlib
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
import random
import pickle
from dotenv import load_dotenv

load_dotenv()

# OpenAI library imports
import asyncio
import openai  # Good for catching error types like openai.RateLimitError
from openai import OpenAI
from openai import AsyncOpenAI
import tiktoken

# Basic libraries
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch

# Visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns

import os, time, hashlib, json, random
from typing import List, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# Transformers
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoTokenizer,
    AutoModel,
)
from sentence_transformers import SentenceTransformer

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Text Scrambler Library
try:
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas
except ImportError:
    print("nlpaug not found, installing...")
    os.system("pip install nlpaug")
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas

# Transformers and sentence-transformers

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from typing import List, Optional

# NLTK data download
import nltk

print("Downloading required NLTK data...")
nltk.download("punkt_tab")
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)
print("NLTK data download completed.")
import logging

# Configuration log
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """Detecting Configuration Classes"""

    revision_model: str = "t5-small"
    embedding_model: str = "all-MiniLM-L6-v2"

    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL")
    )

    perturbation_rate: float = 0.15

    perturbation_methods: List[str] = field(
        default_factory=lambda: ["synonym", "contextual"]
    )

    similarity_threshold: float = 0.95
    use_ml_classifier: bool = True

    batch_size: int = 16
    max_length: int = 512

    def __post_init__(self):
        if self.perturbation_methods is None:
            self.perturbation_methods = ["synonym", "contextual", "backtranslation"]


class EnhancedTextPerturber:

    def __init__(self, config: DetectionConfig):
        self.config = config
        self._init_augmenters()

    def _init_augmenters(self):

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
        """
        Intelligent Perturbation Strategies

        Args.
            text: original text
            method: specified method, None is chosen randomly
        Returns.
            Scrambled text
        """
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

        return text


class EnhancedLLMReviser:
    """Enhanced LLM Rewriter"""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = None
        self.model = None
        self.tokenizer = None
        self._init_reviser()

    def _init_model(self):
        model_name = self.config.revision_model
        """Initialising the rewrite model"""
        if self.config.revision_model.startswith("t5"):

            self.tokenizer = T5Tokenizer.from_pretrained(self.config.revision_model)
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.config.revision_model
            )
        elif self.config.revision_model == "gpt2":
            # GPT-2 as an option
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif model_name.startswith("gpt-"):
            # Initialising the OpenAI API client
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Please provide api_key in DetectionConfig or set the OPENAI_API_KEY environment variable."
                )

            self.client = OpenAI(api_key=api_key, base_url=self.config.base_url)
            self.max_retries = 3
            self.retry_delay = 1.0
            print(f"OpenAI client initialized for model: {model_name}")
            if self.config.base_url:
                print(f"Using custom base URL: {self.config.base_url}")
        else:
            print(
                f"Warning: Unknown revision model '{model_name}'. Will use rule-based fallback."
            )

        self.model.to(self.device)
        self.model.eval()

    def _init_reviser(self):
        """Initialise rewrite model or API client according to configuration"""
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
            # Initialising the OpenAI API client
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Please provide api_key in DetectionConfig or set the OPENAI_API_KEY environment variable."
                )

            # self.client = OpenAI(api_key=api_key, base_url=self.config.base_url)
            self.client = AsyncOpenAI(
                api_key=api_key, base_url=self.config.base_url
            )  # Using Asynchronous Clients
            self.max_retries = 3
            self.retry_delay = 1.0
            print(f"OpenAI client initialized for model: {model_name}")
            if self.config.base_url:
                print(f"Using custom base URL: {self.config.base_url}")
        else:
            print(
                f"Warning: Unknown revision model '{model_name}'. Will use rule-based fallback."
            )

    # def revise(self, text: str, cache: Dict[str, str]) -> str:
    async def revise(
        self, original_text: str, perturbed_text: str, cache: Dict[str, str]
    ) -> str:  # change to async def
        """
        Rewriting Text with LLM

        Args.
            original_text: text to be rewritten
            cache: Cached dictionary
        Returns.
            Rewritten text
        """
        # 生成缓存键
        cache_key = hashlib.md5(
            f"{original_text}_{self.config.revision_model}".encode()
        ).hexdigest()

        # Check if the result for the original text is already in the cache
        if cache_key in cache:
            return cache[cache_key]

        # print(f"\norigin text: {original_text}")

        try:
            model_name = self.config.revision_model
            revised = ""
            if model_name.startswith("t5") or model_name == "gpt2":
                if model_name.startswith("t5"):
                    revised = self._revise_with_t5(perturbed_text)
                    # print(f"t5 rewrite: {revised}")
                else:  # gpt2
                    revised = self._revise_with_gpt2(perturbed_text)
                    # print(f"gpt2 rewrite: {revised}")
            elif model_name.startswith("gpt-"):
                # api rewrite
                # revised = self._revise_with_api(perturbed_text)
                revised = await self._revise_with_api(perturbed_text)  # change to await
                # print(f"gpt3.5 rewrite: {revised}")
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
        """Rewrite using the T5 model"""

        prompt = f"paraphrase: {text}"

        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=512,  # Maximum length of input truncation
            truncation=True,
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
        """Rewrite using the GPT-2 model"""
        try:
            prompt_templates = [
                f"Paraphrase the following text while keeping the same meaning: {text}\n\nParaphrased version:",
                f"Rewrite this sentence in a different way: {text}\n\nRewritten:",
                f"Express this differently: {text}\n\nAlternative expression:",
            ]
            prompt = np.random.choice(prompt_templates)
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,  # Ensure padding is enabled to handle the mask
            )
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,  # Pass input_ids explicitly
                    attention_mask=attention_mask,
                    max_length=min(len(input_ids[0]) + 100, 1024),
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    # early_stopping=True,
                )

            #
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
            print(f"  Fixed Rewrite Failure: {e}")
            return text

    # def _revise_with_api(self, text: str) -> str:
    async def _revise_with_api(self, text: str) -> str:  # change to async def
        """Use OpenAI API with better prompting strategy"""
        if not self.client:
            raise ValueError("OpenAI client not initialized.")

        # Better prompt that encourages standardization
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
                response = await self.client.chat.completions.create(  # change to await
                    model=self.config.revision_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,  # Lower temperature for more consistent output
                    top_p=0.9,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                print(f"API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.time.sleep(
                        self.retry_delay
                    )  # Using asynchronous sleep
                else:
                    return text

        return text

    def _rule_based_revision(self, text: str) -> str:

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
        """Intelligent Sentence Rearrangement"""
        # Keep the first and last sentences
        if len(sentences) <= 2:
            return sentences

        first = sentences[0]
        last = sentences[-1]
        middle = sentences[1:-1]

        # break up the middle of a sentence
        np.random.shuffle(middle)

        return [first] + middle + [last]


class MultiFeatureExtractor:
    """Multi-dimensional feature extractor"""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.sentence_model = SentenceTransformer(config.embedding_model)

        if self.sentence_model.tokenizer.pad_token is None:
            print(
                f"Warning: Tokenizer for '{config.embedding_model}' is missing a pad token. Setting it to eos_token."
            )
            self.sentence_model.tokenizer.pad_token = (
                self.sentence_model.tokenizer.eos_token
            )
        # --- Add these lines to view parameters ---
        print(f"\n--- Parameters for Embedding Model: {config.embedding_model} ---")
        # Print the model architecture (all layers)
        print(self.sentence_model)

        # Calculate and print the total number of trainable parameters
        total_params = sum(
            p.numel() for p in self.sentence_model.parameters() if p.requires_grad
        )
        print(f"Total Trainable Parameters: {total_params:,}")
        self._init_linguistic_features()

    def _init_linguistic_features(self):
        """Initialised Linguistic Feature Extraction"""
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("stopwords", quiet=True)
            self.stop_words = set(stopwords.words("english"))
        except:
            self.stop_words = set()

    def extract_features(self, original: str, revised: str) -> Dict[str, float]:
        """
        Extract multi-dimensional features

        Returns:
            feature dictionary
        """
        features = {}

        # 1. semantic similarity
        features["semantic_similarity"] = self._compute_semantic_similarity(
            original, revised
        )

        # 2. lexical level feature
        features.update(self._extract_lexical_features(original, revised))

        # 3. Syntactic features
        features.update(self._extract_syntactic_features(original, revised))

        # 4. style
        features.update(self._extract_stylistic_features(original, revised))

        # 5. Edit Distance Characteristics
        features.update(self._extract_edit_features(original, revised))

        return features

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculating semantic similarity"""
        emb1 = self.sentence_model.encode(text1)
        emb2 = self.sentence_model.encode(text2)

        from sklearn.metrics.pairwise import cosine_similarity

        return float(cosine_similarity([emb1], [emb2])[0, 0])

    def _extract_lexical_features(
        self, original: str, revised: str
    ) -> Dict[str, float]:
        """Extraction of vocabulary-level features"""
        orig_words = set(word_tokenize(original.lower()))
        rev_words = set(word_tokenize(revised.lower()))

        # overlap rate of vocabulary
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

        # Change in proportion of deactivated words
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
        """Extracting syntactic features"""
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

        try:
            orig_pos = nltk.pos_tag(word_tokenize(original))
            rev_pos = nltk.pos_tag(word_tokenize(revised))

            # Calculate the percentage change in the main lexical categories
            for pos_type in [
                "NN",
                "VB",
                "JJ",
                "RB",
            ]:  # Nouns, verbs, adjectives, adverbs
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
        """Extracting stylistic features"""
        features = {}

        # Changes in the use of punctuation
        orig_punct = sum(1 for c in original if c in ".,!?;:")
        rev_punct = sum(1 for c in revised if c in ".,!?;:")

        features["punctuation_ratio_change"] = abs(
            (orig_punct / len(original) if original else 0)
            - (rev_punct / len(revised) if revised else 0)
        )

        # Change in proportion of capital letters
        orig_caps = sum(1 for c in original if c.isupper())
        rev_caps = sum(1 for c in revised if c.isupper())

        features["capitalization_ratio_change"] = abs(
            (orig_caps / len(original) if original else 0)
            - (rev_caps / len(revised) if revised else 0)
        )

        # Change in average word length
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
        """Extraction of edit distance related features"""
        from difflib import SequenceMatcher

        # Character-level similarity
        char_similarity = SequenceMatcher(None, original, revised).ratio()

        # word-level similarity
        orig_words = word_tokenize(original)
        rev_words = word_tokenize(revised)
        word_similarity = SequenceMatcher(None, orig_words, rev_words).ratio()

        features = {
            "char_level_similarity": char_similarity,
            "word_level_similarity": word_similarity,
            "length_ratio": len(revised) / len(original) if original else 1,
        }

        return features


class AITextDetector:
    """Main detector class"""

    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.perturber = EnhancedTextPerturber(self.config)
        self.reviser = EnhancedLLMReviser(self.config)
        self.feature_extractor = MultiFeatureExtractor(self.config)
        self.classifier = None
        self.scaler = StandardScaler()
        self.cache = {}
        self.api_concurrency_limit = asyncio.Semaphore(10)

    # In the AITextDetector class

    async def detect_batch(
        self, texts: List[str], labels: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Asynchronously detects text in a batch, with robust caching and parallel API calls.
        """
        all_features = []
        all_scores = []

        if not texts:
            # Handle the case where an empty list is passed
            return {
                "predictions": np.array([]),
                "probabilities": np.array([]),
                "features": np.array([]),
                "similarity_scores": np.array([]),
            }

        print("Processing texts...")

        # Step 1: Perform synchronous, CPU-bound perturbation first.
        perturbed_texts = [
            self.perturber.perturb(text) for text in tqdm(texts, desc="Perturbing")
        ]

        # Step 2: Define a helper function to manage semaphore and call the new revise method.
        # This helper will be used to create our list of concurrent tasks.
        async def revise_with_limit(original_text, perturbed_text):
            async with self.api_concurrency_limit:
                # Call the updated revise function with both original and perturbed text
                return await self.reviser.revise(
                    original_text, perturbed_text, self.cache
                )

        # Step 3: Create a list of NEW coroutine tasks.
        # Each task is a call to our helper function with a pair of original and perturbed texts.
        print(f"Revising texts with {self.reviser.config.revision_model}...")
        revision_tasks = [
            revise_with_limit(orig, pert) for orig, pert in zip(texts, perturbed_texts)
        ]

        # Step 4: Concurrently run all revision tasks using asyncio.gather.
        # This preserves the order of the results and is wrapped in tqdm for a progress bar.
        all_revised_texts = await asyncio.gather(
            *tqdm(revision_tasks, desc="Revising", total=len(revision_tasks))
        )

        # Step 5: Process the results. This part is synchronous again.
        print("Extracting features...")
        for i, original_text in enumerate(texts):
            # The order is preserved, so we can safely pair original and revised texts.
            revised_text = all_revised_texts[i]
            features = self.feature_extractor.extract_features(
                original_text, revised_text
            )
            all_features.append(features)
            all_scores.append(features.get("semantic_similarity", 0.5))

        # --- The rest of the function remains the same ---
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
                # Fallback if classifier isn't trained
                predictions = (
                    np.array(all_scores) > self.config.similarity_threshold
                ).astype(int)
                probabilities = np.array(all_scores)
        else:
            # Logic for when the ML classifier is turned off
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
        """Training Machine Learning Classifiers"""
        print("Training classifier...")

        # Standardised features
        features_scaled = self.scaler.fit_transform(features)

        # Using Random Forests
        self.classifier = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )

        print(f"\n--- RandomForestClassifier Parameters ---")
        print(self.classifier.get_params())

        self.classifier.fit(features_scaled, labels)

        # Characteristic importance
        feature_names = list(pd.DataFrame(features).columns)
        importances = self.classifier.feature_importances_

        print("\nTop 10 Most Important Features:")
        for feat, imp in sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        )[:10]:
            print(f"  {feat}: {imp:.4f}")

    def save_model(self, path: str):
        """Save the model"""
        import pickle

        model_data = {
            "classifier": self.classifier,
            "scaler": self.scaler,
            "config": self.config,
            "cache": self.cache,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Loading Models"""
        import pickle

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.classifier = model_data["classifier"]
        self.scaler = model_data["scaler"]
        self.config = model_data["config"]
        self.cache = model_data["cache"]

        print(f"Model loaded from {path}")


class DataLoader:
    """Data Loader Class"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created data directory: {data_dir}")

    def load_data(self, path: str) -> pd.DataFrame:
        """
        Reads a dataset with 'text' and 'label' columns and returns a DataFrame.

        Args.
            path: path to the data file, 'finance' means use financial dataset, 'sample' means create sample data.
        Returns.
            df: columns = ['id', 'text', 'label']
        """
        if path == "finance":
            return self.load_finance_dataset()
        if path == "sample":
            return self.create_sample_dataset()
        elif path == "test":
            return self.load_origin_dataset()

        # If it is a relative path, add a data catalogue prefix
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

            # Ensure that the necessary columns are available
            if "text" not in df.columns:
                raise ValueError("Missing 'text' column in dataset")
            if "label" not in df.columns:
                raise ValueError("Missing 'label' column in dataset")

            # If there is no id column, automatically generate
            if "id" not in df.columns:
                df["id"] = range(len(df))

            # Data Cleaning
            df = df.dropna(subset=["text", "label"])
            df = df[df["text"].str.len() > 10]  # Filtering text that is too short

            logger.info(f"Loaded {len(df)} samples from {path}")
            logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

            return df[["id", "text", "label"]]

        except FileNotFoundError:
            logger.warning(f"File {path} not found. Creating sample dataset...")
            return self.create_sample_dataset()

    def find_finance_data_files(self) -> Dict[str, str]:
        """Find financial dataset files"""
        logger.info("Searching for finance data files...")

        # Local environment path
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
                logger.debug(f"Checking path: {path}")
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
        """
        Load financial dataset - use revised text as data

        Returns.
            df: columns = ['id', 'text', 'label']
            where text is the revised text, label: 0=human, 1=chatgpt
        """
        files = self.find_finance_data_files()

        logger.info("Loading finance datasets...")
        logger.info(f"Found files: {list(files.keys())}")

        all_texts = []
        all_labels = []

        if "revised_human" in files:
            try:
                logger.info(
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
                                logger.warning(f"  Line {line_no} parse error: {e}")
                logger.info(
                    f"  Loaded {len([l for l in all_labels if l == 0])} human texts"
                )
            except Exception as e:
                logger.error(f"Error loading revised human texts: {e}")

        # Load revised ChatGPT text (one JSON object per line)
        if "revised_chatgpt" in files:
            try:
                logger.info(
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
                                logger.warning(f"  Line {line_no} parse error: {e}")
                logger.info(
                    f"  Loaded {len([l for l in all_labels if l == 1])} ChatGPT texts"
                )
            except Exception as e:
                logger.error(f"Error loading revised ChatGPT texts: {e}")

        # Creating a DataFrame
        df = pd.DataFrame(
            {"id": range(len(all_texts)), "text": all_texts, "label": all_labels}
        )

        # Disrupt the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df["id"] = range(len(df))

        logger.info(f"\n=== Dataset Summary ===")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Human texts: {len(df[df['label'] == 0])}")
        logger.info(f"ChatGPT texts: {len(df[df['label'] == 1])}")

        # Show Example
        if len(df) > 0:
            logger.info("\nSample texts:")
            for label in [0, 1]:
                label_name = "Human" if label == 0 else "ChatGPT"
                samples = df[df["label"] == label]
                if len(samples) > 0:
                    sample_text = samples.iloc[0]["text"]
                    logger.info(f"[{label_name}]: {sample_text[:100]}...")

        return df[["id", "text", "label"]]

    def create_sample_dataset(self) -> pd.DataFrame:
        """
        Create a sample dataset for testing

        Returns:
            df: columns = ['id', 'text', 'label']
        """
        sample_data = {
            "id": range(6),
            "text": [
                "There is most likely an error in the WSJ's data.  Yahoo! Finance reports the PE on the Russell 2000 to be 15 as of 83111 and SP 500 PE to be 13 (about the same as WSJ). Good catch, though!  E-mail WSJ, perhaps they will be grateful.",
                "I know this question has a lot of answers already, but I feel the answers are phrased either strongly against, or mildly for, co-signing. What it amounts down to is that this is a personal choice. You cannot receive reliable information as to whether or not co-signing this loan is a good move due to lack of information. The person involved is going to know the person they would be co-signing for, and the people on this site will only have their own personal preferences of experiences to draw from. You know if they are reliable, if they will be able to pay off the loan without need for the banks to come after you.  This site can offer general theories, but I think it should be kept in mind that this is wholly a personal decision for the person involved, and them alone to make based on the facts that they know and we do not.",
                "I think the best investment strategy is to diversify your portfolio across different asset classes.",
                "Historical price-to-earnings (PE) ratios for small-cap and large-cap stocks can vary significantly over time and may not be directly comparable due to the different characteristics of these two categories of stocks.Small-cap stocks, which are defined as stocks with a market capitalization of less than $2 billion, tend to be riskier and more volatile than large-cap stocks, which have a market capitalization of $10 billion or more. As a result, investors may be willing to pay a higher price for the potential growth opportunities offered by small-cap stocks, which can lead to higher PE ratios.On the other hand, large-cap stocks tend to be more established and stable, with a longer track record of earnings and revenue growth. As a result, these stocks may trade at lower PE ratios, as investors may be less willing to pay a premium for their growth potential.It is important to note that PE ratios are just one factor to consider when evaluating a stock and should not be used in isolation. Other factors, such as the company's financial health, industry trends, and macroeconomic conditions, can also impact a stock's PE ratio.",
                "Co-signing a personal loan for a friend or family member can be a risky proposition. When you co-sign a loan, you are agreeing to be responsible for the loan if the borrower is unable to make the payments. This means that if your friend or family member defaults on the loan, you will be on the hook for the remaining balance.There are a few things to consider before co-signing a personal loan for someone:Do you trust the borrower to make the payments on time and in full? If you are not confident that the borrower will be able to make the payments, it may not be a good idea to co-sign the loan.Can you afford to make the payments if the borrower defaults? If you are unable to make the payments, co-signing the loan could put your own financial stability at risk.What is the purpose of the loan? If the borrower is using the loan for a risky or questionable venture, it may not be worth the risk to co-sign.Is there another way for the borrower to get the loan without a co-signer? If the borrower has a good credit score and is able to qualify for a loan on their own, it may not be necessary for you to co-sign.In general, it is important to carefully consider the risks and potential consequences before co-signing a loan for someone. If you do decide to co-sign, it is a good idea to have a conversation with the borrower about their plans for making the loan payments and to have a clear understanding of your responsibilities as a co-signer.",
                "The optimal approach to risk management involves careful assessment of market conditions.",
            ],
            "label": [0, 0, 0, 1, 1, 1],  # 0=human, 1=AI
        }

        df = pd.DataFrame(sample_data)
        logger.info(f"Created sample dataset with {len(df)} examples")
        return df

    def load_origin_dataset(self) -> pd.DataFrame:
        """Load finance dataset from JSON Lines format"""
        files = self.find_finance_data_files()

        logger.info("Loading finance datasets...")
        logger.info(f"Found files: {list(files.keys())}")

        all_texts = []
        all_labels = []

        if "original" in files:
            finance_path = files["original"]
            logger.info(f"Loading from: {finance_path}")

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

                            # Log progress every 1000 lines
                            if line_num % 1000 == 0:
                                logger.info(
                                    f"Processed {line_num} lines, extracted {len(all_texts)} texts"
                                )

                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing line {line_num}: {e}")
                            logger.error(f"Line content: {line[:100]}...")
                            continue

                logger.info(f"Successfully loaded {line_count} JSON objects")
                logger.info(f"Total texts extracted: {len(all_texts)}")

            except FileNotFoundError:
                logger.error(f"File not found: {finance_path}")
                return self.create_sample_dataset()
            except Exception as e:
                logger.error(f"Error loading finance data: {e}")
                return self.create_sample_dataset()
        else:
            logger.warning("No finance data file found. Creating sample dataset...")
            return self.create_sample_dataset()

        if len(all_texts) == 0:
            logger.warning(
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

        # Print summary
        logger.info(f"\n=== Dataset Summary ===")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Human texts: {len(df[df['label'] == 0])}")
        logger.info(f"ChatGPT texts: {len(df[df['label'] == 1])}")

        # Show balance
        balance = len(df[df["label"] == 0]) / len(df) * 100
        logger.info(
            f"Dataset balance: {balance:.1f}% human, {100-balance:.1f}% ChatGPT"
        )

        # Display samples
        if len(df) > 0:
            logger.info("\nSample texts:")
            for label in [0, 1]:
                label_name = "Human" if label == 0 else "ChatGPT"
                samples = df[df["label"] == label]
                if len(samples) > 0:
                    sample_text = samples.iloc[0]["text"]
                    logger.info(f"\n[{label_name}]: {sample_text[:200]}...")

        return df[["id", "text", "label"]]

    def save_dataset(self, df: pd.DataFrame, filename: str, format: str = "csv"):
        """
        Saving a dataset to a file

        Args:
            df: the DataFrame to be saved
            filename: the name of the file (without extension)
            format: file format ('csv', 'json', 'jsonl')
        """
        filepath = os.path.join(self.data_dir, f"{filename}.{format}")

        if format == "csv":
            df.to_csv(filepath, index=False)
        elif format == "json":
            df.to_json(filepath, orient="records", indent=2)
        elif format == "jsonl":
            df.to_json(filepath, orient="records", lines=True)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved dataset to {filepath}")
        return filepath


def load_data(path: str) -> pd.DataFrame:

    loader = DataLoader()
    return loader.load_data(path)


def load_finance_dataset() -> pd.DataFrame:

    loader = DataLoader()
    return loader.load_finance_dataset()


def load_origin_dataset() -> pd.DataFrame:
    loader = DataLoader()
    return loader.load_origin_dataset


def create_sample_dataset() -> pd.DataFrame:

    loader = DataLoader()
    return loader.create_sample_dataset()


def evaluate_detector(
    detector: AITextDetector, texts: List[str], labels: List[int]
) -> Dict[str, float]:
    """
    Evaluating detector performance
    """
    results = detector.detect_batch(texts, labels)

    predictions = results["predictions"]
    probabilities = results["probabilities"]

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "auc": roc_auc_score(labels, probabilities),
    }

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")

    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=["Human", "AI"]))

    human_scores = results["similarity_scores"][np.array(labels) == 0]
    ai_scores = results["similarity_scores"][np.array(labels) == 1]

    print(f"\nSimilarity Score Distribution:")
    print(
        f"Human texts - Mean: {np.mean(human_scores):.3f}, Std: {np.std(human_scores):.3f}"
    )
    print(f"AI texts - Mean: {np.mean(ai_scores):.3f}, Std: {np.std(ai_scores):.3f}")

    return metrics


def plot_roc_curves(results: Dict):

    plt.figure(figsize=(10, 8))
    for name, result_data in results.items():
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
    plt.show()


def plot_precision_recall_curves(results: Dict):

    plt.figure(figsize=(10, 8))
    for name, result_data in results.items():
        precision, recall, _ = precision_recall_curve(
            result_data["test_labels"], result_data["test_results"]["probabilities"]
        )
        plt.plot(recall, precision, label=f"{name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrices(results: Dict):

    n_results = len(results)
    fig, axes = plt.subplots(1, n_results, figsize=(6 * n_results, 5))
    if n_results == 1:
        axes = [axes]  # Make it iterable
    for ax, (name, result_data) in zip(axes, results.items()):
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
    plt.show()


def plot_classification_metrics(results: Dict):

    metrics_data = []
    for name, result_data in results.items():
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
    plt.show()


def find_optimal_threshold(labels: List[int], scores: np.ndarray) -> float:
    """
    Find the optimal classification threshold using the ROC curve.

    The optimal threshold is the one that maximizes the difference
    between the true positive rate (TPR) and the false positive rate (FPR).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Calculate the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(
        f"\nFound optimal similarity threshold from training data: {optimal_threshold:.4f}"
    )
    return optimal_threshold


# 主执行函数
async def main(
    data_path: str = "finance",
    max_samples: Optional[int] = None,
    use_cache: bool = True,
    config: Optional[DetectionConfig] = None,
):

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("=== Enhanced AI Text Detection ===")
    print(
        f"Method: Multi-feature extraction with {config.revision_model if config else 'default model'}"
    )

    logger.info(f"Loading data from: {data_path}")
    loader = DataLoader()
    df = loader.load_data(data_path)
    if max_samples:
        df = df.head(max_samples)
        print(f"Using first {max_samples} samples for testing")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # Data segmentation
    split_idx = int(len(texts) * 0.6)
    train_texts, test_texts = texts[:split_idx], texts[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    print(f"\nDataset split: {len(train_texts)} training, {len(test_texts)} testing")

    # Creating Detectors
    if config is None:
        config = DetectionConfig(
            revision_model="t5-small",  # 或 "gpt2" t5-small
            embedding_model="all-MiniLM-L6-v2",
            perturbation_rate=0.15,
            use_ml_classifier=True,
        )

    detector = AITextDetector(config)

    # Load Cache
    cache_path = f"enhanced_cache_{config.revision_model}.json"
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            detector.cache = json.load(f)
        print(f"Loaded cache with {len(detector.cache)} entries")

    # training phase
    print("\n=== Training Phase ===")
    print(
        "Train labels distribution:", pd.Series(train_labels).value_counts().to_dict()
    )
    train_results = await detector.detect_batch(train_texts, train_labels)
    # If not using the ML classifier, find the best threshold from the training run
    if not config.use_ml_classifier:
        optimal_threshold = find_optimal_threshold(
            train_labels, train_results["similarity_scores"]
        )
        # Update the detector's config with the new, optimal threshold
        detector.config.similarity_threshold = optimal_threshold

    # testing phase
    print("\n=== Testing Phase ===")
    test_results = await detector.detect_batch(test_texts)

    # Assessment results
    test_predictions = test_results["predictions"]
    test_probabilities = test_results["probabilities"]

    # Calculation of indicators
    accuracy = accuracy_score(test_labels, test_predictions)
    auc = roc_auc_score(test_labels, test_probabilities)
    f1 = f1_score(test_labels, test_predictions)

    print(f"\n=== Test Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1 Scores: {f1:.4f}")
    # Detailed report
    print("\nClassification Report:")
    print(
        classification_report(
            test_labels, test_predictions, target_names=["Human", "AI"]
        )
    )

    # confusion matrix (math.)
    cm = confusion_matrix(test_labels, test_predictions)
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"            Human    AI")
    print(f"Actual Human  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       AI     {cm[1,0]:4d}  {cm[1,1]:4d}")

    # character analysis
    features_df = test_results["features"]
    similarity_scores = test_results["similarity_scores"]

    # Subgroup statistics
    human_scores = similarity_scores[np.array(test_labels) == 0]
    ai_scores = similarity_scores[np.array(test_labels) == 1]

    print(f"\n=== Similarity Score Analysis ===")
    print(f"Human texts:")
    print(f"  Mean: {np.mean(human_scores):.3f}, Std: {np.std(human_scores):.3f}")
    print(f"  Min: {np.min(human_scores):.3f}, Max: {np.max(human_scores):.3f}")

    print(f"AI texts:")
    print(f"  Mean: {np.mean(ai_scores):.3f}, Std: {np.std(ai_scores):.3f}")
    print(f"  Min: {np.min(ai_scores):.3f}, Max: {np.max(ai_scores):.3f}")

    print("\n=== Misclassified Examples ===")
    misclassified_idx = np.where(test_predictions != test_labels)[0]

    for i, idx in enumerate(misclassified_idx[:50]):  # 显示前5个
        true_label = "AI" if test_labels[idx] == 1 else "Human"
        pred_label = "AI" if test_predictions[idx] == 1 else "Human"
        text = (
            test_texts[idx][:100] + "..."
            if len(test_texts[idx]) > 100
            else test_texts[idx]
        )

        print(f"\nExample {i+1}:")
        print(f"Text: {text}")
        print(f"True: {true_label}, Predicted: {pred_label}")
        print(
            f"Similarity: {similarity_scores[idx]:.3f}, Probability: {test_probabilities[idx]:.3f}"
        )

    if use_cache:
        with open(cache_path, "w") as f:
            json.dump(detector.cache, f)
        print(f"\nCache saved to {cache_path}")

    model_path = f"enhanced_detector_{config.revision_model}.pkl"
    detector.save_model(model_path)

    return {
        "detector": detector,
        "test_results": test_results,
        "test_labels": test_labels,
        "accuracy": accuracy,
        "auc": auc,
        "f1": f1,
    }


# 实验对比函数
async def run_comparison_experiment(
    data_path: str = "finance", max_samples: Optional[int] = None
):

    print("=== Comparison Experiment ===")

    results = {}

    configurations = {
        # "t5 with Allmini&perturbation": DetectionConfig(
        #     revision_model="t5-small",
        #     embedding_model="all-MiniLM-L6-v2",
        #     perturbation_rate=0.15,
        #     use_ml_classifier=False,
        # ),
        # "GPT-2 with AllMini&perturbation": DetectionConfig(
        #     revision_model="gpt2",
        #     embedding_model="all-MiniLM-L6-v2",
        #     perturbation_rate=0,
        #     use_ml_classifier=False,
        # ),
        # "t5 with Allmini": DetectionConfig(
        #     revision_model="t5-small",
        #     embedding_model="all-MiniLM-L6-v2",
        #     perturbation_rate=0.15,
        #     use_ml_classifier=False,
        # ),
        # "GPT-2 with AllMini": DetectionConfig(
        #     revision_model="gpt2",
        #     embedding_model="all-MiniLM-L6-v2",
        #     perturbation_rate=0,
        #     use_ml_classifier=False,
        # ),
        # "t5 with Allmini&feature&perturbation": DetectionConfig(
        #     revision_model="t5-small",
        #     embedding_model="all-MiniLM-L6-v2",
        #     perturbation_rate=0.15,
        #     use_ml_classifier=True,
        # ),
        # "GPT-2 with AllMini&feature&perturbation": DetectionConfig(
        #     revision_model="gpt2",
        #     embedding_model="all-MiniLM-L6-v2",
        #     perturbation_rate=0.15,
        #     use_ml_classifier=True,
        # ),
        "gpt3.5 with Bert model": DetectionConfig(
            revision_model="gpt-3.5-turbo",
            embedding_model="bert-base-uncased",
            perturbation_rate=0.15,
            use_ml_classifier=True,
        ),
        "gpt3.5 with gpt2": DetectionConfig(
            revision_model="gpt-3.5-turbo",
            embedding_model="gpt2",
            perturbation_rate=0.15,
            use_ml_classifier=True,
        ),
        "gpt3.5 with T5": DetectionConfig(
            revision_model="gpt-3.5-turbo",
            embedding_model="t5-small",
            perturbation_rate=0.15,
            use_ml_classifier=True,
        ),
        "gpt3.5 with RoBEATa model": DetectionConfig(
            revision_model="gpt-3.5-turbo",
            embedding_model="roberta-base",
            perturbation_rate=0.15,
            use_ml_classifier=True,
        ),
        # "gpt3.5 with paraphrase model": DetectionConfig(
        #     revision_model="gpt-3.5-turbo",
        #     embedding_model="paraphrase-MiniLM-L6-v2",
        #     perturbation_rate=0.15,
        #     use_ml_classifier=True,
        # ),
        # "gpt3.5 with funed model": DetectionConfig(
        #     revision_model="gpt-3.5-turbo",  # t5-small,gpt-3.5-turbo,gpt2
        #     embedding_model="./models/paraphrase-MiniLM-L6-v2-ai-detector-incomplete",
        #     perturbation_rate=0.15,
        #     use_ml_classifier=True,
        # ),
        # "High_Perturb": DetectionConfig(
        #     revision_model="t5-small",
        #     embedding_model="all-MiniLM-L6-v2",
        #     perturbation_rate=0.25,
        #     perturbation_methods=["synonym", "contextual", "random_swap"],
        #     use_ml_classifier=True,
        # ),
    }

    # Run experiments
    for name, config in configurations.items():
        print(f"\n{'='*20} Running: {name} {'='*20}")
        try:
            results[name] = await main(
                data_path=data_path,
                max_samples=max_samples,
                use_cache=True,
                config=config,
            )
        except Exception as e:
            print(f"!!!!!! Experiment '{name}' failed: {e} !!!!!!")

    # Compare results textually
    print("\n\n=== Final Comparison Summary ===")
    print(f"{'Configuration':<30} {'Accuracy':<10} {'AUC':<10} {'F1-Score':<10}")
    print("-" * 40)
    for name, result_data in results.items():
        print(
            f"{name:<20} {result_data['accuracy']:<10.4f} {result_data['auc']:<10.4f}{result_data['f1']:<10.4f}"
        )

    # Visualize results
    print("\n\nGenerating visualizations...")
    if results:
        plot_roc_curves(results)
        plot_precision_recall_curves(results)
        plot_confusion_matrices(results)
        plot_classification_metrics(results)
    else:
        print("No successful experiments to visualize.")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "default"

    if mode == "compare":

        print("Running comparison experiment...")
        asyncio.run(run_comparison_experiment(data_path="test", max_samples=None))

    elif mode == "quick":

        print("Running quick test with sample data...")
        config = DetectionConfig(
            revision_model="gpt-3.5-turbo",
            embedding_model="all-MiniLM-L6-v2",
            perturbation_rate=0.15,
            use_ml_classifier=False,
        )
        asyncio.run(main(data_path="sample", max_samples=500, config=config))

    elif mode == "api":
        print("Running api experiment with finance dataset...")
        try:

            config = DetectionConfig(
                revision_model="gpt-3.5-turbo",  # t5-small,gpt-3.5-turbo,gpt2
                embedding_model="./models/paraphrase-MiniLM-L6-v2-ai-detector-incomplete",
                perturbation_rate=0.15,
                use_ml_classifier=False,
                similarity_threshold=0.7,  # 0.705,.698
            )
            asyncio.run(
                main(data_path="finance", max_samples=50, use_cache=True, config=config)
            )  # use asyncio.run()
        except Exception as e:
            print(f"\nError with finance dataset: {e}")
            print("Falling back to sample dataset...")
            asyncio.run(main(data_path="sample", max_samples=200))  # use asyncio.run()
