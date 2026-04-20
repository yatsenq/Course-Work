from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import pickle
import re
import threading

import numpy as np
import pymorphy3
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast


class BertClassifierHeadA(nn.Module):
    def __init__(self, bert_model_name: str, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))


class BertClassifierHeadB(nn.Module):
    def __init__(self, bert_model_name: str, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        scores = self.attention(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


class BertClassifierHeadC(nn.Module):
    def __init__(self, bert_model_name: str, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden = self.bert.config.hidden_size
        self.attention_pool = AttentionPooling(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.attention_pool(outputs.last_hidden_state, attention_mask)
        return self.classifier(pooled)


HEAD_CLASSES = {
    "HeadA": BertClassifierHeadA,
    "HeadB": BertClassifierHeadB,
    "HeadC": BertClassifierHeadC,
    "Head A (Linear)": BertClassifierHeadA,
    "Head B (BN-MLP)": BertClassifierHeadB,
    "Head C (AttPool)": BertClassifierHeadC,
}

DEFAULT_FAKE_CHECKPOINT = Path("experiments") / "models" / "02" / "bert_fake_headc_best.pt"
DEFAULT_TOPIC_CHECKPOINT = Path("experiments") / "models" / "07" / "bert_topic_expb_detector_best.pt"
DEFAULT_TOPIC_LABEL_ENCODER = Path("experiments") / "models" / "07" / "label_encoder.pkl"

_MODEL_RESOURCES_LOCK = threading.Lock()
_MODEL_RESOURCES_CACHE: dict[str, dict[str, Any]] = {}


def _load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device)


def _resolve_path(base: Path, candidate: str | Path) -> Path:
    path = Path(candidate)
    return path if path.is_absolute() else base / path


def _infer_head_type_from_state_dict(state_dict: dict[str, Any]) -> str | None:
    keys = set(state_dict.keys())

    if any(key.startswith("attention_pool.") for key in keys):
        return "HeadC"

    if "classifier.weight" in keys and "classifier.bias" in keys:
        return "HeadA"

    if any(key.startswith("classifier.0.") for key in keys) and any(key.startswith("classifier.4.") for key in keys):
        return "HeadB"

    return None


def _checkpoint_head_type(checkpoint_path: Path, checkpoint: dict[str, Any], state_dict: dict[str, Any] | None = None) -> str:
    if state_dict:
        inferred = _infer_head_type_from_state_dict(state_dict)
        if inferred:
            return inferred

    if isinstance(checkpoint, dict):
        head_type = checkpoint.get("head_type") or checkpoint.get("head")
        if head_type:
            normalized = str(head_type).replace(" ", "")
            for key in HEAD_CLASSES:
                if normalized.lower() in key.replace(" ", "").lower():
                    return key if key in {"HeadA", "HeadB", "HeadC"} else "HeadC"

    filename = checkpoint_path.name.lower()
    if "headc" in filename:
        return "HeadC"
    if "headb" in filename:
        return "HeadB"
    if "heada" in filename:
        return "HeadA"
    return "HeadC"


def _backbone_name(checkpoint: dict[str, Any]) -> str:
    for key in ("bert_model_name", "backbone", "encoder_name", "pretrained_model_name"):
        value = checkpoint.get(key)
        if isinstance(value, str) and value:
            return value

    if isinstance(checkpoint.get("model_name"), str):
        candidate = checkpoint["model_name"]
        if "bert" in candidate.lower() or "mbert" in candidate.lower():
            return candidate

    return "bert-base-multilingual-cased"


def _extract_linear_coefficients(model: Any) -> np.ndarray | None:
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_)

    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        coefficients = []
        for folded_model in calibrated:
            estimator = getattr(folded_model, "estimator", None) or getattr(folded_model, "base_estimator", None)
            if estimator is not None and hasattr(estimator, "coef_"):
                coefficients.append(np.asarray(estimator.coef_))
        if coefficients:
            return np.mean(np.stack(coefficients, axis=0), axis=0)

    estimator = getattr(model, "estimator", None)
    if estimator is not None and hasattr(estimator, "coef_"):
        return np.asarray(estimator.coef_)

    return None


def _is_marker_keyword_valid(keyword: str) -> bool:
    normalized = keyword.strip().lower()
    if not normalized:
        return False

    blocked_tokens = {
        "новина",
        "новини",
        "повідомляти",
        "повідомити",
        "повідомлення",
        "нагадати",
        "верховний",
    }

    if normalized in blocked_tokens:
        return False

    if any(part in blocked_tokens for part in normalized.split()):
        return False

    return len(normalized.replace(" ", "")) >= 3


def _build_theme_markers(
    theme_model: Any,
    theme_vectorizer: Any,
    theme_name: str,
    theme_vector: Any | None = None,
    top_n: int = 7,
) -> list[dict[str, float | str]]:
    class_names = list(getattr(theme_model, "classes_", []))
    coefficients = _extract_linear_coefficients(theme_model)
    if coefficients is None or not class_names:
        return []

    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(1, -1)

    if theme_name not in class_names:
        class_index = int(np.argmax(np.abs(coefficients).sum(axis=1)))
    else:
        class_index = class_names.index(theme_name)

    feature_names = np.asarray(theme_vectorizer.get_feature_names_out())
    class_coefficients = np.asarray(coefficients[class_index])

    marker_candidates: list[tuple[int, float]] = []
    if theme_vector is not None:
        nonzero_indices = theme_vector.nonzero()[1]
        if len(nonzero_indices):
            tfidf_values = np.asarray(theme_vector[0, nonzero_indices].toarray()).ravel()
            contributions = class_coefficients[nonzero_indices] * tfidf_values
            ranked = sorted(
                zip(nonzero_indices.tolist(), contributions.tolist()),
                key=lambda pair: pair[1],
                reverse=True,
            )
            marker_candidates.extend(ranked)

    if not marker_candidates:
        top_indices = np.argsort(class_coefficients)[-max(top_n * 4, top_n):][::-1]
        marker_candidates = [(int(index), float(class_coefficients[index])) for index in top_indices]

    markers = []
    for index, marker_score in marker_candidates:
        if marker_score <= 0:
            continue

        keyword = str(feature_names[index])
        if not _is_marker_keyword_valid(keyword):
            continue

        markers.append({"keyword": keyword, "score": float(marker_score)})
        if len(markers) >= top_n:
            break

    return markers[:top_n]


def load_all_models(model_base_dir: str) -> dict[str, Any]:
    model_key = str(Path(model_base_dir).resolve())
    cached = _MODEL_RESOURCES_CACHE.get(model_key)
    if cached is not None:
        return cached

    with _MODEL_RESOURCES_LOCK:
        cached = _MODEL_RESOURCES_CACHE.get(model_key)
        if cached is not None:
            return cached

        base = Path(model_base_dir)
        detector_dir = base / "Models" / "DETECTOR" / "BERT"
        theme_dir = base / "Models" / "THEME"
        stopwords_path = base / "Models" / "stopwords_ua.txt"

        fake_checkpoint_path = _resolve_path(base, Path(os.getenv("FAKE_CHECKPOINT", str(DEFAULT_FAKE_CHECKPOINT))))
        topic_checkpoint_path = _resolve_path(base, Path(os.getenv("TOPIC_CHECKPOINT", str(DEFAULT_TOPIC_CHECKPOINT))))
        topic_label_encoder_path = _resolve_path(base, Path(os.getenv("TOPIC_LABEL_ENCODER", str(DEFAULT_TOPIC_LABEL_ENCODER))))
        tokenizer_path = detector_dir / "bert_tokenizer"
        vectorizer_path = theme_dir / "theme_vectorizer.pkl"
        theme_model_path = theme_dir / "theme_model.pkl"

        for path in [fake_checkpoint_path, topic_checkpoint_path, topic_label_encoder_path, tokenizer_path, vectorizer_path, theme_model_path, stopwords_path]:
            if not path.exists():
                raise FileNotFoundError(f"Не знайдено файл моделі: {path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fake_checkpoint = _load_checkpoint(fake_checkpoint_path, device)
        fake_head_type = _checkpoint_head_type(
            fake_checkpoint_path,
            fake_checkpoint,
            fake_checkpoint.get("model_state_dict", fake_checkpoint),
        )
        fake_model_name = _backbone_name(fake_checkpoint)
        fake_max_len = fake_checkpoint.get("max_len", 256)
        fake_num_classes = int(fake_checkpoint.get("num_classes", 2))

        topic_checkpoint = _load_checkpoint(topic_checkpoint_path, device)
        topic_head_type = _checkpoint_head_type(
            topic_checkpoint_path,
            topic_checkpoint,
            topic_checkpoint.get("model_state_dict", topic_checkpoint),
        )
        topic_model_name = _backbone_name(topic_checkpoint)
        topic_max_len = topic_checkpoint.get("max_len", 256)
        topic_num_classes = int(topic_checkpoint.get("num_classes", 2))

        fake_model_cls = HEAD_CLASSES[fake_head_type]
        topic_model_cls = HEAD_CLASSES[topic_head_type]

        fake_detector = fake_model_cls(fake_model_name, num_classes=fake_num_classes).to(device)
        topic_detector = topic_model_cls(topic_model_name, num_classes=topic_num_classes).to(device)

        fake_state = fake_checkpoint.get("model_state_dict", fake_checkpoint)
        topic_state = topic_checkpoint.get("model_state_dict", topic_checkpoint)
        fake_detector.load_state_dict(fake_state)
        topic_detector.load_state_dict(topic_state)

        fake_detector.eval()
        topic_detector.eval()

        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

        with open(vectorizer_path, "rb") as f:
            theme_vectorizer = pickle.load(f)

        with open(theme_model_path, "rb") as f:
            theme_model = pickle.load(f)

        with open(topic_label_encoder_path, "rb") as f:
            topic_label_encoder = pickle.load(f)

        with open(stopwords_path, "r", encoding="utf-8") as f:
            stopwords = set(f.read().split())

        morph = pymorphy3.MorphAnalyzer(lang="uk")

        resources = {
            "device": device,
            "fake_detector": fake_detector,
            "topic_detector": topic_detector,
            "tokenizer": tokenizer,
            "theme_vectorizer": theme_vectorizer,
            "theme_model": theme_model,
            "topic_label_encoder": topic_label_encoder,
            "stopwords": stopwords,
            "morph": morph,
            "fake_max_len": fake_max_len,
            "topic_max_len": topic_max_len,
        }
        _MODEL_RESOURCES_CACHE[model_key] = resources
        return resources


def preprocess_for_theme(text: str, stopwords: set[str], morph: pymorphy3.MorphAnalyzer) -> str:
    text = re.sub(r"[^\w\s]", "", text.lower())
    words = [word for word in text.split() if word not in stopwords]
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lemmas)


def predict_news_bundle(text: str, model_base_dir: str) -> dict[str, Any]:
    resources = load_all_models(model_base_dir)
    device = resources["device"]
    tokenizer = resources["tokenizer"]

    fake_encoding = tokenizer(
        text,
        max_length=resources["fake_max_len"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    fake_input_ids = fake_encoding["input_ids"].to(device)
    fake_attention_mask = fake_encoding["attention_mask"].to(device)

    with torch.no_grad():
        fake_logits = resources["fake_detector"](fake_input_ids, fake_attention_mask)
        fake_probs = torch.softmax(fake_logits, dim=1)[0].cpu().numpy()

    prob_fake = float(fake_probs[0])
    prob_true = float(fake_probs[1])
    fake_label = "TRUE" if prob_true >= prob_fake else "FAKE"
    fake_confidence = max(prob_true, prob_fake)

    topic_encoding = tokenizer(
        text,
        max_length=resources["topic_max_len"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        theme_logits = resources["topic_detector"](
            topic_encoding["input_ids"].to(device),
            topic_encoding["attention_mask"].to(device),
        )
        theme_probs = torch.softmax(theme_logits, dim=1)[0].cpu().numpy()

    topic_classes = list(getattr(resources["topic_label_encoder"], "classes_", []))
    if not topic_classes:
        topic_classes = [str(index) for index in range(len(theme_probs))]

    theme_index = int(theme_probs.argmax())
    theme = topic_classes[theme_index]
    theme_confidence = float(theme_probs[theme_index])

    theme_distribution = sorted(
        [{"theme": cls, "probability": float(prob)} for cls, prob in zip(topic_classes, theme_probs)],
        key=lambda item: item["probability"],
        reverse=True,
    )

    clean_text = preprocess_for_theme(text, resources["stopwords"], resources["morph"])
    theme_vector = resources["theme_vectorizer"].transform([clean_text])
    theme_markers = _build_theme_markers(resources["theme_model"], resources["theme_vectorizer"], theme, theme_vector)

    return {
        "fake_label": fake_label,
        "fake_confidence": fake_confidence,
        "prob_true": prob_true,
        "prob_fake": prob_fake,
        "theme": theme,
        "theme_confidence": theme_confidence,
        "theme_distribution": theme_distribution,
        "theme_markers": theme_markers,
    }
