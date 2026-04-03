from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pickle
import re
from typing import Any

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
import pymorphy3


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
    "Head A (Linear)": BertClassifierHeadA,
    "Head B (BN-MLP)": BertClassifierHeadB,
    "Head C (AttPool)": BertClassifierHeadC,
}


@lru_cache(maxsize=1)
def load_all_models(model_base_dir: str) -> dict[str, Any]:
    base = Path(model_base_dir)
    detector_dir = base / "Models" / "DETECTOR" / "BERT"
    theme_dir = base / "Models" / "THEME"
    stopwords_path = base / "Models" / "stopwords_ua.txt"

    checkpoint_path = detector_dir / "bert_fake_detector_best.pt"
    tokenizer_path = detector_dir / "bert_tokenizer"
    vectorizer_path = theme_dir / "theme_vectorizer.pkl"
    theme_model_path = theme_dir / "theme_model.pkl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    head_type = checkpoint["head_type"]
    model_name = checkpoint["model_name"]
    max_len = checkpoint["max_len"]

    model_cls = HEAD_CLASSES[head_type]
    detector = model_cls(model_name).to(device)
    detector.load_state_dict(checkpoint["model_state_dict"])
    detector.eval()

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    with open(vectorizer_path, "rb") as f:
        theme_vectorizer = pickle.load(f)

    with open(theme_model_path, "rb") as f:
        theme_model = pickle.load(f)

    with open(stopwords_path, "r", encoding="utf-8") as f:
        stopwords = set(f.read().split())

    morph = pymorphy3.MorphAnalyzer(lang="uk")

    return {
        "device": device,
        "detector": detector,
        "tokenizer": tokenizer,
        "theme_vectorizer": theme_vectorizer,
        "theme_model": theme_model,
        "stopwords": stopwords,
        "morph": morph,
        "max_len": max_len,
    }


def preprocess_for_theme(text: str, stopwords: set[str], morph: pymorphy3.MorphAnalyzer) -> str:
    text = re.sub(r"[^\w\s]", "", text.lower())
    words = [w for w in text.split() if w not in stopwords]
    lemmas = [morph.parse(w)[0].normal_form for w in words]
    return " ".join(lemmas)


def predict_news_bundle(text: str, model_base_dir: str) -> dict[str, Any]:
    resources = load_all_models(model_base_dir)
    device = resources["device"]
    detector = resources["detector"]
    tokenizer = resources["tokenizer"]
    max_len = resources["max_len"]

    encoding = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = detector(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    prob_fake = float(probs[0])
    prob_true = float(probs[1])
    label = "TRUE" if prob_true >= prob_fake else "FAKE"
    confidence = max(prob_true, prob_fake)

    clean_text = preprocess_for_theme(text, resources["stopwords"], resources["morph"])
    vector = resources["theme_vectorizer"].transform([clean_text])
    theme_probs = resources["theme_model"].predict_proba(vector)[0]
    classes = list(resources["theme_model"].classes_)

    idx = int(theme_probs.argmax())
    theme = classes[idx]
    theme_conf = float(theme_probs[idx])

    theme_distribution = sorted(
        [{"theme": cls, "probability": float(prob)} for cls, prob in zip(classes, theme_probs)],
        key=lambda x: x["probability"],
        reverse=True,
    )

    return {
        "fake_label": label,
        "fake_confidence": confidence,
        "prob_true": prob_true,
        "prob_fake": prob_fake,
        "theme": theme,
        "theme_confidence": theme_conf,
        "theme_distribution": theme_distribution,
    }
