from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import pickle
import joblib
import re
import threading

import numpy as np
import pymorphy3
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
from huggingface_hub import snapshot_download


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
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # older PyTorch without weights_only param
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


# --- Multi-model per-run inference -------------------------------------------------
_MULTI_MODEL_CACHE: dict[str, dict[str, Any]] = {}


def _pretty_ckpt_name(path: Path) -> str:
    return path.stem.replace("_", " ")


def predict_all_models(text: str, model_base_dir: str) -> list[dict[str, Any]]:
    """Run all available saved models (fake & topic) on the given text and
    return a list of per-model result dicts. This loads checkpoints and pickles
    found under `experiments/models/*` and caches them by path.
    """
    base = Path(model_base_dir) / "experiments" / "models"
    results: list[dict[str, Any]] = []

    # load tokenizer if present
    tokenizer = None
    tok_path = base / "bert_tokenizer"
    if tok_path.exists():
        try:
            tokenizer = BertTokenizerFast.from_pretrained(tok_path)
        except Exception:
            tokenizer = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Helper to run a pytorch checkpoint (fake or topic)
    def _run_pt_checkpoint(path: Path, task: str):
        cache_key = str(path.resolve())
        entry = _MULTI_MODEL_CACHE.get(cache_key)
        if entry is None:
            ckpt = _load_checkpoint(path, device)
            head_type = _checkpoint_head_type(path, ckpt, ckpt.get("model_state_dict", ckpt))
            backbone = _backbone_name(ckpt)
            max_len = int(ckpt.get("max_len", 256))
            num_classes = int(ckpt.get("num_classes", 2))
            model_cls = HEAD_CLASSES.get(head_type, BertClassifierHeadC)
            model = model_cls(backbone, num_classes=num_classes).to(device)
            state = ckpt.get("model_state_dict", ckpt)
            try:
                model.load_state_dict(state)
            except Exception:
                # try direct assignment if weights_only style
                model.load_state_dict(state)
            model.eval()
            entry = {"model": model, "max_len": max_len}
            _MULTI_MODEL_CACHE[cache_key] = entry

        model = entry["model"]
        max_len = entry["max_len"]

        if tokenizer is None:
            raise RuntimeError("Tokenizer not available for BERT model inference")

        encoding = tokenizer(text, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        if task == "fake":
            prob_true = float(probs[1]) if len(probs) > 1 else float(probs[0])
            prob_fake = float(probs[0]) if len(probs) > 1 else 1.0 - prob_true
            label = "TRUE" if prob_true >= prob_fake else "FAKE"
            return {
                "name": _pretty_ckpt_name(path),
                "task": "fake",
                "prob_true": prob_true,
                "prob_fake": prob_fake,
                "fake_label": label,
                "fake_confidence": max(prob_true, prob_fake),
            }
        else:
            # topic
            # try to locate a label encoder sibling (pkl) in same folder or parent
            label_encoder = None
            candidate = path.with_name("topic_label_encoder.pkl")
            if not candidate.exists():
                candidate = base / "07" / "label_encoder.pkl"
            if candidate.exists():
                try:
                    with open(candidate, "rb") as f:
                        label_encoder = pickle.load(f)
                except Exception:
                    label_encoder = None

            theme_probs = probs.tolist()
            classes = list(getattr(label_encoder, "classes_", [])) if label_encoder else [str(i) for i in range(len(theme_probs))]
            idx = int(np.argmax(theme_probs))
            theme = classes[idx] if idx < len(classes) else str(idx)
            return {
                "name": _pretty_ckpt_name(path),
                "task": "topic",
                "theme": theme,
                "theme_confidence": float(theme_probs[idx]),
                "theme_distribution": sorted([
                    {"theme": cls, "probability": float(p)} for cls, p in zip(classes, theme_probs)
                ], key=lambda x: x["probability"], reverse=True),
            }

    # 1) Fake detectors (all bert_fake_*.pt)
    for ckpt in sorted(base.rglob("bert_fake*.pt")):
        try:
            out = _run_pt_checkpoint(ckpt, "fake")
            results.append(out)
        except Exception:
            continue

    # 2) Baseline logistic regression (if present)
    logreg_path = base / "01" / "baseline_logreg_model.pkl"
    vec_path = base / "01" / "baseline_logreg_vectorizer.pkl"
    if logreg_path.exists() and vec_path.exists():
        try:
            with open(logreg_path, "rb") as f:
                logreg = pickle.load(f)
            with open(vec_path, "rb") as f:
                vec = pickle.load(f)
            vec_text = vec.transform([text])
            proba = None
            if hasattr(logreg, "predict_proba"):
                proba = logreg.predict_proba(vec_text)[0]
            else:
                pred = logreg.predict(vec_text)[0]
                # fallback: treat predicted class as 1.0
                classes = getattr(logreg, "classes_", [0, 1])
                proba = [0.0, 1.0] if pred == classes[-1] else [1.0, 0.0]

            # assume class index 1 corresponds to TRUE
            prob_true = float(proba[1]) if len(proba) > 1 else float(proba[0])
            prob_fake = float(proba[0]) if len(proba) > 1 else 1.0 - prob_true
            label = "TRUE" if prob_true >= prob_fake else "FAKE"
            results.append(
                {
                    "name": "Baseline LogReg",
                    "task": "fake",
                    "prob_true": prob_true,
                    "prob_fake": prob_fake,
                    "fake_label": label,
                    "fake_confidence": max(prob_true, prob_fake),
                }
            )
        except Exception:
            pass

    # 3) Topic -> SVM pickles (topic_svm_*.pkl)
    for model_path in sorted(base.rglob("topic_svm*_model.pkl")):
        try:
            vec_path = model_path.with_name(model_path.name.replace("_model.pkl", "_vectorizer.pkl"))
            with open(model_path, "rb") as f:
                svm = pickle.load(f)
            with open(vec_path, "rb") as f:
                vec = pickle.load(f)
            _morph = pymorphy3.MorphAnalyzer(lang="uk")
            clean = preprocess_for_theme(text, set(), _morph)
            X = vec.transform([clean])
            probs = None
            if hasattr(svm, "predict_proba"):
                probs = svm.predict_proba(X)[0]
                classes = list(getattr(svm, "classes_", []))
                idx = int(np.argmax(probs))
                theme = classes[idx] if idx < len(classes) else str(idx)
                results.append(
                    {
                        "name": _pretty_ckpt_name(model_path),
                        "task": "topic",
                        "theme": theme,
                        "theme_confidence": float(probs[idx]),
                        "theme_distribution": sorted([
                            {"theme": cls, "probability": float(p)} for cls, p in zip(classes, probs)
                        ], key=lambda x: x["probability"], reverse=True),
                    }
                )
            else:
                pred = svm.predict(X)[0]
                results.append({"name": _pretty_ckpt_name(model_path), "task": "topic", "theme": str(pred), "theme_confidence": 1.0, "theme_distribution": []})
        except Exception:
            continue

    # 4) BERT topic checkpoints
    for ckpt in sorted(base.rglob("bert_topic*.pt")):
        try:
            out = _run_pt_checkpoint(ckpt, "topic")
            results.append(out)
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
#  Optimised selected-models inference
# ---------------------------------------------------------------------------
_SELECTED_MODELS_CACHE: dict[str, Any] = {}
_SELECTED_MODELS_LOCK = threading.Lock()

def _checkpoint_head_type(path: Path, ckpt: dict, weights: dict) -> str:
    """Detect if the checkpoint uses Head A, B, or C based on keys."""
    if "head_type" in ckpt:
        return ckpt["head_type"]
    keys = list(weights.keys())
    has_attn = any("attention_pool" in k for k in keys)
    has_bn = any("classifier.1.weight" in k for k in keys)
    if has_attn: return "C"
    if has_bn: return "B"
    return "A"

HEAD_CLASSES = {
    "A": BertClassifierHeadA,
    "B": BertClassifierHeadB,
    "C": BertClassifierHeadC
}

def load_selected_models(model_base_dir: str) -> dict[str, Any]:
    """Load only the 4 selected models for production inference."""
    cache_key = str(Path(model_base_dir).resolve())
    with _SELECTED_MODELS_LOCK:
        if cache_key in _SELECTED_MODELS_CACHE:
            return _SELECTED_MODELS_CACHE[cache_key]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- HUGGING FACE HUB SYNC ---
        REPO_ID = "yatsenq/news-detector-weights"
        print(f"Syncing models from {REPO_ID}...")
        
        try:
            weights_path = Path(snapshot_download(
                repo_id=REPO_ID, 
                repo_type="model",
                token=os.getenv("HF_TOKEN")
            ))
            exp = weights_path / "models"
        except Exception as e:
            print(f"Warning: Could not sync from HF Hub ({e}). Falling back to local paths.")
            base = Path(model_base_dir)
            exp = base / "experiments" / "models"

        # --- tokenizer --------------------------------------------------------
        tokenizer_path = str(exp / "bert_tokenizer")
        try:
            if os.path.exists(tokenizer_path):
                tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
            else:
                print(f"Tokenizer not found at {tokenizer_path}, falling back to online version.")
                tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
        except Exception as te:
            print(f"Error loading local tokenizer ({te}), using online fallback.")
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

        # --- 1. Baseline LogReg (fake) ----------------------------------------
        logreg_model = joblib.load(exp / "01" / "baseline_logreg_model.pkl")
        logreg_vec = joblib.load(exp / "01" / "baseline_logreg_vectorizer.pkl")

        # --- 2. BERT HeadC (fake) ---------------------------------------------
        fake_path = exp / "02" / "bert_fake_headc_best.pt"
        fake_ckpt = _load_checkpoint(fake_path, device)
        fake_weights = fake_ckpt.get("model_state_dict", fake_ckpt)
        fake_backbone = _backbone_name(fake_ckpt)
        fake_max_len = int(fake_ckpt.get("max_len", 256))
        
        # Авто-визначення типу голови
        fake_ht = _checkpoint_head_type(fake_path, fake_ckpt, fake_weights)
        fake_cls = HEAD_CLASSES.get(fake_ht, BertClassifierHeadC)
        
        bert_fake = fake_cls(fake_backbone, num_classes=2).to(device)
        bert_fake.load_state_dict(fake_weights, strict=False)
        bert_fake.eval()

        # --- 3. SVM expB (topic) ----------------------------------------------
        with open(exp / "05" / "topic_svm_expB_model.pkl", "rb") as f:
            svm_model = pickle.load(f)
        with open(exp / "05" / "topic_svm_expB_vectorizer.pkl", "rb") as f:
            svm_vec = pickle.load(f)

        # --- 4. BERT topic expB -----------------------------------------------
        topic_path = exp / "07" / "bert_topic_expb_detector_best.pt"
        topic_ckpt = _load_checkpoint(topic_path, device)
        topic_weights = topic_ckpt.get("model_state_dict", topic_ckpt)
        topic_backbone = _backbone_name(topic_ckpt)
        topic_max_len = int(topic_ckpt.get("max_len", 256))
        
        # Авто-визначення типу голови
        topic_ht = _checkpoint_head_type(topic_path, topic_ckpt, topic_weights)
        topic_cls = HEAD_CLASSES.get(topic_ht, BertClassifierHeadC)
        
        bert_topic = topic_cls(topic_backbone, num_classes=2).to(device)
        bert_topic.load_state_dict(topic_weights, strict=False)
        bert_topic.eval()

        with open(exp / "07" / "label_encoder.pkl", "rb") as f:
            topic_le = pickle.load(f)

        # --- shared helpers ---------------------------------------------------
        stopwords_path = exp / "stopwords_ua.txt"
        if not stopwords_path.exists():
            # fallback if it's in the root of the repo
            stopwords_path = weights_path / "stopwords_ua.txt"

        try:
            with open(stopwords_path, "r", encoding="utf-8") as f:
                stopwords = set(f.read().split())
        except Exception:
            stopwords = set()
            
        morph = pymorphy3.MorphAnalyzer(lang="uk")

        # Використовуємо SVM з експерименту 05 як основну модель для маркерів тем
        theme_vectorizer = svm_vec
        theme_model = svm_model

        resources = {
            "device": device, "tokenizer": tokenizer,
            "logreg_model": logreg_model, "logreg_vec": logreg_vec,
            "bert_fake": bert_fake, "bert_fake_max_len": fake_max_len,
            "svm_model": svm_model, "svm_vec": svm_vec,
            "bert_topic": bert_topic, "bert_topic_max_len": topic_max_len,
            "topic_le": topic_le,
            "stopwords": stopwords, "morph": morph,
            "theme_vectorizer": theme_vectorizer, "theme_model": theme_model,
        }
        _SELECTED_MODELS_CACHE[cache_key] = resources
        return resources


def predict_selected_models(text: str, model_base_dir: str) -> dict[str, Any]:
    """Run the 4 selected models and return a combined result dict."""
    res = load_selected_models(model_base_dir)
    device, tok = res["device"], res["tokenizer"]
    per_model: list[dict[str, Any]] = []

    # 1. LogReg fake
    try:
        X = res["logreg_vec"].transform([text])
        lr = res["logreg_model"]
        if hasattr(lr, "predict_proba"):
            proba = lr.predict_proba(X)[0]
        else:
            pred = lr.predict(X)[0]
            cl = getattr(lr, "classes_", [0, 1])
            proba = [0.0, 1.0] if pred == cl[-1] else [1.0, 0.0]
        pt = float(proba[1]) if len(proba) > 1 else float(proba[0])
        pf = float(proba[0]) if len(proba) > 1 else 1.0 - pt
        per_model.append({"name": "Baseline LogReg", "task": "fake",
                          "prob_true": pt, "prob_fake": pf,
                          "fake_label": "TRUE" if pt >= pf else "FAKE",
                          "fake_confidence": max(pt, pf)})
    except Exception:
        pass

    # 2. BERT HeadC fake
    try:
        enc = tok(text, max_length=res["bert_fake_max_len"],
                  padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = res["bert_fake"](enc["input_ids"].to(device),
                                     enc["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pt = float(probs[1]) if len(probs) > 1 else float(probs[0])
        pf = float(probs[0]) if len(probs) > 1 else 1.0 - pt
        per_model.append({"name": "BERT Head C (AttPool)", "task": "fake",
                          "prob_true": pt, "prob_fake": pf,
                          "fake_label": "TRUE" if pt >= pf else "FAKE",
                          "fake_confidence": max(pt, pf)})
    except Exception:
        pass

    # 3. SVM expB topic
    try:
        clean = preprocess_for_theme(text, res["stopwords"], res["morph"])
        X = res["svm_vec"].transform([clean])
        svm = res["svm_model"]
        if hasattr(svm, "predict_proba"):
            probs = svm.predict_proba(X)[0]
            classes = list(getattr(svm, "classes_", []))
            idx = int(np.argmax(probs))
            theme = classes[idx] if idx < len(classes) else str(idx)
            per_model.append({"name": "SVM Exp-B", "task": "topic",
                              "theme": theme,
                              "theme_confidence": float(probs[idx]),
                              "theme_distribution": sorted(
                                  [{"theme": c, "probability": float(p)}
                                   for c, p in zip(classes, probs)],
                                  key=lambda x: x["probability"], reverse=True)})
        else:
            pred = svm.predict(X)[0]
            per_model.append({"name": "SVM Exp-B", "task": "topic",
                              "theme": str(pred), "theme_confidence": 1.0,
                              "theme_distribution": []})
    except Exception:
        pass

    # 4. BERT topic expB
    try:
        enc = tok(text, max_length=res["bert_topic_max_len"],
                  padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = res["bert_topic"](enc["input_ids"].to(device),
                                      enc["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        classes = list(getattr(res["topic_le"], "classes_", []))
        if not classes:
            classes = [str(i) for i in range(len(probs))]
        idx = int(np.argmax(probs))
        theme = classes[idx] if idx < len(classes) else str(idx)
        per_model.append({"name": "BERT Topic Exp-B", "task": "topic",
                          "theme": theme,
                          "theme_confidence": float(probs[idx]),
                          "theme_distribution": sorted(
                              [{"theme": c, "probability": float(p)}
                               for c, p in zip(classes, probs)],
                              key=lambda x: x["probability"], reverse=True)})
    except Exception:
        pass

    # --- build combined result ------------------------------------------------
    fake_r = [m for m in per_model if m["task"] == "fake"]
    topic_r = [m for m in per_model if m["task"] == "topic"]
    
    result: dict[str, Any] = {}
    
    if fake_r:
        avg_pt = sum(m["prob_true"] for m in fake_r) / len(fake_r)
        avg_pf = sum(m["prob_fake"] for m in fake_r) / len(fake_r)
        result["prob_true"] = avg_pt
        result["prob_fake"] = avg_pf
        result["fake_label"] = "TRUE" if avg_pt >= avg_pf else "FAKE"
        result["fake_confidence"] = max(avg_pt, avg_pf)
        
    best_t = max(topic_r, key=lambda x: x["theme_confidence"], default=None)
    if best_t:
        result.update({k: best_t[k] for k in
                       ("theme", "theme_confidence", "theme_distribution")})

    try:
        clean_text = preprocess_for_theme(text, res["stopwords"], res["morph"])
        tv = res["theme_vectorizer"].transform([clean_text])
        result["theme_markers"] = _build_theme_markers(
            res["theme_model"], res["theme_vectorizer"],
            result.get("theme", ""), tv)
    except Exception:
        result["theme_markers"] = []

    try:
        lv = res["logreg_vec"].transform([text])
        lr = res["logreg_model"]
        feature_names = res["logreg_vec"].get_feature_names_out()
        non_zero = lv.nonzero()[1]
        fake_markers = []
        for idx in non_zero:
            coef = float(lr.coef_[0][idx])
            # Assuming negative is FAKE, positive is TRUE (typical for binary logreg)
            fake_markers.append({"keyword": feature_names[idx], "score": coef})
        # take top 15 most impactful words
        result["fake_markers"] = sorted(fake_markers, key=lambda x: abs(x["score"]), reverse=True)[:15]
    except Exception:
        result["fake_markers"] = []

    result["input_text"] = text
    result["per_model_results"] = per_model
    return result
