"""Inference runtime matching the preprocessing and artifacts from the training notebook."""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class DeploymentConfig:
    model_file: str = "best_model_final.keras"
    tokenizer_file: str = "best_tokenizer_final.pkl"
    label_encoder_file: str = "label_encoder.pkl"
    max_len: int = 300
    text_column: str = "text"
    label_column: str = "text_category"
    model_name: str = "Best deep-learning model"
    task_type: str = "single_label_priority"
    priority_order: tuple[str, ...] = ("Problem", "Suggestion", "Appreciation", "Neutral")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeploymentConfig":
        known = {
            "model_file": data.get("model_file", cls.model_file),
            "tokenizer_file": data.get("tokenizer_file", cls.tokenizer_file),
            "label_encoder_file": data.get("label_encoder_file", cls.label_encoder_file),
            "max_len": int(data.get("max_len", cls.max_len)),
            "text_column": data.get("text_column", cls.text_column),
            "label_column": data.get("label_column", cls.label_column),
            "model_name": data.get("model_name", cls.model_name),
            "task_type": data.get("task_type", cls.task_type),
            "priority_order": tuple(data.get("priority_order", cls.priority_order)),
        }
        return cls(**known)


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    label_encoder: Any
    config: DeploymentConfig
    run_summary: dict[str, Any]


def remove_upper_case(text: Any) -> str:
    if pd.isna(text):
        return ""
    sentences = str(text).split("\n")
    new_sentences = []
    for sentence in sentences:
        words = sentence.split()
        new_sentences.append(" ".join(word.title() if word.isupper() else word for word in words))
    return "\n".join(new_sentences)


def basic_clean(text: Any) -> str:
    """Replicate the exact basic_clean pipeline from the notebook."""
    cleaned = remove_upper_case(text)
    cleaned = URL_PATTERN.sub("", cleaned)
    cleaned = HTML_PATTERN.sub("", cleaned)
    cleaned = EMOJI_PATTERN.sub("", cleaned)
    return str(cleaned).strip()


def read_deployment_config(model_dir: str | Path) -> DeploymentConfig:
    model_dir = Path(model_dir)
    config_path = model_dir / "deployment_config.json"
    if not config_path.exists():
        return DeploymentConfig()
    with config_path.open("r", encoding="utf-8") as file:
        return DeploymentConfig.from_dict(json.load(file))


def missing_artifacts(model_dir: str | Path, config: DeploymentConfig | None = None) -> list[str]:
    model_dir = Path(model_dir)
    config = config or read_deployment_config(model_dir)
    required = [config.model_file, config.tokenizer_file, config.label_encoder_file]
    return [name for name in required if not (model_dir / name).exists()]


def _load_keras_model(model_path: Path):
    import tensorflow as tf
    from custom_layers import MultiHeadSelfAttention, TokenAndPositionEmbedding, TransformerBlock

    custom_objects = {
        "MultiHeadSelfAttention": MultiHeadSelfAttention,
        "TransformerBlock": TransformerBlock,
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "PeerFeedback>MultiHeadSelfAttention": MultiHeadSelfAttention,
        "PeerFeedback>TransformerBlock": TransformerBlock,
        "PeerFeedback>TokenAndPositionEmbedding": TokenAndPositionEmbedding,
    }
    try:
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,
        )
    except TypeError:
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)


def load_bundle(model_dir: str | Path) -> ModelBundle:
    model_dir = Path(model_dir)
    config = read_deployment_config(model_dir)
    missing = missing_artifacts(model_dir, config)
    if missing:
        raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))

    model = _load_keras_model(model_dir / config.model_file)
    with (model_dir / config.tokenizer_file).open("rb") as file:
        tokenizer = pickle.load(file)
    with (model_dir / config.label_encoder_file).open("rb") as file:
        label_encoder = pickle.load(file)

    summary_path = model_dir / "run_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as file:
            run_summary = json.load(file)
    else:
        run_summary = {}

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        config=config,
        run_summary=run_summary,
    )


def prepare_inputs(tokenizer: Any, texts: Iterable[str], max_len: int) -> np.ndarray:
    from tensorflow.keras.preprocessing import sequence

    sequences = tokenizer.texts_to_sequences(list(texts))
    return sequence.pad_sequences(sequences, maxlen=max_len)


def predict_dataframe(bundle: ModelBundle, data: pd.DataFrame, text_column: str) -> pd.DataFrame:
    if text_column not in data.columns:
        raise KeyError(f"Column '{text_column}' was not found.")

    output = data.copy()
    output["cleaned_text"] = output[text_column].apply(basic_clean)
    valid_mask = output["cleaned_text"].str.strip().ne("")
    output["predicted_label"] = pd.NA
    output["confidence"] = np.nan

    classes = [str(label) for label in bundle.label_encoder.classes_]
    for label in classes:
        output[f"score_{label}"] = np.nan

    if not valid_mask.any():
        return output

    X = prepare_inputs(
        bundle.tokenizer,
        output.loc[valid_mask, "cleaned_text"].tolist(),
        bundle.config.max_len,
    )
    scores = np.asarray(bundle.model.predict(X, verbose=0))

    if scores.ndim == 1 or (scores.ndim == 2 and scores.shape[1] == 1):
        positive = scores.reshape(-1)
        predicted_ids = (positive >= 0.5).astype(int)
        score_matrix = np.column_stack([1.0 - positive, positive])
    else:
        score_matrix = scores
        predicted_ids = np.argmax(score_matrix, axis=1)

    predicted_labels = bundle.label_encoder.inverse_transform(predicted_ids)
    confidences = np.max(score_matrix, axis=1)
    valid_indices = output.index[valid_mask]
    output.loc[valid_indices, "predicted_label"] = predicted_labels
    output.loc[valid_indices, "confidence"] = confidences

    if score_matrix.shape[1] != len(classes):
        raise ValueError(
            f"Model returned {score_matrix.shape[1]} scores, but label encoder has {len(classes)} classes."
        )
    for class_id, label in enumerate(classes):
        output.loc[valid_indices, f"score_{label}"] = score_matrix[:, class_id]

    return output
