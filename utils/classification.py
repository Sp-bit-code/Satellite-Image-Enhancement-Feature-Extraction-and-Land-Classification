from __future__ import annotations

import io
import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow import keras

load_model = keras.models.load_model
Model = keras.Model


class ClassificationError(Exception):
    """Custom exception for classification-related errors."""
    pass


LabelInput = Union[List[str], Tuple[str, ...], str]


DEFAULT_EUROSAT_LABELS = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


@dataclass
class PredictionConfig:
    """
    Configuration for model inference.

    target_size:
        Model input size as (width, height)

    normalize:
        Whether model preprocessing should be applied

    color_mode:
        'rgb' or 'bgr'

    confidence_threshold:
        Threshold for accepted prediction

    top_k:
        Number of top predictions to return

    preprocessing_mode:
        'mobilenet_v2', 'divide_255', or 'none'
    """
    target_size: Tuple[int, int] = (224, 224)
    normalize: bool = True
    color_mode: str = "rgb"
    confidence_threshold: float = 0.0
    top_k: int = 3
    preprocessing_mode: str = "mobilenet_v2"


def _validate_uint8_image(image: np.ndarray) -> np.ndarray:
    """
    Validate image input and convert to uint8 if needed.
    """
    if image is None:
        raise ClassificationError("Input image is None.")

    if not isinstance(image, np.ndarray):
        raise ClassificationError("Input image must be a numpy array.")

    if image.size == 0:
        raise ClassificationError("Input image is empty.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim not in (2, 3):
        raise ClassificationError(
            f"Unsupported image shape: {image.shape}. Expected 2D or 3D."
        )

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ClassificationError(
            f"Unsupported number of channels: {image.shape[2]}."
        )

    return image


def _load_image_from_bytes_with_exif_fix(file_bytes: bytes) -> np.ndarray:
    """
    Load image from bytes with EXIF orientation fix.
    Returns OpenCV BGR image.
    """
    try:
        pil_image = Image.open(io.BytesIO(file_bytes))
        pil_image = ImageOps.exif_transpose(pil_image)

        if pil_image.mode not in ("RGB", "RGBA", "L"):
            pil_image = pil_image.convert("RGB")

        image_np = np.array(pil_image)

        if image_np.ndim == 2:
            return cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

        if image_np.shape[2] == 4:
            return cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)

        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    except Exception as exc:
        raise ClassificationError(f"Failed to decode uploaded image: {exc}") from exc


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB.
    """
    image = _validate_uint8_image(image)

    if image.ndim == 2:
        return image

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert grayscale/RGBA/RGB-like input into 3-channel BGR.
    """
    image = _validate_uint8_image(image)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image.copy()


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resize image to model input size.
    target_size = (width, height)
    """
    image = _validate_uint8_image(image)

    if (
        not isinstance(target_size, (tuple, list))
        or len(target_size) != 2
        or target_size[0] <= 0
        or target_size[1] <= 0
    ):
        raise ClassificationError(
            f"Invalid target_size: {target_size}. Must be (width, height)."
        )

    return cv2.resize(image, tuple(target_size), interpolation=interpolation)


def safe_load_model(model_path: str) -> Model:
    """
    Load a saved Keras model safely.
    Prefer .keras format.
    """
    if not isinstance(model_path, str) or not model_path.strip():
        raise ClassificationError("Model path must be a non-empty string.")

    if not os.path.exists(model_path):
        raise ClassificationError(f"Model file not found: {model_path}")

    errors: List[str] = []

    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as exc:
        errors.append(f"Attempt 1 failed: {exc}")

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False,
        )
        return model
    except TypeError:
        errors.append("Attempt 2 skipped: safe_mode not supported in this TensorFlow version.")
    except Exception as exc:
        errors.append(f"Attempt 2 failed: {exc}")

    joined_errors = " | ".join(errors)
    raise ClassificationError(f"Failed to load model: {joined_errors}")


def load_labels(labels_source: LabelInput) -> List[str]:
    """
    Load class labels from:
    - Python list / tuple
    - .json file
    - .pkl file
    - .txt file
    """
    if isinstance(labels_source, (list, tuple)):
        labels = [str(x).strip() for x in labels_source if str(x).strip()]
        if not labels:
            raise ClassificationError("Provided label list is empty.")
        return labels

    if not isinstance(labels_source, str):
        raise ClassificationError(
            "labels_source must be a list/tuple of labels or a file path."
        )

    if not os.path.exists(labels_source):
        raise ClassificationError(f"Labels file not found: {labels_source}")

    ext = os.path.splitext(labels_source)[1].lower()

    try:
        if ext == ".json":
            with open(labels_source, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                try:
                    items = sorted(data.items(), key=lambda x: int(x[0]))
                    return [str(v) for _, v in items]
                except Exception:
                    return [str(v) for v in data.values()]

            if isinstance(data, list):
                return [str(x) for x in data]

            raise ClassificationError("Unsupported JSON label structure.")

        if ext == ".pkl":
            with open(labels_source, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, (list, tuple)):
                return [str(x) for x in data]

            if hasattr(data, "classes_"):
                return [str(x) for x in data.classes_]

            raise ClassificationError("Unsupported pickle label structure.")

        if ext == ".txt":
            with open(labels_source, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]
            if not labels:
                raise ClassificationError("Text label file is empty.")
            return labels

    except Exception as exc:
        raise ClassificationError(f"Failed to load labels: {exc}") from exc

    raise ClassificationError(
        f"Unsupported label file format: {ext}. Use .json, .pkl, or .txt"
    )


def get_labels_for_model(
    labels_path: Optional[str] = None,
    fallback_labels: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Prefer saved labels file. Fall back to provided labels or EuroSAT defaults.
    """
    if labels_path and os.path.exists(labels_path):
        labels = load_labels(labels_path)
        if labels:
            return labels

    if fallback_labels:
        labels = [str(x).strip() for x in fallback_labels if str(x).strip()]
        if labels:
            return labels

    return DEFAULT_EUROSAT_LABELS.copy()


def infer_model_target_size(model: Model) -> Tuple[int, int]:
    """
    Infer target input size from model input shape.
    Returns (width, height)
    """
    try:
        input_shape = model.input_shape
    except Exception as exc:
        raise ClassificationError(f"Could not read model input shape: {exc}") from exc

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if (
        not isinstance(input_shape, tuple)
        or len(input_shape) != 4
        or input_shape[1] is None
        or input_shape[2] is None
    ):
        raise ClassificationError(
            f"Unsupported model input shape: {input_shape}"
        )

    height = int(input_shape[1])
    width = int(input_shape[2])

    return (width, height)


def apply_model_preprocessing(
    image: np.ndarray,
    mode: str = "mobilenet_v2",
) -> np.ndarray:
    """
    Apply the exact preprocessing expected by the trained model.
    """
    if mode == "mobilenet_v2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(image)

    if mode == "divide_255":
        return image / 255.0

    if mode == "none":
        return image

    raise ClassificationError(
        f"Unsupported preprocessing_mode: {mode}. "
        "Use 'mobilenet_v2', 'divide_255', or 'none'."
    )


def preprocess_image_for_model(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    color_mode: str = "rgb",
    preprocessing_mode: str = "mobilenet_v2",
    center_crop: bool = False,
) -> np.ndarray:
    """
    Preprocess a single image for model prediction.

    Steps:
    - validate
    - convert to 3-channel BGR
    - resize
    - convert to RGB if needed
    - apply model-specific preprocessing
    - add batch dimension
    """
    image = _validate_uint8_image(image)

    if color_mode not in {"rgb", "bgr"}:
        raise ClassificationError("color_mode must be 'rgb' or 'bgr'.")

    image = convert_to_bgr(image)

    # Important:
    # Do not force center crop by default.
    # EuroSAT training from directory loader usually resizes directly,
    # so center crop can change semantics and hurt predictions.
    image = resize_image(image, target_size)

    if color_mode == "rgb":
        image = convert_bgr_to_rgb(image)

    image = image.astype(np.float32)

    if normalize:
        image = apply_model_preprocessing(image, mode=preprocessing_mode)

    image = np.expand_dims(image, axis=0)
    return image


def predict_probabilities(
    model: Model,
    image: np.ndarray,
    config: Optional[PredictionConfig] = None,
) -> np.ndarray:
    """
    Predict raw class probabilities for one image.
    """
    config = config or PredictionConfig()

    batch = preprocess_image_for_model(
        image=image,
        target_size=config.target_size,
        normalize=config.normalize,
        color_mode=config.color_mode,
        preprocessing_mode=config.preprocessing_mode,
        center_crop=False,
    )

    try:
        preds = model.predict(batch, verbose=0)
    except Exception as exc:
        raise ClassificationError(f"Model prediction failed: {exc}") from exc

    preds = np.asarray(preds)

    if preds.ndim == 2:
        if preds.shape[0] != 1:
            raise ClassificationError(
                f"Unexpected prediction shape: {preds.shape}. Expected (1, num_classes)."
            )
        return preds[0]

    if preds.ndim == 1:
        return preds

    raise ClassificationError(
        f"Unexpected prediction shape: {preds.shape}. Expected (1, num_classes) or (num_classes,)."
    )


def predict_single_image(
    model: Model,
    image: np.ndarray,
    class_labels: Sequence[str],
    config: Optional[PredictionConfig] = None,
) -> Dict[str, Any]:
    """
    Predict class for a single image.
    """
    config = config or PredictionConfig()
    class_labels = list(class_labels)

    probabilities = predict_probabilities(model, image, config=config)

    if len(class_labels) != len(probabilities):
        raise ClassificationError(
            f"Number of labels ({len(class_labels)}) does not match "
            f"model outputs ({len(probabilities)})."
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_confidence = float(probabilities[predicted_index])
    predicted_label = class_labels[predicted_index]

    top_k = max(1, min(config.top_k, len(class_labels)))
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    top_predictions = [
        {
            "class_index": int(idx),
            "class_label": class_labels[idx],
            "confidence": float(probabilities[idx]),
            "confidence_percent": float(probabilities[idx] * 100.0),
        }
        for idx in top_indices
    ]

    accepted = predicted_confidence >= config.confidence_threshold

    return {
        "predicted_index": predicted_index,
        "predicted_label": predicted_label,
        "confidence": predicted_confidence,
        "confidence_percent": predicted_confidence * 100.0,
        "accepted": accepted,
        "threshold": config.confidence_threshold,
        "probabilities": probabilities,
        "top_predictions": top_predictions,
    }


def predict_batch_images(
    model: Model,
    images: Sequence[np.ndarray],
    class_labels: Sequence[str],
    config: Optional[PredictionConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Predict classes for multiple images.
    """
    if not images:
        raise ClassificationError("No images provided for batch prediction.")

    results: List[Dict[str, Any]] = []
    for image in images:
        result = predict_single_image(
            model=model,
            image=image,
            class_labels=class_labels,
            config=config,
        )
        results.append(result)

    return results


def format_prediction_for_streamlit(prediction: Dict[str, Any]) -> Dict[str, str]:
    """
    Format prediction output for clean Streamlit display.
    """
    return {
        "Predicted Class": str(prediction["predicted_label"]),
        "Confidence": f'{prediction["confidence_percent"]:.2f}%',
        "Accepted": "Yes" if prediction["accepted"] else "No",
        "Threshold": f'{prediction["threshold"] * 100:.2f}%',
    }


def get_probability_table(prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build table from top predictions.
    """
    rows = []
    for item in prediction["top_predictions"]:
        rows.append(
            {
                "Class Index": item["class_index"],
                "Class Label": item["class_label"],
                "Confidence": item["confidence"],
                "Confidence (%)": item["confidence_percent"],
            }
        )
    return rows


def get_bar_chart_data(
    prediction: Dict[str, Any],
    class_labels: Sequence[str],
) -> Dict[str, List[Any]]:
    """
    Build chart-ready probability data.
    """
    probabilities = prediction["probabilities"]
    return {
        "labels": list(class_labels),
        "probabilities": [float(x) for x in probabilities],
        "probabilities_percent": [float(x * 100.0) for x in probabilities],
    }


def build_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Build confusion matrix without sklearn.
    """
    if len(y_true) != len(y_pred):
        raise ClassificationError("y_true and y_pred must have the same length.")

    if len(y_true) == 0:
        raise ClassificationError("y_true and y_pred cannot be empty.")

    if num_classes is None:
        num_classes = max(max(y_true), max(y_pred)) + 1

    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1

    return cm


def compute_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """
    Compute accuracy.
    """
    if len(y_true) != len(y_pred):
        raise ClassificationError("y_true and y_pred must have the same length.")

    if len(y_true) == 0:
        raise ClassificationError("Input arrays cannot be empty.")

    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def get_classification_report_data(
    confusion_matrix: np.ndarray,
    class_labels: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Compute precision/recall/F1 from confusion matrix.
    """
    cm = np.asarray(confusion_matrix)

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ClassificationError("Confusion matrix must be square.")

    n_classes = cm.shape[0]
    if len(class_labels) != n_classes:
        raise ClassificationError(
            "Number of class labels must match confusion matrix size."
        )

    report = []

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        report.append(
            {
                "class_index": i,
                "class_label": class_labels[i],
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "support": int(cm[i, :].sum()),
            }
        )

    return report


def predict_from_uploaded_file(
    uploaded_file: Any,
    model: Model,
    class_labels: Sequence[str],
    config: Optional[PredictionConfig] = None,
) -> Dict[str, Any]:
    """
    Convenience function for Streamlit uploaded file prediction.
    """
    if uploaded_file is None:
        raise ClassificationError("No uploaded file provided.")

    file_bytes = uploaded_file.read()
    if not file_bytes:
        raise ClassificationError("Uploaded file is empty.")

    image = _load_image_from_bytes_with_exif_fix(file_bytes)

    return predict_single_image(
        model=model,
        image=image,
        class_labels=class_labels,
        config=config,
    )


if __name__ == "__main__":
    MODEL_PATH = "model/land_classifier_model.keras"
    LABELS_PATH = "model/class_labels.json"
    SAMPLE_IMAGE_PATH = "data/sample_images/sample.jpg"

    try:
        model = safe_load_model(MODEL_PATH)
        inferred_size = infer_model_target_size(model)
        labels = get_labels_for_model(LABELS_PATH, DEFAULT_EUROSAT_LABELS)

        print("Labels used for model:")
        for idx, label in enumerate(labels):
            print(f"{idx}: {label}")

        if not os.path.exists(SAMPLE_IMAGE_PATH):
            raise ClassificationError(f"Sample image not found: {SAMPLE_IMAGE_PATH}")

        image = cv2.imread(SAMPLE_IMAGE_PATH)
        if image is None:
            raise ClassificationError("Failed to load sample image.")

        config = PredictionConfig(
            target_size=inferred_size,
            normalize=True,
            color_mode="rgb",
            confidence_threshold=0.0,
            top_k=3,
            preprocessing_mode="mobilenet_v2",
        )

        prediction = predict_single_image(
            model=model,
            image=image,
            class_labels=labels,
            config=config,
        )

        print("\nPredicted Label:", prediction["predicted_label"])
        print("Confidence: {:.2f}%".format(prediction["confidence_percent"]))
        print("Top Predictions:")
        for item in prediction["top_predictions"]:
            print(f" - {item['class_label']}: {item['confidence_percent']:.2f}%")

    except Exception as exc:
        print(f"Error: {exc}")