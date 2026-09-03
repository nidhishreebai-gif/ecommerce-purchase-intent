"""Exploratory analysis and reproducible model training for purchase intent.

The dataset is split before the sklearn pipeline is fitted so that feature
scaling cannot learn anything from the held-out test set.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "target"
FEATURES = [
    "pages_viewed",
    "session_minutes",
    "products_viewed",
    "cart_additions",
    "discount_seen",
    "previous_orders",
]


def _save_eda_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Save a compact overview of distributions and class balance."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for axis, feature in zip(axes, FEATURES):
        for label, group in df.groupby(TARGET):
            axis.hist(
                group[feature],
                bins=12,
                alpha=0.6,
                label=f"Intent {label}",
                edgecolor="white",
            )
        axis.set_title(feature.replace("_", " ").title())
        axis.set_xlabel("Value")
        axis.set_ylabel("Sessions")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("E-commerce purchase intent: feature distributions", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_data_quality_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Save explicit class-balance and feature-correlation visualizations."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    df[TARGET].value_counts().sort_index().plot.bar(
        ax=axes[0], color=["#94a3b8", "#75aa37"], rot=0, edgecolor="white"
    )
    axes[0].set_title("Target class distribution")
    axes[0].set_xlabel("Target (0 = no purchase, 1 = purchase)")
    axes[0].set_ylabel("Sessions")
    axes[0].grid(axis="y", alpha=0.2)

    correlation = df[FEATURES + [TARGET]].corr()
    image = axes[1].imshow(correlation, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("Feature correlation heatmap")
    axes[1].set_xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
    axes[1].set_yticks(range(len(correlation.columns)), correlation.columns)
    for row in range(len(correlation)):
        for column in range(len(correlation)):
            axes[1].text(column, row, f"{correlation.iloc[row, column]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_and_correlation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_evaluation_plot(
    y_test: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """Save confusion matrix and ROC curve for the held-out test set."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["No purchase", "Purchase"],
        cmap="Blues",
        colorbar=False,
        ax=axes[0],
    )
    axes[0].set_title("Confusion matrix")
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)
    axes[1].plot(false_positive_rate, true_positive_rate, color="#2563eb", lw=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1)
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_and_report() -> dict:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "dataset_04_ecommerce_purchase_intent.csv"
    model_dir = project_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    expected_columns = FEATURES + [TARGET]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Unexpected dataset columns. Expected {expected_columns}, got {list(df.columns)}")
    if df[expected_columns].isnull().any().any():
        raise ValueError("The dataset contains missing values.")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("Data types:")
    print(df.dtypes.to_string())
    print(f"Missing values: {int(df.isnull().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print(f"Unique values:\n{df.nunique().to_string()}")
    print(f"Target distribution:\n{df[TARGET].value_counts().sort_index().to_string()}")
    print(f"Target percentages:\n{(df[TARGET].value_counts(normalize=True).sort_index() * 100).round(2).to_string()}")
    print(f"Feature ranges:\n{df[FEATURES].agg(['min', 'max']).to_string()}")

    X = df[FEATURES]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    matrix = confusion_matrix(y_test, predictions).tolist()

    coefficients = pipeline.named_steps["logistic_regression"].coef_[0]
    feature_importance = [
        {
            "feature": feature,
            "coefficient": float(coefficient),
            "absolute_coefficient": float(abs(coefficient)),
        }
        for feature, coefficient in zip(FEATURES, coefficients)
    ]
    feature_importance.sort(key=lambda item: item["absolute_coefficient"], reverse=True)

    summary = {}
    for feature in FEATURES:
        summary[feature] = {
            "min": float(df[feature].min()),
            "max": float(df[feature].max()),
            "mean": float(df[feature].mean()),
            "median": float(df[feature].median()),
        }
    report = {
        "dataset": {
            "file": "data/dataset_04_ecommerce_purchase_intent.csv",
            "rows": int(len(df)),
            "features": FEATURES,
            "target": TARGET,
            "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "feature_summary": summary,
        },
        "training": {
            "model": "Logistic Regression",
            "pipeline": ["StandardScaler", "LogisticRegression"],
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "stratified": True,
        },
        "metrics": metrics,
        "confusion_matrix": matrix,
        "roc_curve": {
            "false_positive_rate": [float(value) for value in false_positive_rate],
            "true_positive_rate": [float(value) for value in true_positive_rate],
        },
        "feature_importance": feature_importance,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    dump(pipeline, model_dir / "model.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (model_dir / "feature_metadata.json").write_text(
        json.dumps(
            {
                "features": FEATURES,
                "target": TARGET,
                "feature_summary": summary,
                "class_labels": {"0": "No purchase", "1": "Purchase"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _save_eda_plot(df, model_dir / "eda_distributions.png")
    _save_data_quality_plots(df, model_dir)
    _save_evaluation_plot(y_test, predictions, probabilities, model_dir / "model_evaluation.png")

    print(json.dumps({"metrics": metrics, "confusion_matrix": matrix}, indent=2))
    return report


if __name__ == "__main__":
    train_and_report()
