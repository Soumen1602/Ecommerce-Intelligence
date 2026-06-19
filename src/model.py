"""
EcomIQ Model Training & Evaluation Module
==========================================
Provides training wrappers for Logistic Regression, Random Forest, and
XGBoost classifiers, an evaluation helper that computes standard
classification metrics, SHAP explainability, and model persistence via
joblib.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
) -> LogisticRegression:
    """Train a Logistic Regression classifier.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training target vector.

    Returns
    -------
    LogisticRegression
        Fitted scikit-learn Logistic Regression model.
    """
    try:
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train, y_train)
        logger.info("Logistic Regression trained successfully.")
        return model
    except Exception as exc:
        logger.error("Error training Logistic Regression: %s", exc)
        raise


def train_random_forest(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
) -> RandomForestClassifier:
    """Train a Random Forest classifier.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training target vector.

    Returns
    -------
    RandomForestClassifier
        Fitted scikit-learn Random Forest model.
    """
    try:
        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train, y_train)
        logger.info("Random Forest trained successfully.")
        return model
    except Exception as exc:
        logger.error("Error training Random Forest: %s", exc)
        raise


def train_xgboost(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
) -> XGBClassifier:
    """Train an XGBoost classifier with automatic class-imbalance handling.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training target vector.

    Returns
    -------
    XGBClassifier
        Fitted XGBoost model.
    """
    try:
        y = np.asarray(y_train)
        n_negative = int((y == 0).sum())
        n_positive = int((y == 1).sum())
        scale_pos = n_negative / n_positive if n_positive > 0 else 1.0

        model = XGBClassifier(
            scale_pos_weight=scale_pos,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        logger.info(
            "XGBoost trained successfully (scale_pos_weight=%.2f).",
            scale_pos,
        )
        return model
    except Exception as exc:
        logger.error("Error training XGBoost: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
) -> dict[str, float]:
    """Evaluate a trained classifier on test data.

    Parameters
    ----------
    model : estimator
        Any fitted scikit-learn-compatible classifier.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True test labels.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``accuracy``, ``precision``, ``recall``,
        ``f1``, ``roc_auc``.
    """
    try:
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )

        metrics: dict[str, float] = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        }

        logger.info("Model evaluation: %s", metrics)
        return metrics

    except Exception as exc:
        logger.error("Error evaluating model: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def generate_shap_values(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[list[str]] = None,
) -> tuple:
    """Generate SHAP values for a trained model.

    Uses :class:`shap.TreeExplainer` for tree-based models (Random Forest,
    XGBoost) and :class:`shap.LinearExplainer` for linear models (Logistic
    Regression).

    Parameters
    ----------
    model : estimator
        Fitted model.
    X : array-like
        Feature matrix to explain.
    feature_names : list[str], optional
        Human-readable feature names.

    Returns
    -------
    tuple[np.ndarray, shap.Explainer]
        ``(shap_values, explainer)``
    """
    try:
        if isinstance(model, (RandomForestClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        elif isinstance(model, LogisticRegression):
            # LinearExplainer needs a background dataset / summary
            if isinstance(X, pd.DataFrame):
                background = shap.maskers.Independent(X, max_samples=100)
            else:
                background = shap.maskers.Independent(X, max_samples=100)
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X)
        else:
            # Fallback to KernelExplainer
            background = shap.sample(X, min(100, len(X)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X)

        logger.info("SHAP values generated for %d samples.", len(X))
        return shap_values, explainer

    except Exception as exc:
        logger.error("Error generating SHAP values: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, filepath: Union[str, Path]) -> None:
    """Persist a trained model to disk via joblib.

    Parameters
    ----------
    model : estimator
        Fitted model to save.
    filepath : str or Path
        Destination path (e.g. ``models/xgboost_churn.joblib``).
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, filepath)
        logger.info("Model saved to %s", filepath)
    except Exception as exc:
        logger.error("Error saving model to %s: %s", filepath, exc)
        raise


def load_model(filepath: Union[str, Path]) -> Any:
    """Load a persisted model from disk.

    Parameters
    ----------
    filepath : str or Path
        Path to the saved model file.

    Returns
    -------
    estimator
        The deserialized model object.
    """
    try:
        filepath = Path(filepath)
        model = joblib.load(filepath)
        logger.info("Model loaded from %s", filepath)
        return model
    except Exception as exc:
        logger.error("Error loading model from %s: %s", filepath, exc)
        raise


# ---------------------------------------------------------------------------
# Compare all models
# ---------------------------------------------------------------------------

def train_and_compare_all(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Train Logistic Regression, Random Forest, and XGBoost, then evaluate.

    Parameters
    ----------
    X_train, X_test : array-like
        Feature matrices.
    y_train, y_test : array-like
        Target vectors.

    Returns
    -------
    tuple[dict, dict]
        ``(models_dict, results_dict)`` where keys are
        ``'Logistic Regression'``, ``'Random Forest'``, ``'XGBoost'``.
        ``models_dict`` values are fitted estimators; ``results_dict``
        values are metric dicts from :func:`evaluate_model`.
    """
    try:
        models_dict: dict[str, Any] = {
            "Logistic Regression": train_logistic_regression(X_train, y_train),
            "Random Forest": train_random_forest(X_train, y_train),
            "XGBoost": train_xgboost(X_train, y_train),
        }

        results_dict: dict[str, dict[str, float]] = {}
        for name, model in models_dict.items():
            results_dict[name] = evaluate_model(model, X_test, y_test)
            logger.info("%-20s → %s", name, results_dict[name])

        return models_dict, results_dict

    except Exception as exc:
        logger.error("Error in train_and_compare_all: %s", exc)
        raise
