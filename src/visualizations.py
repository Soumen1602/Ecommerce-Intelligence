"""
EcomIQ Visualizations Module
=============================
Plotly-based charting functions for the EcomIQ dashboard.  Every figure is
rendered with a consistent dark theme (``#000000`` background, electric-teal
accents) before being returned.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT: dict = dict(
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(family="Inter", color="#a1a1aa", size=12),
    title=dict(font=dict(size=14, color="#ededed", family="Inter")),
    legend=dict(bgcolor="#0a0a0a", font=dict(color="#a1a1aa")),
    xaxis=dict(gridcolor="#111111", zerolinecolor="#111111"),
    yaxis=dict(gridcolor="#111111", zerolinecolor="#111111"),
    hoverlabel=dict(
        bgcolor="#111111", bordercolor="#0070f3", font=dict(color="#ededed")
    ),
    margin=dict(l=40, r=40, t=50, b=40),
)

# Palette
TEAL = "#0070f3"
VIOLET = "#7928ca"
AMBER = "#F59E0B"
RED = "#EF4444"


# ---------------------------------------------------------------------------
# Theme helper
# ---------------------------------------------------------------------------

def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply the EcomIQ dark theme to a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        Any Plotly figure.

    Returns
    -------
    go.Figure
        The same figure with the dark theme layout applied.
    """
    try:
        fig.update_layout(**PLOTLY_LAYOUT)
        return fig
    except Exception as exc:
        logger.error("Error applying dark theme: %s", exc)
        raise


# ---------------------------------------------------------------------------
# EDA charts
# ---------------------------------------------------------------------------

def create_monthly_revenue_trend(df: pd.DataFrame) -> go.Figure:
    """Line + area chart of monthly revenue over time.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with ``order_purchase_timestamp``, ``price``,
        and ``freight_value`` columns.

    Returns
    -------
    go.Figure
    """
    try:
        df = df.copy()
        df["order_purchase_timestamp"] = pd.to_datetime(
            df["order_purchase_timestamp"], errors="coerce"
        )
        df["revenue"] = df["price"].fillna(0) + df["freight_value"].fillna(0)
        df["month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

        monthly = (
            df.groupby("month", sort=True)["revenue"]
            .sum()
            .reset_index()
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=monthly["month"],
                y=monthly["revenue"],
                mode="lines+markers",
                line=dict(color=TEAL, width=2),
                marker=dict(size=5, color=TEAL),
                fill="tozeroy",
                fillcolor="rgba(0, 112, 243, 0.15)",
                name="Revenue",
            )
        )
        fig.update_layout(
            title="Monthly Revenue Trend",
            xaxis_title="Month",
            yaxis_title="Revenue (BRL)",
        )
        return apply_dark_theme(fig)

    except Exception as exc:
        logger.error("Error creating monthly revenue trend: %s", exc)
        raise


def create_top_categories_chart(
    df: pd.DataFrame, top_n: int = 10
) -> go.Figure:
    """Horizontal bar chart of the top-selling product categories.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with ``product_category_name_english``.
    top_n : int, optional
        Number of categories to display (default 10).

    Returns
    -------
    go.Figure
    """
    try:
        cat_counts = (
            df["product_category_name_english"]
            .value_counts()
            .head(top_n)
            .sort_values()
        )

        fig = go.Figure(
            go.Bar(
                x=cat_counts.values,
                y=cat_counts.index,
                orientation="h",
                marker_color=TEAL,
            )
        )
        fig.update_layout(
            title=f"Top {top_n} Product Categories",
            xaxis_title="Number of Orders",
            yaxis_title="Category",
        )
        return apply_dark_theme(fig)

    except Exception as exc:
        logger.error("Error creating top categories chart: %s", exc)
        raise


def create_order_status_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart of order status distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with ``order_status``.

    Returns
    -------
    go.Figure
    """
    try:
        status_counts = df["order_status"].value_counts()

        fig = go.Figure(
            go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                hole=0.45,
                marker=dict(colors=[TEAL, VIOLET, AMBER, RED, "#64748B"]),
                textinfo="percent+label",
                textfont=dict(color="#ededed"),
            )
        )
        fig.update_layout(title="Order Status Distribution")
        return apply_dark_theme(fig)

    except Exception as exc:
        logger.error("Error creating order status distribution: %s", exc)
        raise


def create_state_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of the top 15 customer states.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with ``customer_state``.

    Returns
    -------
    go.Figure
    """
    try:
        state_counts = (
            df["customer_state"]
            .value_counts()
            .head(15)
            .sort_values(ascending=True)
        )

        fig = go.Figure(
            go.Bar(
                x=state_counts.values,
                y=state_counts.index,
                orientation="h",
                marker_color=TEAL,
            )
        )
        fig.update_layout(
            title="Top 15 Customer States",
            xaxis_title="Number of Orders",
            yaxis_title="State",
        )
        return apply_dark_theme(fig)

    except Exception as exc:
        logger.error("Error creating state distribution chart: %s", exc)
        raise


def create_review_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of review scores (1–5) with a red-to-teal gradient.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with ``review_score``.

    Returns
    -------
    go.Figure
    """
    try:
        score_counts = (
            df["review_score"]
            .value_counts()
            .sort_index()
        )
        # Ensure scores 1-5 are present
        score_counts = score_counts.reindex([1, 2, 3, 4, 5], fill_value=0)

        # Gradient from RED (1) → TEAL (5)
        colors = [RED, "#f5a623", AMBER, "#50e3c2", TEAL]

        fig = go.Figure(
            go.Bar(
                x=score_counts.index.astype(str),
                y=score_counts.values,
                marker_color=colors,
                text=score_counts.values,
                textposition="auto",
                textfont=dict(color="#ededed"),
            )
        )
        fig.update_layout(
            title="Review Score Distribution",
            xaxis_title="Review Score",
            yaxis_title="Count",
        )
        return apply_dark_theme(fig)

    except Exception as exc:
        logger.error("Error creating review distribution chart: %s", exc)
        raise


def create_payment_breakdown(df: pd.DataFrame) -> go.Figure:
    """Donut chart of payment type distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with ``payment_type``.

    Returns
    -------
    go.Figure
    """
    try:
        payment_counts = df["payment_type"].value_counts()

        fig = go.Figure(
            go.Pie(
                labels=payment_counts.index,
                values=payment_counts.values,
                hole=0.45,
                marker=dict(colors=[TEAL, VIOLET, AMBER, RED, "#64748B"]),
                textinfo="percent+label",
                textfont=dict(color="#ededed"),
            )
        )
        fig.update_layout(title="Payment Type Breakdown")
        return apply_dark_theme(fig)

    except Exception as exc:
        logger.error("Error creating payment breakdown chart: %s", exc)
        raise


def create_delivery_performance(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing actual vs. estimated delivery days.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with ``order_purchase_timestamp``,
        ``order_delivered_customer_date``, and
        ``order_estimated_delivery_date``.

    Returns
    -------
    go.Figure
    """
    try:
        df = df.copy()
        for col in [
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]:
            df[col] = pd.to_datetime(df[col], errors="coerce")

        df["actual_days"] = (
            df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
        ).dt.days
        df["estimated_days"] = (
            df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
        ).dt.days

        # Group by month
        df["month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
        monthly = (
            df.groupby("month")[["actual_days", "estimated_days"]]
            .mean()
            .reset_index()
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=monthly["month"],
                y=monthly["actual_days"],
                name="Actual Days",
                marker_color=TEAL,
            )
        )
        fig.add_trace(
            go.Bar(
                x=monthly["month"],
                y=monthly["estimated_days"],
                name="Estimated Days",
                marker_color=VIOLET,
            )
        )
        fig.update_layout(
            title="Delivery Performance: Actual vs Estimated",
            xaxis_title="Month",
            yaxis_title="Days",
            barmode="group",
        )
        return apply_dark_theme(fig)

    except Exception as exc:
        logger.error("Error creating delivery performance chart: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Model / prediction charts
# ---------------------------------------------------------------------------

def create_churn_gauge(probability: float) -> go.Figure:
    """Gauge chart displaying a single customer's churn probability.

    Parameters
    ----------
    probability : float
        Churn probability in the range ``[0, 1]``.  Displayed as a
        percentage (0–100).

    Returns
    -------
    go.Figure
    """
    try:
        pct = probability * 100

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pct,
                number=dict(suffix="%", font=dict(color="#ededed", size=36)),
                title=dict(
                    text="Churn Probability",
                    font=dict(color="#ededed", size=14),
                ),
                gauge=dict(
                    axis=dict(
                        range=[0, 100],
                        tickcolor="#a1a1aa",
                        tickfont=dict(color="#a1a1aa"),
                    ),
                    bar=dict(color="#111111"),
                    bgcolor="#111111",
                    steps=[
                        dict(range=[0, 40], color=TEAL),
                        dict(range=[40, 70], color=AMBER),
                        dict(range=[70, 100], color=RED),
                    ],
                    threshold=dict(
                        line=dict(color="#ededed", width=3),
                        thickness=0.8,
                        value=pct,
                    ),
                ),
            )
        )
        return apply_dark_theme(fig)

    except Exception as exc:
        logger.error("Error creating churn gauge: %s", exc)
        raise


def create_roc_curves(
    models_dict: dict[str, Any],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
) -> go.Figure:
    """Overlay ROC curves for multiple models.

    Parameters
    ----------
    models_dict : dict[str, estimator]
        Keyed by model name; values are fitted classifiers.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True test labels.

    Returns
    -------
    go.Figure
    """
    try:
        colors = {
            "Logistic Regression": TEAL,
            "Random Forest": VIOLET,
            "XGBoost": AMBER,
        }
        fig = go.Figure()

        for name, model in models_dict.items():
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            color = colors.get(name, "#64748B")

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2),
                )
            )

        # Diagonal reference
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#64748B", dash="dash", width=1),
                showlegend=False,
            )
        )

        fig.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        return apply_dark_theme(fig)

    except Exception as exc:
        logger.error("Error creating ROC curves: %s", exc)
        raise


def create_confusion_matrix_chart(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> go.Figure:
    """Heatmap visualisation of the confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    go.Figure
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        labels = ["Not Churned", "Churned"]

        # Custom colorscale from dark background to teal
        colorscale = [
            [0.0, "#000000"],
            [0.5, "#00326e"],
            [1.0, TEAL],
        ]

        fig = go.Figure(
            go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale=colorscale,
                text=cm,
                texttemplate="%{text}",
                textfont=dict(color="#ededed", size=16),
                showscale=False,
            )
        )
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
        )
        return apply_dark_theme(fig)

    except Exception as exc:
        logger.error("Error creating confusion matrix chart: %s", exc)
        raise


def create_feature_importance(
    model: Any,
    feature_names: list[str],
) -> go.Figure:
    """Horizontal bar chart of feature importances.

    Parameters
    ----------
    model : estimator
        A fitted model with ``feature_importances_`` (tree models) or
        ``coef_`` (linear models) attribute.
    feature_names : list[str]
        Feature names matching the model's input order.

    Returns
    -------
    go.Figure
    """
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            raise AttributeError(
                "Model has neither 'feature_importances_' nor 'coef_'."
            )

        indices = np.argsort(importances)
        sorted_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        fig = go.Figure(
            go.Bar(
                x=sorted_importances,
                y=sorted_names,
                orientation="h",
                marker_color=TEAL,
            )
        )
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
        )
        return apply_dark_theme(fig)

    except Exception as exc:
        logger.error("Error creating feature importance chart: %s", exc)
        raise
