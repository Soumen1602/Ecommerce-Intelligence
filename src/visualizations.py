"""
EcomIQ Visualizations Module
=============================
Plotly-based charting functions for the EcomIQ dashboard.  Every figure is
rendered with a consistent light theme (``#000000`` background, electric-teal
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
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Source Serif 4", color="#6B6358", size=12),
    title=dict(font=dict(size=14, color="#2B2620", family="Source Serif 4")),
    legend=dict(bgcolor="#FFFFFF", font=dict(color="#6B6358")),
    xaxis=dict(gridcolor="#D8D0C2", zerolinecolor="#D8D0C2"),
    yaxis=dict(gridcolor="#D8D0C2", zerolinecolor="#D8D0C2"),
    hoverlabel=dict(
        bgcolor="#FFFFFF", bordercolor="#B5562F", font=dict(color="#2B2620")
    ),
    margin=dict(l=40, r=40, t=50, b=40),
)

# Palette
TEAL = "#B5562F"  # Terracotta
VIOLET = "#4A6651"  # Forest
AMBER = "#C19A3F"  # Gold
RED = "#B5562F"  # Terracotta


# ---------------------------------------------------------------------------
# Theme helper
# ---------------------------------------------------------------------------

def apply_theme(fig: go.Figure, is_dark: bool = False) -> go.Figure:
    """Apply dynamic theme to a Plotly figure."""
    
    # Text position for pie charts to prevent overlapping
    for trace in fig.data:
        if isinstance(trace, go.Pie):
            trace.textposition = 'outside'
            trace.textinfo = 'label+percent'

    if is_dark:
        bg = "#0F172A"
        surface = "#1E293B"
        text = "#F8FAFC"
        text_muted = "#94A3B8"
        rule = "#334155"
        accent = "#38BDF8"
    else:
        bg = "#F6F2EA"
        surface = "#FFFFFF"
        text = "#2B2620"
        text_muted = "#6B6358"
        rule = "#D8D0C2"
        accent = "#B5562F"

    layout = dict(
        paper_bgcolor=surface, # Or 'rgba(0,0,0,0)' if we want transparent
        plot_bgcolor=surface,
        font=dict(family="Inter", color=text_muted, size=12),
        title=dict(font=dict(size=14, color=text, family="Source Serif 4")),
        legend=dict(bgcolor=surface, font=dict(color=text_muted)),
        xaxis=dict(gridcolor=rule, zerolinecolor=rule, tickfont=dict(color=text_muted)),
        yaxis=dict(gridcolor=rule, zerolinecolor=rule, tickfont=dict(color=text_muted)),
        hoverlabel=dict(bgcolor=surface, bordercolor=accent, font=dict(color=text)),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    
    try:
        fig.update_layout(**layout)
        return fig
    except Exception as exc:
        logger.error("Error applying theme: %s", exc)
        raise


# ---------------------------------------------------------------------------
# EDA charts
# ---------------------------------------------------------------------------

def create_monthly_revenue_trend(df: pd.DataFrame, is_dark: bool = False) -> go.Figure:
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
        return apply_theme(fig, is_dark)

    except Exception as exc:
        logger.error("Error creating monthly revenue trend: %s", exc)
        raise


def create_top_categories_chart(
    df: pd.DataFrame, top_n: int = 10, is_dark: bool = False
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
        return apply_theme(fig, is_dark)

    except Exception as exc:
        logger.error("Error creating top categories chart: %s", exc)
        raise


def create_order_status_distribution(df: pd.DataFrame, is_dark: bool = False) -> go.Figure:
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
                marker=dict(colors=[TEAL, VIOLET, AMBER, RED, "#64748B", "#94A3B8", "#A78BFA", "#FB923C"]),
                textposition="outside",
                textinfo="label+percent",
                automargin=True,
            )
        )
        fig.update_layout(title="Order Status Distribution", margin=dict(l=80, r=80, t=60, b=60))
        return apply_theme(fig, is_dark)

    except Exception as exc:
        logger.error("Error creating order status distribution: %s", exc)
        raise


def create_state_distribution(df: pd.DataFrame, is_dark: bool = False) -> go.Figure:
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
        return apply_theme(fig, is_dark)

    except Exception as exc:
        logger.error("Error creating state distribution chart: %s", exc)
        raise


def create_review_distribution(df: pd.DataFrame, is_dark: bool = False) -> go.Figure:
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
                textfont=dict(color="#2B2620"),
            )
        )
        fig.update_layout(
            title="Review Score Distribution",
            xaxis_title="Review Score",
            yaxis_title="Count",
        )
        return apply_theme(fig, is_dark)

    except Exception as exc:
        logger.error("Error creating review distribution chart: %s", exc)
        raise


def create_payment_breakdown(df: pd.DataFrame, is_dark: bool = False) -> go.Figure:
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
                textposition="outside",
                textinfo="label+percent",
                automargin=True,
            )
        )
        fig.update_layout(title="Payment Type Breakdown", margin=dict(l=80, r=80, t=60, b=60))
        return apply_theme(fig, is_dark)

    except Exception as exc:
        logger.error("Error creating payment breakdown chart: %s", exc)
        raise


def create_delivery_performance(df: pd.DataFrame, is_dark: bool = False) -> go.Figure:
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
        return apply_theme(fig, is_dark)

    except Exception as exc:
        logger.error("Error creating delivery performance chart: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Model / prediction charts
# ---------------------------------------------------------------------------

def create_churn_gauge(probability: float, is_dark: bool = False) -> go.Figure:
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
        text_color = "#F8FAFC" if is_dark else "#2B2620"
        tick_color = "#94A3B8" if is_dark else "#6B6358"
        bar_color = "#38BDF8" if is_dark else "#2B2620"
        bg_color = "#1E293B" if is_dark else "#FFFFFF"

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pct,
                number=dict(suffix="%", font=dict(color=text_color, size=36)),
                title=dict(
                    text="Churn Probability",
                    font=dict(color=text_color, size=14),
                ),
                gauge=dict(
                    axis=dict(
                        range=[0, 100],
                        tickcolor=tick_color,
                        tickfont=dict(color=tick_color),
                    ),
                    bar=dict(color=bar_color),
                    bgcolor=bg_color,
                    steps=[
                        dict(range=[0, 40], color="#4A6651"),
                        dict(range=[40, 70], color="#C19A3F"),
                        dict(range=[70, 100], color="#B5562F"),
                    ],
                    threshold=dict(
                        line=dict(color=text_color, width=3),
                        thickness=0.8,
                        value=pct,
                    ),
                ),
            )
        )
        return apply_theme(fig, is_dark)

    except Exception as exc:
        logger.error("Error creating churn gauge: %s", exc)
        raise


def create_roc_curves(
    models_dict: dict[str, Any],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    is_dark: bool = False,
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
        return apply_theme(fig, is_dark)

    except Exception as exc:
        logger.error("Error creating ROC curves: %s", exc)
        raise


def create_confusion_matrix_chart(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    is_dark: bool = False,
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
            [0.0, "#1E293B" if is_dark else "#F6F2EA"],
            [0.5, "#334155" if is_dark else "#E3CCB9"],
            [1.0, "#38BDF8" if is_dark else "#B5562F"],
        ]

        fig = go.Figure(
            go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale=colorscale,
                text=cm,
                texttemplate="%{text}",
                textfont=dict(color="#2B2620", size=16),
                showscale=False,
            )
        )
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
        )
        return apply_theme(fig, is_dark)

    except Exception as exc:
        logger.error("Error creating confusion matrix chart: %s", exc)
        raise


def create_feature_importance(
    model: Any,
    feature_names: list[str],
    is_dark: bool = False,
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
        return apply_theme(fig, is_dark)

    except Exception as exc:
        logger.error("Error creating feature importance chart: %s", exc)
        raise
