"""
EcomIQ Data Loader Module
=========================
Handles loading, merging, and summarizing the Olist e-commerce dataset.
All 9 CSV files are loaded via pathlib, timestamp columns are parsed,
and a master dataframe is produced by joining across entities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
TIMESTAMP_COLS: dict[str, list[str]] = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
    "items": ["shipping_limit_date"],
}

CSV_MAP: dict[str, str] = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all_datasets(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all 9 Olist CSV files from *data_dir* and return a keyed dict.

    Parameters
    ----------
    data_dir : Path
        Path to the directory containing the raw CSV files
        (e.g. ``data/raw``).

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary keyed by short name:
        ``orders``, ``items``, ``customers``, ``payments``, ``reviews``,
        ``products``, ``sellers``, ``geolocation``, ``category_translation``.

    Raises
    ------
    FileNotFoundError
        If any expected CSV file is missing from *data_dir*.
    """
    data_dir = Path(data_dir)
    datasets: dict[str, pd.DataFrame] = {}

    for short_name, filename in CSV_MAP.items():
        filepath = data_dir / filename
        try:
            parse_dates = TIMESTAMP_COLS.get(short_name, False) or False
            df = pd.read_csv(filepath, parse_dates=parse_dates)
            datasets[short_name] = df
            logger.info(
                "Loaded %-25s → %d rows, %d cols",
                filename,
                len(df),
                len(df.columns),
            )
        except FileNotFoundError:
            logger.error("File not found: %s", filepath)
            raise FileNotFoundError(
                f"Expected CSV not found: {filepath}"
            ) from None
        except Exception as exc:
            logger.error("Error loading %s: %s", filepath, exc)
            raise

    return datasets


def merge_master_dataframe(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge individual dataframes into a single master dataframe.

    Join order:
        orders → customers   (on ``customer_id``)
        orders → items       (on ``order_id``)
        items  → products    (on ``product_id``)
        orders → payments    (on ``order_id``)
        orders → reviews     (on ``order_id``)

    Product category names are translated to English using
    ``category_translation``.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary returned by :func:`load_all_datasets`.

    Returns
    -------
    pd.DataFrame
        Merged master dataframe.
    """
    try:
        # Start with orders ← customers
        master = datasets["orders"].merge(
            datasets["customers"], on="customer_id", how="left"
        )

        # orders ← items
        master = master.merge(
            datasets["items"], on="order_id", how="left"
        )

        # items ← products
        master = master.merge(
            datasets["products"], on="product_id", how="left"
        )

        # orders ← payments
        master = master.merge(
            datasets["payments"], on="order_id", how="left"
        )

        # orders ← reviews
        master = master.merge(
            datasets["reviews"], on="order_id", how="left"
        )

        # Translate product categories to English
        translation = datasets["category_translation"]
        master = master.merge(
            translation, on="product_category_name", how="left"
        )

        logger.info(
            "Master dataframe built: %d rows × %d cols",
            len(master),
            len(master.columns),
        )
        return master

    except KeyError as exc:
        logger.error("Missing required dataset key: %s", exc)
        raise
    except Exception as exc:
        logger.error("Error during merge: %s", exc)
        raise


def get_data_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute high-level KPIs from the master dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe produced by :func:`merge_master_dataframe`.

    Returns
    -------
    dict
        Keys: ``total_orders``, ``total_revenue``, ``avg_order_value``,
        ``total_customers``, ``date_range`` (tuple of min/max timestamps).
    """
    try:
        total_orders = int(df["order_id"].nunique())

        # Revenue = sum of price + freight_value per unique order-item row
        total_revenue = float(
            df[["price", "freight_value"]].sum().sum()
        )

        avg_order_value = (
            total_revenue / total_orders if total_orders > 0 else 0.0
        )

        total_customers = int(df["customer_unique_id"].nunique())

        purchase_col = "order_purchase_timestamp"
        if purchase_col in df.columns:
            ts = pd.to_datetime(df[purchase_col], errors="coerce")
            date_range = (ts.min(), ts.max())
        else:
            date_range = (None, None)

        summary: dict[str, Any] = {
            "total_orders": total_orders,
            "total_revenue": round(total_revenue, 2),
            "avg_order_value": round(avg_order_value, 2),
            "total_customers": total_customers,
            "date_range": date_range,
        }

        logger.info("Data summary: %s", summary)
        return summary

    except Exception as exc:
        logger.error("Error computing summary: %s", exc)
        raise
