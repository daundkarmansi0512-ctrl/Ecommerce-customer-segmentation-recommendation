"""
segmentation_pipeline.py: Reusable functions for customer segmentation.
This module handles loading, cleaning, feature engineering, preprocessing,
clustering, and persisting models and metadata.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# Constants
RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "AOV",
    "ProductVariety",
    "ReturnRatio",
    "AvgQtyPerOrder",
    "ActiveSpan",
]
RAW_REQUIRED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


class Preprocessor:
    """Handles clipping, log transformation, and scaling using fitted parameters."""
    
    def __init__(self, feature_columns: list, lower_bounds: dict, upper_bounds: dict, 
                 log_features: list, scaler: StandardScaler, reference_date: pd.Timestamp):
        self.feature_columns = feature_columns
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.log_features = log_features
        self.scaler = scaler
        self.reference_date = reference_date

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply clipping, log1p transformation, and standard scaling to the feature DataFrame."""
        df = features.copy()
        
        # Verify required columns are present
        missing = set(self.feature_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # 1. Clip outliers using bounds computed during training
        for col in self.feature_columns:
            df[col] = df[col].clip(
                lower=self.lower_bounds[col], 
                upper=self.upper_bounds[col]
            )
            
        # 2. Log-transform skewed features
        for col in self.log_features:
            df[col] = np.log1p(df[col])

        # 3. Standard scale the features
        scaled_values = self.scaler.transform(df[self.feature_columns])
        
        return pd.DataFrame(
            scaled_values, 
            columns=self.feature_columns, 
            index=features.index
        )


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load the raw retail CSV dataset and verify its columns."""
    df = pd.read_csv(csv_path, low_memory=False)
    missing = set(RAW_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df


def audit_raw_data(df: pd.DataFrame) -> dict:
    """Audit the raw dataset to identify quality issues (missing values, duplicates, etc.)."""
    invoice_str = df["Invoice"].astype(str)
    dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    
    # Check returns (cancelled orders start with 'C', or have negative quantity)
    is_cancelled = invoice_str.str.startswith("C")
    is_negative_qty = df["Quantity"] < 0
    is_non_product = ~df["StockCode"].astype(str).str.match(r"^\d")

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "unique_customers": df["Customer ID"].nunique(),
        "unique_invoices": df["Invoice"].nunique(),
        "unique_stock_codes": df["StockCode"].nunique(),
        "unique_descriptions": df["Description"].nunique(),
        "unique_countries": df["Country"].nunique(),
        "date_min": dates.min(),
        "date_max": dates.max(),
        "invalid_dates": int(dates.isna().sum()),
        "missing_values": df.isna().sum().to_dict(),
        "exact_duplicates": int(df.duplicated().sum()),
        "cancelled_rows": int(is_cancelled.sum()),
        "cancelled_invoices": df.loc[is_cancelled, "Invoice"].nunique(),
        "negative_quantity_rows": int(is_negative_qty.sum()),
        "non_cancelled_negative_quantity_rows": int((~is_cancelled & is_negative_qty).sum()),
        "zero_quantity_rows": int((df["Quantity"] == 0).sum()),
        "zero_price_rows": int((df["Price"] == 0).sum()),
        "negative_price_rows": int((df["Price"] < 0).sum()),
        "non_product_rows": int(is_non_product.sum()),
        "non_product_codes": df.loc[is_non_product, "StockCode"].nunique(),
    }


def clean_transactions(df: pd.DataFrame) -> tuple:
    """Clean the raw transactions by filtering customer IDs, non-product codes, and prices."""
    cleaned = df.copy()
    report = {"starting_rows": len(cleaned)}

    # Drop missing customer IDs
    cleaned = cleaned.dropna(subset=["Customer ID"])
    cleaned["Customer ID"] = cleaned["Customer ID"].astype(int)
    report["after_customer_id"] = len(cleaned)

    # Drop exact duplicates
    cleaned = cleaned.drop_duplicates()
    report["after_deduplication"] = len(cleaned)

    # Convert invoice dates
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="raise")

    # Filter to product stock codes only (must start with a digit)
    is_product = cleaned["StockCode"].astype(str).str.match(r"^\d")
    cleaned = cleaned[is_product].copy()
    report["after_product_filter"] = len(cleaned)

    # Filter to positive unit prices
    cleaned = cleaned[cleaned["Price"] > 0].copy()
    report["after_price_filter"] = len(cleaned)

    # Mark cancellations and returns
    cleaned["is_cancelled"] = cleaned["Invoice"].astype(str).str.startswith("C")
    cleaned["is_return"] = cleaned["Quantity"] < 0
    cleaned["LineTotal"] = cleaned["Quantity"] * cleaned["Price"]

    # Split into purchases (positive quantity) and returns (cancellations/returns)
    is_return_txn = cleaned["is_cancelled"] | cleaned["is_return"]
    purchases = cleaned[~is_return_txn].copy()
    returns = cleaned[is_return_txn].copy()

    # Integrity assertions
    assert not (purchases["Quantity"] <= 0).any(), "Purchases must have positive quantities"
    assert not (purchases["Price"] <= 0).any(), "Purchases must have positive unit prices"

    report.update({
        "purchases": len(purchases),
        "returns": len(returns),
        "model_customers": purchases["Customer ID"].nunique(),
        "model_products": purchases["StockCode"].nunique(),
        "description_missing_after_cleaning": int(cleaned["Description"].isna().sum()),
    })

    return purchases, returns, cleaned, report


def build_customer_features(purchases_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 8 customer-level features (RFM + behavioral metrics)."""
    if purchases_df.empty:
        raise ValueError("At least one purchase transaction is required.")

    # Reference date is the day after the last transaction in the dataset
    reference_date = purchases_df["InvoiceDate"].max() + pd.Timedelta(days=1)
    cust_groups = purchases_df.groupby("Customer ID")

    # Core RFM Features
    recency = (reference_date - cust_groups["InvoiceDate"].max()).dt.days.rename("Recency")
    frequency = cust_groups["Invoice"].nunique().rename("Frequency")
    monetary = cust_groups["LineTotal"].sum().rename("Monetary")
    
    # Extended Behavioral Features
    aov = (monetary / frequency).rename("AOV")
    variety = cust_groups["StockCode"].nunique().rename("ProductVariety")

    # Return Ratio (returned quantity / purchased quantity)
    purchased_qty = cust_groups["Quantity"].sum()
    returned_qty = (
        returns_df.assign(abs_return_qty=returns_df["Quantity"].abs())
        .groupby("Customer ID")["abs_return_qty"]
        .sum()
        .reindex(purchased_qty.index, fill_value=0.0)
    )
    return_ratio = (returned_qty / purchased_qty).fillna(0.0).rename("ReturnRatio")

    # Order metrics
    order_quantities = purchases_df.groupby(["Customer ID", "Invoice"])["Quantity"].sum()
    avg_qty_per_order = order_quantities.groupby("Customer ID").mean().rename("AvgQtyPerOrder")
    active_span = (cust_groups["InvoiceDate"].max() - cust_groups["InvoiceDate"].min()).dt.days.rename("ActiveSpan")

    # Combine all features
    features_df = pd.concat([
        recency, frequency, monetary, aov, variety, return_ratio, avg_qty_per_order, active_span
    ], axis=1).dropna()
    
    features_df.index.name = "CustomerID"
    return features_df.reset_index()


def fit_preprocessor(customer_df: pd.DataFrame) -> tuple:
    """Fit boundaries and standard scaler to prepare features for clustering."""
    model_df = customer_df[FEATURE_COLUMNS].copy()

    # Determine 1st and 99th percentiles for winsorization clipping
    lower_bounds = {col: float(model_df[col].quantile(0.01)) for col in FEATURE_COLUMNS}
    upper_bounds = {col: float(model_df[col].quantile(0.99)) for col in FEATURE_COLUMNS}

    # Clip values
    for col in FEATURE_COLUMNS:
        model_df[col] = model_df[col].clip(lower=lower_bounds[col], upper=upper_bounds[col])

    # Identify features with high skewness (|skew| > 1) for log transformation
    log_features = [col for col in FEATURE_COLUMNS if abs(model_df[col].skew()) > 1]
    for col in log_features:
        model_df[col] = np.log1p(model_df[col])

    # Fit StandardScaler
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(model_df)
    scaled_df = pd.DataFrame(scaled_values, columns=FEATURE_COLUMNS, index=customer_df.index)

    preprocessor = Preprocessor(
        feature_columns=FEATURE_COLUMNS,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        log_features=log_features,
        scaler=scaler,
        reference_date=pd.NaT
    )

    return preprocessor, scaled_df


def evaluate_kmeans(scaled_features: pd.DataFrame, k_range=range(2, 11), random_state=RANDOM_STATE) -> pd.DataFrame:
    """Evaluate K-Means on multiple k values using inertia, silhouette, CH, and DB scores."""
    rows = []
    X = scaled_features.values
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=random_state)
        labels = kmeans.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": kmeans.inertia_,
            "silhouette": silhouette_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        })
    return pd.DataFrame(rows).set_index("k")


def evaluate_agglomerative(scaled_features: pd.DataFrame, k_range=range(2, 11)) -> pd.DataFrame:
    """Evaluate Ward Agglomerative clustering on multiple k values for comparison."""
    rows = []
    X = scaled_features.values
    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(X)
        rows.append({
            "k": k,
            "silhouette": silhouette_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        })
    return pd.DataFrame(rows).set_index("k")


def select_final_k(kmeans_evaluation: pd.DataFrame, preferred_k=4) -> int:
    """Select the best number of clusters, defaulting to preferred_k if silhouette is close."""
    if preferred_k not in kmeans_evaluation.index:
        raise ValueError(f"Preferred k ({preferred_k}) not present in evaluation data.")
    
    best_sil = kmeans_evaluation.loc[3:, "silhouette"].max()
    # If the silhouette score of preferred_k is within 85% of the absolute best score (for k>=3), use it
    if kmeans_evaluation.loc[preferred_k, "silhouette"] >= 0.85 * best_sil:
        return preferred_k
    return int(kmeans_evaluation.loc[3:, "silhouette"].idxmax())


def build_cluster_profiles(customer_df: pd.DataFrame) -> tuple:
    """Build cluster profiles using both medians and means on original unscaled features."""
    profile_median = customer_df.groupby("Cluster")[FEATURE_COLUMNS].median().round(2)
    profile_median["CustomerCount"] = customer_df.groupby("Cluster").size()
    profile_median["Pct"] = (profile_median["CustomerCount"] / len(customer_df) * 100).round(1)
    profile_median = profile_median[["CustomerCount", "Pct"] + FEATURE_COLUMNS]

    profile_mean = customer_df.groupby("Cluster")[FEATURE_COLUMNS].mean().round(2)
    return profile_median, profile_mean


def assign_personas(profile_median: pd.DataFrame) -> dict:
    """Assign data-driven business persona names and descriptions based on cluster medians."""
    personas = {}
    n_clusters = len(profile_median)
    
    for cluster_id in profile_median.index:
        row = profile_median.loc[cluster_id]
        
        # Rank features to implement rule-based logic
        recency_rank = profile_median["Recency"].rank().loc[cluster_id] # Lower is better
        monetary_rank = profile_median["Monetary"].rank(ascending=False).loc[cluster_id]
        frequency_rank = profile_median["Frequency"].rank(ascending=False).loc[cluster_id]
        return_rank = profile_median["ReturnRatio"].rank(ascending=False).loc[cluster_id]

        is_recent = recency_rank <= n_clusters * 0.4
        is_high_value = monetary_rank <= 2
        is_frequent = frequency_rank <= 2
        is_dormant = recency_rank >= n_clusters * 0.75
        is_high_return = return_rank == 1 and row["ReturnRatio"] > 0.1

        if is_high_return:
            name = "High-Return Risk"
            desc = "Highest median observed return ratio; investigate quality or satisfaction."
        elif is_high_value and is_frequent and is_recent:
            name = "Champions"
            desc = "High spend, frequent, and recent purchasers."
        elif is_high_value and is_dormant:
            name = "At-Risk High-Value"
            desc = "High historical spend but less recently active."
        elif is_recent and not is_high_value and is_frequent:
            name = "Loyal Regulars"
            desc = "Recently active, frequent purchasers with moderate spend."
        elif is_recent and not is_frequent:
            name = "Recent / Growing"
            desc = "Recently active customers with limited purchase history."
        elif is_dormant and not is_high_value:
            name = "Dormant / Lapsed"
            desc = "Longer time since purchase with lower engagement."
        else:
            name = "Occasional Buyers"
            desc = "Moderate engagement across the observed behaviours."

        personas[int(cluster_id)] = {
            "name": name,
            "description": desc,
            "customer_count": int(row["CustomerCount"]),
            "recency_median": float(row["Recency"]),
            "frequency_median": float(row["Frequency"]),
            "monetary_median": float(row["Monetary"]),
            "aov_median": float(row["AOV"]),
            "product_variety_median": float(row["ProductVariety"]),
            "return_ratio_median": float(row["ReturnRatio"]),
        }
    return personas


def fit_segmentation(purchases_df: pd.DataFrame, returns_df: pd.DataFrame, 
                     preferred_k=4, random_state=RANDOM_STATE) -> dict:
    """Fit the full customer segmentation model and return a dictionary of result objects."""
    customer_df = build_customer_features(purchases_df, returns_df)
    preprocessor, scaled_features = fit_preprocessor(customer_df)
    preprocessor.reference_date = purchases_df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # Evaluate models
    kmeans_evaluation = evaluate_kmeans(scaled_features, random_state=random_state)
    agg_evaluation = evaluate_agglomerative(scaled_features)
    
    # Determine best K and fit K-Means
    final_k = select_final_k(kmeans_evaluation, preferred_k=preferred_k)
    kmeans = KMeans(n_clusters=final_k, init="k-means++", n_init=10, max_iter=300, random_state=random_state)
    customer_df["Cluster"] = kmeans.fit_predict(scaled_features.values)

    # Profile segments and assign personas
    profile_median, profile_mean = build_cluster_profiles(customer_df)
    personas = assign_personas(profile_median)
    customer_df["Persona"] = customer_df["Cluster"].map({cid: val["name"] for cid, val in personas.items()})

    # Run PCA for visualization coordinates
    pca = PCA(n_components=2, random_state=random_state)
    pca_coordinates = pd.DataFrame(
        pca.fit_transform(scaled_features.values), 
        columns=["PC1", "PC2"], 
        index=customer_df.index
    )
    pca_coordinates["Cluster"] = customer_df["Cluster"].values

    return {
        "customer_df": customer_df,
        "preprocessor": preprocessor,
        "kmeans": kmeans,
        "final_k": final_k,
        "random_state": random_state,
        "kmeans_evaluation": kmeans_evaluation,
        "agglomerative_evaluation": agg_evaluation,
        "profile_median": profile_median,
        "profile_mean": profile_mean,
        "personas": personas,
        "pca": pca,
        "pca_coordinates": pca_coordinates,
    }


def calculate_new_customer_features(purchase_transactions: pd.DataFrame, 
                                    return_transactions: pd.DataFrame, 
                                    reference_date: pd.Timestamp) -> pd.DataFrame:
    """Calculate the 8 features for a single customer based on their transaction history."""
    if purchase_transactions.empty:
        raise ValueError("At least one purchase is required.")
    if purchase_transactions["Customer ID"].nunique() != 1:
        raise ValueError("Requires transactions for exactly one customer.")
    if (purchase_transactions["Quantity"] <= 0).any() or (purchase_transactions["Price"] <= 0).any():
        raise ValueError("Requires cleaned positive purchases only.")

    customer_id = int(purchase_transactions["Customer ID"].iloc[0])
    first_purchase = purchase_transactions["InvoiceDate"].min()
    last_purchase = purchase_transactions["InvoiceDate"].max()
    frequency = purchase_transactions["Invoice"].nunique()
    monetary = float(purchase_transactions["LineTotal"].sum())
    purchased_qty = float(purchase_transactions["Quantity"].sum())

    returned_qty = 0.0
    if return_transactions is not None and not return_transactions.empty:
        returned_qty = float(return_transactions["Quantity"].abs().sum())

    row = {
        "CustomerID": customer_id,
        "Recency": int((reference_date - last_purchase).days),
        "Frequency": int(frequency),
        "Monetary": monetary,
        "AOV": monetary / frequency,
        "ProductVariety": int(purchase_transactions["StockCode"].nunique()),
        "ReturnRatio": returned_qty / purchased_qty if purchased_qty > 0 else 0.0,
        "AvgQtyPerOrder": float(purchase_transactions.groupby("Invoice")["Quantity"].sum().mean()),
        "ActiveSpan": int((last_purchase - first_purchase).days),
    }
    return pd.DataFrame([row])


def assign_new_customer(purchase_transactions: pd.DataFrame, 
                        return_transactions: pd.DataFrame, 
                        preprocessor: Preprocessor, 
                        kmeans: KMeans, 
                        personas: dict) -> dict:
    """Predict the cluster and persona for a new customer using a fitted preprocessor and model."""
    features = calculate_new_customer_features(
        purchase_transactions, return_transactions, preprocessor.reference_date
    )
    scaled = preprocessor.transform(features)
    cluster = int(kmeans.predict(scaled.values)[0])
    
    return {
        "customer_id": int(features.loc[0, "CustomerID"]),
        "cluster": cluster,
        "persona": str(personas[cluster]["name"]),
        "features": features.loc[0, ["CustomerID"] + FEATURE_COLUMNS].to_dict(),
        "scaled_features": scaled.loc[0].to_dict(),
    }


def save_segmentation_artifacts(result: dict, audit: dict, cleaning_report: dict, 
                                output_dir="artifacts") -> dict:
    """Persist the preprocessor, model, personas, and metrics to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save fitted model & parameters using joblib
    model_path = output_path / "segmentation_model.joblib"
    joblib.dump(
        {
            "preprocessor": result["preprocessor"],
            "kmeans": result["kmeans"],
            "personas": result["personas"],
            "feature_columns": FEATURE_COLUMNS,
            "final_k": result["final_k"],
            "random_state": result["random_state"],
        },
        model_path,
    )

    # Convert complex values to standard python types for JSON dumping
    def to_json_type(val):
        if isinstance(val, dict):
            return {str(k): to_json_type(v) for k, v in val.items()}
        if isinstance(val, pd.Timestamp):
            return val.isoformat()
        if isinstance(val, (np.integer, int)):
            return int(val)
        if isinstance(val, (np.floating, float)):
            return float(val)
        if isinstance(val, (list, tuple)):
            return [to_json_type(item) for item in val]
        return val

    metadata = {
        "random_state": result["random_state"],
        "feature_columns": FEATURE_COLUMNS,
        "final_k": result["final_k"],
        "reference_date": result["preprocessor"].reference_date.isoformat(),
        "lower_bounds": result["preprocessor"].lower_bounds,
        "upper_bounds": result["preprocessor"].upper_bounds,
        "log_features": result["preprocessor"].log_features,
        "raw_data_audit": to_json_type(audit),
        "cleaning_report": to_json_type(cleaning_report),
        "personas": {str(k): v for k, v in result["personas"].items()},
    }
    
    # Save metadata as JSON
    metadata_path = output_path / "segmentation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Save datasets
    segments_path = output_path / "customer_segments.csv"
    result["customer_df"].to_csv(segments_path, index=False)
    
    kmeans_path = output_path / "kmeans_evaluation.csv"
    result["kmeans_evaluation"].to_csv(kmeans_path)
    
    agglomerative_path = output_path / "agglomerative_evaluation.csv"
    result["agglomerative_evaluation"].to_csv(agglomerative_path)

    return {
        "model": model_path,
        "metadata": metadata_path,
        "segments": segments_path,
        "kmeans_evaluation": kmeans_path,
        "agglomerative_evaluation": agglomerative_path,
    }


def load_segmentation_artifacts(model_path: str) -> dict:
    """Load persisted models and parameters from disk."""
    return joblib.load(model_path)
