# E-Commerce Customer Segmentation & Personalized Recommendation System

A Python-based customer analytics project that combines behavioral customer segmentation with a simple personalized product recommendation system.

The project analyzes e-commerce transaction data, creates customer-level behavioral features, identifies customer segments using clustering, generates customer personas, and recommends products using both individual purchase history and segment preferences.

## Project Pipeline

```text
Raw E-Commerce Data
        ↓
Data Audit & Cleaning
        ↓
Customer-Level Feature Engineering
        ↓
Data Transformation & Scaling
        ↓
K-Means Clustering Evaluation
        ↓
Customer Segments & Personas
        ↓
New-Customer Cluster Assignment
        ↓
Personalized Product Recommendation
        ↓
Recommendation Evaluation
```

## Features

### Customer Segmentation

* Transaction data cleaning and return handling
* Customer-level behavioral feature engineering
* RFM-based and behavioral features
* Outlier clipping and skew-aware transformation
* StandardScaler preprocessing
* K-Means clustering with multiple `k` values
* Ward Agglomerative clustering comparison
* Silhouette, Calinski-Harabasz and Davies-Bouldin evaluation
* PCA-based visualization
* Rule-based customer personas

### Recommendation System

* Item-item cosine similarity from customer purchase history
* Segment-level product preferences
* Hybrid recommendation scoring
* Cold-start recommendations for customers without purchase history
* Top-5 personalized product recommendations
* Recommendation reasoning
* Temporal offline evaluation

## Customer Features

The segmentation model uses eight customer-level features:

* Recency
* Frequency
* Monetary Value
* Average Order Value (AOV)
* Product Variety
* Return Ratio
* Average Quantity per Order
* Active Span

## Technologies

* Python
* Pandas
* NumPy
* scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Jupyter / Google Colab

## Project Structure

```text
customer-segmentation-recommendation/
│
├── Customer_Segmentation_Project.ipynb
├── segmentation_pipeline.py
├── online_retail_II.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the dataset

Keep the dataset in the project directory:

```text
online_retail_II.csv
```

### 3. Run the notebook

Open:

```text
Customer_Segmentation_Project.ipynb
```

and run the cells from top to bottom.

The notebook performs the complete segmentation analysis, generates cluster visualizations and customer personas, demonstrates new-customer assignment, and runs the recommendation workflow.

## Recommendation Approach

The recommendation system uses a simple and interpretable hybrid approach.

For existing customers:

```text
70% → Similarity to previous purchases
30% → Preference within the customer's segment
```

The final score is used to rank products that the customer has not already purchased.

For customers without sufficient purchase history, the system falls back to popularity-based recommendations.

## Evaluation

The project evaluates clustering quality using:

* Silhouette Score
* Calinski-Harabasz Score
* Davies-Bouldin Score

The recommendation workflow includes temporal offline evaluation of the item-similarity component using held-out future purchases.

## Reproducibility

The segmentation pipeline uses a fixed random state to make clustering results reproducible.

The reusable segmentation logic is implemented in:

```text
segmentation_pipeline.py
```

while the notebook provides the analysis, visualizations, examples and recommendation workflow.

## Limitations

* The dataset represents one e-commerce retailer and a fixed observation period.
* Customer personas are rule-based interpretations of cluster profiles.
* Recommendation quality depends on the available purchase history and product interactions.
* The system is intended as an educational and portfolio project rather than a production recommendation platform.

## Future Improvements

* More advanced recommendation evaluation
* Additional recommendation signals
* Better handling of sparse customer-product interactions
* Periodic model retraining for changing customer behavior
* Deployment as an interactive application or API
