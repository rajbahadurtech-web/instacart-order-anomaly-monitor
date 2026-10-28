from pathlib import Path
import json
from datetime import datetime, timezone

import duckdb
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "instacart"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ARTIFACTS / "isolation_forest.joblib"
REPORT_PATH = ARTIFACTS / "training_report.json"
SAMPLE_SIZE = 200_000
FEATURES = [
    "order_number", "order_dow", "order_hour_of_day",
    "days_since_prior_order", "items_in_order", "reorder_rate"
]

def main():
    orders = DATA / "orders.csv"
    prior = DATA / "order_products__prior.csv"
    if not orders.exists() or not prior.exists():
        raise FileNotFoundError(
            f"Expected {orders} and {prior}. Extract the dataset ZIP into data/instacart/ first."
        )
    con = duckdb.connect()
    # DuckDB aggregates the large relationship CSV without loading all rows into pandas.
    query = f"""
    SELECT
        CAST(o.order_number AS DOUBLE) AS order_number,
        CAST(o.order_dow AS DOUBLE) AS order_dow,
        CAST(o.order_hour_of_day AS DOUBLE) AS order_hour_of_day,
        TRY_CAST(o.days_since_prior_order AS DOUBLE) AS days_since_prior_order,
        CAST(p.items_in_order AS DOUBLE) AS items_in_order,
        CAST(p.reorder_rate AS DOUBLE) AS reorder_rate
    FROM read_csv_auto('{orders.as_posix()}', header=true) o
    INNER JOIN (
        SELECT order_id,
               COUNT(*) AS items_in_order,
               AVG(CAST(reordered AS DOUBLE)) AS reorder_rate
        FROM read_csv_auto('{prior.as_posix()}', header=true)
        GROUP BY order_id
    ) p USING (order_id)
    WHERE o.eval_set = 'prior'
    """
    print("Aggregating order-product rows with DuckDB...")
    total = con.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0]
    sample_query = f"SELECT * FROM ({query}) USING SAMPLE {SAMPLE_SIZE} ROWS"
    frame = con.execute(sample_query).df()
    con.close()
    if frame.empty:
        raise RuntimeError("No training rows were produced. Check the dataset files.")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(
            n_estimators=150,
            contamination=0.01,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    print(f"Training on {len(frame):,} sampled order rows (from {total:,} eligible orders)...")
    model.fit(frame[FEATURES])
    joblib.dump(model, MODEL_PATH)
    report = {
        "model": "IsolationForest",
        "purpose": "Unsupervised Instacart order anomaly detection",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "eligible_orders_with_prior_products": int(total),
        "training_rows": int(len(frame)),
        "features": FEATURES,
        "n_estimators": 150,
        "contamination": 0.01,
        "random_state": 42,
        "ground_truth_available": False,
        "accuracy_claimed": False,
        "model_file": MODEL_PATH.name,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved model: {MODEL_PATH}")

if __name__ == "__main__":
    main()
