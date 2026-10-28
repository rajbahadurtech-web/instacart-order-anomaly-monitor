from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "instacart"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "isolation_forest.joblib"
REPORT_PATH = ARTIFACT_DIR / "training_report.json"
FEATURES = [
    "order_number", "order_dow", "order_hour_of_day",
    "days_since_prior_order", "items_in_order", "reorder_rate"
]

st.set_page_config(page_title="Instacart Order Anomaly Monitor", page_icon="🛒", layout="wide")
st.title("🛒 Instacart Order Anomaly Monitor")
st.caption("Unsupervised machine learning demo using the Instacart Market Basket dataset")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

def load_report():
    if REPORT_PATH.exists():
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    return {}

def make_features(orders: pd.DataFrame, products: pd.DataFrame | None = None) -> pd.DataFrame:
    orders = orders.copy()
    required = {"order_id", "order_number", "order_dow", "order_hour_of_day"}
    missing = required - set(orders.columns)
    if missing:
        raise ValueError("orders file is missing required columns: " + ", ".join(sorted(missing)))
    orders["days_since_prior_order"] = pd.to_numeric(
        orders.get("days_since_prior_order", pd.Series(index=orders.index, dtype="float64")),
        errors="coerce"
    )
    for c in ["order_number", "order_dow", "order_hour_of_day"]:
        orders[c] = pd.to_numeric(orders[c], errors="coerce")
    if products is not None:
        needed = {"order_id", "reordered"}
        missing_products = needed - set(products.columns)
        if missing_products:
            raise ValueError("order products file is missing required columns: " + ", ".join(sorted(missing_products)))
        products["reordered"] = pd.to_numeric(products["reordered"], errors="coerce").fillna(0)
        summary = products.groupby("order_id", as_index=False).agg(
            items_in_order=("product_id", "count"),
            reorder_rate=("reordered", "mean"),
        )
        orders = orders.merge(summary, on="order_id", how="left")
    else:
        orders["items_in_order"] = 0.0
        orders["reorder_rate"] = 0.0
    orders["items_in_order"] = pd.to_numeric(orders["items_in_order"], errors="coerce").fillna(0)
    orders["reorder_rate"] = pd.to_numeric(orders["reorder_rate"], errors="coerce").fillna(0)
    orders["source_order_id"] = orders["order_id"]
    orders["source_user_id"] = orders["user_id"] if "user_id" in orders.columns else None
    for c in FEATURES:
        orders[c] = pd.to_numeric(orders[c], errors="coerce")
    return orders

def score(frame: pd.DataFrame, model):
    scored = frame.copy()
    scored = scored.dropna(subset=["order_number", "order_dow", "order_hour_of_day"])
    scored["anomaly_prediction"] = model.predict(scored[FEATURES])
    scored["anomaly_score"] = model.decision_function(scored[FEATURES])
    scored["is_anomaly"] = scored["anomaly_prediction"] == -1
    return scored

model = load_model()
if model is None:
    st.error("Trained model not found. Run `docker compose run --rm app python train_model.py` first.")
    st.stop()

report = load_report()
p1, p2, p3 = st.tabs(["Overview", "Detect anomalies", "Project methodology"])

with p1:
    st.subheader("Project overview")
    st.write("This dashboard uses an Isolation Forest model to flag statistically unusual grocery-order patterns. A flagged order is a review candidate, not proof of fraud or an incorrect order.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "Isolation Forest")
    c2.metric("Training sample", f"{report.get('training_rows', 0):,}" if report else "Not reported")
    c3.metric("Features", len(FEATURES))
    c4.metric("Threshold setting", str(report.get("contamination", "0.01")) if report else "0.01")
    st.markdown("#### Workflow")
    st.markdown("1. Read Instacart order and product CSV files.  \n2. Aggregate product lines into order-level features.  \n3. Train an unsupervised Isolation Forest.  \n4. Score orders and inspect unusual records.  \n5. Export the flagged orders for further review.")
    st.info("This project uses order-level patterns, not personally identifying information, as model inputs. Keep the original data local and do not upload private datasets to public services.")

with p2:
    st.subheader("Score an orders CSV")
    st.write("Upload `orders.csv` and a matching `order_products__prior.csv` or `order_products__train.csv`. Both are needed because the trained model uses items-per-order and reorder-rate features. For large files, use the Docker training workflow rather than the browser uploader.")
    orders_upload = st.file_uploader("Upload orders.csv", type=["csv"], key="orders")
    products_upload = st.file_uploader("Upload a matching order-products CSV", type=["csv"], key="products")
    can_score = orders_upload is not None and products_upload is not None
    if not can_score:
        st.info("Choose both CSV files to enable scoring. The order-products file should contain records for the orders being scored.")
    if st.button("Run anomaly detection", type="primary", disabled=not can_score):
        try:
            with st.spinner("Reading data and building order features..."):
                orders = pd.read_csv(orders_upload)
                products = pd.read_csv(products_upload)
                features = make_features(orders, products)
            with st.spinner("Scoring orders with the saved model..."):
                result = score(features, model)
            flagged = result[result["is_anomaly"]].copy()
            total = len(result)
            anomaly_count = len(flagged)
            a, b, c = st.columns(3)
            a.metric("Orders scored", f"{total:,}")
            b.metric("Potential anomalies", f"{anomaly_count:,}")
            c.metric("Flagged rate", f"{anomaly_count / total:.2%}" if total else "N/A")
            st.caption("The observed rate depends on the uploaded data and the model's learned boundary; it is not a validated fraud rate.")
            chart = result["is_anomaly"].value_counts().rename(index={True: "Potential anomaly", False: "Not flagged"}).rename_axis("result").reset_index(name="orders")
            st.plotly_chart(px.bar(chart, x="result", y="orders", title="Scoring results"), use_container_width=True)
            if not flagged.empty:
                st.markdown("#### Most unusual orders")
                show_cols = ["source_order_id", "source_user_id", *FEATURES, "anomaly_score"]
                st.dataframe(flagged.sort_values("anomaly_score")[show_cols].head(500), use_container_width=True)
                st.download_button("Download flagged orders CSV", flagged.to_csv(index=False).encode("utf-8"), "instacart_anomalies.csv", "text/csv")
                st.download_button("Download all scored orders CSV", result.to_csv(index=False).encode("utf-8"), "instacart_scored_orders.csv", "text/csv")
            else:
                st.success("No anomalies were flagged in this upload.")
            st.session_state["last_result_summary"] = {
                "orders_scored": int(total),
                "potential_anomalies": int(anomaly_count),
                "observed_flagged_rate": float(anomaly_count / total) if total else 0.0,
            }
        except Exception as exc:
            st.error(f"Could not process this upload: {exc}")

with p3:
    st.subheader("How the project works")
    st.markdown("""
**Dataset.** The provided archive contains Instacart order, product, aisle, department, and order-product relationship tables.

**Feature engineering.** Order-level features include order sequence number, day of week, hour of day, days since the previous order, items in an order, and the fraction of items marked as reordered.

**Model.** Isolation Forest is an unsupervised method that isolates unusual observations using randomized splits. The saved scikit-learn pipeline performs median imputation, scaling, and anomaly detection.

**Advantages**
- Does not require labelled fraud examples.
- Can score new orders without retraining on every upload.
- Produces row-level scores that can be reviewed and exported.

**Limitations**
- Anomaly flags are not proof of fraud or errors.
- The model is trained on this dataset's distribution and may behave differently on another dataset.
- There are no ground-truth labels here to calculate accuracy, precision, or recall.
- Uploading a very large order-products file can consume significant memory; process full-size data in Docker with the training script instead.

**Reproducibility.** The training script records row counts, model settings, and the date of training in `artifacts/training_report.json`.
""")
