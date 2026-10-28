# Instacart Order Anomaly Monitor

A small Dockerized Streamlit project that trains and deploys an Isolation Forest model to identify unusual grocery-order patterns in the Instacart Market Basket dataset.

## Dataset

The supplied ZIP contains these CSVs:

- `orders.csv`
- `order_products__prior.csv`
- `order_products__train.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

This application uses `orders.csv` and `order_products__prior.csv` to create order-level features. It does not include the dataset in the Git repository.

## Run on macOS with Docker Desktop

1. Install and start Docker Desktop for Mac.
2. Extract the supplied dataset archive.
3. Put the six CSV files in `data/instacart/` so the paths include:
   - `data/instacart/orders.csv`
   - `data/instacart/order_products__prior.csv`
4. Build the Docker image:

   ```bash
   docker compose build
   ```

5. Train the model in a one-off container:

   ```bash
   docker compose run --rm app python train_model.py
   ```

6. Start the dashboard:

   ```bash
   docker compose up
   ```

7. Open http://localhost:8501.

The model and training report are written to `artifacts/` on the Mac through a bind mount.

## Dashboard pages

- **Overview:** project context and workflow.
- **Detect anomalies:** upload `orders.csv`, optionally upload an order-products CSV, score rows with the saved model, inspect results, and download CSV outputs.
- **Project methodology:** dataset, feature engineering, model explanation, advantages, and limitations.

## Model details

The pipeline uses median imputation, standard scaling, and `IsolationForest`. Features are order number, day of week, hour, days since prior order, items per order, and reorder rate. The training script uses DuckDB to aggregate the large order-products CSV and samples up to 200,000 eligible orders for training.

An anomaly is a statistically unusual record, not proof of fraud. The dataset does not provide ground-truth anomaly labels for validating model accuracy.

## Notes

- The training process can take several minutes depending on your Mac.
- The app's CSV uploader is intended for manageable test files. For the full dataset, run the Docker training workflow rather than uploading the complete order-products table through the browser.
- Keep the supplied dataset local; it is not copied into the Docker image.
