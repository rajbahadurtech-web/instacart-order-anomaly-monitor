# Instacart Order Anomaly Monitor Using Machine Learning

An unsupervised machine learning project that identifies unusual patterns in grocery orders using Isolation Forest, Python, Streamlit, and Docker.

## Project Overview

The Instacart Order Anomaly Monitor analyzes grocery-order data and identifies orders with patterns that differ from typical orders.

The project uses the Instacart Market Basket dataset to prepare order-level features, train an Isolation Forest model, and display anomaly-scoring results through an interactive Streamlit dashboard.

**Important:** An anomalous order is an unusual data pattern, not proof of fraud or suspicious activity.

## Objectives

- Load and process the Instacart dataset.
- Prepare meaningful order-level features.
- Train an unsupervised anomaly-detection model.
- Score orders and identify unusual patterns.
- Develop an interactive dashboard for reviewing results.
- Use Docker to simplify application setup and execution.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| DuckDB | Efficiently query and aggregate CSV data |
| pandas | Data processing and manipulation |
| scikit-learn | Machine learning and Isolation Forest |
| Streamlit | Interactive web dashboard |
| Plotly | Data visualization |
| joblib | Saving and loading the trained model |
| Docker | Running the application in a container |
| Docker Compose | Managing the application environment |

## Dataset

The project uses the Instacart Market Basket dataset.

The dataset includes these CSV files:

- `orders.csv` — order information, including order sequence, day, hour, and time since the previous order.
- `order_products__prior.csv` — products associated with prior orders and whether products were reordered.
- `order_products__train.csv` — product-level records for the training evaluation set.
- `products.csv` — product information.
- `aisles.csv` — aisle information.
- `departments.csv` — department information.

The training script uses order data and prior-product records to construct order-level features.

### Dataset setup

Download the dataset from its original source, then place the extracted CSV files in:

```text
data/instacart/
```

The directory should contain:

```text
data/instacart/
├── orders.csv
├── order_products__prior.csv
├── order_products__train.csv
├── products.csv
├── aisles.csv
└── departments.csv
```

The raw dataset is not included in this repository because of its size.

## Machine Learning Approach

### Algorithm: Isolation Forest

Isolation Forest is an unsupervised machine learning algorithm used to identify observations that differ from common patterns in a dataset.

The model assigns anomaly scores to orders and uses its configured decision threshold to flag unusual observations.

### Features

The model uses six order-level features:

1. `order_number` — the order's sequence number.
2. `order_dow` — the day of the week.
3. `order_hour_of_day` — the hour the order was placed.
4. `days_since_prior_order` — days since the previous order.
5. `items_in_order` — number of products in the order.
6. `reorder_rate` — proportion of products in the order that were reordered.

### Training configuration

The project is configured to:

- Sample up to 200,000 eligible orders for training.
- Use a scikit-learn pipeline with missing-value imputation and feature scaling.
- Train an Isolation Forest with 150 estimators.
- Use a contamination parameter of `0.01`.
- Save the trained model as `artifacts/isolation_forest.joblib`.
- Save training information as `artifacts/training_report.json`.

The contamination parameter is a model configuration, not a measured fraud rate. The model has not been validated against confirmed fraud labels.

## Project Structure

```text
BD Act02/
├── app.py
├── train_model.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── instacart/
│       └── Dataset CSV files
└── artifacts/
    ├── isolation_forest.joblib
    └── training_report.json
```

The `data/` and `artifacts/` folders may contain additional files or placeholder files depending on your local setup.

## Prerequisites

- A Mac, Windows, or Linux computer.
- Docker Desktop installed and running.
- The project files.
- The Instacart dataset extracted into `data/instacart/`.

Download Docker Desktop from:

https://www.docker.com/products/docker-desktop/

## Installation and Execution

### Step 1: Clone the repository

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/instacart-order-anomaly-monitor.git
```

### Step 2: Open the project directory

```bash
cd instacart-order-anomaly-monitor
```

### Step 3: Prepare the dataset

Place the six CSV files inside:

```text
data/instacart/
```

Ensure that the filenames match the expected names.

### Step 4: Build the Docker image

```bash
docker compose build
```

### Step 5: Train the model

```bash
docker compose run --rm app python train_model.py
```

After successful training, the model and training report should be available in the `artifacts/` directory.

### Step 6: Start the dashboard

```bash
docker compose up
```

### Step 7: Open the application

Visit:

http://localhost:8501

Keep the Docker process running while using the dashboard.

To stop the application, press `Control + C` in the Terminal window running Docker Compose.

To restart the application later, run:

```bash
docker compose up
```

## Dashboard Features

The Streamlit dashboard contains three main sections.

### 1. Overview

Provides an introduction to the project and its anomaly-detection workflow.

### 2. Detect Anomalies

Allows users to upload the required order and order-product CSV files, score orders, inspect flagged observations, and download results.

### 3. Project Methodology

Explains the data preparation process, model features, Isolation Forest algorithm, and project limitations.

## Results and Interpretation

The initial training run reported:

- 3,214,874 eligible orders with prior-product information.
- A training sample of 200,000 orders.
- Six model features.
- An Isolation Forest model configured with a contamination parameter of 1%.

These figures describe the dataset and training configuration, not model accuracy or business performance.

A flagged order should be reviewed in context. The project does not establish whether an order is fraudulent, and no verified precision, recall, F1-score, or fraud-detection rate is claimed.

## Limitations

- The model is trained without confirmed anomaly labels.
- Unusual orders may be legitimate.
- The contamination parameter is a configuration choice, not a real-world fraud probability.
- Model performance has not been established using a labelled evaluation dataset.
- Results depend on data quality and the features supplied to the model.

## Future Improvements

- Evaluate the model using labelled or expert-reviewed data where available.
- Measure precision, recall, and other appropriate metrics.
- Explore additional order-level and customer-level features.
- Improve dashboard visualizations and filtering.
- Add repeatable model evaluation and monitoring.
- Investigate model drift as order patterns change.

## Learning Outcomes

This project demonstrates practical experience with:

- Data preprocessing and feature engineering.
- Unsupervised machine learning.
- Anomaly scoring and interpretation.
- Python-based dashboard development.
- Saving and loading trained machine learning models.
- Containerizing and running an application with Docker.

## Author

**Student:** Raj Bahadur

**Project:** Instacart Order Anomaly Monitor Using Machine Learning

**Institution:** [Add your college or institution name]

**Department:** [Add your department]

**Academic Year:** [Add your academic year]

## References

- Instacart Market Basket Analysis dataset: https://www.kaggle.com/competitions/instacart-market-basket-analysis/data
- scikit-learn Isolation Forest documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- Streamlit documentation: https://docs.streamlit.io/
- Docker documentation: https://docs.docker.com/
- DuckDB documentation: https://duckdb.org/docs/

---

*This project is developed for educational purposes to demonstrate an end-to-end machine learning workflow. Anomaly detection results should not be interpreted as proof of fraud.*
