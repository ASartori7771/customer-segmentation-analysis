# Customer Segmentation Analysis

Unsupervised machine learning pipeline that segments retail customers into behavioural clusters and uses an LLM to generate data quality reports and targeted marketing strategies per segment.

---

## What this project does

1. **Ingests and cleans** raw customer CSV files from a folder — handling missing values, duplicates, outliers, and inconsistent formatting
2. **Segments customers** into 3 clusters using K-Means and Hierarchical Clustering across age, income, and spending score
3. **Visualises** cluster structure via PCA scatter plot and hierarchical dendrogram
4. **Calls GPT-4o-mini** to auto-generate a plain-English data quality report and tailored marketing campaign ideas per cluster

---

## Project structure

```
customer-segmentation-analysis/
│
├── main-customer-segmentation-analysis.ipynb   # main notebook — run this
├── pipeline.py                                 # data ingestion & cleaning
├── llm.py                                      # OpenAI integration
├── .env                                        # your API key (never commit this)
│
├── data/
│   ├── raw/                                    # drop input CSVs here
│   │   ├── customers_jan.csv
│   │   ├── customers_feb.csv
│   │   └── customers_mar.csv
│   └── processed/                              # clean output files (auto-created)
│
└── logs/                                       # pipeline run logs & summaries (auto-created)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy openai python-dotenv
```

### 2. Add your OpenAI API key

Create a `.env` file in the project folder:

```
OPENAI_API_KEY=sk-proj-your-key-here
```

Never hardcode your key in the notebook or commit it to GitHub.

### 3. Add your data

Drop one or more CSV files into `data/raw/`. Each file must have these columns:

| Column | Type | Valid range |
|---|---|---|
| CustomerID | integer | unique per customer |
| Edad | numeric | 10 – 100 |
| Ingresos Anuales (k$) | numeric | 0 – 1000 |
| Puntuación de Gasto (1-100) | numeric | 1 – 100 |
| Categoría de Producto Favorito | string | any |

### 4. Run the notebook

Open `main-customer-segmentation-analysis.ipynb` and run cells top to bottom.

---

## How each file works

### `pipeline.py`

Handles everything before analysis touches the data:

- Scans `data/raw/` for all CSV files
- Validates that required columns exist
- Coerces numeric types and standardises string casing
- Removes duplicate CustomerIDs
- Imputes missing numeric values with the column median
- Drops rows with outlier values outside defined bounds
- Merges multiple files into a single clean DataFrame
- Saves a timestamped clean CSV to `data/processed/`
- Writes a JSON run summary to `logs/`

### `llm.py`

Two functions powered by GPT-4o-mini:

**`explain_anomalies(log_dir)`** — reads the latest pipeline summary JSON and returns a plain-English paragraph describing any data quality issues found, written for a non-technical audience.

**`generate_marketing_strategies(cluster_info)`** — takes the cluster DataFrame and returns a persona name, one-line description, and 3 campaign ideas (with channel) per cluster, structured as JSON.

---

## Cluster results

Three segments identified from 168 customers across 3 monthly CSV files:

| Cluster | Avg Age | Avg Income | Avg Spending Score | Profile |
|---|---|---|---|---|
| 0 | 46.2 | $40.7k | 34.4 | Older practical shoppers — Home & Books |
| 1 | 34.0 | $77.9k | 28.8 | High-income selective buyers — evenly spread |
| 2 | 28.6 | $40.2k | 64.4 | Young high spenders — Electronics & Food |

Key finding: income does not predict spending. Cluster 2 earns the least but spends the most. Cluster 1 earns the most but has the lowest spending score — representing the highest untapped revenue opportunity.

---

## Tech stack

- **Python 3.10+**
- **pandas / numpy** — data manipulation
- **scikit-learn** — K-Means clustering, MinMaxScaler, PCA
- **scipy** — hierarchical clustering and dendrogram
- **matplotlib / seaborn** — visualisation
- **openai** — GPT-4o-mini API
- **python-dotenv** — environment variable management

---

## Possible next steps

- Add RFM features (recency, frequency, monetary value) to enrich clustering
- Wrap the pipeline as a REST API with FastAPI
- Build a RAG system over customer feedback documents
- Add an evaluation pipeline with MLflow to track cluster stability across runs
- Orchestrate LLM steps as a multi-agent workflow with LangGraph