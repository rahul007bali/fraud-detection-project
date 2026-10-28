# Credit Card Fraud Detection System

An end-to-end machine learning project that identifies potentially fraudulent credit card transactions using Python, Scikit-learn, Flask, SQLite, and Docker.

## Project Overview

Credit card fraud detection is an imbalanced classification problem because fraudulent transactions represent only a small fraction of all transactions. This project explores transaction data, trains machine learning models, and provides an API for making fraud predictions.

## Features

- Exploratory data analysis using Pandas and NumPy
- Fraud classification using machine learning
- Comparison of Logistic Regression and Random Forest
- Prediction API built with Flask
- Prediction logging with SQLite
- Docker containerization
- Git and GitHub version control

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas and NumPy | Data processing |
| Scikit-learn | Machine learning |
| Flask | REST API |
| SQLite | Prediction storage |
| Docker | Containerization |
| Git and GitHub | Version control |

## Project Workflow

1. Load and explore transaction data.
2. Analyze the distribution of legitimate and fraudulent transactions.
3. Train and evaluate machine learning models.
4. Save the trained model.
5. Serve predictions through a Flask API.
6. Store prediction results in SQLite.
7. Package the API using Docker.

## Model Evaluation

The initial model comparison produced the following results on the evaluation split:

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Logistic Regression | 0.0610 | 0.9184 | 0.1144 |
| Random Forest | 0.9059 | 0.7857 | 0.8415 |

The Random Forest model achieved a substantially higher F1-score and precision in this experiment. Logistic Regression detected a higher proportion of fraud cases but generated more false positives.

These results depend on the evaluation split and classification threshold. Further threshold tuning and evaluation are planned.

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/rahul007bali/fraud-detection-project.git
cd fraud-detection-project
```

### 2. Install dependencies

Create and activate a virtual environment, then install the required packages:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the virtual environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Prepare the dataset and model

Place the `creditcard.csv` dataset in the project directory. The dataset is not included in this repository.

Train and save the model using the project's training script:

```bash
python save_model.py
```

Ensure the model file is available before starting the API.

### 4. Start the API

```bash
python app.py
```

The API should be available at:

`http://127.0.0.1:5000`

The root endpoint can be opened in a browser. Use the project's test script to test prediction requests.

### 5. Run with Docker

After preparing the model and confirming the Docker configuration:

```bash
docker build -t fraud-detection-api .
docker run -p 5000:5000 fraud-detection-api
```

## Dataset

This project uses a credit card transaction dataset containing anonymized features and a binary target column named `Class`:

- `0`: Legitimate transaction
- `1`: Fraudulent transaction

The dataset file is excluded from version control and must be obtained separately from its original source.

## Future Improvements

- Add a dashboard for reviewing predictions and model performance.
- Integrate AWS for cloud deployment and storage.
- Improve threshold selection using precision-recall analysis.
- Add automated tests, input validation, and API error handling.
- Document deployment and database persistence.

## Disclaimer

This is an educational machine learning project. Predictions are probabilistic and should not be used as the sole basis for real financial decisions.