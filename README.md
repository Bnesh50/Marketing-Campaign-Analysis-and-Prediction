# Marketing Campaign Performance Analysis and Prediction

## Project Overview

This project analyzes marketing campaign data from three beauty and cosmetics brands: **Nykaa, Purplle, and Tira**.

The objective is to evaluate marketing campaign performance, identify revenue and ROI trends, analyze profitable and loss-making campaigns, and use Machine Learning to predict campaign revenue and profitability.

The project combines data cleaning, exploratory data analysis, feature engineering, Machine Learning, and interactive visualization using Streamlit.

---

## Brands Analyzed

- Nykaa
- Purplle
- Tira

The three datasets were combined into a single dataset containing **166,665 campaign records**.

---

## Objectives

- Combine campaign datasets from multiple brands.
- Handle missing values and duplicate records.
- Perform data cleaning and preprocessing.
- Conduct Exploratory Data Analysis (EDA).
- Analyze Revenue, ROI, campaign types, and profitability.
- Build a Regression model to predict campaign revenue.
- Build a Classification model to predict campaign profitability.
- Evaluate Machine Learning model performance.
- Develop an interactive dashboard using Streamlit.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Git
- GitHub

---

## Project Workflow

### 1. Data Collection

The datasets were collected separately for Nykaa, Purplle, and Tira in CSV format.

### 2. Data Cleaning

The following preprocessing steps were performed:

- Combined the three datasets.
- Removed duplicate records.
- Handled missing Campaign IDs.
- Handled missing numerical values using median imputation.
- Handled missing categorical values.
- Converted the Date column into datetime format.
- Converted numerical columns into appropriate data types.

### 3. Feature Engineering

New features were created to support analysis and Machine Learning:

- Brand identification
- Profit_Flag
- Profit_Status
- Multi-label encoding of marketing channels

### 4. Exploratory Data Analysis

The analysis includes:

- Revenue comparison between brands.
- Average ROI comparison between brands.
- Campaign type distribution.
- Revenue by campaign type.
- Profit and loss campaign analysis.
- Revenue distribution.
- ROI distribution.

### 5. Machine Learning

#### Regression

A Regression model was developed to predict campaign revenue.

Evaluation metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

#### Classification

A Classification model was developed to predict whether a campaign is profitable or loss-making.

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score

---

## Model Evaluation Results

### Regression Model

| Metric | Result |
|---|---:|
| MAE | 170,214.73 |
| MSE | 69,703,012,731.20 |
| RMSE | 264,013.28 |
| R² Score | 0.6952 |

### Classification Model

| Metric | Result |
|---|---:|
| Accuracy | 82.09% |
| Precision | 95.75% |
| Recall | 80.48% |
| F1 Score | 87.45% |

The Classification model achieved a precision of approximately 95.75%, indicating strong performance in identifying predicted profitable campaigns.

---

## Interactive Dashboard

The Streamlit dashboard provides:

- Brand filter
- Campaign type filter
- Total campaigns
- Total revenue
- Average revenue
- Average ROI
- Profitable campaigns
- Loss campaigns
- Revenue analysis
- Campaign type analysis
- Profit and loss analysis
- Revenue and ROI distributions
- Campaign data table
- Download filtered campaign data
- Machine Learning model evaluation results

---

## Project Structure

```text
Marketing_Campaign_Prediction/
│
├── data/
│   └── raw/
│       ├── nykaa_campaign_data_with_nulls.csv
│       ├── purplle_campaign_data_with_nulls.csv
│       └── tira_campaign_data_with_nulls.csv
│
├── models/
│   ├── classification_model_parameters.npz
│   └── revenue_regression_model.npz
│
├── outputs/
│   ├── cleaned_campaign_data.csv
│   ├── feature_engineered_data.csv
│   ├── model_evaluation_results.csv
│   └── visualizations/
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/Bnesh50/Marketing-Campaign-Analysis-and-Prediction.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Marketing-Campaign-Analysis-and-Prediction
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Data Processing and Machine Learning

```bash
python main.py
```

### Step 5: Launch the Dashboard

```bash
python -m streamlit run app.py
```

The dashboard will open in your browser.

---

## Conclusion

This project provides an analysis of marketing campaigns across Nykaa, Purplle, and Tira. By combining data analysis and Machine Learning, it helps understand campaign performance, revenue trends, ROI, and profitability.

The interactive dashboard makes the analysis easier to explore and supports data-driven marketing decisions.

---


