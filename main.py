# ============================================================
# MARKETING CAMPAIGN PERFORMANCE ANALYSIS AND PREDICTION
# ============================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("\nPROJECT STARTED")

# ============================================================
# 2. LOAD DATASETS
# ============================================================

nykaa = pd.read_csv(
    os.path.join(DATA_DIR, "nykaa_campaign_data_with_nulls.csv")
)

purplle = pd.read_csv(
    os.path.join(DATA_DIR, "purplle_campaign_data_with_nulls.csv")
)

tira = pd.read_csv(
    os.path.join(DATA_DIR, "tira_campaign_data_with_nulls.csv")
)

print("\nDataset Shapes:")
print("Nykaa:", nykaa.shape)
print("Purplle:", purplle.shape)
print("Tira:", tira.shape)

# ============================================================
# 3. ADD BRAND COLUMN
# ============================================================

nykaa["Brand"] = "Nykaa"
purplle["Brand"] = "Purplle"
tira["Brand"] = "Tira"

# ============================================================
# 4. COMBINE DATASETS
# ============================================================

df = pd.concat(
    [nykaa, purplle, tira],
    ignore_index=True
)

print("\nCombined Dataset Shape:")
print(df.shape)

# ============================================================
# 5. INITIAL DATA INSPECTION
# ============================================================

print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

df.drop_duplicates(inplace=True)

print("\nDuplicates Removed")
print("Updated Dataset Shape:", df.shape)

# ============================================================
# 7. HANDLE MISSING CAMPAIGN IDs
# ============================================================

missing_ids = df["Campaign_ID"].isnull()

df.loc[missing_ids, "Campaign_ID"] = [
    f"CAM_MISSING_{i}"
    for i in range(missing_ids.sum())
]

print("\nMissing Campaign IDs Handled")

# ============================================================
# 8. HANDLE NUMERICAL MISSING VALUES
# ============================================================

numeric_columns = [
    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Revenue",
    "Acquisition_Cost",
    "ROI",
    "Engagement_Score"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    df[column] = df[column].fillna(
        df[column].median()
    )

print("\nNumerical Missing Values Handled")

# ============================================================
# 9. HANDLE CATEGORICAL MISSING VALUES
# ============================================================

categorical_columns = [
    "Campaign_Type",
    "Target_Audience",
    "Channel_Used",
    "Language",
    "Customer_Segment"
]

for column in categorical_columns:

    df[column] = df[column].fillna("Unknown")

print("\nCategorical Missing Values Handled")

# ============================================================
# 10. DATE CONVERSION
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
    dayfirst=True
)

df["Campaign_Month"] = df["Date"].dt.month

df["Campaign_Year"] = df["Date"].dt.year

df["Campaign_Day"] = df["Date"].dt.day

df["Campaign_Weekday"] = df["Date"].dt.dayofweek

df["Campaign_Month"] = df["Campaign_Month"].fillna(0)

df["Campaign_Year"] = df["Campaign_Year"].fillna(0)

df["Campaign_Day"] = df["Campaign_Day"].fillna(0)

df["Campaign_Weekday"] = df["Campaign_Weekday"].fillna(0)

print("\nDate Feature Engineering Completed")

# ============================================================
# 11. CREATE PROFIT / LOSS TARGET
# ============================================================

df["Profit_Flag"] = np.where(
    df["ROI"] > 0,
    1,
    0
)

df["Profit_Status"] = df["Profit_Flag"].map(
    {
        1: "Profit",
        0: "Loss"
    }
)

print("\nProfit/Loss Distribution:")
print(df["Profit_Status"].value_counts())

# ============================================================
# 12. MULTI-LABEL CHANNEL ENCODING
# ============================================================

df["Channel_Original"] = df["Channel_Used"].astype(str)

df["Channel_Used"] = df["Channel_Used"].apply(
    lambda x: [
        channel.strip()
        for channel in str(x).split(",")
    ]
)

all_channels = set()

for channels in df["Channel_Used"]:

    all_channels.update(channels)

all_channels = sorted(all_channels)

for channel in all_channels:

    df["Channel_" + channel] = df["Channel_Used"].apply(
        lambda channels: 1
        if channel in channels
        else 0
    )

print("\nMulti-label Channel Encoding Completed")

# ============================================================
# 13. SAVE CLEANED DATASET
# ============================================================

cleaned_file = os.path.join(
    OUTPUT_DIR,
    "cleaned_campaign_data.csv"
)

df.to_csv(
    cleaned_file,
    index=False
)

print("\nCleaned Dataset Saved:")
print(cleaned_file)

# ============================================================
# 14. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n========== EDA SUMMARY ==========")

print("\nTotal Campaign Records:")
print(len(df))

print("\nCampaigns by Brand:")
print(df["Brand"].value_counts())

print("\nAverage Revenue by Brand:")
print(df.groupby("Brand")["Revenue"].mean())

print("\nTotal Revenue by Brand:")
print(df.groupby("Brand")["Revenue"].sum())

print("\nAverage ROI by Brand:")
print(df.groupby("Brand")["ROI"].mean())

print("\nCampaign Type Distribution:")
print(df["Campaign_Type"].value_counts())

print("\nProfit/Loss Distribution:")
print(df["Profit_Status"].value_counts())

# ============================================================
# 15. VISUALIZATION - REVENUE BY BRAND
# ============================================================

revenue_by_brand = df.groupby("Brand")["Revenue"].sum()

plt.figure(figsize=(8, 5))

revenue_by_brand.plot(kind="bar")

plt.title("Total Revenue by Brand")
plt.xlabel("Brand")
plt.ylabel("Total Revenue")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "revenue_by_brand.png")
)

plt.close()

print("\nRevenue by Brand Chart Saved")

# ============================================================
# 16. VISUALIZATION - PROFIT AND LOSS
# ============================================================

profit_counts = df["Profit_Status"].value_counts()

plt.figure(figsize=(8, 5))

profit_counts.plot(kind="bar")

plt.title("Profit vs Loss Campaigns")
plt.xlabel("Campaign Status")
plt.ylabel("Number of Campaigns")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "profit_loss_chart.png")
)

plt.close()

print("\nProfit/Loss Chart Saved")

# ============================================================
# 17. VISUALIZATION - CAMPAIGN TYPE
# ============================================================

campaign_counts = df["Campaign_Type"].value_counts()

plt.figure(figsize=(10, 6))

campaign_counts.plot(kind="bar")

plt.title("Campaign Type Distribution")
plt.xlabel("Campaign Type")
plt.ylabel("Number of Campaigns")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "campaign_type_chart.png")
)

plt.close()

print("\nCampaign Type Chart Saved")

# ============================================================
# 18. VISUALIZATION - AVERAGE ROI BY BRAND
# ============================================================

roi_by_brand = df.groupby("Brand")["ROI"].mean()

plt.figure(figsize=(8, 5))

roi_by_brand.plot(kind="bar")

plt.title("Average ROI by Brand")
plt.xlabel("Brand")
plt.ylabel("Average ROI")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "average_roi_by_brand.png")
)

plt.close()

print("\nAverage ROI Chart Saved")

# ============================================================
# 19. VISUALIZATION - REVENUE BY CAMPAIGN TYPE
# ============================================================

revenue_by_campaign = df.groupby(
    "Campaign_Type"
)["Revenue"].sum()

plt.figure(figsize=(10, 6))

revenue_by_campaign.plot(kind="bar")

plt.title("Total Revenue by Campaign Type")
plt.xlabel("Campaign Type")
plt.ylabel("Total Revenue")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "revenue_by_campaign_type.png")
)

plt.close()

print("\nRevenue by Campaign Type Chart Saved")

# ============================================================
# 20. PREPARE MACHINE LEARNING DATA
# ============================================================

print("\n========== MACHINE LEARNING DATA ==========")

# Features used for prediction

feature_columns = [
    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Acquisition_Cost",
    "Engagement_Score",
    "Campaign_Month",
    "Campaign_Year",
    "Campaign_Day",
    "Campaign_Weekday"
]

# Add channel encoding features

channel_features = [
    column
    for column in df.columns
    if column.startswith("Channel_")
    and column != "Channel_Used"
    and column != "Channel_Original"
]

feature_columns.extend(channel_features)

# Convert categorical variables into numerical columns

categorical_features = [
    "Brand",
    "Campaign_Type",
    "Target_Audience",
    "Language",
    "Customer_Segment"
]

df_encoded = pd.get_dummies(
    df,
    columns=categorical_features,
    drop_first=True,
    dtype=int
)

encoded_features = [
    column
    for column in df_encoded.columns
    if column.startswith("Brand_")
    or column.startswith("Campaign_Type_")
    or column.startswith("Target_Audience_")
    or column.startswith("Language_")
    or column.startswith("Customer_Segment_")
]

feature_columns.extend(encoded_features)

# Remove duplicate feature names

feature_columns = list(dict.fromkeys(feature_columns))

X = df_encoded[feature_columns].values.astype(float)

# Regression target

y_regression = df["Revenue"].values.astype(float)

# Classification target

y_classification = df["Profit_Flag"].values.astype(int)

print("\nFeature Dataset Shape:")
print(X.shape)

print("\nRegression Target Shape:")
print(y_regression.shape)

print("\nClassification Target Shape:")
print(y_classification.shape)

# ============================================================
# 21. TRAIN TEST SPLIT
# ============================================================

np.random.seed(42)

indices = np.random.permutation(len(X))

split_index = int(len(X) * 0.8)

train_indices = indices[:split_index]

test_indices = indices[split_index:]

X_train = X[train_indices]

X_test = X[test_indices]

y_reg_train = y_regression[train_indices]

y_reg_test = y_regression[test_indices]

y_clf_train = y_classification[train_indices]

y_clf_test = y_classification[test_indices]

print("\n========== TRAIN TEST SPLIT ==========")

print("Training Data:", X_train.shape)

print("Testing Data:", X_test.shape)

print("\nTrain-Test Split Completed Successfully")

# ============================================================
# 22. STANDARDIZE FEATURES
# ============================================================

X_mean = X_train.mean(axis=0)

X_std = X_train.std(axis=0)

X_std[X_std == 0] = 1

X_train_scaled = (X_train - X_mean) / X_std

X_test_scaled = (X_test - X_mean) / X_std

# ============================================================
# 23. REVENUE PREDICTION - LINEAR REGRESSION
# ============================================================

print("\n========== REVENUE PREDICTION MODEL ==========")

print("\nTraining Linear Regression Model...")

# Add intercept column

X_train_reg = np.column_stack(
    (
        np.ones(len(X_train_scaled)),
        X_train_scaled
    )
)

X_test_reg = np.column_stack(
    (
        np.ones(len(X_test_scaled)),
        X_test_scaled
    )
)

# Calculate coefficients using least squares

regression_coefficients = np.linalg.lstsq(
    X_train_reg,
    y_reg_train,
    rcond=None
)[0]

# Make predictions

y_reg_pred = X_test_reg @ regression_coefficients

# Regression Evaluation

mae = np.mean(
    np.abs(y_reg_test - y_reg_pred)
)

mse = np.mean(
    (y_reg_test - y_reg_pred) ** 2
)

rmse = np.sqrt(mse)

ss_total = np.sum(
    (y_reg_test - y_reg_test.mean()) ** 2
)

ss_residual = np.sum(
    (y_reg_test - y_reg_pred) ** 2
)

r2_score = 1 - (
    ss_residual / (ss_total + 1e-10)
)

print("\nRegression Model Evaluation:")

print("MAE:", mae)

print("MSE:", mse)

print("RMSE:", rmse)

print("R2 Score:", r2_score)

# Save regression model

np.savez(
    os.path.join(
        MODEL_DIR,
        "revenue_regression_model.npz"
    ),
    coefficients=regression_coefficients,
    mean=X_mean,
    std=X_std,
    features=np.array(feature_columns)
)

print("\nRegression Model Saved Successfully")

# ============================================================
# 24. PROFIT / LOSS CLASSIFICATION
# ============================================================

print("\n========== PROFIT/LOSS CLASSIFICATION MODEL ==========")

print("\nTraining Logistic Regression Model...")

# Add intercept

X_train_clf = np.column_stack(
    (
        np.ones(len(X_train_scaled)),
        X_train_scaled
    )
)

X_test_clf = np.column_stack(
    (
        np.ones(len(X_test_scaled)),
        X_test_scaled
    )
)

# Initialize weights

weights = np.zeros(
    X_train_clf.shape[1]
)

# Calculate class weights

number_of_zeros = np.sum(y_clf_train == 0)

number_of_ones = np.sum(y_clf_train == 1)

weight_zero = len(y_clf_train) / (
    2 * number_of_zeros
)

weight_one = len(y_clf_train) / (
    2 * number_of_ones
)

sample_weights = np.where(
    y_clf_train == 1,
    weight_one,
    weight_zero
)

# Initialize intercept

positive_ratio = np.mean(y_clf_train)

weights[0] = np.log(
    positive_ratio / (1 - positive_ratio)
)

# Sigmoid function

def sigmoid(z):

    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))

# Train model

learning_rate = 0.01

iterations = 1000

for iteration in range(iterations):

    predictions = sigmoid(
        X_train_clf @ weights
    )

    errors = (
        predictions - y_clf_train
    ) * sample_weights

    gradient = (
        X_train_clf.T @ errors
    ) / len(y_clf_train)

    weights = weights - (
        learning_rate * gradient
    )

# ============================================================
# 25. CLASSIFICATION PREDICTIONS
# ============================================================

probabilities = sigmoid(
    X_test_clf @ weights
)

# Classification threshold

threshold = 0.5

y_pred = (
    probabilities >= threshold
).astype(int)

# ============================================================
# 26. CLASSIFICATION EVALUATION
# ============================================================

accuracy = np.mean(
    y_pred == y_clf_test
)

true_positive = np.sum(
    (y_clf_test == 1) & (y_pred == 1)
)

true_negative = np.sum(
    (y_clf_test == 0) & (y_pred == 0)
)

false_positive = np.sum(
    (y_clf_test == 0) & (y_pred == 1)
)

false_negative = np.sum(
    (y_clf_test == 1) & (y_pred == 0)
)

precision = true_positive / (
    true_positive + false_positive + 1e-10
)

recall = true_positive / (
    true_positive + false_negative + 1e-10
)

f1_score = (
    2 * precision * recall
) / (
    precision + recall + 1e-10
)

print("\nClassification Model Evaluation:")

print("Accuracy:", accuracy)

print("Precision:", precision)

print("Recall:", recall)

print("F1 Score:", f1_score)

print("\nActual Test Class Distribution:")

print("Loss:", np.sum(y_clf_test == 0))

print("Profit:", np.sum(y_clf_test == 1))

print("\nPredicted Class Distribution:")

print("Loss:", np.sum(y_pred == 0))

print("Profit:", np.sum(y_pred == 1))

print("\nConfusion Matrix:")

print("True Negative:", true_negative)

print("False Positive:", false_positive)

print("False Negative:", false_negative)

print("True Positive:", true_positive)

# ============================================================
# 27. SAVE CLASSIFICATION MODEL
# ============================================================

np.savez(
    os.path.join(
        MODEL_DIR,
        "classification_model_parameters.npz"
    ),
    weights=weights,
    mean=X_mean,
    std=X_std,
    features=np.array(feature_columns),
    threshold=threshold
)

print("\nClassification Model Saved Successfully")

# ============================================================
# 28. SAVE MODEL RESULTS
# ============================================================

results = pd.DataFrame(
    {
        "Metric": [
            "Regression MAE",
            "Regression MSE",
            "Regression RMSE",
            "Regression R2",
            "Classification Accuracy",
            "Classification Precision",
            "Classification Recall",
            "Classification F1 Score"
        ],
        "Value": [
            mae,
            mse,
            rmse,
            r2_score,
            accuracy,
            precision,
            recall,
            f1_score
        ]
    }
)

results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_evaluation_results.csv"
    ),
    index=False
)

print("\nModel Evaluation Results Saved")

# ============================================================
# 29. SAVE FEATURE DATASET
# ============================================================

df_encoded.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_engineered_data.csv"
    ),
    index=False
)

print("\nFeature Engineered Dataset Saved")

# ============================================================
# 30. FINAL PROJECT STATUS
# ============================================================

print("\n===================================================")

print("MARKETING CAMPAIGN ANALYSIS COMPLETED SUCCESSFULLY")

print("===================================================")

print("\nFiles Generated:")

print("1. Cleaned Campaign Dataset")

print("2. Feature Engineered Dataset")

print("3. Revenue by Brand Chart")

print("4. Profit/Loss Chart")

print("5. Campaign Type Chart")

print("6. Average ROI Chart")

print("7. Revenue by Campaign Type Chart")

print("8. Regression Model")

print("9. Classification Model")

print("10. Model Evaluation Results")

print("\nOutput Folder:")
print(OUTPUT_DIR)

print("\nModel Folder:")
print(MODEL_DIR)

print("\nPROJECT COMPLETED")