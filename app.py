
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Marketing Campaign Analysis",
    page_icon="📊",
    layout="wide"
)

# ==============================
# PROJECT PATHS
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

DATA_FILE = os.path.join(
    OUTPUT_DIR,
    "cleaned_campaign_data.csv"
)

# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

df = load_data()

# ==============================
# DASHBOARD TITLE
# ==============================

st.title("📊 Marketing Campaign Performance Analysis")
st.caption("Nykaa | Purplle | Tira")

# ==============================
# SIDEBAR FILTERS
# ==============================

st.sidebar.header("Dashboard Filters")

brands = ["All"] + sorted(df["Brand"].unique().tolist())

selected_brand = st.sidebar.selectbox(
    "Select Brand",
    brands
)

campaign_types = ["All"] + sorted(
    df["Campaign_Type"].dropna().unique().tolist()
)

selected_campaign = st.sidebar.selectbox(
    "Select Campaign Type",
    campaign_types
)

# ==============================
# FILTER DATA
# ==============================

filtered_df = df.copy()

if selected_brand != "All":
    filtered_df = filtered_df[
        filtered_df["Brand"] == selected_brand
    ]

if selected_campaign != "All":
    filtered_df = filtered_df[
        filtered_df["Campaign_Type"] == selected_campaign
    ]

# ==============================
# KPI METRICS
# ==============================

total_campaigns = len(filtered_df)
total_revenue = filtered_df["Revenue"].sum()
average_revenue = filtered_df["Revenue"].mean()
average_roi = filtered_df["ROI"].mean()

total_profit_campaigns = (
    filtered_df["Profit_Status"] == "Profit"
).sum()

total_loss_campaigns = (
    filtered_df["Profit_Status"] == "Loss"
).sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Campaigns", f"{total_campaigns:,}")
col2.metric("Total Revenue", f"₹{total_revenue:,.0f}")
col3.metric("Average Revenue", f"₹{average_revenue:,.0f}")
col4.metric("Average ROI", f"{average_roi:.2f}")

st.divider()

col1, col2 = st.columns(2)

col1.metric("Profitable Campaigns", f"{total_profit_campaigns:,}")
col2.metric("Loss Campaigns", f"{total_loss_campaigns:,}")

# ==============================
# CHART SETTINGS
# ==============================

plt.rcParams["figure.figsize"] = (6, 3.5)
plt.rcParams["font.size"] = 9

# ==============================
# REVENUE ANALYSIS
# ==============================

st.header("Revenue Analysis")

col1, col2 = st.columns(2)

with col1:

    revenue_by_brand = filtered_df.groupby(
        "Brand"
    )["Revenue"].sum()

    fig, ax = plt.subplots(figsize=(6, 3.5))

    revenue_by_brand.plot(kind="bar", ax=ax)

    ax.set_title("Revenue by Brand")
    ax.set_xlabel("")
    ax.set_ylabel("Revenue")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col2:

    revenue_by_campaign = filtered_df.groupby(
        "Campaign_Type"
    )["Revenue"].sum()

    fig, ax = plt.subplots(figsize=(6, 3.5))

    revenue_by_campaign.plot(kind="bar", ax=ax)

    ax.set_title("Revenue by Campaign Type")
    ax.set_xlabel("")
    ax.set_ylabel("Revenue")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ==============================
# ROI ANALYSIS
# ==============================

st.header("ROI Analysis")

col1, col2 = st.columns(2)

with col1:

    roi_by_brand = filtered_df.groupby(
        "Brand"
    )["ROI"].mean()

    fig, ax = plt.subplots(figsize=(6, 3.5))

    roi_by_brand.plot(kind="bar", ax=ax)

    ax.set_title("Average ROI by Brand")
    ax.set_xlabel("")
    ax.set_ylabel("Average ROI")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col2:

    fig, ax = plt.subplots(figsize=(6, 3.5))

    sns.histplot(
        filtered_df["ROI"],
        bins=25,
        ax=ax
    )

    ax.set_title("ROI Distribution")
    ax.set_xlabel("ROI")
    ax.set_ylabel("Campaign Count")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ==============================
# CAMPAIGN ANALYSIS
# ==============================

st.header("Campaign Analysis")

col1, col2 = st.columns(2)

with col1:

    campaign_distribution = filtered_df[
        "Campaign_Type"
    ].value_counts()

    fig, ax = plt.subplots(figsize=(6, 3.5))

    campaign_distribution.plot(kind="bar", ax=ax)

    ax.set_title("Campaign Type Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Campaigns")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col2:

    profit_distribution = filtered_df[
        "Profit_Status"
    ].value_counts()

    fig, ax = plt.subplots(figsize=(6, 3.5))

    profit_distribution.plot(kind="bar", ax=ax)

    ax.set_title("Profit vs Loss Campaigns")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Campaigns")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ==============================
# REVENUE DISTRIBUTION
# ==============================

st.header("Revenue Distribution")

fig, ax = plt.subplots(figsize=(8, 3.5))

sns.histplot(
    filtered_df["Revenue"],
    bins=30,
    ax=ax
)

ax.set_title("Revenue Distribution")
ax.set_xlabel("Revenue")
ax.set_ylabel("Frequency")

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ==============================
# CAMPAIGN DATA
# ==============================

st.header("Campaign Data")

st.dataframe(
    filtered_df.head(1000),
    use_container_width=True,
    height=350
)

st.download_button(
    label="📥 Download Filtered Campaign Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_campaign_data.csv",
    mime="text/csv"
)

# ==============================
# MODEL PERFORMANCE
# ==============================

st.header("Machine Learning Model Performance")

results_file = os.path.join(
    OUTPUT_DIR,
    "model_evaluation_results.csv"
)

if os.path.exists(results_file):

    results_df = pd.read_csv(results_file)

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning("Model evaluation results file was not found.")
