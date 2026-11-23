import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DATA_FILE = os.path.join(OUTPUT_DIR, "cleaned_campaign_data.csv")
RESULT_FILE = os.path.join(OUTPUT_DIR, "model_evaluation_results.csv")

REGRESSION_MODEL_FILE = os.path.join(
    MODEL_DIR, "revenue_regression_model.npz"
)

CLASSIFICATION_MODEL_FILE = os.path.join(
    MODEL_DIR, "classification_model_parameters.npz"
)

st.set_page_config(
    page_title="Marketing Campaign Analysis",
    page_icon="📊",
    layout="wide"
)


# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    return data


@st.cache_data
def load_results():
    if os.path.exists(RESULT_FILE):
        return pd.read_csv(RESULT_FILE)

    return pd.DataFrame()


@st.cache_resource
def load_models():
    regression = np.load(
        REGRESSION_MODEL_FILE,
        allow_pickle=True
    )

    classification = np.load(
        CLASSIFICATION_MODEL_FILE,
        allow_pickle=True
    )

    return regression, classification


try:
    df = load_data()
    regression_model, classification_model = load_models()
    evaluation_results = load_results()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# =========================
# TITLE
# =========================

st.title("📊 Marketing Campaign Performance Analysis")

st.subheader("Nykaa | Purplle | Tira")

st.write(
    "Analyze marketing campaign performance "
    "and predict revenue and profitability."
)


# =========================
# SIDEBAR FILTERS
# =========================

st.sidebar.header("Dashboard Filters")

brand_options = ["All"] + sorted(
    df["Brand"].dropna().unique().tolist()
)

campaign_options = ["All"] + sorted(
    df["Campaign_Type"].dropna().unique().tolist()
)

selected_brand = st.sidebar.selectbox(
    "Select Brand",
    brand_options
)

selected_campaign = st.sidebar.selectbox(
    "Select Campaign Type",
    campaign_options
)


filtered = df.copy()

if selected_brand != "All":
    filtered = filtered[
        filtered["Brand"] == selected_brand
    ]

if selected_campaign != "All":
    filtered = filtered[
        filtered["Campaign_Type"] == selected_campaign
    ]


# =========================
# KPI METRICS
# =========================

st.header("Campaign Performance Overview")

total_campaigns = len(filtered)

total_revenue = (
    filtered["Revenue"].sum()
    if total_campaigns else 0
)

average_revenue = (
    filtered["Revenue"].mean()
    if total_campaigns else 0
)

average_roi = (
    filtered["ROI"].mean()
    if total_campaigns else 0
)

profitable = (
    (filtered["Profit_Flag"] == 1).sum()
    if total_campaigns else 0
)

losses = (
    (filtered["Profit_Flag"] == 0).sum()
    if total_campaigns else 0
)

profit_rate = (
    profitable / total_campaigns * 100
    if total_campaigns else 0
)


c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Campaigns",
    f"{total_campaigns:,}"
)

c2.metric(
    "Total Revenue",
    f"₹{total_revenue:,.2f}"
)

c3.metric(
    "Average Revenue",
    f"₹{average_revenue:,.2f}"
)

c4.metric(
    "Average ROI",
    f"{average_roi:,.2f}"
)


c1, c2, c3 = st.columns(3)

c1.metric(
    "Profitable Campaigns",
    f"{profitable:,}"
)

c2.metric(
    "Loss Campaigns",
    f"{losses:,}"
)

c3.metric(
    "Profitability Rate",
    f"{profit_rate:.2f}%"
)


# =========================
# REVENUE ANALYSIS
# =========================

st.header("Revenue Analysis")


# Revenue by Brand - converted to billions

brand_revenue = (
    filtered.groupby("Brand", as_index=False)["Revenue"]
    .sum()
)

brand_revenue["Revenue_Billion"] = (
    brand_revenue["Revenue"] / 1_000_000_000
)

brand_revenue["Revenue_Label"] = (
    "₹" +
    brand_revenue["Revenue_Billion"].map(
        lambda x: f"{x:.2f}B"
    )
)


fig = px.bar(
    brand_revenue,
    x="Brand",
    y="Revenue_Billion",
    color="Brand",
    title="Total Revenue by Brand",
    text="Revenue_Label"
)

fig.update_layout(
    yaxis_title="Revenue (₹ Billion)",
    xaxis_title="Brand"
)

fig.update_traces(
    textposition="outside",
    hovertemplate=
    "<b>%{x}</b><br>"
    "Revenue: ₹%{customdata:,.2f}"
    "<extra></extra>",
    customdata=brand_revenue["Revenue"]
)

fig.update_yaxes(
    tickformat=".2f"
)

st.plotly_chart(fig, width="stretch")


# =========================
# REVENUE BY CAMPAIGN TYPE
# =========================

campaign_revenue = (
    filtered.groupby("Campaign_Type", as_index=False)["Revenue"]
    .sum()
)

campaign_revenue["Revenue_Billion"] = (
    campaign_revenue["Revenue"] / 1_000_000_000
)

campaign_revenue["Revenue_Label"] = (
    "₹" +
    campaign_revenue["Revenue_Billion"].map(
        lambda x: f"{x:.2f}B"
    )
)


fig = px.bar(
    campaign_revenue,
    x="Campaign_Type",
    y="Revenue_Billion",
    color="Campaign_Type",
    title="Revenue by Campaign Type",
    text="Revenue_Label"
)

fig.update_layout(
    yaxis_title="Revenue (₹ Billion)",
    xaxis_title="Campaign Type"
)

fig.update_traces(
    textposition="outside",
    hovertemplate=
    "<b>%{x}</b><br>"
    "Revenue: ₹%{customdata:,.2f}"
    "<extra></extra>",
    customdata=campaign_revenue["Revenue"]
)

fig.update_yaxes(
    tickformat=".2f"
)

st.plotly_chart(fig, width="stretch")


# =========================
# ROI BY BRAND
# =========================

roi_brand = (
    filtered.groupby("Brand", as_index=False)["ROI"]
    .mean()
)

fig = px.bar(
    roi_brand,
    x="Brand",
    y="ROI",
    color="Brand",
    title="Average ROI by Brand",
    text_auto=".2f"
)

fig.update_layout(
    yaxis_title="Average ROI",
    xaxis_title="Brand"
)

st.plotly_chart(fig, width="stretch")


# =========================
# PROFIT AND LOSS ANALYSIS
# =========================

st.header("Profit and Loss Analysis")

profit_data = (
    filtered["Profit_Status"]
    .value_counts()
    .rename_axis("Status")
    .reset_index(name="Count")
)

fig = px.pie(
    profit_data,
    names="Status",
    values="Count",
    title="Profitable vs Loss Campaigns",
    hole=0.4
)

st.plotly_chart(fig, width="stretch")


# =========================
# REVENUE DISTRIBUTION
# =========================

fig = px.histogram(
    filtered,
    x="Revenue",
    nbins=40,
    title="Revenue Distribution"
)

fig.update_layout(
    xaxis_title="Revenue (₹)",
    yaxis_title="Number of Campaigns"
)

st.plotly_chart(fig, width="stretch")


# =========================
# CAMPAIGN DATA
# =========================

st.header("Campaign Data")

columns_to_show = [
    "Brand",
    "Campaign_Type",
    "Target_Audience",
    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Revenue",
    "Acquisition_Cost",
    "ROI",
    "Profit_Status"
]

columns_to_show = [
    col for col in columns_to_show
    if col in filtered.columns
]

st.dataframe(
    filtered[columns_to_show],
    width="stretch",
    hide_index=True
)


# =========================
# DOWNLOAD DATA
# =========================

st.download_button(
    "Download Filtered Campaign Data",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_campaign_data.csv",
    mime="text/csv"
)


# =========================
# MODEL PERFORMANCE
# =========================

st.header("Machine Learning Model Performance")

if not evaluation_results.empty:

    st.dataframe(
        evaluation_results,
        width="stretch",
        hide_index=True
    )

else:

    st.info(
        "Model evaluation results are unavailable."
    )


# =========================
# INTERACTIVE PREDICTION
# =========================

st.header("🔮 Interactive Campaign Prediction")

st.write(
    "Enter campaign details to predict Revenue "
    "and Profit/Loss."
)


brands = sorted(
    df["Brand"].dropna().unique().tolist()
)

campaign_types = sorted(
    df["Campaign_Type"].dropna().unique().tolist()
)

audiences = sorted(
    df["Target_Audience"].dropna().unique().tolist()
)

languages = sorted(
    df["Language"].dropna().unique().tolist()
)

segments = sorted(
    df["Customer_Segment"].dropna().unique().tolist()
)


reg_features = regression_model["features"].tolist()

clf_features = classification_model["features"].tolist()


channel_features = [
    feature for feature in reg_features
    if feature.startswith("Channel_")
]

channel_options = [
    feature.replace("Channel_", "", 1)
    for feature in channel_features
]


# =========================
# INPUT FORM
# =========================

with st.form("prediction_form"):

    st.subheader("Campaign Details")

    left, right = st.columns(2)

    with left:

        brand = st.selectbox(
            "Brand",
            brands
        )

        campaign_type = st.selectbox(
            "Campaign Type",
            campaign_types
        )

        audience = st.selectbox(
            "Target Audience",
            audiences
        )

        duration = st.number_input(
            "Duration (Days)",
            min_value=1,
            value=30
        )

        language = st.selectbox(
            "Language",
            languages
        )

        segment = st.selectbox(
            "Customer Segment",
            segments
        )

    with right:

        impressions = st.number_input(
            "Impressions",
            min_value=0,
            value=10000
        )

        clicks = st.number_input(
            "Clicks",
            min_value=0,
            value=1000
        )

        leads = st.number_input(
            "Leads",
            min_value=0,
            value=500
        )

        conversions = st.number_input(
            "Conversions",
            min_value=0,
            value=100
        )

        cost = st.number_input(
            "Acquisition Cost / Campaign Spend",
            min_value=0.0,
            value=5000.0
        )

        engagement = st.number_input(
            "Engagement Score",
            min_value=0.0,
            value=50.0
        )


    date_col1, date_col2, date_col3 = st.columns(3)

    with date_col1:

        month = st.selectbox(
            "Campaign Month",
            list(range(1, 13)),
            index=0
        )

    with date_col2:

        year = st.number_input(
            "Campaign Year",
            min_value=2000,
            max_value=2100,
            value=2026
        )

    with date_col3:

        day = st.selectbox(
            "Campaign Day",
            list(range(1, 32)),
            index=0
        )


    selected_channels = st.multiselect(
        "Marketing Channels",
        channel_options
    )


    submit_prediction = st.form_submit_button(
        "Predict Campaign Performance"
    )


# =========================
# FEATURE VECTOR
# =========================

def create_feature_vector(features):

    values = {
        feature: 0.0
        for feature in features
    }


    numeric_inputs = {

        "Duration": duration,

        "Impressions": impressions,

        "Clicks": clicks,

        "Leads": leads,

        "Conversions": conversions,

        "Acquisition_Cost": cost,

        "Engagement_Score": engagement,

        "Campaign_Month": month,

        "Campaign_Year": year,

        "Campaign_Day": day,

        "Campaign_Weekday": 0
    }


    for key, value in numeric_inputs.items():

        if key in values:

            values[key] = float(value)


    category_inputs = {

        "Brand": brand,

        "Campaign_Type": campaign_type,

        "Target_Audience": audience,

        "Language": language,

        "Customer_Segment": segment
    }


    for category, chosen_value in category_inputs.items():

        dummy_name = f"{category}_{chosen_value}"

        if dummy_name in values:

            values[dummy_name] = 1.0


    for channel in selected_channels:

        channel_dummy = f"Channel_{channel}"

        if channel_dummy in values:

            values[channel_dummy] = 1.0


    return np.array(
        [values[name] for name in features],
        dtype=float
    ).reshape(1, -1)


# =========================
# PREDICTION
# =========================

if submit_prediction:

    try:

        # REGRESSION PREDICTION

        X_reg = create_feature_vector(reg_features)

        reg_mean = regression_model["mean"].astype(float)

        reg_std = regression_model["std"].astype(float)

        reg_std[reg_std == 0] = 1

        X_reg_scaled = (
            X_reg - reg_mean
        ) / reg_std

        X_reg_intercept = np.column_stack([
            np.ones(1),
            X_reg_scaled
        ])

        coefficients = regression_model["coefficients"]

        predicted_revenue = float(
            (X_reg_intercept @ coefficients)[0]
        )


        # CLASSIFICATION PREDICTION

        X_clf = create_feature_vector(clf_features)

        clf_mean = classification_model["mean"].astype(float)

        clf_std = classification_model["std"].astype(float)

        clf_std[clf_std == 0] = 1

        X_clf_scaled = (
            X_clf - clf_mean
        ) / clf_std

        X_clf_intercept = np.column_stack([
            np.ones(1),
            X_clf_scaled
        ])


        weights = classification_model["weights"]

        threshold = float(
            classification_model["threshold"]
        )


        z = float(
            (X_clf_intercept @ weights)[0]
        )

        z = np.clip(z, -500, 500)

        profit_probability = float(
            1 / (1 + np.exp(-z))
        )

        loss_probability = 1 - profit_probability

        predicted_profit = (
            profit_probability >= threshold
        )


        # =========================
        # DISPLAY RESULTS
        # =========================

        st.divider()

        st.subheader("Prediction Results")

        p1, p2 = st.columns(2)

        p1.metric(
            "Predicted Revenue",
            f"₹{predicted_revenue:,.2f}"
        )

        p2.metric(
            "Profit Probability",
            f"{profit_probability * 100:.2f}%"
        )


        if predicted_profit:

            st.success(
                "✅ The model predicts this campaign will be PROFITABLE."
            )

        else:

            st.error(
                "⚠️ The model predicts this campaign may result in a LOSS."
            )


        # =========================
        # PROFIT / LOSS PROBABILITY
        # =========================

        probability_df = pd.DataFrame({

            "Outcome": ["Profit", "Loss"],

            "Probability": [
                profit_probability * 100,
                loss_probability * 100
            ]

        })


        fig_probability = px.bar(

            probability_df,

            x="Outcome",

            y="Probability",

            color="Outcome",

            title="Predicted Profit / Loss Probability",

            text=probability_df["Probability"].map(
                lambda x: f"{x:.2f}%"
            )

        )


        fig_probability.update_layout(
            yaxis_range=[0, 100],
            yaxis_title="Probability (%)"
        )


        fig_probability.update_traces(
            textposition="outside"
        )


        st.plotly_chart(
            fig_probability,
            width="stretch"
        )


        # =========================
        # INPUT SUMMARY
        # =========================

        input_summary = pd.DataFrame({

            "Input": [

                "Brand",

                "Campaign Type",

                "Target Audience",

                "Duration",

                "Impressions",

                "Clicks",

                "Leads",

                "Conversions",

                "Acquisition Cost",

                "Engagement Score",

                "Language",

                "Customer Segment",

                "Selected Channels"

            ],


            "Value": [

                brand,

                campaign_type,

                audience,

                duration,

                impressions,

                clicks,

                leads,

                conversions,

                cost,

                engagement,

                language,

                segment,

                ", ".join(selected_channels)

            ]

        })


        st.subheader("Campaign Input Summary")

        st.dataframe(
            input_summary,
            width="stretch",
            hide_index=True
        )


        # =========================
        # REVENUE VS COST
        # =========================

        compare_df = pd.DataFrame({

            "Metric": [
                "Predicted Revenue",
                "Campaign Cost"
            ],

            "Amount": [
                predicted_revenue,
                cost
            ]

        })


        compare_df["Amount_Billion"] = (
            compare_df["Amount"] / 1_000_000_000
        )


        compare_df["Label"] = compare_df["Amount"].map(
            lambda x: f"₹{x:,.2f}"
        )


        fig_compare = px.bar(

            compare_df,

            x="Metric",

            y="Amount_Billion",

            color="Metric",

            title="Predicted Revenue vs Campaign Cost",

            text="Label"

        )


        fig_compare.update_layout(
            yaxis_title="Amount (₹ Billion)"
        )


        fig_compare.update_traces(
            textposition="outside",
            hovertemplate=
            "<b>%{x}</b><br>"
            "Amount: ₹%{customdata:,.2f}"
            "<extra></extra>",
            customdata=compare_df["Amount"]
        )


        fig_compare.update_yaxes(
            tickformat=".4f"
        )


        st.plotly_chart(
            fig_compare,
            width="stretch"
        )


    except Exception as error:

        st.error(
            f"Prediction error: {error}"
        )


st.divider()

st.caption(
    "Marketing Campaign Performance Analysis"
)