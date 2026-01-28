import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# -------------------------------------------------
# Load CSS
# -------------------------------------------------
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

# -------------------------------------------------
# Title & Description
# -------------------------------------------------
st.title("🟢 Customer Segmentation Dashboard")
st.markdown(
    """
    This system uses **K-Means Clustering** to group customers based on their  
    **purchasing behavior and similarities**.

    👉 Discover hidden customer groups **without predefined labels**.
    """
)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Wholesale customers data.csv")

df = load_data()

# Select numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
st.sidebar.header("🔧 Clustering Controls")

feature_1 = st.sidebar.selectbox(
    "Select Feature 1",
    numeric_cols
)

feature_2 = st.sidebar.selectbox(
    "Select Feature 2",
    [col for col in numeric_cols if col != feature_1]
)

k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=3
)

random_state = st.sidebar.number_input(
    "Random State (optional)",
    value=42,
    step=1
)

run_button = st.sidebar.button("🟦 Run Clustering")

# -------------------------------------------------
# Run Clustering
# -------------------------------------------------
if run_button:

    # Prepare data
    X = df[[feature_1, feature_2]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans Model
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state
    )

    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # -------------------------------------------------
    # Visualization Section
    # -------------------------------------------------
    st.subheader("📊 Cluster Visualization")

    fig, ax = plt.subplots()

    for i in range(k):
        ax.scatter(
            X[df["Cluster"] == i][feature_1],
            X[df["Cluster"] == i][feature_2],
            s=50,
            label=f"Cluster {i}"
        )

    # Cluster centers (inverse scaled)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=300,
        marker="X",
        c="black",
        label="Centroids"
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.legend()

    st.pyplot(fig)

    # -------------------------------------------------
    # Cluster Summary
    # -------------------------------------------------
    st.subheader("📋 Cluster Summary")

    summary = (
        df.groupby("Cluster")
        .agg(
            Count=("Cluster", "count"),
            Avg_Feature_1=(feature_1, "mean"),
            Avg_Feature_2=(feature_2, "mean")
        )
        .round(2)
    )

    st.dataframe(summary)

    # -------------------------------------------------
    # Business Interpretation
    # -------------------------------------------------
    st.subheader("💡 Business Interpretation")

    mean_f1 = summary["Avg_Feature_1"].mean()
    mean_f2 = summary["Avg_Feature_2"].mean()

    for cluster_id, row in summary.iterrows():
        if row["Avg_Feature_1"] > mean_f1 and row["Avg_Feature_2"] > mean_f2:
            st.success(
                f"🟢 Cluster {cluster_id}: High-spending customers across both categories."
            )
        elif row["Avg_Feature_1"] < mean_f1 and row["Avg_Feature_2"] < mean_f2:
            st.warning(
                f"🟡 Cluster {cluster_id}: Budget-conscious customers with lower spending."
            )
        else:
            st.info(
                f"🔵 Cluster {cluster_id}: Moderate spenders with selective purchasing behavior."
            )

    # -------------------------------------------------
    # Insight Box
    # -------------------------------------------------
    st.info(
        "📌 Customers in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted with similar business strategies."
    )

else:
    st.warning("👈 Select features, choose K, and click **Run Clustering** to begin.")
