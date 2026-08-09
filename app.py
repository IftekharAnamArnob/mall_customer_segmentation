import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from kneed import KneeLocator

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# ── Helper ────────────────────────────────────────────────────
def set_k(val:int):
    st.session_state['k_value']=int(val)


 # ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
    'Annual Income (k$)'    : 'Annual Income',
    'Spending Score (1-100)': 'Spending Score',
    'CustomerID'            : 'Customer ID'
    })
    return df

df = load_data()

# Title─────────────────────────────────────────────────
st.title("🛍️ Customer Segmentation")
st.caption("This app uses K-Means clustering to segment customers based on their annual income and spending score.")
st.markdown("---")

# ── Dataset Overview ──────────────────────────────────────────
st.header("Dataset Overview")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Total Customers", len(df))
col2.metric("Age Range", f"{df['Age'].min()} - {df['Age'].max()}")
col3.metric("Income Range", f"{df['Annual Income'].min()} - {df['Annual Income'].max()}")
col4.metric("Spending Score Range", f"{df['Spending Score'].min()} - {df['Spending Score'].max()}")

st.dataframe(df.head(10), use_container_width=True)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────

st.sidebar.header("Settings")

if "k_value" not in st.session_state:
    st.session_state['k_value'] = 3

st.sidebar.slider(
     label="Select number of clusters (k)",
     min_value=2,
     max_value=10,
     step=1,
     key="k_value"
 )   

# Features ──────────────────────────────────────────────────
X = df[["Annual Income", "Spending Score"]].values

# ── Elbow Method ──────────────────────────────────────────────
st.header("Elbow Method for Optimal k")

inertias = []
k_range = range(2,11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(X)
    inertias.append(km.inertia_)

knee = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")

optimal_k = knee.knee

col_left, col_right = st.columns([2,1])

with col_left:
    fig1, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(k_range, inertias, color='steelblue', markersize=8, linewidth=2, marker='o')
    ax1.axvline(st.session_state['k_value'], color='red', linestyle='--', linewidth=2, label= f"Selected k = {st.session_state['k_value']}" )
    ax1.set_xlabel('Number of Clusters(k)', fontsize=12)
    ax1.set_ylabel('Inertia', fontsize=12)
    ax1.set_title('Elbow Curve', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)


with col_right:
    st.markdown("### Suggested K")
    if optimal_k is not None:
        st.metric("Optimal K (KneeLocator)", int(optimal_k))
        st.button("Use Suggested K", on_click=set_k, args=(int(optimal_k),))

    else:
        st.warning("No Clear Knee found")   

# ── Run K-Means ───────────────────────────────────────────────
st.markdown('---')
st.header(f"K-Means Clustering ─ K = {st.session_state['k_value']}") 

km = KMeans(n_clusters=st.session_state['k_value'], random_state=42, n_init='auto')

km.fit_predict(X)

df_out = df.copy()

df_out['Cluster'] = km.labels_
df_out["Cluster"] = df_out["Cluster"].apply(lambda x: f"Cluster {x+1}")

centroids = km.cluster_centers_

# ── Scatter Plot ──────────────────────────────────────────────
st.subheader("Cluster Visualization")

palette = sns.color_palette('Set2', st.session_state['k_value'])
fig, ax = plt.subplots(figsize=(10,6))

for i, cluster in enumerate(sorted(df_out['Cluster'].unique())):
    subset = df_out[df_out['Cluster']==cluster]
    ax.scatter(
        subset['Annual Income'],
        subset['Spending Score'],
        label=cluster,
        s=70,
        color=palette[i],
        alpha=0.85,
        edgecolor='white',
        linewidth=0.5
    )
  
ax.scatter(
  centroids[:, 0], centroids[:, 1],
  marker='*', color='black',s=300,
  label='Centroid', zorder=5        
    )   

ax.set_title(f"Annual Income vs Spending Score", fontsize=15)
ax.set_xlabel(f"Annual Income",fontsize=12)
ax.set_ylabel(f"Spending Score", fontsize=12)
ax.legend(title='Cluster')
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)


# ── Cluster Details ────────────────────────

st.markdown("### Cluster Profiles")

profile = df_out[['Age', 'Annual Income', 'Spending Score','Cluster']]\
    .groupby('Cluster').mean().round(1) 

profile.insert(0,'Count',df_out.groupby('Cluster').size())

st.dataframe(profile, use_container_width=True)

# ── Download ─────────────────────
st.markdown("---")
csv = df_out.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Clustered Data", csv,
                   "clustered_customers.csv", "text/csv")