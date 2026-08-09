# 🛍️ Customer Segmentation with K-Means

An interactive **Streamlit** application for segmenting mall customers using **K-Means clustering**. The app uses **Annual Income** and **Spending Score** as the clustering features and provides an interactive way to explore different values of K.

## 📌 Project Overview

This project applies unsupervised machine learning to group customers with similar income and spending behavior.

The Streamlit application allows users to:

- View an overview of the customer dataset
- Select the number of clusters (**K**) from a sidebar slider
- Analyze the **Elbow Method** to understand a suitable K
- Automatically identify a suggested K using **KneeLocator**
- Apply K-Means clustering with the selected K
- Visualize customer clusters and their centroids
- View average customer characteristics for each cluster
- Download the clustered dataset as a CSV file

## 📊 Dataset

The project uses `Mall_Customers.csv`, containing **200 customers** and the following 5 columns:

| Column | Description |
|---|---|
| `CustomerID` | Unique identifier for each customer |
| `Gender` | Customer gender |
| `Age` | Customer age |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Spending score assigned to the customer |

### Dataset Statistics

| Feature | Range |
|---|---:|
| Age | 18 – 70 |
| Annual Income | 15k – 137k ($) |
| Spending Score | 1 – 99 |

### 🤖 K-Means Clustering

The clustering model uses these two features:

```text
 Annual Income, Spending Score
```

The customer clusters are therefore based specifically on **annual income** and **spending score**.

### Choosing K

The application evaluates K values from **2 through 10** using the Elbow Method.

For each K, the model's **inertia** is calculated and plotted. The application also uses `KneeLocator` to automatically identify a possible optimal K.

Users can either:

- Select K manually using the sidebar slider, or
- Click **Use Suggested K** to use the K detected by `KneeLocator`

## 📈 Visualizations

### Elbow Curve

The Elbow Method visualization shows:

- Number of clusters (K) on the x-axis
- Inertia on the y-axis
- The currently selected K as a vertical reference line

### Customer Cluster Visualization

The application creates a scatter plot with:

- **X-axis:** Annual Income
- **Y-axis:** Spending Score
- Different colors for different clusters
- Black star markers representing cluster centroids

This makes it easy to visually understand the customer groups formed by K-Means.

## 👥 Cluster Profiles

After clustering, the application creates a cluster profile table containing:

- Number of customers in each cluster
- Average age
- Average annual income
- Average spending score


This helps interpret the practical characteristics of each customer segment.

## ⬇️ Download Clustered Data

The application provides a **Download Clustered Data** button.

The downloaded `clustered_customers.csv` file contains the original customer information along with the assigned cluster for every customer.

## 🖥️ Application Structure

```text
mall_customer_segmentation/
│
├── app.py
├── Mall_Customers.csv
├── README.md
└── requirements.txt
```

`app.py` contains the complete Streamlit application, while `Mall_Customers.csv` is the dataset used by the application.

## 🛠️ Technologies Used

- **Python**
- **Streamlit** — interactive web application
- **Pandas** — data loading and manipulation
- **NumPy** — numerical operations
- **Matplotlib** — plotting
- **Seaborn** — cluster visualization colors
- **Scikit-learn** — K-Means clustering
- **Kneed** — automatic knee/elbow point detection

## Installation

## 1.📦 Clone the repository

```bash
git clone https://github.com/IftekharAnamArnob/mall_customer_segmentation
cd mall_customer_segmentation
```

## 2.📦 Install dependencies

```bash
pip install -r requirements.txt
```

## 3.▶️ Run the Application

```bash
streamlit run app.py
```

Streamlit will start a local web server and provide the application in your browser.

> **Important:** Keep `Mall_Customers.csv` in the same directory as `app.py`, because the application loads the dataset using the relative path `Mall_Customers.csv`.

## 🔄 Application Workflow

```text
Mall_Customers.csv
        ↓
Load and clean dataset
        ↓
Select Annual Income + Spending Score
        ↓
Calculate K-Means inertia for K = 2–10
        ↓
Find suggested K using KneeLocator
        ↓
User selects K or uses suggested K
        ↓
Train K-Means model
        ↓
Assign customers to clusters
        ↓
Visualize clusters + centroids
        ↓
Generate cluster profiles
        ↓
Download clustered dataset
```

## ⚙️ Streamlit Features

### Session State

The selected K value is stored using Streamlit's `session_state`. This allows the application to retain the selected cluster count across Streamlit reruns.

### Data Caching

The dataset-loading function uses Streamlit's `@st.cache_data` so the CSV does not need to be unnecessarily reloaded every time the application reruns.

## 👤 Project

**Customer Segmentation using K-Means Clustering**

Built as an interactive machine learning application with Streamlit.

### 🚀 Live Demo

👉 **[Try the Customer Segmentation App](https://mallcustomersegmentation-yfgrec3w3n3szynj3y5f3h.streamlit.app/)**

Click the link above to open the deployed application and interact with the customer segmentation model directly.
