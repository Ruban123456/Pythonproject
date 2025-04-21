import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import zscore
import numpy as np

# Load dataset
df = pd.read_csv("edacleaneddata.csv")

# Clean data
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df.dropna(subset=['numerator', 'reportyear', 'vap'], inplace=True)
df['reportyear'] = df['reportyear'].astype(int)
df['numerator'] = pd.to_numeric(df['numerator'], errors='coerce')
df['vap'] = pd.to_numeric(df['vap'], errors='coerce')
sns.set(style="whitegrid")

# Objective 1: Voter Registration Trends Over Time
plt.figure(figsize=(8, 5))
trend_df = df.groupby('reportyear')['numerator'].sum().reset_index()
sns.barplot(data=trend_df, x='reportyear', y='numerator')
plt.title("Objective 1: Voter Registration Trend Over Time")
plt.ylabel("Registered Voters")
plt.xlabel("Year")
plt.tight_layout()
plt.show()

# Objective 2: Registration by Race/Ethnicity
plt.figure(figsize=(12, 6))
race_df = df.groupby(['reportyear', 'race_eth_name'])['numerator'].sum().reset_index()
sns.barplot(data=race_df, x='race_eth_name', y='numerator', hue='reportyear')
plt.title("Objective 2: Registration by Race/Ethnicity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Objective 3: Top 10 Counties by Registered Voters
plt.figure(figsize=(12, 6))
top_counties = df.groupby('county_name')['numerator'].sum().nlargest(10).reset_index()
sns.barplot(data=top_counties, x='county_name', y='numerator')
plt.title("Objective 3: Top 10 Counties by Registered Voters")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Objective 4: Missing Data Overview
plt.figure(figsize=(12, 6))
missing_df = df.isnull().mean().sort_values(ascending=False).reset_index()
missing_df.columns = ['Column', 'Missing Ratio']
sns.barplot(data=missing_df.head(10), x='Column', y='Missing Ratio')
plt.title("Objective 4: Top 10 Columns with Most Missing Data")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Objective 5: Registered Voters vs VAP
df_valid = df[(df['vap'].notnull()) & (df['vap'] > 0) & (df['numerator'].notnull())].copy()
df_valid['registration_rate'] = df_valid['numerator'] / df_valid['vap']
rate_df = df_valid.groupby('reportyear')['registration_rate'].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.lineplot(data=rate_df, x='reportyear', y='registration_rate', marker='o')
plt.title("Objective 5: Avg Registration Rate (Registered / VAP)")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

#Objective 6: Top 10 Counties by Registration Rate
plt.figure(figsize=(12, 6))
top_rate = df_valid.groupby('county_name')['registration_rate'].mean().nlargest(10).reset_index()
sns.barplot(data=top_rate, x='county_name', y='registration_rate')
plt.title("Objective 6: Top 10 Counties by Avg Registration Rate")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

#Objective 7: Trend of Registration Rate by Race
if 'race_eth_name' in df_valid.columns:
    plt.figure(figsize=(12, 6))
    race_rate = df_valid.groupby(['reportyear', 'race_eth_name'])['registration_rate'].mean().reset_index()
    sns.lineplot(data=race_rate, x='reportyear', y='registration_rate', hue='race_eth_name', marker='o')
    plt.title("Objective 7: Trend of Registration Rate by Race")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

#Objective 8: Outlier Detection using Z-score
df_valid['zscore'] = zscore(df_valid['registration_rate'].fillna(0))
plt.figure(figsize=(10, 5))
sns.histplot(df_valid['zscore'], bins=50, kde=True)
plt.axvline(3, color='red', linestyle='--')
plt.axvline(-3, color='red', linestyle='--')
plt.title(" Objective 8: Z-score Distribution of Registration Rate")
plt.xlabel("Z-score")
plt.tight_layout()
plt.show()

#Objective 9: Clustering Counties (KMeans)
from sklearn.preprocessing import StandardScaler

cluster_df = df_valid.groupby('county_name')[['vap', 'numerator']].mean().dropna().copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df)

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_df['Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=cluster_df, x='vap', y='numerator', hue='Cluster', palette='Set2', s=100)
plt.title(" Objective 9: County Clustering based on VAP and Registered Voters")
plt.xlabel("Avg Voting Age Population")
plt.ylabel("Avg Registered Voters")
plt.tight_layout()
plt.show()
#Objective 10: Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df_valid[['vap', 'numerator', 'registration_rate']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Objective 10: Correlation Heatmap")
plt.tight_layout()
plt.show()







