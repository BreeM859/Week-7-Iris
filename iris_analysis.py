import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset
try:
    # Load the dataset
    df = pd.read_csv("iris.csv")

    # Drop the 'Id' column if it exists
    if "Id" in df.columns:
        df.drop(columns=["Id"], inplace=True)



    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Check data types
    print("\nData Types:")
    print(df.dtypes)

    # Handle missing values if any
    df.fillna(df.mean(numeric_only=True), inplace=True)

except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis
print("\nBasic Statistics:")
print(df.describe())

# Group by species and compute mean of numerical columns
species_means = df.groupby('Species').mean()
print("\nMean Values per Species:")
print(species_means)

# Identifying key observations
print("\nKey Observations:")
print("- Setosa has significantly smaller petal length & width compared to other species.")
print("- Virginica tends to have the largest petal and sepal sizes.")

# Task 3: Data Visualization
sns.set(style="whitegrid")

# Line chart: Sepal and Petal Length Trends
plt.figure(figsize=(7, 5))
x = np.arange(len(df))
plt.plot(x, df['SepalLengthCm'], label='Sepal Length', linestyle='dashed', color='b')
plt.plot(x, df['PetalLengthCm'], label='Petal Length', linestyle='solid', color='r')
plt.xlabel("Sample Index")
plt.ylabel("Length (cm)")
plt.title("Trend of Sepal and Petal Lengths")
plt.legend()
plt.show()

# Bar chart: Average petal length per species (Fixed Warning)
plt.figure(figsize=(6, 4))
sns.barplot(x=species_means.index, y=species_means['PetalLengthCm'], hue=species_means.index, legend=False, palette="viridis")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.title("Average Petal Length by Species")
plt.show()

# Histogram: Distribution of Sepal Width
plt.figure(figsize=(6, 4))
sns.histplot(df['SepalWidthCm'], bins=20, kde=True, color="blue")
plt.xlabel("Sepal Width (cm)")
plt.title("Distribution of Sepal Width")
plt.show()

# Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['SepalLengthCm'], y=df['PetalLengthCm'], hue=df['Species'], palette="coolwarm")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal Length vs Petal Length")
plt.legend()
plt.show()

# Pair plot for a comprehensive visual analysis
sns.pairplot(df, hue="Species", palette="husl")
plt.show()

# Box plot for Sepal Length distribution per species
plt.figure(figsize=(8, 6))
sns.boxplot(x=df["Species"], y=df["SepalLengthCm"], palette="Set2")
plt.title("Sepal Length Distribution per Species")
plt.show()

print("\nAnalysis completed successfully! 🚀")
