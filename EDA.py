import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog


root = tk.Tk()
root.withdraw()


input_file = filedialog.askopenfilename(
    title="Select the Parquet file",
    filetypes=[("Parquet files", "*.parquet")]
)
if not input_file:
    print("No file selected. Exiting script.")
    exit()


df = pd.read_parquet(input_file)


print("Dataset Information:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe(include='all'))


for col in ['main_country', 'main_city']:
    if col in df.columns:
        print(f"\nTop 10 values for column '{col}':")
        print(df[col].value_counts().head(10))


sns.set_theme(style="whitegrid")


if 'main_country' in df.columns:
    plt.figure(figsize=(10, 6))
    country_counts = df['main_country'].value_counts(dropna=True)
    top_n = 10
    top_countries = country_counts.head(top_n)
    others_count = country_counts.iloc[top_n:].sum()
    if others_count > 0:
        top_countries['Others'] = others_count
    sns.barplot(x=top_countries.index, y=top_countries.values, palette="mako")
    plt.xticks(rotation=45)
    plt.title("Distribution of Records by Region (Top 10 + Others)", fontsize=14)
    plt.xlabel("Country/Region", fontsize=12)
    plt.ylabel("Number of Records", fontsize=12)
    plt.tight_layout()
    plt.show()


missing = df.isnull().mean() * 100
missing = missing.sort_values(ascending=False)
top_missing = missing.head(10)
plt.figure(figsize=(8, 6))
top_missing.plot(kind='barh', color='salmon')
plt.xlabel('Missing Percentage (%)', fontsize=12)
plt.title('Top 10 Columns with Highest Missing Values', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


if 'main_country' in df.columns and 'main_city' in df.columns:
    top_country = df['main_country'].value_counts().idxmax()
    df_top = df[df['main_country'] == top_country]
    plt.figure(figsize=(10, 6))
    city_counts = df_top['main_city'].value_counts(dropna=True).head(10)
    sns.barplot(x=city_counts.values, y=city_counts.index, palette="coolwarm")
    plt.title(f"Top 10 Cities in {top_country}", fontsize=14)
    plt.xlabel("Number of Records", fontsize=12)
    plt.ylabel("City", fontsize=12)
    plt.tight_layout()
    plt.show()
