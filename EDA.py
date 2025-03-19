import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog


root = tk.Tk()
root.withdraw()

input_file = filedialog.askopenfilename(title="Selectează fișierul Parquet", filetypes=[("Parquet files", "*.parquet")])
if not input_file:
    print("Nu a fost selectat niciun fișier. Scriptul se încheie.")
    exit()


df = pd.read_parquet(input_file)


print("Informații despre dataset:")
print(df.info())
print("\nStatistici descriptive:")
print(df.describe(include='all'))


columns_to_analyze = ['company_name', 'website_domain', 'primary_phone', 'main_country']
for col in columns_to_analyze:
    if col in df.columns:
        print(f"\nTop 10 valori pentru coloana '{col}':")
        print(df[col].value_counts().head(10))


if 'main_country' in df.columns:
    plt.figure(figsize=(10,6))
    country_counts = df['main_country'].value_counts(dropna=True)
    top_n = 10
    top_countries = country_counts.head(top_n)
    others_count = country_counts.iloc[top_n:].sum()
    if others_count > 0:
        top_countries['Altele'] = others_count
    sns.barplot(x=top_countries.index, y=top_countries.values, palette="viridis")
    plt.xticks(rotation=45)
    plt.title("Distribuția înregistrărilor pe țări (Top 10 + Altele)")
    plt.xlabel("Țară")
    plt.ylabel("Număr înregistrări")
    plt.tight_layout()
    plt.show()


if 'website_domain' in df.columns:
    plt.figure(figsize=(10,6))
    df['domain_length'] = df['website_domain'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    sns.histplot(df['domain_length'], bins=30, kde=True, color="teal")
    plt.title("Distribuția lungimii domeniilor web")
    plt.xlabel("Lungimea domeniului")
    plt.ylabel("Frecvență")
    plt.tight_layout()
    plt.show()


if 'company_name' in df.columns:
    plt.figure(figsize=(10,6))
    df['name_length'] = df['company_name'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    sns.histplot(df['name_length'], bins=30, kde=True, color="coral")
    plt.title("Distribuția lungimii numelor de companii")
    plt.xlabel("Lungimea numelui")
    plt.ylabel("Frecvență")
    plt.tight_layout()
    plt.show()
