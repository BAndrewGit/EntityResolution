import pandas as pd
import tldextract
import phonenumbers
from rapidfuzz import fuzz, process
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

# Inițializare Tkinter
root = tk.Tk()
root.withdraw()

input_file = filedialog.askopenfilename(
    title="Selectează fișierul Parquet",
    filetypes=[("Parquet files", "*.parquet")]
)
if not input_file:
    print("Nu a fost selectat niciun fișier. Scriptul se încheie.")
    exit()

def normalize_domain(urls):
    result = []
    for url in urls:
        if pd.isna(url) or isinstance(url, (float, int)):
            result.append(None)
            continue
        extracted = tldextract.extract(str(url))
        result.append(f"{extracted.domain}.{extracted.suffix}".lower())
    return pd.Series(result, index=urls.index)

def normalize_names(names):
    suffixes = {"ltd", "inc", "llc", "gmbh", "pty", "plc", "co", "corp"}
    names = names.str.lower().str.strip().fillna('')
    for suffix in suffixes:
        names = names.str.replace(fr'\b{suffix}\b,?', '', regex=True)
    return names.str.strip()

def normalize_phones(phones):
    def _normalize(phone):
        try:
            return phonenumbers.format_number(
                phonenumbers.parse(str(phone), None),
                phonenumbers.PhoneNumberFormat.E164
            )
        except:
            return ''.join(filter(str.isdigit, str(phone)))
    return phones.apply(_normalize)

def preprocess_data(df):
    df = df.copy()
    df['domain'] = normalize_domain(df['website_domain'])
    df['name_norm'] = normalize_names(df['company_name'])
    df['phone_norm'] = normalize_phones(df['primary_phone'])
    address_cols = ['main_street', 'main_street_number', 'main_city', 'main_postcode', 'main_country']
    df[address_cols] = df[address_cols].fillna('').astype(str)
    df['address_norm'] = (df[address_cols[0]] + ' ' +
                          df[address_cols[1]] + ' ' +
                          df[address_cols[2]] + ' ' +
                          df[address_cols[3]] + ' ' +
                          df[address_cols[4]]).str.lower().str.strip()
    return df

def find_matches(df, threshold=80):
    matches = []
    clustered = set()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Procesare duplicate"):
        if idx in clustered:
            continue
        group = []
        current = row.to_dict()
        if current['domain']:
            domain_mask = df['domain'] == current['domain']
            domain_group = df[domain_mask]
            group.extend(domain_group.to_dict('records'))
            clustered.update(domain_group.index.tolist())
        else:
            phone_group = df[df['phone_norm'] == current['phone_norm']]
            if not phone_group.empty:
                names = phone_group['name_norm'].tolist()
                results = process.extract(
                    current['name_norm'], names,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=threshold
                )
                for _, score, res_idx in results:
                    if score >= threshold:
                        match_idx = phone_group.index[res_idx]
                        if match_idx not in clustered:
                            group.append(df.loc[match_idx].to_dict())
                            clustered.add(match_idx)
        if group:
            matches.append(pd.DataFrame(group).drop_duplicates())
    return matches

def evaluate_duplicates(groups, df):
    total_records = len(df)
    duplicate_indices = set()
    for group in groups:
        duplicate_indices.update(group.index)
    total_duplicates = len(duplicate_indices)
    total_groups = len(groups)
    print("\nEvaluare rezultate duplicate:")
    print(f"Total înregistrări: {total_records}")
    print(f"Total grupuri duplicate: {total_groups}")
    print(f"Total înregistrări duplicate (unice): {total_duplicates}")
    print(f"Procentaj duplicate: {total_duplicates / total_records * 100:.2f}%")


df = pd.read_parquet(input_file)
df = preprocess_data(df)
groups = find_matches(df, threshold=80)
evaluate_duplicates(groups, df)


for i, group in enumerate(groups):
    if len(group) > 1:
        print(f"\nGrup {i + 1} ({len(group)} duplicate):")
        print(group[['company_name', 'website_domain', 'primary_phone']])
        print("-" * 80)
