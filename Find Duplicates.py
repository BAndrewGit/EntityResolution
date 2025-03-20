import pandas as pd
import tldextract
import phonenumbers
from rapidfuzz import fuzz, process
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm


root = tk.Tk()
root.withdraw()

input_file = filedialog.askopenfilename(
    title="Select Parquet file",
    filetypes=[("Parquet files", "*.parquet")]
)
if not input_file:
    print("No file selected. Closing...")
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
    df['address_norm'] = df[address_cols].agg(' '.join, axis=1).str.lower().str.strip()

    return df


def match_by_domain(current, df):
    domain_mask = df['domain'] == current['domain']
    domain_group = df[domain_mask]
    group_records = domain_group.to_dict('records')
    idxs = set(domain_group.index.tolist())
    return group_records, idxs


def match_by_phone(current, df, threshold):
    phone_group = df[df['phone_norm'] == current['phone_norm']]
    if phone_group.empty:
        return [], set()
    names = phone_group['name_norm'].tolist()
    results = process.extract(
        current['name_norm'], names,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold
    )
    group_records = []
    idxs = set()
    for _, score, res_idx in results:
        if score >= threshold:
            match_idx = phone_group.index[res_idx]
            if match_idx not in idxs:
                group_records.append(df.loc[match_idx].to_dict())
                idxs.add(match_idx)
    return group_records, idxs


def find_matches(df, threshold=80):
    matches = []
    clustered = set()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Finding duplicates"):
        if idx in clustered:
            continue
        current = row.to_dict()
        if current['domain']:
            group, idxs = match_by_domain(current, df)
        else:
            group, idxs = match_by_phone(current, df, threshold)
        if group:
            clustered.update(idxs)
            matches.append(pd.DataFrame(group).drop_duplicates())
    return matches


def evaluate_similarity(groups, field='name_norm'):
    total_similarity = 0
    total_pairs = 0
    for group in groups:
        values = group[field].dropna().tolist()
        n = len(values)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                sim = fuzz.token_sort_ratio(values[i], values[j])
                total_similarity += sim
                total_pairs += 1
    avg_similarity = total_similarity / total_pairs if total_pairs > 0 else 0
    return avg_similarity



df = pd.read_parquet(input_file)
df = preprocess_data(df)
groups = find_matches(df, threshold=80)


for i, group in enumerate(groups):
    if len(group) > 1:
        print(f"\nGroup {i + 1} ({len(group)} duplicates):")
        print(group[['company_name', 'website_domain', 'primary_phone']])
        print("-" * 80)


avg_similarity = evaluate_similarity(groups, field='name_norm')
print(f"\nInternal Evaluation: Mean similarity of names in groups: {avg_similarity:.2f}%")
