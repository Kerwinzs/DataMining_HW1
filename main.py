import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm
import unicodedata
import re

# 第一步：加载数据
primary_df = pd.read_csv("primary.csv")
alternate_df = pd.read_csv("alternate.csv")

# 第二步：清洗函数
def clean_text(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 第三步：清洗并添加首字母
primary_df['cleaned_name'] = primary_df['NAME'].apply(clean_text)
alternate_df['cleaned_name'] = alternate_df['NAME'].apply(clean_text)
primary_df['initial'] = primary_df['cleaned_name'].str[0]
alternate_df['initial'] = alternate_df['cleaned_name'].str[0]

# 第四步：执行阻断去重逻辑
results = []
threshold = 90

for letter in tqdm(sorted(primary_df['initial'].dropna().unique())):
    p_sub = primary_df[primary_df['initial'] == letter]
    a_sub = alternate_df[alternate_df['initial'] == letter]

    for _, p_row in p_sub.iterrows():
        for _, a_row in a_sub.iterrows():
            score = fuzz.token_sort_ratio(p_row['cleaned_name'], a_row['cleaned_name'])
            if score >= threshold:
                results.append({
                    'Primary_ID': p_row['ID'],
                    'Primary_NAME': p_row['NAME'],
                    'Alternate_ID': a_row['ID'],
                    'Alternate_NAME': a_row['NAME'],
                    'Score': score,
                    'Match': 'Y' if score >= 95 else 'Maybe'
                })

dedup_df_blocking = pd.DataFrame(results)
dedup_df_blocking.to_csv("dedup_blocking_result.csv", index=False)
