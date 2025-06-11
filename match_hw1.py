import pandas as pd
from rapidfuzz.fuzz import token_sort_ratio
from tqdm import tqdm
import unicodedata
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# ===== 清洗函数 =====
def clean_text(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===== 读取主数据 =====
primary_df = pd.read_csv("primary.csv")
alternate_df = pd.read_csv("alternate.csv")

primary_df['cleaned_name'] = primary_df['NAME'].apply(clean_text)
alternate_df['cleaned_name'] = alternate_df['NAME'].apply(clean_text)

# 合并 primary 和 alternate 为搜索池
pool_df = pd.concat([
    primary_df[['ID', 'cleaned_name']].assign(SOURCE='primary'),
    alternate_df[['ID', 'cleaned_name']].assign(SOURCE='alternate')
], ignore_index=True)

# ===== 匹配函数 =====
def match_variants(test_df, pool_df, variant_col='VARIANT'):
    test_df['cleaned_variant'] = test_df[variant_col].apply(clean_text)
    predictions = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Matching test_01"):
        best_score = -1
        best_id = None
        best_source = None
        for _, pool_row in pool_df.iterrows():
            score = token_sort_ratio(row['cleaned_variant'], pool_row['cleaned_name'])
            if score > best_score:
                best_score = score
                best_id = pool_row['ID']
                best_source = pool_row['SOURCE']
        predictions.append({
            'VARIANT': row[variant_col],
            'PREDICTED_ID': best_id,
            'PREDICTED_SOURCE': best_source,
            'TRUE_ID': row['ID'],
            'SCORE': best_score,
            'MATCH': 'Y' if best_id == row['ID'] else 'N'
        })

    return pd.DataFrame(predictions)

# ===== 匹配 test_01.csv =====
print("\n=== Matching test_01.csv ===")
test_01 = pd.read_csv("test_01.csv")
test_01_results = match_variants(test_01, pool_df)
test_01_results.to_csv("test_01_match_result.csv", index=False)

# ===== 精度评估 =====
y_true = test_01_results['TRUE_ID']
y_pred = test_01_results['PREDICTED_ID']
precision = precision_score(y_true, y_pred, average='micro')
recall = recall_score(y_true, y_pred, average='micro')
f1 = f1_score(y_true, y_pred, average='micro')

print(f"\n📊 test_01 精度评估：")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("\n✅ test_01 匹配完成并导出结果。")
