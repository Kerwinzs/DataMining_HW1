import pandas as pd
from rapidfuzz.fuzz import token_sort_ratio
from sklearn.metrics import precision_score, recall_score, f1_score
import unicodedata
import re
from tqdm import tqdm

# ===== 清洗函数 =====
def clean_text(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===== 加载数据 =====
primary_df = pd.read_csv("primary.csv")
test_02 = pd.read_excel("test_02.xlsx", sheet_name=None)

primary_df['cleaned_name'] = primary_df['NAME'].apply(clean_text)

# ===== 匹配和评估函数 =====
def match_and_evaluate(test_df, sheet_name):
    test_df = test_df.dropna(subset=["VARIANT", "ID"])
    test_df['cleaned_variant'] = test_df['VARIANT'].apply(clean_text)
    results = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Matching {sheet_name}"):
        best_score = -1
        best_id = None
        for _, p_row in primary_df.iterrows():
            score = token_sort_ratio(row['cleaned_variant'], p_row['cleaned_name'])
            if score > best_score:
                best_score = score
                best_id = p_row['ID']
        results.append({
            'VARIANT': row['VARIANT'],
            'TRUE_ID': row['ID'],
            'PREDICTED_ID': best_id,
            'SCORE': best_score,
            'MATCH': 'Y' if best_id == row['ID'] else 'N'
        })

    result_df = pd.DataFrame(results)
    precision = precision_score(result_df['TRUE_ID'], result_df['PREDICTED_ID'], average='micro')
    recall = recall_score(result_df['TRUE_ID'], result_df['PREDICTED_ID'], average='micro')
    f1 = f1_score(result_df['TRUE_ID'], result_df['PREDICTED_ID'], average='micro')

    return result_df, {
        'Sheet': sheet_name,
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4)
    }

# ===== 主程序 =====
all_scores = []

for sheet_name, df in test_02.items():
    if sheet_name.lower() == 'desc':
        continue  # 跳过说明页

    # 清洗列名，统一为大写
    df.columns = [col.strip().upper() for col in df.columns]

    if 'ID' not in df.columns or ('VARIANT' not in df.columns and 'NAME' not in df.columns):
        print(f"⚠️ 跳过 {sheet_name}，因为缺少 ID 或 VARIANT/NAME 列")
        continue

    # 重命名 NAME 为 VARIANT（如果没有 VARIANT）
    if 'VARIANT' not in df.columns and 'NAME' in df.columns:
        df.rename(columns={'NAME': 'VARIANT'}, inplace=True)

    result_df, metrics = match_and_evaluate(df, sheet_name)
    result_df.to_csv(f"test_02_{sheet_name}_match_result.csv", index=False)
    all_scores.append(metrics)

# 保存精度评估结果
score_df = pd.DataFrame(all_scores)
score_df.to_csv("test_02_sheet_scores.csv", index=False)

print("\n✅ 所有匹配和评估完成，结果已保存。")
