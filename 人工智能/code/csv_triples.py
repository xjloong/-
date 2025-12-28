import csv
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv(r"D:\Code\人工智能\dataset\query.csv")
df.head()
# 去掉完全重复行
df = df.drop_duplicates()
# 缺失值处理，比如 height 缺失用 0 表示
df['height'] = df['height'].fillna(0)

triples = []

for _, row in df.iterrows():
    entity = row['playerLabel']
    triples.append((entity, 'birthDate', row['birthDate']))
    triples.append((entity, 'country', row['countryLabel']))
    triples.append((entity, 'height', str(row['height'])))
    triples.append((entity, 'sex', row['sexLabel']))

with open(r"D:\Code\人工智能\dataset\football_triples.txt", "w", encoding="utf-8") as f:
    for h, r, t in triples:
        f.write(f"{h}\t{r}\t{t}\n")

# 读取 CSV 转换好的三元组
df = pd.read_csv(r"D:\Code\人工智能\dataset\football_triples.txt", sep="\t", header=None, names=["h","r","t"])

# 划分训练/验证/测试集
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42)

# 保存到文件
os.makedirs(r"D:\Code\人工智能\dataset\train", exist_ok=True)
train_df.to_csv(r"D:\Code\人工智能\dataset\train\train.txt", sep="\t", header=False, index=False)
valid_df.to_csv(r"D:\Code\人工智能\dataset\train\valid.txt", sep="\t", header=False, index=False)
test_df.to_csv(r"D:\Code\人工智能\dataset\train\test.txt", sep="\t", header=False, index=False)