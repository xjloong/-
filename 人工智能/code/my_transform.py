import torch
import torch.nn as nn
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ======================
# 路径配置
# ======================
QA_PATH = r"D:\Code\人工智能\dataset\qa.csv"
ENTITY_EMB_PATH = r"D:\Code\人工智能\output\model\entity_embeddings.pt"
ENTITY2ID_PATH = r"D:\Code\人工智能\output\entity_to_id.pkl"
MODEL_SAVE_PATH = r"D:\Code\人工智能\output\model\kg_t5_model.pt"

# ======================
# 加载 TransE 实体向量
# ======================
entity_embeddings = torch.load(ENTITY_EMB_PATH, weights_only=True)

with open(ENTITY2ID_PATH, "rb") as f:
    entity_to_id = pickle.load(f)

def get_entity_vec(name):
    if name not in entity_to_id:
        return None
    return entity_embeddings[entity_to_id[name]]


# ======================
# Dataset
# ======================
class QADataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        q_enc = self.tokenizer(
            row["question"],
            max_length=32,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        a_enc = self.tokenizer(
            str(row["answer"]),
            max_length=16,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        kg_vec = get_entity_vec(row["playerLabel"])
        if kg_vec is None:
            kg_vec = torch.zeros(128)

        return {
            "input_ids": q_enc["input_ids"].squeeze(0),
            "attention_mask": q_enc["attention_mask"].squeeze(0),
            "labels": a_enc["input_ids"].squeeze(0),
            "kg_vec": kg_vec
        }


# ======================
# KG + T5 模型
# ======================
class KG_T5(nn.Module):
    def __init__(self, kg_dim=128, prefix_len=5):
        super().__init__()
        self.prefix_len = prefix_len
        self.t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.kg_proj = nn.Linear(kg_dim, prefix_len * self.t5.config.d_model)

    def forward(self, input_ids, attention_mask, kg_vec, labels=None):
        B = kg_vec.size(0)

        prefix = self.kg_proj(kg_vec)
        prefix = prefix.view(B, self.prefix_len, -1)

        token_embeds = self.t5.encoder.embed_tokens(input_ids)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

        prefix_mask = torch.ones(B, self.prefix_len, device=input_ids.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        return self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

# ======================
# 训练流程
# ======================
tokenizer = T5Tokenizer.from_pretrained("t5-small")
dataset = QADataset(QA_PATH, tokenizer)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = KG_T5()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

model.train()
for epoch in range(5):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            kg_vec=batch["kg_vec"],
            labels=batch["labels"]
        )

        loss = out.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

# ======================
# 保存模型
# ======================
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("模型训练并保存完成！")
