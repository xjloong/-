import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pickle

# ---------------------------
# 1️⃣ 定义 KG-T5 类（必须和训练时一致）
# ---------------------------
class KG_T5(torch.nn.Module):
    def __init__(self, kg_dim=128):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.kg_proj = torch.nn.Linear(kg_dim, self.t5.config.d_model)

    def forward(self, input_ids, attention_mask, kg_vec=None, labels=None):
        """
        kg_vec: [batch, kg_dim]
        """
        # 如果有 KG 向量，把它投影到 d_model
        if kg_vec is not None:
            kg_hidden = self.kg_proj(kg_vec)  # [batch, d_model]
            # 简单把 KG token 拼到 encoder embedding 后面
            emb = self.t5.encoder.embed_tokens(input_ids)  # [batch, seq_len, d_model]
            # 拼接 KG
            emb = torch.cat([emb, kg_hidden.unsqueeze(1)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)], dim=1)
            return self.t5(
                inputs_embeds=emb,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            return self.t5(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

# ---------------------------
# 2️⃣ 初始化模型
# ---------------------------
model_path = r"D:\Code\人工智能\output\model\kg_t5_model.pt"  # 训练好的 KG-T5
model = KG_T5(kg_dim=128)

# 加载权重
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# ---------------------------
# 3️⃣ 加载 tokenizer
# ---------------------------
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# ---------------------------
# 4️⃣ 加载实体向量和映射
# ---------------------------
entity_embeddings = torch.load(r"D:\Code\人工智能\output\model\entity_embeddings.pt")
with open(r"D:\Code\人工智能\output\entity_to_id.pkl", "rb") as f:
    entity_to_id = pickle.load(f)

embedding_dim = entity_embeddings.shape[1]

def get_entity_vec(name):
    if name not in entity_to_id:
        return torch.zeros(embedding_dim)
    return entity_embeddings[entity_to_id[name]]

# ---------------------------
# 5️⃣ 问题 + KG 输入生成
# ---------------------------
def prepare_input(question, entity_name):
    inputs = tokenizer(question, return_tensors="pt")
    kg_vec = get_entity_vec(entity_name).unsqueeze(0)  # [1, kg_dim]
    return inputs["input_ids"], inputs["attention_mask"], kg_vec

# ---------------------------
# 6️⃣ 推理生成答案
# ---------------------------
def answer_question(question, entity_name, max_len=50):
    input_ids, attention_mask, kg_vec = prepare_input(question, entity_name)
    with torch.no_grad():
        output_ids = model.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_len,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

# ---------------------------
# 7️⃣ 测试
# ---------------------------
question = "弗朗齐歇克·普拉尼奇卡是哪国人？"
entity_name = "弗朗齐歇克·普拉尼奇卡"
ans = answer_question(question, entity_name)
print("问题:", question)
print("答案:", ans)
