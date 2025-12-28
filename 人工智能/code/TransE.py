import torch
import pickle
import torch.nn.functional as F

# 加载实体向量
entity_embeddings = torch.load(
    r"D:\Code\人工智能\output\model\entity_embeddings.pt",
    weights_only=True
)

# 加载实体映射
with open(r"D:\Code\人工智能\output\entity_to_id.pkl", "rb") as f:
    entity_to_id = pickle.load(f)

id_to_entity = {v: k for k, v in entity_to_id.items()}

def get_entity_vector(entity_name):
    idx = entity_to_id.get(entity_name)
    if idx is None:
        return None
    return entity_embeddings[idx]

print(get_entity_vector("阿纳托利·季莫什丘克"))

v1 = get_entity_vector("Piet Kraak")
v2 = get_entity_vector("弗朗齐歇克·普拉尼奇卡")

sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
print("相似度：", sim.item())


# 加载关系向量
relation_embeddings = torch.load(
    r"D:\Code\人工智能\output\model\relation_embeddings.pt",
    weights_only=True
)


with open(r"D:\Code\人工智能\output\relation_to_id.pkl", "rb") as f:
    relation_to_id = pickle.load(f)

import torch

def predict_tail(head, relation, top_k=5):
    h_id = entity_to_id[head]
    r_id = relation_to_id[relation]

    h = entity_embeddings[h_id]
    r = relation_embeddings[r_id]

    predicted = h + r

    # 计算与所有实体的距离
    distances = torch.norm(entity_embeddings - predicted, dim=1)

    topk = torch.topk(distances, k=top_k, largest=False)
    return [(id_to_entity[i.item()], distances[i].item()) for i in topk.indices]

print(predict_tail("弗朗齐歇克·普拉尼奇卡", "birthDate"))
