from pykeen.pipeline import pipeline
import os
import torch
import pickle

# 训练 TransE
result = pipeline(
    training=r"D:\Code\人工智能\dataset\train\train.txt",
    testing=r"D:\Code\人工智能\dataset\train\test.txt",
    validation=r"D:\Code\人工智能\dataset\train\valid.txt",
    model='TransE',
    model_kwargs=dict(embedding_dim=128),
    training_kwargs=dict(num_epochs=200),

)

# 保存模型实体向量
model = result.model
entity_embeddings = model.entity_representations[0](indices=None).detach().cpu()
os.makedirs(r"D:\Code\人工智能\output\model", exist_ok=True)
torch.save(entity_embeddings, r"D:\Code\人工智能\output\model\entity_embeddings.pt")

# 从 result.training 获取实体映射
triples_factory = result.training
entity_to_id = triples_factory.entity_to_id

# 保存实体映射
os.makedirs(r"D:\Code\人工智能\output", exist_ok=True)
with open(r"D:\Code\人工智能\output\entity_to_id.pkl", "wb") as f:
    pickle.dump(entity_to_id, f)

# 保存关系向量
relation_embeddings = model.relation_representations[0](indices=None).detach().cpu()
torch.save(
    relation_embeddings,
    r"D:\Code\人工智能\output\model\relation_embeddings.pt"
)

# 保存关系映射
relation_to_id = triples_factory.relation_to_id
with open(r"D:\Code\人工智能\output\relation_to_id.pkl", "wb") as f:
    pickle.dump(relation_to_id, f)

print("保存完成！")
print(f"实体数量: {len(entity_to_id)}")
print(f"实体Embedding 形状: {entity_embeddings.shape}")
print(f"关系数量: {len(relation_to_id )}")
print(f"关系Embedding 形状: {relation_embeddings.shape}")



