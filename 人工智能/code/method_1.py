import pandas as pd

# ------------------------------
# 1. 读取 CSV 数据并清洗
# ------------------------------
df = pd.read_csv(r"D:\Code\人工智能\dataset\query.csv")

# 去掉重复行
df = df.drop_duplicates()

# 填充缺失值
df['height'] = df['height'].fillna(0)
df['countryLabel'] = df['countryLabel'].fillna("未知")
df['birthDate'] = df['birthDate'].fillna("未知")
df['sexLabel'] = df['sexLabel'].fillna("未知")

# ------------------------------
# 2. 构建知识图谱三元组
# ------------------------------
triples = []

for _, row in df.iterrows():
    entity = row['playerLabel']
    triples.append((entity, 'birthDate', row['birthDate']))
    triples.append((entity, 'country', row['countryLabel']))
    triples.append((entity, 'height', str(row['height'])))
    triples.append((entity, 'sex', row['sexLabel']))


# ------------------------------
# 3. 管道式问答系统函数
# ------------------------------

# 3.1 实体识别（简单匹配）
def entity_linking(question, df):
    for player in df['playerLabel'].unique():
        if player in question:
            return player
    return None


# 3.2 关系检测（规则匹配）
def relation_detection(question):
    if any(x in question for x in ['出生', '生日', '出生日期']):
        return 'birthDate'
    elif any(x in question for x in ['国家', '国籍']):
        return 'country'
    elif any(x in question for x in ['身高', '高度']):
        return 'height'
    elif any(x in question for x in ['性别']):
        return 'sex'
    else:
        return None


# 3.3 知识图谱查询
def kg_query(entity, relation, triples):
    answers = [t for h, r, t in triples if h == entity and r == relation]
    return list(set(answers))  # 去重


# 3.4 答案生成
def generate_answer(entity, relation, answers):
    if not entity:
        return "未识别到问题中的球员。"
    if not relation:
        return "无法识别问题中的属性。"
    if not answers:
        return f"未找到{entity}的{relation}信息。"

    # 根据属性生成自然语言
    if relation == 'birthDate':
        ans_text = "、".join([a[:10] for a in answers])  # 只保留日期前10位
        return f"{entity}的出生日期是：{ans_text}。"
    elif relation == 'country':
        ans_text = "、".join(answers)
        return f"{entity}的国家/国籍是：{ans_text}。"
    elif relation == 'height':
        ans_text = "、".join(answers)
        return f"{entity}的身高是：{ans_text} cm。"
    elif relation == 'sex':
        ans_text = "、".join(answers)
        return f"{entity}的性别是：{ans_text}。"
    else:
        return f"{entity}的{relation}是：{'、'.join(answers)}。"


# 3.5 整合问答系统
def answer_question(question):
    entity = entity_linking(question, df)
    relation = relation_detection(question)
    answers = kg_query(entity, relation, triples)
    return generate_answer(entity, relation, answers)


# ------------------------------
# 4. 测试示例
# ------------------------------
if __name__ == "__main__":
    questions = [
        "利昂内尔·梅西出生在哪一年？",
        "齐内丁·齐达内的国籍是什么？",
        "亚历山德罗·德尔·皮耶罗的性别？",
        "弗朗齐歇克·普拉尼奇卡出生日期？",

        "克里斯蒂亚诺·罗纳尔多出生在哪一年？",
        "内马尔的国籍是什么？",
        "安德烈亚·皮尔洛的身高是多少？",
        "卢卡·莫德里奇出生日期？",

        "基利安·姆巴佩出生在哪一年？",
        "罗伯特·莱万多夫斯基的国籍是什么？",
        "哈维·埃尔南德斯的性别？",
        "伊涅斯塔出生日期？",

        "卡里姆·本泽马的身高是多少？",
        "埃尔林·哈兰德出生在哪一年？",
        "穆罕默德·萨拉赫的国籍是什么？",
        "孙兴慜的身高是多少？",

        "保罗·博格巴出生日期？",
        "安托万·格列兹曼的国籍是什么？",
        "凯文·德布劳内出生在哪一年？",
        "托尼·克罗斯的身高是多少？",

        "塞尔吉奥·拉莫斯的性别？",
        "维吉尔·范戴克出生日期？",
        "马内的国籍是什么？",
        "布斯克茨的身高是多少？",

        "加雷斯·贝尔出生在哪一年？",
        "卡塞米罗的国籍是什么？",
        "罗纳尔迪尼奥出生日期？",
        "菲利普·拉姆的身高是多少？",

        "乔治·维阿的国籍是什么？",
        "卡卡出生在哪一年？",
        "布冯的性别？",
        "伊克尔·卡西利亚斯出生日期？",

        "贝克汉姆的身高是多少？",
        "亨利的国籍是什么？",
        "苏亚雷斯出生在哪一年？",
        "阿圭罗的身高是多少？",

        "范巴斯滕出生日期？",
        "舍甫琴科的国籍是什么？",
        "内斯塔的身高是多少？",
        "皮克出生在哪一年？",

        "阿尔维斯的国籍是什么？",
        "马尔蒂尼的性别？",
        "法布雷加斯出生日期？",
        "罗本的身高是多少？"
    ]

    for q in questions:
        print("Q:", q)
        print("A:", answer_question(q))
        print()
