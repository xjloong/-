import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import os
import data


# ================= 1. 分词函数 (Tokenizer) =================
def mixed_tokenizer(text):
    """
    你原有的分词逻辑：中文按字拆，英文数字按块拆
    """
    pattern = re.compile(r'[\u4e00-\u9fa5]|[^\s\u4e00-\u9fa5]+')
    return pattern.findall(text)

# ================= 2. BLEU分数计算函数 =================
def calculate_bleu_list(references, hypotheses, model_label):
    """
    计算每对句子的 BLEU-4 分数列表
    返回格式：[{"Model": label, "Score": score}, ...]
    """
    results = []
    smoothie = SmoothingFunction().method1
    for ref, hyp in zip(references, hypotheses):
        r_t = [mixed_tokenizer(ref)]
        h_t = mixed_tokenizer(hyp)
        score = sentence_bleu(r_t, h_t, smoothing_function=smoothie)
        results.append({"Model": model_label, "BLEU": score * 100})
    return results


# ================= 3. ROUGE分数计算函数 =================
# def calculate_rouge_list(references, hypotheses, model_label):
#     """
#     计算每对句子的 ROUGE-L 分数列表
#     返回格式：[{"Model": label, "Score": score}, ...]
#     """
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     results = []
#     for ref, hyp in zip(references, hypotheses):
#         ref_sep = " ".join(mixed_tokenizer(ref))
#         hyp_sep = " ".join(mixed_tokenizer(hyp))
#         score = scorer.score(ref_sep, hyp_sep)
#         results.append({"Model": model_label, "ROUGE-L": score['rougeL'].fmeasure * 100})
#     return results
def calculate_rouge_list(references, hypotheses, model_label):
    """
    使用 rouge-score 库计算 ROUGE-L
    """
    # 禁用 stemmer，因为混有中文
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    results = []

    for ref, hyp in zip(references, hypotheses):
        # 将 token 列表用空格连接，这是 rouge-score 库要求的输入格式
        ref_sep = " ".join(mixed_tokenizer(ref))
        hyp_sep = " ".join(mixed_tokenizer(hyp))

        # 计算得分
        score = scorer.score(ref_sep, hyp_sep)

        # 获取 ROUGE-L 的 F1 分数 (fmeasure)
        # 如果你依然得到 100，说明两个句子的核心 token 序列在分词后确实一致
        rouge_l_score = score['rougeL'].fmeasure * 100

        results.append({
            "Model": model_label,
            "ROUGE-L": round(rouge_l_score, 2)
        })

    return results


# ================= 4. BLEU 绘图函数 =================
def plot_bleu(df, save_path):
    """
    专门绘制 BLEU 分数对比图
    """
    plt.figure(figsize=(6, 5), dpi=300)
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']

    # 绘制箱线图 + 散点
    ax = sns.boxplot(x="Model", y="BLEU", data=df, palette="Blues", width=0.5, showfliers=False)
    sns.stripplot(x="Model", y="BLEU", data=df, color=".25", size=5, alpha=0.6, jitter=True)

    plt.title("BLEU Score Distribution", fontweight='bold')
    plt.ylabel("BLEU Score (%)", fontweight='bold')
    plt.xlabel("")
    plt.ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ BLEU图表已保存至: {save_path}")
    plt.show()


# ================= 5. ROUGE 绘图函数 =================
def plot_rouge(df, save_path):
    """
    专门绘制 ROUGE-L 分数对比图
    """
    plt.figure(figsize=(6, 5), dpi=300)
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']

    # 绘制箱线图 + 散点 (换一个色系以示区别)
    ax = sns.boxplot(x="Model", y="ROUGE-L", data=df, palette="Reds", width=0.5, showfliers=False)
    sns.stripplot(x="Model", y="ROUGE-L", data=df, color=".25", size=5, alpha=0.6, jitter=True)

    plt.title("ROUGE Score Distribution", fontweight='bold')
    plt.ylabel("ROUGE Score (%)", fontweight='bold')
    plt.xlabel("")
    plt.ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ ROUGE图表已保存至: {save_path}")
    plt.show()



# ================= 主程序 (Execution) =================
if __name__ == "__main__":
    # 数据定义

    # 1. 计算数据
    bleu_results = calculate_bleu_list(data.references, data.hypotheses_pipe, "Pipeline") + \
                   calculate_bleu_list(data.references, data.hypotheses_e2e, "End-to-End")
    print(bleu_results)
    rouge_results = calculate_rouge_list(data.references, data.hypotheses_pipe, "Pipeline") + \
                    calculate_rouge_list(data.references, data.hypotheses_e2e, "End-to-End")
    print(rouge_results)
    # 2. 转换为 DataFrame
    df_bleu = pd.DataFrame(bleu_results)
    print(df_bleu)
    df_rouge = pd.DataFrame(rouge_results)
    print(df_rouge)

    # 3. 调用独立绘图函数
    # 路径请根据你的电脑实际情况修改，或使用相对路径
    plot_bleu(df_bleu, r"D:\Code\人工智能\picture\bleu_comparison.png")
    plot_rouge(df_rouge, r"D:\Code\人工智能\picture\rouge_comparison.png")

