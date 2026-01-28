import json
from sklearn.metrics import precision_recall_fscore_support, classification_report

# 文件路径
gt_file = "data/guba_random_128_gt.txt"
pred_file = "data/guba_random_128_gemini_p1.jsonl"

# 读取真实标签
gt_labels = []
with open(gt_file, "r", encoding="utf-8") as f:
    for line in f:
        # 获取最后一个'<sep>'后面的内容并去除换行
        label = line.rstrip().split("<sep>")[-1].strip()
        gt_labels.append(label)

# 读取模型预测标签
pred_labels = []
with open(pred_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        label = obj.get("label", "")
        
        # 如果 label 是 JSON 字符串（如 '{"label": "positive"}'），则解析它
        if label and isinstance(label, str) and label.startswith("{") and label.endswith("}"):
            try:
                label_obj = json.loads(label)
                label = label_obj.get("label", "")
            except json.JSONDecodeError:
                pass  # 如果不是有效 JSON，使用原值
        
        pred_labels.append(label)

# 保证长度一致
assert len(gt_labels) == len(pred_labels), "标签数量不一致，数据可能有误"

# 定义标签顺序
labels = ["positive", "negative", "neutral"]

# 计算精确率、召回率、F1（macro平均）
precision, recall, f1, _ = precision_recall_fscore_support(
    gt_labels, pred_labels, labels=labels, average=None, zero_division=0
)
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    gt_labels, pred_labels, labels=labels, average="macro", zero_division=0
)

# 输出每一类的精确率、召回率、F1
for i, label in enumerate(labels):
    print(f"类别: {label}")
    print(f"  精确率(Precision): {precision[i]:.4f}")
    print(f"  召回率(Recall):    {recall[i]:.4f}")
    print(f"  F1分数:           {f1[i]:.4f}")

# 输出macro平均
print("\n整体 Macro 平均：")
print(f"  宏平均精确率(Precision): {macro_precision:.4f}")
print(f"  宏平均召回率(Recall):    {macro_recall:.4f}")
print(f"  宏平均F1分数:           {macro_f1:.4f}")

# 也可以直接打印详细分类报告
print("\n详细分类报告：")
print(classification_report(gt_labels, pred_labels, labels=labels, digits=4))
correct_count = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == pred)
success_rate = correct_count / len(gt_labels)
print(f"\n总共{len(gt_labels)}个样本，成功匹配{correct_count}个，成功率: {success_rate:.2%}")
