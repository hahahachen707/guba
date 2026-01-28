import json
from sklearn.metrics import precision_recall_fscore_support, classification_report

# 文件路径
gt_file = "guba_random_128_gt.txt"
pred_file = "guba_random_128_gemini_p1_confidence.jsonl"

# 读取真实标签
gt_labels = []
with open(gt_file, "r", encoding="utf-8") as f:
    for line in f:
        # 获取最后一个'<sep>'后面的内容并去除换行
        label = line.rstrip().split("<sep>")[-1].strip()
        gt_labels.append(label)

# 读取模型预测标签和confidence
pred_labels = []
confidences = []
with open(pred_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        label_field = obj.get("label", "").strip()

        # 默认将整个字段作为label，置信度为None
        pred_label = label_field
        conf = None

        # 检查并解析 "label+confidence" 格式
        if '+' in label_field:
            parts = label_field.rsplit('+', 1)
            if len(parts) == 2:
                label_part, conf_part = parts
                try:
                    # 尝试将置信度部分转换为浮点数
                    conf_val = float(conf_part)
                    pred_label = label_part
                    conf = conf_val
                except ValueError:
                    # 如果转换失败，则保持原样，整个字段为label
                    pass
        
        pred_labels.append(pred_label)
        confidences.append(conf)

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

# 统计confidence<0.5和>=0.5的情况
low_conf_total = 0
low_conf_correct = 0
high_conf_total = 0
high_conf_correct = 0

for gt, pred, conf in zip(gt_labels, pred_labels, confidences):
    if conf is None:
        continue  # 跳过无法判断的
    if conf < 0.5:
        low_conf_total += 1
        if gt == pred:
            low_conf_correct += 1
    else:
        high_conf_total += 1
        if gt == pred:
            high_conf_correct += 1

if low_conf_total > 0:
    print(f"\nconfidence < 0.5 条目: {low_conf_total}，命中 {low_conf_correct}，成功率: {low_conf_correct / low_conf_total:.2%}")
else:
    print("\n没有 confidence < 0.5 的条目")
if high_conf_total > 0:
    print(f"confidence >= 0.5 条目: {high_conf_total}，命中 {high_conf_correct}，成功率: {high_conf_correct / high_conf_total:.2%}")
else:
    print("没有 confidence >= 0.5 的条目")
