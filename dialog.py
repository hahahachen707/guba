"""
与Deepseek对话，分析股吧评论的情绪
"""
import os
from openai import OpenAI

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=YOUR_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

system_prompt = """你是一个情绪分析专家，专门处理东方财富股吧的评论内容。你的任务是根据每条评论的文本，判断其情绪是正面（positive）、负面（negative）还是中性（neutral）。请遵循以下指南：

- **正面（positive）**：评论表达乐观、积极的情感，例如对股票或市场表示看好、赞扬、推荐购买、期待上涨等。
- **负面（negative）**：评论表达悲观、消极的情感，例如对股票或市场表示看跌、批评、抱怨、警告风险、预测下跌等。
- **中性（neutral）**：评论没有明显的情感倾向，或只提供事实信息、数据、中性讨论而不带情感色彩。

请基于评论内容本身进行客观分析，避免主观偏见。如果评论涉及多个情感，以主导情感为准。如果情感不明确，则归类为中性。输出中的confidence是你对自己判断正确性的信心，取值范围是0.0～1.0。当你不确定时，confidence可以设置为0.5以下，当你感觉确定时，confidence可以设置为0.5以上。
输入：用户将提供一条评论文本。
输出：label+confidence
输出示例：positive+0.9
不要添加任何额外解释或文本。label是positive，negative，neutral三种中的一个，confidence是0~1.0之间的一个浮点数。


"""

# system_prompt = """你是一个专门分析东方财富股吧评论情绪的专业AI助手。请严格按照以下要求对每条评论进行情绪分类：

# ## 任务说明
# 对给定的股票论坛评论进行情绪分析，判断其情绪倾向为以下三类之一：
# - **positive（积极）**：表达乐观、看好、赞赏等正面情绪
# - **negative（消极）**：表达悲观、看空、批评等负面情绪  
# - **neutral（中性）**：无明显情绪倾向，或包含矛盾情绪难以判断

# ## 分类标准

# ### 积极情绪特征：
# - 使用正面词汇：大涨、看好、牛股、抄底、机会、支持、感谢等
# - 表达乐观预期：明天涨停、还会涨、长期持有等
# - 赞赏语气：老师厉害、分析到位、谢谢分享等
# - 鼓励性语句：加油、坚持住、看好你等

# ### 消极情绪特征：
# - 使用负面词汇：暴跌、被套、垃圾、割肉、跑路、骗人等
# - 表达悲观预期：要跌、完蛋、清仓、远离等
# - 批评抱怨：主力出货、庄家坑人、后悔买了等
# - 愤怒指责：骗子、害人不浅、去死吧等

# ### 中性情绪特征：
# - 单纯事实陈述：今天涨了3%、成交量放大等
# - 无明显情感词汇的提问：明天走势如何？什么价格？
# - 包含矛盾信息但无明显倾向的评论
# - 无法明确判断情绪的简短语句

# 输入：用户将提供一条评论文本。
# 输出：直接返回情绪标签，且只能是“positive”、“negative”或“neutral”中的一个。不要添加任何额外解释或文本。

# 请基于评论内容本身进行客观分析，避免主观偏见。如果评论涉及多个情感，以主导情感为准。如果情感不明确，则归类为中性。
# """


# system_prompt = """你是一个专门分析东方财富股吧评论情绪的专业AI助手。请严格按照以下要求对每条评论进行情绪分类：

# ## 任务说明
# 对给定的股票论坛评论进行情绪分析，判断其情绪倾向为以下三类之一：
# - **positive（积极）**：表达乐观、看好、赞赏等正面情绪
# - **negative（消极）**：表达悲观、看空、批评等负面情绪  
# - **neutral（中性）**：无明显情绪倾向，或包含矛盾情绪难以判断

# ## 分类标准

# ### 积极情绪特征：
# - 使用正面词汇：大涨、看好、牛股、抄底、机会、支持、感谢等
# - 表达乐观预期：明天涨停、还会涨、长期持有等
# - 赞赏语气：老师厉害、分析到位、谢谢分享等
# - 鼓励性语句：加油、坚持住、看好你等

# ### 消极情绪特征：
# - 使用负面词汇：暴跌、被套、垃圾、割肉、跑路、骗人等
# - 表达悲观预期：要跌、完蛋、清仓、远离等
# - 批评抱怨：主力出货、庄家坑人、后悔买了等
# - 愤怒指责：骗子、害人不浅、去死吧等

# ### 中性情绪特征：
# - 单纯事实陈述：今天涨了3%、成交量放大等
# - 无明显情感词汇的提问：明天走势如何？什么价格？
# - 包含矛盾信息但无明显倾向的评论
# - 无法明确判断情绪的简短语句

# 输入：用户将提供一条评论文本。
# 输出：直接返回情绪标签，且只能是“positive”、“negative”或“neutral”中的一个。不要添加任何额外解释或文本。

# 请基于评论内容本身进行客观分析，避免主观偏见。如果评论涉及多个情感，以主导情感为准。如果情感不明确，则归类为中性。
# """



import time

def read_guba_file(filepath):
    """
    读取guba数据文件，返回每一行的字段字典列表
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t<sep>\t')
            if len(parts) != 6 and len(parts) != 7:
                # 兼容有些行可能缺字段
                raise ValueError(f"行数据格式不正确: {line}")
                # continue
            # 兼容6字段和7字段
            if len(parts) == 6:
                title, date, comment_num, read_num, user_comment, subpage = parts
            else:
                title, date, comment_num, read_num, user_comment, subpage = parts[:6]
            data.append({
                'title': title,
                'date': date,
                'comment_num': comment_num,
                'read_num': read_num,
                'user_comment': user_comment,
                'subpage': subpage
            })
    return data

def ask_deepseek(client, user_comment, system_prompt="你是一个有用的助手。请帮我分析下面的股吧评论内容，并简要总结其主要观点或情绪。"):
    """
    向deepseek发送请求，返回回复内容
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-v3.2-exp",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_comment},
            ],
            extra_body={"enable_thinking": False},
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Deepseek请求失败: {e}")
        return ""

def save_results(results, save_path):
    """
    保存结果到文件，每行为jsonl格式
    """
    import json
    with open(save_path, 'a', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main_guba_dialog():
    input_path = '/home/tione/notebook/workspace/xiaoyangchen/work/guba/guba_random_128.txt'
    output_path = '/home/tione/notebook/workspace/xiaoyangchen/work/guba/guba_random_128_deepseek_3.1_p1_confidence.jsonl'
    batch_size = 2

    all_data = read_guba_file(input_path)
    print(f"共读取到{len(all_data)}条数据。")
    results = []
    for idx, item in enumerate(all_data):
        user_comment = item.get('user_comment', '').strip()
        if not user_comment:
            summary = ""
        else:
            summary = ask_deepseek(client, user_comment, system_prompt)
            # 可适当sleep防止QPS过高
            time.sleep(0.5)
        result_item = {
            "title": item.get('title', ''),
            "date": item.get('date', ''),
            "comment_num": item.get('comment_num', ''),
            "read_num": item.get('read_num', ''),
            "user_comment": user_comment,
            "subpage": item.get('subpage', ''),
            "label": summary
        }
        results.append(result_item)
        if (idx + 1) % batch_size == 0:
            save_results(results, output_path)
            print(f"已保存{idx + 1}条结果到{output_path}")
            results = []
    # 保存剩余未保存的
    if results:
        save_results(results, output_path)
        print(f"已保存剩余{len(results)}条结果到{output_path}")

# 运行
if __name__ == "__main__":
    main_guba_dialog()
