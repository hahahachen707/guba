"""
与Gemini对话，分析股吧评论的情绪
"""

import os
import json
import time
from google import genai
from google.genai import types

# 初始化 Google Gemini 客户端
client = genai.Client()
print("Google Gemini client initialized")

# system_prompt = """你是一个情绪分析专家，专门处理东方财富股吧的评论内容。你的任务是根据每条评论的文本，判断其情绪是正面（positive）、负面（negative）还是中性（neutral）。请遵循以下指南：

# - **正面（positive）**：评论表达乐观、积极的情感，例如对股票或市场表示看好、赞扬、推荐购买、期待上涨等。
# - **负面（negative）**：评论表达悲观、消极的情感，例如对股票或市场表示看跌、批评、抱怨、警告风险、预测下跌等。
# - **中性（neutral）**：评论没有明显的情感倾向，或只提供事实信息、数据、中性讨论而不带情感色彩。

# 请基于评论内容本身进行客观分析，避免主观偏见。如果评论涉及多个情感，以主导情感为准。如果情感不明确，则归类为中性。

# 输入：用户将提供一条评论文本。
# 输出：请以JSON格式输出，包含两个字段：
# 1. "label": 情绪标签，必须是"positive"、"negative"或"neutral"中的一个
# 2. "reasoning": 简短的推理过程，说明为什么判断为该情绪（1-2句话即可）

# 输出示例：
# {"label": "positive", "reasoning": "评论表达了对股票看好的态度和上涨的期待，属于积极情绪。"}
# """

system_prompt = """你是一个情绪分析专家，专门处理东方财富股吧的评论内容。你的任务是根据每条评论的文本，判断其情绪是正面（positive）、负面（negative）还是中性（neutral）。请遵循以下指南：

- **正面（positive）**：评论表达乐观、积极的情感，例如对股票或市场表示看好、赞扬、推荐购买、期待上涨等。
- **负面（negative）**：评论表达悲观、消极的情感，例如对股票或市场表示看跌、批评、抱怨、警告风险、预测下跌等。
- **中性（neutral）**：评论没有明显的情感倾向，或只提供事实信息、数据、中性讨论而不带情感色彩。

请基于评论内容本身进行客观分析，避免主观偏见。如果评论涉及多个情感，以主导情感为准。如果情感不明确，则归类为中性。

输入：用户将提供一条评论文本。
输出：请以JSON格式输出，包含一个字段：
1. "label": 情绪标签，必须是"positive"、"negative"或"neutral"中的一个


输出示例：
{"label": "positive"}
"""


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

def ask_gemini(client, user_comment, system_prompt="你是一个有用的助手。请帮我分析下面的股吧评论内容，并简要总结其主要观点或情绪。"):
    """
    向 Google Gemini 发送请求，返回回复内容（JSON格式）
    """
    try:
        # 组合 system prompt 和 user comment
        prompt_text = f"{system_prompt}\n\n请分析以下评论：\n{user_comment}"
        
        # 配置生成参数，强制输出 JSON 格式
        config = types.GenerateContentConfig(
            response_mime_type="application/json",  # 强制返回 JSON 格式
            temperature=0.1  # 降低温度，提高输出的确定性
        )
        
        # 调用 Gemini 模型
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt_text,
            config=config
        )
        return response.text
    except Exception as e:
        print(f"Gemini 请求失败: {e}")
        return '{"label": "", "reasoning": ""}'

def save_results(results, save_path):
    """
    保存结果到文件，每行为jsonl格式
    """
    with open(save_path, 'a', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main_guba_dialog():
    input_path = '/home/tione/notebook/workspace/xiaoyangchen/work/guba/data/guba_1_1000.txt'
    output_path = '/home/tione/notebook/workspace/xiaoyangchen/work/guba/data/guba_1_1000_gemini.jsonl'
    batch_size = 16

    all_data = read_guba_file(input_path)
    print(f"共读取到{len(all_data)}条数据。")
    results = []
    for idx, item in enumerate(all_data):
        user_comment = item.get('user_comment', '').strip()
        # 如果 user_comment 为空，使用 title 作为输入
        if not user_comment:
            submit = item.get('title', '').strip()
        else:
            submit = user_comment
        
        if submit:
            print(f"submit: {submit}")
            response_text = ask_gemini(client, submit, system_prompt)
            # 解析 JSON 响应
            try:
                response_json = json.loads(response_text)
                # 处理可能返回列表或字典的情况
                if isinstance(response_json, list):
                    # 如果是列表，取第一个元素
                    if len(response_json) > 0:
                        response_json = response_json[0]
                    else:
                        response_json = {}
                # 如果是字典，直接使用
                if isinstance(response_json, dict):
                    label = response_json.get("label", "")
                    reasoning = response_json.get("reasoning", "")
                else:
                    print(f"意外的响应格式: {type(response_json)}, 原始响应: {response_text}")
                    label = ""
                    reasoning = ""
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}, 原始响应: {response_text}")
                label = ""
                reasoning = ""
            except Exception as e:
                print(f"处理响应时出错: {e}, 原始响应: {response_text}")
                label = ""
                reasoning = ""
            # 可适当sleep防止QPS过高
            # time.sleep(0.5)
        else:
            label = ""
            reasoning = ""
        
        result_item = {
            "title": item.get('title', ''),
            "date": item.get('date', ''),
            "comment_num": item.get('comment_num', ''),
            "read_num": item.get('read_num', ''),
            "user_comment": user_comment,
            "subpage": item.get('subpage', ''),
            "label": label,
            "reasoning": reasoning
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
