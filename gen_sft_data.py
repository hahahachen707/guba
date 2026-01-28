"""
生成期货交易 SFT (Supervised Fine-Tuning) 数据集。
利用未来价格信息（上帝视角）构造"专家"轨迹，用于预热模型。
已集成交易成本过滤：只有当收益覆盖成本时，专家才建议开仓。
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

# 确保可以导入同目录下的模块
sys.path.append(str(Path(__file__).parent))

try:
    import gen_rl_data
except ImportError:
    # 尝试从父目录导入（如果在项目根目录运行）
    try:
        from guba import gen_rl_data
    except ImportError:
        raise ImportError("无法导入 gen_rl_data 模块，请检查路径。确保你在项目根目录或 guba 目录下运行。")

def preprocess_cost_only(
    df: pd.DataFrame, 
    price_col: str = '收盘价(元)', 
    cost_rate: float = 0.002  #  (双边手续费 + 滑点)
) -> pd.DataFrame:
    """
    仅考虑交易成本的数据预处理。
    逻辑：
    1. 计算未来一期收益率。
    2. 如果收益率绝对值 <= 成本，视为无利可图，净收益置为 0。
    3. 如果收益率绝对值 > 成本，净收益 = 原始收益 - 成本方向。
    """
    df = df.copy()
    
    # 检查列是否存在
    if price_col not in df.columns:
        raise ValueError(f"数据中缺少列: {price_col}")

    # 1. 计算未来一期收益率 (Raw Return)
    # shift(-1) 将下一行的数据向上平移
    next_price = df[price_col].shift(-1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_ret = (next_price - df[price_col]) / df[price_col]
    
    # 处理除零和空值（最后一行shift后是NaN）
    raw_ret = raw_ret.replace([np.inf, -np.inf], 0).fillna(0)
    
    # 2. 扣除交易成本 (Cost Adjustment)
    # 逻辑：
    # |ret| <= cost  -> 0 (不折腾)
    # ret > cost     -> ret - cost (做多赚的净利)
    # ret < -cost    -> ret + cost (做空赚的净利，注意ret是负数，所以是+cost使其绝对值变小)
    
    abs_ret = np.abs(raw_ret)
    sign_ret = np.sign(raw_ret)
    
    # 核心公式：净收益 = 符号 * (绝对值 - 成本)
    net_ret = sign_ret * (abs_ret - cost_rate)
    
    # 过滤：如果绝对值小于成本，上述公式会算出反向的值，所以要强制置 0
    final_ret = np.where(abs_ret <= cost_rate, 0.0, net_ret)
    
    df['Clean_Ret'] = final_ret
    
    return df

def calculate_expert_positions(
    df: pd.DataFrame, 
    start_idx: int, 
    trajectory_length: int,
    scale_factor: float = 80.0
) -> List[float]:
    """
    构造专家策略（Hindsight Strategy）。
    
    优化：
    不再在循环中计算收益率，而是直接读取预处理好的 'Clean_Ret' 列。
    这不仅考虑了交易成本，而且速度比逐行计算快得多。
    """
    # 直接切片获取未来一段的净收益率
    # 注意：我们要获取从 start_idx 开始的 trajectory_length 个数据
    target_rets = df['Clean_Ret'].iloc[start_idx : start_idx + trajectory_length].values
    
    # 边界处理：如果数据到了末尾，长度不够，补 0
    if len(target_rets) < trajectory_length:
        padding = np.zeros(trajectory_length - len(target_rets))
        target_rets = np.concatenate([target_rets, padding])
    
    # 专家逻辑：tanh 映射
    # 此时的 target_rets 已经是扣除成本后的净收益
    expert_pos = np.tanh(target_rets * scale_factor)
    
    # 转为列表并保留小数
    return np.round(expert_pos, 4).tolist()

def generate_sft_dataset(excel_path: str, output_dir: str, 
                         trajectory_length: int = 5,
                         lookback_window: int = 60,
                         min_valid_features: int = 60,
                         train_ratio: float = 0.9,
                         trajectory_stride: int = 1):
    
    print(f"Loading data from: {excel_path}")
    df = gen_rl_data.load_data(excel_path)
    
    # =======================================================
    # [新增步骤] 全局预处理：计算扣除成本后的收益率
    # =======================================================
    print("Preprocessing data (Calculating Cost-Adjusted Returns)...")
    # 设定成本，例如万分之三 (0.0003)
    # 建议使用 '收盘价(元)' 而非 '结算价(元)'，因为交易通常以现价成交
    df = preprocess_cost_only(df, price_col='收盘价(元)', cost_rate=0.002)
    # =======================================================
    
    samples = []
    
    # 保持与 RL 完全一致的循环逻辑
    start_idx = max(min_valid_features, lookback_window)
    max_start_idx = len(df) - trajectory_length - 1
    
    print(f"Generating SFT samples (Trajectory Length: {trajectory_length})...")
    
    trajectory_count = 0
    for idx in range(start_idx, max_start_idx + 1, trajectory_stride):
        # 检查轨迹窗口内的数据是否完整
        end_idx = idx + trajectory_length
        if end_idx >= len(df):
            break
        
        # 简单的完整性检查
        trajectory_df = df.iloc[idx:end_idx+1]
        if trajectory_df['收盘价(元)'].isna().any():
            continue

        # 1. 生成 Input Prompt (复用 gen_rl_data)
        try:
            prompt = gen_rl_data.generate_problem_prompt_trajectory(
                df, idx, 
                trajectory_length=trajectory_length,
                lookback_window=lookback_window
            )
        except Exception as e:
            print(f"Error generating prompt at {idx}: {e}")
            continue
            
        # 2. 生成 Expert Output (使用预处理后的数据)
        expert_positions = calculate_expert_positions(df, idx, trajectory_length)
        
        output_json = {
            "positions": expert_positions,
            "reasoning": "Expert strategy based on future returns adjusted for transaction costs."
        }
        
        # 3. 构造 Alpaca 格式条目
        entry = {
            "instruction": prompt,
            "input": "",
            "output": json.dumps(output_json, ensure_ascii=False)
        }
        
        samples.append(entry)
        trajectory_count += 1
        
        if trajectory_count % 1000 == 0:
            print(f"Generated {trajectory_count} samples...")

    # 划分训练集和验证集
    print(f"\nTotal samples: {len(samples)}")
    print(f"Splitting train/eval with ratio {train_ratio}")
    
    train_size = int(len(samples) * train_ratio)
    train_samples = samples[:train_size]
    eval_samples = samples[train_size:]
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Eval samples: {len(eval_samples)}")
    
    # 保存文件
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir_path / "guba_sft_train.json"
    eval_path = output_dir_path / "guba_sft_eval.json"
    
    print(f"Saving train set to {train_path}")
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
        
    print(f"Saving eval set to {eval_path}")
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_samples, f, ensure_ascii=False, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    # 配置
    EXCEL_PATH = "./data/RB0.SHF.xlsx"
    OUTPUT_DIR = "/home/tione/notebook/workspace/xiaoyangchen/work/LLaMA-Factory/data"    
    generate_sft_dataset(
        excel_path=EXCEL_PATH,
        output_dir=OUTPUT_DIR,
        trajectory_length=5, 
        lookback_window=15,
        min_valid_features=62,
        train_ratio=0.9,
        trajectory_stride=1
    )