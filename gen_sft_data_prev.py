"""
生成期货交易 SFT (Supervised Fine-Tuning) 数据集。
利用未来价格信息（上帝视角）构造"专家"轨迹，用于预热模型。
复用 guba/gen_rl_data.py 的逻辑以确保数据分布和划分一致。
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

def calculate_expert_positions(df: pd.DataFrame, start_idx: int, trajectory_length: int) -> List[float]:
    """
    构造专家策略（Hindsight Strategy）。
    逻辑：
    1. 获取未来一天的收益率 (Next Day Return)。
    2. 如果未来一天涨，今天应该做多；如果跌，应该做空。
    3. 仓位大小由收益率的幅度决定。使用 tanh 函数将收益率映射到 [-1, 1]。
    """
    positions = []
    
    for i in range(trajectory_length):
        current_idx = start_idx + i
        next_idx = current_idx + 1
        
        if next_idx >= len(df):
            positions.append(0.0)
            continue
            
        price_curr = df.iloc[current_idx]['结算价(元)']
        price_next = df.iloc[next_idx]['结算价(元)']
        
        # 计算未来收益率
        if price_curr <= 1e-8:
            ret = 0.0
        else:
            ret = (price_next - price_curr) / price_curr
        
        # 专家逻辑：根据收益率生成连续仓位
        scale_factor = 80.0 
        position = np.tanh(ret * scale_factor)
        
        positions.append(float(round(position, 4)))
        
    return positions

def generate_sft_dataset(excel_path: str, output_dir: str, 
                         trajectory_length: int = 5,
                         lookback_window: int = 60,
                         min_valid_features: int = 60,
                         train_ratio: float = 0.9,
                         trajectory_stride: int = 1):
    
    print(f"Loading data from: {excel_path}")
    # 复用加载逻辑
    df = gen_rl_data.load_data(excel_path)
    
    # 复用特征准备逻辑 (如果 gen_rl_data 中启用了的话，这里保持一致)
    # 注意：gen_rl_data.py 中 prepare_features 默认被注释掉了，
    # 但 generate_problem_prompt_trajectory 依赖 volatility_60 等字段。
    # 如果 Excel 中没有这些字段，gen_rl_data 会使用默认值。
    # 这里我们不显式调用 prepare_features，完全保持与 gen_rl_data.py 的行为一致。
    
    samples = []
    
    # 保持与 RL 完全一致的循环逻辑
    start_idx = max(min_valid_features, lookback_window)
    max_start_idx = len(df) - trajectory_length - 1
    
    print(f"Generating SFT samples (Trajectory Length: {trajectory_length})...")
    
    trajectory_count = 0
    for idx in range(start_idx, max_start_idx + 1, trajectory_stride):
        # 检查轨迹窗口内的数据是否完整 (复用 RL 逻辑)
        end_idx = idx + trajectory_length
        if end_idx >= len(df):
            break
        
        trajectory_df = df.iloc[idx:end_idx+1]
        if trajectory_df['结算价(元)'].isna().any() or trajectory_df['时间'].isna().any():
            continue

        # 1. 生成 Input Prompt (直接复用 gen_rl_data 的函数)
        try:
            prompt = gen_rl_data.generate_problem_prompt_trajectory(
                df, idx, 
                trajectory_length=trajectory_length,
                lookback_window=lookback_window
            )
        except Exception as e:
            print(f"Error generating prompt at {idx}: {e}")
            continue
            
        # 2. 生成 Expert Output (Hindsight) - 这是 SFT 特有的
        expert_positions = calculate_expert_positions(df, idx, trajectory_length)
        
        output_json = {
            "positions": expert_positions,
            "reasoning": ""
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

    # 划分训练集和验证集 (保持与 RL 一致)
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
        min_valid_features=62, # 保持与 RL 一致
        train_ratio=0.9,       # 保持与 RL 一致
        trajectory_stride=1    # 保持与 RL 一致
    )