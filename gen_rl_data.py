"""
生成 qwen3 强化学习数据集，用于训练期货交易 agent。
从 Excel 文件读取数据，构造训练样本并保存为 jsonl 格式。
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any
from pathlib import Path

# 注意：在强化学习训练中，历史动作应该从实际模型输出中获取
# 这里使用参考策略只是为了生成初始的数据集


def load_data(excel_path: str) -> pd.DataFrame:
    """从 Excel 文件加载数据"""
    df = pd.read_excel(excel_path)
    # Excel 中的除零错误（#DIV/0!）读入后为 NaN，inf 也需处理；统一置为 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
    # 确保按时间排序
    df = df.sort_values('时间')
    df = df.reset_index(drop=True)
    return df


def preprocess_cost_only(
    df: pd.DataFrame, 
    price_col: str = '结算价(元)', 
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



def calculate_volatility(returns: pd.Series, window: int = 60) -> pd.Series:
    """
    计算指数加权移动标准差作为波动率估计
    """
    volatility = returns.ewm(span=window, adjust=False).std()
    return volatility


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    准备特征数据，计算所需的技术指标
    """
    df = df.copy()
    
    # 使用结算价作为价格
    prices = df['结算价(元)']
    
    # 计算收益率（用于波动率计算）
    returns = prices.pct_change(1)
    
    # 计算波动率（60天窗口，如论文中所述）
    if 'volatility_60' not in df.columns:
        df['volatility_60'] = calculate_volatility(returns, window=60)
    
    # 填充缺失值
    df['volatility_60'] = df['volatility_60'].bfill().fillna(0.15)  # 默认15%
    
    return df


def generate_problem_prompt_trajectory(df: pd.DataFrame, start_idx: int, trajectory_length: int = 20, 
                                       lookback_window: int = 60, price_scale: float = 1.0,
                                       use_sentiment: bool = False) -> str:
    """
    生成轨迹级别的问题提示，包含多天的市场状态信息
    
    Args:
        df: 完整数据框
        start_idx: 轨迹起始索引
        trajectory_length: 轨迹长度（天数）
        lookback_window: 回看窗口大小（用于提供历史上下文）
        price_scale: 价格缩放因子（数据增广用）
        use_sentiment: 是否在prompt中加入情绪指标
    """
    # 获取轨迹窗口的数据
    end_idx = start_idx + trajectory_length
    if end_idx > len(df):
        raise ValueError(f"轨迹结束索引 {end_idx} 超出数据范围 {len(df)}")
    
    
    
    # 获取历史数据窗口（用于提供上下文）
    history_start_idx = max(0, start_idx - lookback_window)
    history_df = df.iloc[history_start_idx:start_idx]
    
    # 提取历史价格趋势（用于上下文）
    # 使用完整的lookback_window天数的历史数据（如果可用）
    actual_history_length = len(history_df)    
    # 构建轨迹中每一天的状态信息
    daily_states = []
    for i in range(actual_history_length):
        row = history_df.iloc[i]
        
        # 提取技术指标
        norm_return_1 = row.get('norm_return_1', 0)
        norm_return_7 = row.get('norm_return_7', 0)
        norm_return_30 = row.get('norm_return_30', 0)
        volatility = row.get('volatility_60', 0.15)
        rsi = row.get('RSI_30', 50)
        macd = row.get('MACD_16_48', 0)
        sent_1 = row.get('sent_1', 0.0)
        
        # 应用价格缩放增广
        price = row['结算价(元)'] * price_scale
        time = row['时间']
        
        state = {
            'day': i + 1,
            'time': str(time),
            'price': float(price),
            'volatility': float(volatility),
            'norm_return_1': float(norm_return_1),
            'norm_return_7': float(norm_return_7),
            'norm_return_30': float(norm_return_30),
            'rsi': float(rsi),
            'macd': float(macd)
        }
        
        if use_sentiment:
            state['sent_1'] = float(sent_1)
            
        daily_states.append(state)
    
    # 构建提示
    prompt = f"""# Role
You are an expert futures trading agent. Your task is to make trading decisions for a sequence of {trajectory_length} days based on market state information.

# Objective
Analyze the market state for each day in the trajectory and decide on trading positions. You should consider:
- Price trends and historical context
- Technical indicators (normalized returns, volatility, RSI, MACD) for each day"""

    if use_sentiment:
        prompt += "\n- Daily sentiment indicator : positive values represent bullish, negative values represent bearish, and the absolute value represents the intensity of the sentiment."

    prompt += f"""
- Risk management through position sizing
- Sequential decision-making: each day's decision should consider previous days' market conditions

# Historical market State Information for {lookback_window} Days (for reference)
"""
    
    # 添加每一天的状态信息
    for state in daily_states:
        prompt += f"""Day {state['day']}:
  Time: {state['time']}
  Price: {state['price']:.2f}
  Volatility (60-day EWMA): {state['volatility']:.4f}
  Normalized Return (1 day): {state['norm_return_1']:.4f}
  Normalized Return (7 days): {state['norm_return_7']:.4f}
  Normalized Return (30 days): {state['norm_return_30']:.4f}
  RSI (30-day): {state['rsi']:.2f}
  MACD (16-48): {state['macd']:.4f}
"""
        if use_sentiment:
            prompt += f"  Sentiment: {state['sent_1']:.2f}\n"

        prompt += "\n"
    
    prompt += f"""# Output Format
You must strictly output a valid JSON object following the structure below:
{{
  "positions": [action_0, action_1, ..., action_{trajectory_length-1}],
  "reasoning": "Brief explanation of your trading strategy for this trajectory"
}}

# Position Values (CRITICAL: Use Continuous Values)
**You MUST use continuous values in the range [-1, 1] for position sizing. Avoid using only discrete values like 0.0, 1.0, or -1.0.**

Position value meaning:
- -1.0: Fully short position (maximum bearish, expect price to fall)
- 0.0: No position (neutral, hold cash)
- 1.0: Fully long position (maximum bullish, expect price to rise)
- **Continuous values between -1 and 1 represent partial positions for precise risk management:**
  - 0.5: 50% long position (moderate bullish confidence)
  - -0.3: 30% short position (mild bearish view)
  - 0.75: 75% long position (strong bullish confidence)
  - -0.6: 60% short position (moderate bearish view)
  - 0.25: 25% long position (weak bullish signal)

**Why continuous values matter:**
- Better risk management: Adjust position size based on confidence level
- More precise trading: Reflect nuanced market signals rather than binary decisions
- Optimal performance: Fine-tune exposure to match market conditions

# Example Output (with continuous values):
{{
  "positions": [0.35, -0.15, 0.62, 0.48, -0.22],
  "reasoning": "Day 1: Moderate bullish (0.35) based on positive RSI and MACD convergence. Day 2: Slight bearish shift (-0.15) due to increased volatility. Day 3: Strong bullish (0.62) on breakout signal. Day 4: Moderate bullish (0.48) maintaining position. Day 5: Slight bearish (-0.22) for profit-taking."
}}

Important:
- The "positions" array must contain exactly {trajectory_length} values, one for each day
- Each position value corresponds to the trading decision for that day
- positions[0] corresponds to Day 1, positions[1] to Day 2, and so on
- Consider the sequential nature: decisions should account for market evolution across days
- **Use continuous values that reflect your confidence level, not just 0.0, 1.0, or -1.0**

Reasoning: Brief explanation of your overall trading strategy for this {trajectory_length}-day trajectory

# Formatting Constraints
1. Output **only** the raw JSON string.
2. Do **not** use markdown code blocks (e.g., ```json ... ```).
3. Ensure the JSON is syntactically valid (proper escaping of quotes, no trailing commas).
4. The "positions" array must have exactly {trajectory_length} elements.
5. **Use continuous values (e.g., 0.35, -0.42, 0.78) rather than only discrete values (0.0, 1.0, -1.0).**
"""
    
    return prompt


def compute_simple_position(row: pd.Series) -> float:
    """
    计算简单的交易位置（用于生成参考策略的动作，作为初始历史动作）
    返回 [-1, 1] 范围内的连续值
    实际训练时，这个应该由模型预测
    """
    # 提取特征
    norm_return_7 = row.get('norm_return_7', 0)
    norm_return_30 = row.get('norm_return_30', 0)
    rsi = row.get('RSI_30', 50)
    macd = row.get('MACD_16_48', 0)
    
    # 简单的交易策略示例：返回连续值
    # 基于多个指标的综合判断，输出 [-1, 1] 的连续值
    
    # 趋势信号（归一化到 [-1, 1]）
    trend_signal = np.tanh(norm_return_30 * 0.5)  # 使用 tanh 将趋势信号映射到 [-1, 1]
    
    # RSI 信号（超买/超卖调整）
    rsi_signal = 0.0
    if rsi > 70:
        rsi_signal = -0.3 * (rsi - 70) / 30  # 超买，偏向做空
    elif rsi < 30:
        rsi_signal = 0.3 * (30 - rsi) / 30  # 超卖，偏向做多
    
    # MACD 信号
    macd_signal = np.tanh(macd * 0.3) if not np.isnan(macd) else 0.0
    
    # 综合信号，加权平均
    position = 0.6 * trend_signal + 0.2 * rsi_signal + 0.2 * macd_signal
    
    # 确保在 [-1, 1] 范围内
    position = max(-1.0, min(1.0, position))
    
    return float(position)


def generate_answer_ground_truth_trajectory(df: pd.DataFrame, start_idx: int, 
                                             trajectory_length: int = 20,
                                             price_scale: float = 1.0) -> Dict[str, Any]:
    """
    生成轨迹级别的 ground truth 信息，包含多天的价格和波动率序列
    
    Args:
        df: 完整数据框
        start_idx: 轨迹起始索引
        trajectory_length: 轨迹长度（天数）
        price_scale: 价格缩放因子（数据增广用）
    
    Note:
        volatility_target 和 transaction_cost_bp 等超参数会在 reward 函数中使用默认值
        如果需要自定义这些参数，可以在 reward 函数中设置，或通过配置文件传递
    """
    # import ipdb; ipdb.set_trace()
    # 获取轨迹窗口的数据
    start_idx -= 2
    end_idx = start_idx + (trajectory_length + 1)
    if end_idx >= len(df):
        raise ValueError(f"轨迹结束索引 {end_idx} 超出数据范围 {len(df)}")
    
    trajectory_df = df.iloc[start_idx:end_idx+1]  # +1 因为需要 end_idx 的价格来计算最后一天的收益
    
    # 提取价格序列（需要 trajectory_length + 1 个价格值，用于计算每日收益）
    # 应用价格缩放增广
    prices = (trajectory_df['结算价(元)'].values * price_scale).tolist()[1:]
    
    # 提取波动率序列（trajectory_length 个值，每个对应一天的决策）
    volatilities = trajectory_df['volatility_60'].values.tolist()[:-1]  # 排除最后一个，因为最后一个不需要决策
    
    # 获取初始动作（轨迹开始前一天的参考动作）
    initial_action = 0.0
    if start_idx > 0:
        prev_row = df.iloc[start_idx - 1]
        initial_action = compute_simple_position(prev_row)
    
    # 获取初始波动率（用于计算第一步的调仓成本）
    initial_volatility = volatilities[0]
    volatilities = volatilities[1:]
    
    # 获取专家动作序列 (用于 Reward MSE Loss)
    # 注意：专家动作基于收益率计算，而收益率是相对值，因此不受 price_scale 影响
    expert_positions = calculate_expert_positions(df, start_idx, trajectory_length)

    ground_truth = {
        "sequence_id": "RB0.SHF",  # 序列标识符
        "trajectory_start_idx": int(start_idx),  # 轨迹起始索引
        "time_stamp": str(trajectory_df.iloc[2]['时间']),  # Timestamp 需转为 str 才能 JSON 序列化
        "trajectory_length": int(trajectory_length),  # 轨迹长度
        "prices": [float(p) for p in prices],  # 价格序列（trajectory_length + 1 个值）
        "volatilities": [float(v) for v in volatilities],  # 波动率序列（trajectory_length 个值）
        "initial_action": float(initial_action),  # 初始动作（轨迹开始前的动作）
        "initial_volatility": float(initial_volatility),  # 初始波动率
        "expert_positions": expert_positions,  # 专家动作序列
        "price_scale": float(price_scale), # 记录缩放因子
    }
    
    return {
        "ground_truth": json.dumps(ground_truth, ensure_ascii=False)
    }


def generate_dataset(excel_path: str, output_dir: str, 
                     trajectory_length: int = 20,
                     lookback_window: int = 60,
                     min_valid_features: int = 60,
                     train_ratio: float = 0.9,
                     trajectory_stride: int = 1,
                     aug_times: int = 1,
                     use_sentiment: bool = False):
    """
    生成轨迹级别的数据集，并按比例划分为训练集和测试集
    
    Args:
        excel_path: Excel 文件路径
        output_dir: 输出目录路径
        trajectory_length: 轨迹长度（天数），模型需要一次性生成这么多天的决策
        lookback_window: 回看窗口大小（用于提供历史上下文）
        min_valid_features: 最少需要的有效特征数量（跳过前面数据不足的部分）
        train_ratio: 训练集比例（默认 0.9，即 9:1 划分）
        trajectory_stride: 轨迹滑动步长（默认1，即每个轨迹之间重叠trajectory_length-1天）
                         如果设为trajectory_length，则轨迹之间不重叠
        aug_times: 数据增广倍数（默认1，即不增广）。如果 > 1，会对每个样本生成 aug_times 个变体，
                   价格在 [0.95, 1.05] 范围内随机缩放。
        use_sentiment: 是否在prompt中加入情绪指标
    """
    # 加载数据
    print(f"加载数据从: {excel_path}")
    df = load_data(excel_path)
    print(f"共加载 {len(df)} 条数据")
    
    # 预处理：计算 Clean_Ret 用于专家策略
    print("预处理数据 (计算 Clean_Ret)...")
    df = preprocess_cost_only(df)
    
    # 准备特征
    print("准备特征...")
    # df = prepare_features(df)
    
    # 1. 收集有效索引
    print("正在扫描有效轨迹索引...")
    valid_indices = []
    start_idx = max(min_valid_features, lookback_window)
    max_start_idx = len(df) - trajectory_length - 1
    
    if start_idx >= max_start_idx:
        raise ValueError(f"数据不足：需要至少 {start_idx + trajectory_length + 1} 条数据，但只有 {len(df)} 条")

    for idx in range(start_idx, max_start_idx + 1, trajectory_stride):
        # 检查轨迹窗口内的数据是否完整
        end_idx = idx + trajectory_length
        if end_idx >= len(df):
            break
        
        # 检查必要字段是否存在
        trajectory_df = df.iloc[idx:end_idx+1]
        if trajectory_df['结算价(元)'].isna().any() or trajectory_df['时间'].isna().any():
            continue
            
        valid_indices.append(idx)

    # 2. 划分训练/测试索引
    total_valid = len(valid_indices)
    split_point = int(total_valid * train_ratio)
    train_indices = valid_indices[:split_point]
    eval_indices = valid_indices[split_point:]
    
    print(f"共找到 {total_valid} 个有效轨迹")
    print(f"训练集索引数: {len(train_indices)} (前 {train_ratio*100:.0f}%)")
    print(f"测试集索引数: {len(eval_indices)} (后 {(1-train_ratio)*100:.0f}%)")

    # 定义内部生成函数
    def _generate_samples(indices, is_train):
        samples = []
        count = 0
        # 只有训练集才进行增广
        current_aug_times = aug_times if is_train else 1
        
        mode_str = "训练集(带增广)" if is_train else "测试集(无增广)"
        print(f"\n开始生成 {mode_str}...")

        for idx in indices:
            # 增广循环
            for i in range(current_aug_times):
                # 第一次迭代或者是测试集，使用原始数据
                if i == 0:
                    price_scale = 1.0
                else:
                    # 随机生成 [0.95, 1.05] 之间的缩放因子
                    price_scale = np.random.uniform(0.95, 1.05)

                try:
                    # 生成轨迹级别的问题提示
                    problem = generate_problem_prompt_trajectory(
                        df, idx, 
                        trajectory_length=trajectory_length,
                        lookback_window=lookback_window,
                        price_scale=price_scale,
                        use_sentiment=use_sentiment
                    )
                    
                    # 生成轨迹级别的 ground truth
                    answer_info = generate_answer_ground_truth_trajectory(
                        df, idx,
                        trajectory_length=trajectory_length,
                        price_scale=price_scale
                    )
                    
                    # 构造样本
                    sample = {
                        "problem": problem,
                        "answer": answer_info["ground_truth"],
                    }
                    samples.append(sample)
                    
                except Exception as e:
                    print(f"生成失败 (idx={idx}): {e}")
                    continue
            
            count += 1
            if count % 100 == 0:
                print(f"  已处理 {count}/{len(indices)} 个原始轨迹...")
        
        return samples

    # 3. 生成样本
    train_samples = _generate_samples(train_indices, is_train=True)
    eval_samples = _generate_samples(eval_indices, is_train=False)
    
    print(f"\n生成完成！")
    print(f"训练集样本数: {len(train_samples)}")
    print(f"测试集样本数: {len(eval_samples)}")
    
    # 创建输出目录
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 保存训练集
    train_path = output_dir_path / "guba_train_verl.jsonl"
    print(f"\n保存训练集到: {train_path}")
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 保存测试集
    eval_path = output_dir_path / "guba_eval_verl.jsonl"
    print(f"保存测试集到: {eval_path}")
    with open(eval_path, 'w', encoding='utf-8') as f:
        for sample in eval_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n数据集生成完成！")
    print(f"  - 训练集: {train_path} ({len(train_samples)} 个轨迹样本)")
    print(f"  - 测试集: {eval_path} ({len(eval_samples)} 个轨迹样本)")
    print(f"  - 轨迹长度: {trajectory_length} 天")
    print(f"  - 轨迹滑动步长: {trajectory_stride} 天")


if __name__ == "__main__":
    # 配置路径和参数
    # output_dir = "./data"
    excel_path = "./data/RB0.SHF.xlsx"
    output_dir = "/home/tione/notebook/workspace/xiaoyangchen/work/data/guba"
    
    # 生成轨迹级别数据集
    generate_dataset(
        excel_path=excel_path,
        output_dir=output_dir,
        trajectory_length=5,  # 轨迹长度：模型需要一次性生成20天的决策
        lookback_window=15,  # 历史回看窗口：提供60天的历史上下文
        min_valid_features=62,  # 最少需要的有效特征数量
        train_ratio=0.9,  # 9:1 划分训练集和测试集
        trajectory_stride=1,  # 轨迹滑动步长
        aug_times=5,  # 数据增广：每个样本生成5个变体（1个原始 + 4个随机缩放）
        use_sentiment=True  # 是否在prompt中加入情绪指标
        # 注意：volatility_target 和 transaction_cost_bp 使用 reward 函数的默认值
        # 如需自定义，可在 reward 函数中设置或通过配置文件传递
    )