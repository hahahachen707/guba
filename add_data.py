"""
添加额外特征到Excel文件，如归一化收益率、滚动波动率、移动平均差距、RSI、MACD等
"""
import pandas as pd
import numpy as np

# --- 辅助函数保持不变，它们是正确的 ---

def calculate_rsi(prices, period=30):
    """ 计算相对强弱指数（RSI） """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period-1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_paper_macd(prices, fast_period=8, slow_period=24):
    """ 计算论文版 MACD """
    m_s = prices.ewm(span=fast_period, adjust=False).mean()
    m_l = prices.ewm(span=slow_period, adjust=False).mean()
    std_price = prices.rolling(window=63).std()
    epsilon = 1e-8
    q_t = (m_s - m_l) / (std_price + epsilon)
    std_q = q_t.rolling(window=252).std()
    macd_final = q_t / (std_q + epsilon)
    return macd_final

def calculate_normalized_return(prices, lookback):
    """ 计算归一化收益率 """
    ret = prices.pct_change(lookback)
    daily_returns = prices.pct_change(1)
    sigma_t = daily_returns.ewm(span=60).std()
    normalized_ret = ret / (sigma_t * np.sqrt(lookback) + 1e-8)
    return normalized_ret

# --- 修正后的主函数 ---

def add_features(feats):
    """
    添加符合 Deep RL for Trading 论文的特征及自定义特征
    """
    # 0. 预处理：确保按时间排序，防止 diff 计算错误
    # feats = feats.sort_index() # 如果你不确定输入是否排序，请取消注释这行
    
    # --- 1. 归一化收益率特征 (Normalized Returns) ---
    # 这里的 1 是有意义的，代表单期归一化动量
    return_periods = [1, 7, 30, 60, 90, 180]
    for lag in return_periods:
        feats[f"norm_return_{lag}"] = calculate_normalized_return(feats["close"], lag)
    
    # --- 2. 滚动波动率特征 (Rolling Volatility) ---
    # 修正：移除了 period=1，因为 rolling(1).std() 产生 NaN 且无意义
    # 预计算 Log Returns 避免在循环中重复计算
    log_returns = np.log(feats["close"]).diff()
    
    vol_periods = [7, 30, 60, 90, 180]
    for period in vol_periods:
        feats[f"volatility_{period}"] = log_returns.rolling(period).std()
    
    # --- 3. 移动平均差距特征 (MA Gap) ---
    # 修正：移除了 period=1，因为 Price/MA(1) 恒等于 1
    # 修正：修复了 'minute' 后缀导致的 KeyError
    
    ma_periods = [7, 30, 60, 90, 180]
    for period in ma_periods:
        col_name = f"MA_gap_{period}"
        ma = feats["close"].rolling(period).mean()
        feats[col_name] = feats["close"] / ma
        
        # 处理除以0的情况 (虽然 MA 为 0 很少见)
        feats[col_name] = feats[col_name].replace([np.inf, -np.inf], np.nan)
    
    # --- 4. RSI 特征 ---
    feats["RSI_30"] = calculate_rsi(feats["close"], period=30)
    feats["RSI_15"] = calculate_rsi(feats["close"], period=15)
    
    # --- 5. MACD 特征 (论文版) ---
    macd_scales = [(8, 24), (16, 48), (32, 96)]
    for fast, slow in macd_scales:
        col_name = f"MACD_{fast}_{slow}"
        feats[col_name] = calculate_paper_macd(feats["close"], fast_period=fast, slow_period=slow)
    
    # --- 6. 清洗数据 ---
    # 再次处理可能产生的 inf
    feats = feats.replace([np.inf, -np.inf], np.nan)
    
    # 处理 NaN
    # 1. ffill: 用前一个有效值填充 (模拟实盘，保持最近状态)
    feats = feats.ffill()
    
    # 2. fillna(0): 处理最开始无法 ffill 的数据 (冷启动)
    # 注意：这会导致前 252+ 行数据有很多 0，建议在输入模型前 drop 掉
    feats = feats.fillna(np.nan)
    
    return feats

# 简单测试块
if __name__ == "__main__":
    # 1. 读取Excel文件
    print("正在读取Excel文件...")
    excel_file = 'RB0.SHF.xlsx'
    df = pd.read_excel(excel_file)
    
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 2. 提取"收盘价(元)"列作为close
    # 查找包含"收盘价"的列（支持中文编码）
    close_col = None
    for col in df.columns:
        col_str = str(col)
        # 尝试多种可能的列名匹配
        if '收盘价' in col_str or '收盘' in col_str or 'close' in col_str.lower():
            close_col = col
            break
    
    if close_col is None:
        # 如果找不到，打印所有列名帮助调试
        print("所有列名:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col} (type: {type(col)})")
        raise ValueError("未找到收盘价列，请检查Excel文件列名")
    
    print(f"\n找到收盘价列: {close_col}")
    
    # 检查收盘价列是否有缺失值
    if df[close_col].isna().any():
        print(f"警告: 收盘价列有 {df[close_col].isna().sum()} 个缺失值，将删除这些行")
        df = df.dropna(subset=[close_col])
    
    # 创建新的DataFrame，包含close列
    # 先处理缺失值，确保数据质量
    df_clean = df.copy()
    df_clean[close_col] = pd.to_numeric(df_clean[close_col], errors='coerce')
    
    # 删除收盘价为NaN的行
    initial_len = len(df_clean)
    df_clean = df_clean.dropna(subset=[close_col])
    removed_rows = initial_len - len(df_clean)
    if removed_rows > 0:
        print(f"已删除 {removed_rows} 行包含无效收盘价的数据")
    
    # 创建特征DataFrame，保持与原始df相同的索引
    feats = pd.DataFrame(index=df_clean.index)
    feats['close'] = df_clean[close_col].copy()
    
    # 如果有时间列，也保留（用于后续写回和调试）
    time_col = None
    for col in df_clean.columns:
        col_str = str(col)
        if '时间' in col_str or '日期' in col_str or 'date' in col_str.lower():
            time_col = col
            feats['date'] = df_clean[col].copy()
            break
    
    print(f"有效数据行数: {len(feats)}")
    print(f"close列统计信息:")
    print(feats['close'].describe())
    
    # 3. 计算特征
    print("\n正在计算特征...")
    feats = add_features(feats)
    
    print(f"计算后的数据形状: {feats.shape}")
    new_cols = [col for col in feats.columns if col not in ['close', 'date']]
    print(f"新增特征列数量: {len(new_cols)}")
    print(f"特征列示例: {new_cols[:10]}")
    
    # 4. 将新特征合并到原始DataFrame
    # 只添加新计算的特征列（排除close和date，因为原始数据可能已有）
    new_feature_cols = [col for col in feats.columns 
                       if col not in ['close', 'date'] 
                       and col not in df_clean.columns]
    
    print(f"\n将添加 {len(new_feature_cols)} 个新特征列到Excel文件")
    
    # 将新特征添加到清理后的DataFrame（索引已对齐）
    for col in new_feature_cols:
        df_clean[col] = feats[col].values
    
    # 使用清理后的DataFrame作为最终输出
    df_final = df_clean
    
    # 5. 写回Excel文件
    print(f"\n正在保存到 {excel_file}...")
    try:
        df_final.to_excel(excel_file, index=False)
        print("✓ 成功保存到Excel文件！")
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        print("尝试保存为备份文件...")
        backup_file = excel_file.replace('.xlsx', '_with_features.xlsx')
        df_final.to_excel(backup_file, index=False)
        print(f"✓ 已保存为备份文件: {backup_file}")
    
    print("\n" + "="*50)
    print("处理完成！")
    print("="*50)
    print(f"最终数据形状: {df_final.shape}")
    print(f"原始列数: {len(df.columns)}")
    print(f"新增特征列数: {len(new_feature_cols)}")
    print(f"最终列数: {len(df_final.columns)}")
    print(f"\n新增的特征列（前15个）:")
    for i, col in enumerate(new_feature_cols[:15], 1):
        print(f"  {i}. {col}")
    if len(new_feature_cols) > 15:
        print(f"  ... 还有 {len(new_feature_cols) - 15} 个特征列")
