from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


# ---------- 价格建议计算 (从 select_stock.py 移动至此) ----------

def calculate_price_suggestions(stock_code: str, trade_date: pd.Timestamp, data: Dict[str, pd.DataFrame], price_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算股票的入场价、离场价、止损价建议
    """
    # 从参数中获取收益目标，并提供默认值
    profit_target_min_pct = price_params.get("profit_target_min_pct", 0.10)
    profit_target_max_pct = price_params.get("profit_target_max_pct") # 默认为 None

    if stock_code not in data:
        return {"entry_price": 0.0, "exit_price": 0.0, "stop_loss": 0.0, "actual_date": trade_date.strftime('%Y-%m-%d')}
    
    df = data[stock_code].copy()
    df_sorted = df.sort_values('date')
    
    # 找到交易日期 T 对应的数据
    trade_date_mask = df_sorted['date'].dt.date == trade_date.date()
    if not trade_date_mask.any():
        # 如果当天停牌或无数据，则无法进行判断
        return {"entry_price": 0.0, "exit_price": 0.0, "stop_loss": 0.0, "actual_date": trade_date.strftime('%Y-%m-%d')}
        
    current_idx_loc = df_sorted.index.get_loc(df_sorted[trade_date_mask].index[0])

    # 找到 T+1 日的数据
    next_day_idx_loc = current_idx_loc + 1
    if next_day_idx_loc >= len(df_sorted):
        # 没有下一个交易日的数据，无法生成T+1建议
        return {"entry_price": 0.0, "exit_price": 0.0, "stop_loss": 0.0, "actual_date": "N/A"}

    # --- 获取T日和T+1日的数据 ---
    current_data = df_sorted.iloc[current_idx_loc]
    next_day_data = df_sorted.iloc[next_day_idx_loc]
    
    # 使用 T+1 的开盘价作为入场价
    entry_price = next_day_data['open']
    
    # 获取T日（含）之前的数据用于计算技术指标
    hist_data_t = df_sorted.iloc[:current_idx_loc + 1]
    
    # 获取最近20天的数据用于计算支撑阻力位
    end_idx = len(hist_data_t) - 1
    start_idx = max(0, end_idx - 19)
    recent_data = hist_data_t.iloc[start_idx:end_idx+1]
    
    if len(recent_data) < 5:
        # 数据不足，使用基于T+1开盘价的简单规则
        return {
            "entry_price": round(entry_price, 2), 
            "exit_price": round(entry_price * 1.05, 2), 
            "stop_loss": round(entry_price * 0.95, 2), 
            "actual_date": next_day_data['date'].strftime('%Y-%m-%d')
        }
    
    # 计算支撑位和阻力位 (基于T日及之前的数据)
    support_level = recent_data['low'].min()
    resistance_level = recent_data['high'].max()
    
    # 计算ATR（平均真实波幅）用于止损 (基于T日及之前的数据)
    high_low = recent_data['high'] - recent_data['low']
    if len(recent_data) > 1:
        high_close = abs(recent_data['high'] - recent_data['close'].shift(1))
        low_close = abs(recent_data['low'] - recent_data['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.mean()
    else:
        atr = high_low.iloc[-1]
    
    # --- 基于 T+1 开盘价计算价格建议 ---
    
    # 根据 profit_target_max_pct 决定阻力目标
    if profit_target_max_pct is not None and profit_target_max_pct > 0:
        resistance_target = min(resistance_level, entry_price * (1 + profit_target_max_pct))
    else:
        # 无上限模式
        resistance_target = resistance_level

    exit_price = max(entry_price * (1 + profit_target_min_pct), resistance_target)

    atr_stop = entry_price - (atr * 1.5)
    support_stop = support_level * 0.98 # 支撑位仍然是基于历史价格的绝对值
    stop_loss = max(atr_stop, support_stop, entry_price * 0.95)
    
    return {
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2), 
        "stop_loss": round(stop_loss, 2),
        "actual_date": next_day_data['date'].strftime('%Y-%m-%d')
    }

# --------------------------- 通用指标 --------------------------- #

def compute_kdj(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    if df.empty:
        return df.assign(K=np.nan, D=np.nan, J=np.nan)

    low_n = df["low"].rolling(window=n, min_periods=1).min()
    high_n = df["high"].rolling(window=n, min_periods=1).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-9) * 100

    K = np.zeros_like(rsv, dtype=float)
    D = np.zeros_like(rsv, dtype=float)
    for i in range(len(df)):
        if i == 0:
            K[i] = D[i] = 50.0
        else:
            K[i] = 2 / 3 * K[i - 1] + 1 / 3 * rsv.iloc[i]
            D[i] = 2 / 3 * D[i - 1] + 1 / 3 * K[i]
    J = 3 * K - 2 * D
    return df.assign(K=K, D=D, J=J)


def compute_bbi(df: pd.DataFrame) -> pd.Series:
    ma3 = df["close"].rolling(3).mean()
    ma6 = df["close"].rolling(6).mean()
    ma12 = df["close"].rolling(12).mean()
    ma24 = df["close"].rolling(24).mean()
    return (ma3 + ma6 + ma12 + ma24) / 4


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """计算 MACD 指标 (DIF, DEA, MACD)"""
    df['amount'] = df['amount'].fillna(0) # 填充成交额的NaN
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = (dif - dea) * 2
    return df.assign(DIF=dif, DEA=dea, MACD=macd)


def compute_rsi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """计算 RSI 指标"""
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=n).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_rsv(
    df: pd.DataFrame,
    n: int,
) -> pd.Series:
    """
    按公式：RSV(N) = 100 × (C - LLV(L,N)) ÷ (HHV(C,N) - LLV(L,N))
    - C 用收盘价最高值 (HHV of close)
    - L 用最低价最低值 (LLV of low)
    """
    low_n = df["low"].rolling(window=n, min_periods=1).min()
    high_close_n = df["close"].rolling(window=n, min_periods=1).max()
    rsv = (df["close"] - low_n) / (high_close_n - low_n + 1e-9) * 100.0
    return rsv


def compute_dif(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """计算 MACD 指标中的 DIF (EMA fast - EMA slow)。"""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def compute_bollinger_bands(df: pd.DataFrame, window: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """计算布林带指标"""
    if df.empty:
        return df.assign(BB_MIDDLE=np.nan, BB_UPPER=np.nan, BB_LOWER=np.nan)

    middle_band = df["close"].rolling(window=window).mean()
    std = df["close"].rolling(window=window).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)

    return df.assign(BB_MIDDLE=middle_band, BB_UPPER=upper_band, BB_LOWER=lower_band)


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """计算OBV能量潮指标"""
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv


def bbi_deriv_uptrend(
    bbi: pd.Series,
    *,
    min_window: int,
    max_window: int | None = None,
    q_threshold: float = 0.0,
) -> bool:
    """
    判断 BBI 是否"整体上升"。

    令最新交易日为 T，在区间 [T-w+1, T]（w 自适应，w ≥ min_window 且 ≤ max_window）
    内，先将 BBI 归一化：BBI_norm(t) = BBI(t) / BBI(T-w+1)。

    再计算一阶差分 Δ(t) = BBI_norm(t) - BBI_norm(t-1)。
    若 Δ(t) 的前 q_threshold 分位数 ≥ 0，则认为该窗口通过；只要存在
    **最长** 满足条件的窗口即可返回 True。q_threshold=0 时退化为
    "全程单调不降"（旧版行为）。

    Parameters
    ----------
    bbi : pd.Series
        BBI 序列（最新值在最后一位）。
    min_window : int
        检测窗口的最小长度。
    max_window : int | None
        检测窗口的最大长度；None 表示不设上限。
    q_threshold : float, default 0.0
        允许一阶差分为负的比例（0 ≤ q_threshold ≤ 1）。
    """
    if not 0.0 <= q_threshold <= 1.0:
        raise ValueError("q_threshold 必须位于 [0, 1] 区间内")

    bbi = bbi.dropna()
    if len(bbi) < min_window:
        return False

    longest = min(len(bbi), max_window or len(bbi))

    # 自最长窗口向下搜索，找到任一满足条件的区间即通过
    for w in range(longest, min_window - 1, -1):
        seg = bbi.iloc[-w:]                # 区间 [T-w+1, T]
        norm = seg / seg.iloc[0]           # 归一化
        diffs = np.diff(norm.values)       # 一阶差分
        if np.quantile(diffs, q_threshold) >= 0:
            return True
    return False

# --------------------------- Selector 类 --------------------------- #
class BBIKDJSelector:
    """
    自适应 *BBI(导数)* + *KDJ* 选股器
        • BBI: 允许 bbi_q_threshold 比例的回撤
        • KDJ: J < threshold ；或位于历史 J 的 j_q_threshold 分位及以下
        • MACD: DIF > 0
        • 收盘价波动幅度 ≤ price_range_pct
    """

    def __init__(
        self,
        j_threshold: float = -5,
        bbi_min_window: int = 90,
        max_window: int = 90,
        price_range_pct: float = 100.0,
        bbi_q_threshold: float = 0.05,
        j_q_threshold: float = 0.10,
    ) -> None:
        self.j_threshold = j_threshold
        self.bbi_min_window = bbi_min_window
        self.max_window = max_window
        self.price_range_pct = price_range_pct
        self.bbi_q_threshold = bbi_q_threshold
        self.j_q_threshold = j_q_threshold

    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        hist = hist.copy()
        hist["BBI"] = compute_bbi(hist)

        win = hist.tail(self.max_window)
        high, low = win["close"].max(), win["close"].min()
        if low <= 0 or (high / low - 1) > self.price_range_pct:
            return False

        if not bbi_deriv_uptrend(
            hist["BBI"],
            min_window=self.bbi_min_window,
            max_window=self.max_window,
            q_threshold=self.bbi_q_threshold,
        ):
            return False

        kdj = compute_kdj(hist)
        j_today = float(kdj.iloc[-1]["J"])

        j_window = kdj["J"].tail(self.max_window).dropna()
        if j_window.empty:
            return False
        j_quantile = float(j_window.quantile(self.j_q_threshold))

        if not (j_today < self.j_threshold or j_today <= j_quantile):
            return False

        hist["DIF"] = compute_dif(hist)
        return hist["DIF"].iloc[-1] > 0

    def select(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            hist = hist.tail(self.max_window + 20)
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class MACDGoldenCrossSelector:
    """
    MACD 金叉策略
    - 条件: DIF 从下向上穿过 DEA
    - 过滤: 站上 MA60, 日均成交额 > 1亿
    """
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, ma_period: int = 60, min_avg_amount: float = 1e8) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.ma_period = ma_period
        self.min_avg_amount = min_avg_amount

    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        required_len = max(self.fast, self.slow, self.signal, self.ma_period) + 20
        if len(hist) < required_len:
            return False

        hist = compute_macd(hist, self.fast, self.slow, self.signal)

        hist['MA60'] = hist['close'].rolling(window=self.ma_period).mean()
        hist['Amount20'] = hist['amount'].rolling(window=20).mean()

        today = hist.iloc[-1]
        yesterday = hist.iloc[-2]

        if today['close'] < today['MA60']:
            return False
        if today['Amount20'] < self.min_avg_amount:
            return False

        if yesterday["DIF"] < yesterday["DEA"] and today["DIF"] >= today["DEA"]:
            return True
        return False

    def select(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class RSIOversoldSelector:
    """
    RSI 超卖反弹策略
    - 条件: RSI 从下向上穿过 30
    - 过滤: 站上 MA60, 日均成交额 > 1亿
    """
    def __init__(self, n: int = 14, threshold: int = 30, ma_period: int = 60, min_avg_amount: float = 1e8):
        self.n = n
        self.threshold = threshold
        self.ma_period = ma_period
        self.min_avg_amount = min_avg_amount

    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        required_len = max(self.n + 1, self.ma_period, 20)
        if len(hist) < required_len:
            return False

        hist = hist.copy()
        hist["RSI"] = compute_rsi(hist, self.n)

        hist['MA60'] = hist['close'].rolling(window=self.ma_period).mean()
        hist['Amount20'] = hist['amount'].rolling(window=20).mean()

        today = hist.iloc[-1]
        yesterday = hist.iloc[-2]

        if today['close'] < today['MA60']:
            return False
        if today['Amount20'] < self.min_avg_amount:
            return False

        if yesterday["RSI"] < self.threshold and today["RSI"] >= self.threshold:
            return True
        return False

    def select(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class BBIShortLongSelector:
    """
    BBI 上升 + 短/长期 RSV 条件 + DIF > 0 选股器
    """
    def __init__(
        self,
        n_short: int = 3,
        n_long: int = 21,
        m: int = 3,
        bbi_min_window: int = 90,
        max_window: int = 150,
        bbi_q_threshold: float = 0.05,
    ) -> None:
        if m < 2:
            raise ValueError("m 必须 ≥ 2")
        self.n_short = n_short
        self.n_long = n_long
        self.m = m
        self.bbi_min_window = bbi_min_window
        self.max_window = max_window
        self.bbi_q_threshold = bbi_q_threshold

    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        hist = hist.copy()
        hist["BBI"] = compute_bbi(hist)

        if not bbi_deriv_uptrend(
            hist["BBI"],
            min_window=self.bbi_min_window,
            max_window=self.max_window,
            q_threshold=self.bbi_q_threshold,
        ):
            return False

        hist["RSV_short"] = compute_rsv(hist, self.n_short)
        hist["RSV_long"] = compute_rsv(hist, self.n_long)

        if len(hist) < self.m:
            return False

        win = hist.iloc[-self.m :]
        long_ok = (win["RSV_long"] >= 80).all()

        short_series = win["RSV_short"]
        short_start_end_ok = (
            short_series.iloc[0] >= 80 and short_series.iloc[-1] >= 80
        )
        short_has_below_20 = (short_series < 20).any()

        if not (long_ok and short_start_end_ok and short_has_below_20):
            return False

        hist["DIF"] = compute_dif(hist)
        return hist["DIF"].iloc[-1] > 0

    def select(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class BreakoutVolumeKDJSelector:
    """
    *成交量突破* + *KDJ* 选股器
        • J < threshold ；或位于历史 J 的 j_q_threshold 分位及以下
        • 突破日涨幅 ≥ up_threshold
        • 放量 ≥ 1/(1−volume_threshold) × 窗口内其他日成交量
        • 收盘价波动幅度 ≤ price_range_pct
    """
    def __init__(
        self,
        j_threshold: float = 0.0,
        up_threshold: float = 3.0,
        volume_threshold: float = 2.0 / 3,
        offset: int = 15,
        max_window: int = 120,
        price_range_pct: float = 10.0,
        j_q_threshold: float = 0.10,
    ) -> None:
        self.j_threshold = j_threshold
        self.up_threshold = up_threshold
        self.volume_threshold = volume_threshold
        self.offset = offset
        self.max_window = max_window
        self.price_range_pct = price_range_pct
        self.j_q_threshold = j_q_threshold

    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        if len(hist) < self.offset + 2 or len(hist) < self.max_window:
            return False

        hist = hist.tail(self.max_window).copy()

        high, low = hist["close"].max(), hist["close"].min()
        if low <= 0 or (high / low - 1) > self.price_range_pct:
            return False

        hist = compute_kdj(hist)
        hist["pct_chg"] = hist["close"].pct_change() * 100

        j_today = float(hist["J"].iloc[-1])
        j_window = hist["J"].tail(self.max_window).dropna()
        if j_window.empty:
            return False
        j_quantile = float(j_window.quantile(self.j_q_threshold))

        if not (j_today < self.j_threshold or j_today <= j_quantile):
            return False

        win = hist.tail(self.offset)
        passes = False
        for i in range(len(win)):
            if win["pct_chg"].iloc[i] < self.up_threshold:
                continue

            vol_T = win["volume"].iloc[i]
            if vol_T <= 0:
                continue
            vols_except_T = win["volume"].drop(index=win.index[i])
            if not (vols_except_T <= self.volume_threshold * vol_T).all():
                continue

            if win["close"].iloc[i] <= win["close"].iloc[:i].max():
                continue

            passes = True
            break
        return passes

    def select(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class BollingerBandsSelector:
    """
    布林带下轨策略
    - 条件: 收盘价触及或跌破布林带下轨
    - 过滤: 日均成交额 > 1亿
    """
    def __init__(self, window: int = 20, std_dev: int = 2, min_avg_amount: float = 1e8):
        self.window = window
        self.std_dev = std_dev
        self.min_avg_amount = min_avg_amount

    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        required_len = max(self.window, 20)
        if len(hist) < required_len:
            return False

        hist = hist.copy()
        hist['Amount20'] = hist['amount'].rolling(window=20).mean()
        if hist.iloc[-1]['Amount20'] < self.min_avg_amount:
            return False

        hist = compute_bollinger_bands(hist, self.window, self.std_dev)
        today = hist.iloc[-1]
        
        if pd.isna(today['BB_LOWER']):
            return False

        if today['close'] <= today['BB_LOWER']:
            return True
        return False

    def select(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class GoldenCrossSelector:
    """
    均线黄金交叉策略
    - 条件: 短期均线从下向上穿过长期均线
    - 过滤: 日均成交额 > 1亿
    """
    def __init__(self, short_window: int = 10, long_window: int = 30, min_avg_amount: float = 1e8):
        self.short_window = short_window
        self.long_window = long_window
        self.min_avg_amount = min_avg_amount

    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        required_len = max(self.long_window + 1, 20)
        if len(hist) < required_len:
            return False

        hist = hist.copy()
        hist['Amount20'] = hist['amount'].rolling(window=20).mean()
        if hist.iloc[-1]['Amount20'] < self.min_avg_amount:
            return False

        hist['MA_short'] = hist['close'].rolling(window=self.short_window).mean()
        hist['MA_long'] = hist['close'].rolling(window=self.long_window).mean()

        today = hist.iloc[-1]
        yesterday = hist.iloc[-2]

        if pd.isna(today['MA_short']) or pd.isna(today['MA_long']):
            return False

        if yesterday['MA_short'] < yesterday['MA_long'] and today['MA_short'] >= today['MA_long']:
            return True
        return False

    def select(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class OBVSelector:
    """
    OBV 能量潮策略 (价涨量增)
    - 条件: 股价和OBV都处于短期均线之上
    - 过滤: 日均成交额 > 1亿
    """
    def __init__(self, ma_window: int = 20, min_avg_amount: float = 1e8):
        self.ma_window = ma_window
        self.min_avg_amount = min_avg_amount

    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        required_len = max(self.ma_window, 20)
        if len(hist) < required_len:
            return False

        hist = hist.copy()
        hist['Amount20'] = hist['amount'].rolling(window=20).mean()
        if hist.iloc[-1]['Amount20'] < self.min_avg_amount:
            return False

        hist['OBV'] = compute_obv(hist)
        hist['Close_MA'] = hist['close'].rolling(window=self.ma_window).mean()
        hist['OBV_MA'] = hist['OBV'].rolling(window=self.ma_window).mean()
        
        today = hist.iloc[-1]
        
        if pd.isna(today['Close_MA']) or pd.isna(today['OBV_MA']):
            return False

        if today['close'] > today['Close_MA'] and today['OBV'] > today['OBV_MA']:
            return True
        return False

    def select(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class CombinedStrategySelector:
    """
    一个组合多个策略并根据加权分数进行选择的元选择器。
    """

    STRATEGIES = {
        "bbikdj": BBIKDJSelector,
        "macd": MACDGoldenCrossSelector,
        "rsi": RSIOversoldSelector,
        "breakout": BreakoutVolumeKDJSelector,
        "bollinger": BollingerBandsSelector,
        "goldencross": GoldenCrossSelector,
        "obv": OBVSelector
    }

    def __init__(self, score_threshold: float = 1.0, weights: Optional[Dict[str, float]] = None, top_n: Optional[int] = None, price_params: Optional[Dict[str, Any]] = None, **kwargs):
        self.score_threshold = score_threshold
        self.weights = weights if weights is not None else {}
        self.top_n = top_n
        self.price_params = price_params if price_params is not None else {}
        self.selectors: Dict[str, Any] = {}

        for key, selector_class in self.STRATEGIES.items():
            if key in kwargs:
                # 使用 kwargs 中为特定策略提供的参数来实例化
                self.selectors[key] = selector_class(**kwargs[key])

    def select(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        all_scores: Dict[str, float] = {}
        # 预先计算所有子策略的结果
        sub_results: Dict[str, List[str]] = {}
        for name, selector in self.selectors.items():
            sub_results[name] = selector.select(date, data)

        # 为每只股票计算总分
        all_stocks = data.keys()
        for stock_code in all_stocks:
            score = 0.0
            strategy_details = []
            for name, weight in self.weights.items():
                is_selected = stock_code in sub_results.get(name, [])
                if is_selected:
                    score += weight
                strategy_details.append(f"S{list(self.weights.keys()).index(name)+1}({name.upper()}):{is_selected}")

            if score > 0:
                all_scores[stock_code] = score
                # 调试日志 (在回测时暂时禁用，以保持输出清洁)
                # print(f"[DEBUG {date.date()}] Stock: {stock_code} | Score: {score:.2f} | Threshold: {self.score_threshold} | {', '.join(strategy_details)}")

        # 为所有有得分的股票计算详细信息（包括风险收益比）
        stocks_with_details = []
        for stock_code, score in all_scores.items():
            prices = calculate_price_suggestions(stock_code, date, data, self.price_params)
            risk_reward_ratio = 0
            # 确保价格有效再计算
            if prices.get("entry_price", 0) > 0 and prices.get("entry_price") > prices.get("stop_loss", 0):
                potential_return = prices["exit_price"] - prices["entry_price"]
                potential_loss = prices["entry_price"] - prices["stop_loss"]
                if potential_loss > 0:
                    risk_reward_ratio = potential_return / potential_loss
            
            details = {
                'code': stock_code, 
                'score': score, 
                'risk_reward_ratio': risk_reward_ratio,
                'prices': prices
            }
            stocks_with_details.append(details)
            
        # 根据得分（主）和风险收益比（次）对股票进行排序
        sorted_stocks = sorted(
            stocks_with_details, 
            key=lambda item: (item['score'], item.get('risk_reward_ratio', 0)), 
            reverse=True
        )

        # 根据 top_n 或 score_threshold 返回最终结果
        if self.top_n is not None and self.top_n > 0:
            return sorted_stocks[:self.top_n]
        else:
            return [s for s in sorted_stocks if s['score'] >= self.score_threshold]
