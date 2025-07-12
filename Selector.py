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
    current_data = df_sorted.iloc[current_idx_loc]

    # --- 尝试获取 T+1 日数据，若无则进行估算 ---
    next_day_idx_loc = current_idx_loc + 1
    if next_day_idx_loc >= len(df_sorted):
        # T+1 数据不存在，使用 T 日收盘价作为估算入场价
        entry_price = current_data['close']
        actual_date_str = f"{current_data['date'].strftime('%Y-%m-%d')} (估算)"
    else:
        # T+1 数据存在，使用 T+1 开盘价作为实际入场价
        next_day_data = df_sorted.iloc[next_day_idx_loc]
        entry_price = next_day_data['open']
        actual_date_str = next_day_data['date'].strftime('%Y-%m-%d')
    
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
            "actual_date": actual_date_str
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
        "actual_date": actual_date_str
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


class LongTermValueSelector:
    """
    长线价值策略: 结合长期趋势、低波动性和持续动量进行选股。
    - 趋势: 50日均线持续在200日均线之上
    - 波动性: ATR占收盘价百分比低于阈值
    - 动量: RSI > 50, MACD金叉且在0轴之上
    """
    def __init__(
        self,
        ma_short: int = 50,
        ma_long: int = 200,
        trend_stability_window: int = 20,
        atr_window: int = 20,
        max_atr_pct: float = 0.05,
        rsi_window: int = 14,
        rsi_threshold: float = 50,
        min_avg_amount: float = 1e8,
    ):
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.trend_stability_window = trend_stability_window
        self.atr_window = atr_window
        self.max_atr_pct = max_atr_pct
        self.rsi_window = rsi_window
        self.rsi_threshold = rsi_threshold
        self.min_avg_amount = min_avg_amount

    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        """检查单个股票是否符合长线价值策略"""
        # 需要足够的数据来计算最长的均线
        if len(hist) < self.ma_long + self.trend_stability_window:
            return False
            
        # 1. 成交额过滤
        if hist['amount'].rolling(window=60).mean().iloc[-1] < self.min_avg_amount:
            return False

        # 2. 计算技术指标
        # MA
        ma_short = hist["close"].rolling(window=self.ma_short).mean()
        ma_long = hist["close"].rolling(window=self.ma_long).mean()
        
        # ATR
        high_low = hist['high'] - hist['low']
        high_close = abs(hist['high'] - hist['close'].shift(1))
        low_close = abs(hist['low'] - hist['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_window).mean()
        atr_pct = atr.iloc[-1] / hist['close'].iloc[-1]
        
        # RSI
        rsi = compute_rsi(hist, n=self.rsi_window)
        
        # MACD
        macd_data = compute_macd(hist)

        # 3. 应用策略规则
        # 趋势: MA short > MA long
        if ma_short.iloc[-1] < ma_long.iloc[-1]:
            return False
            
        # 趋势稳定性: 过去 trend_stability_window 天内，MA short 始终 > MA long
        trend_stable = (ma_short.iloc[-self.trend_stability_window:] > ma_long.iloc[-self.trend_stability_window:]).all()
        if not trend_stable:
            return False
            
        # 低波动性: ATR百分比低于阈值
        if atr_pct > self.max_atr_pct:
            return False
            
        # 动量: RSI > 阈值
        if rsi.iloc[-1] < self.rsi_threshold:
            return False
            
        # MACD 确认: DIF > DEA and DIF > 0
        if not (macd_data['DIF'].iloc[-1] > macd_data['DEA'].iloc[-1] and macd_data['DIF'].iloc[-1] > 0):
            return False
            
        return True

    def select(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """根据长线价值策略筛选股票"""
        selected_stocks = []
        for stock_code, df in data.items():
            hist = df[df["date"] <= date].copy()
            if self._passes_filters(hist):
                # 注意：这里的 price_params 是一个空字典，因为独立选择器没有自己的价格参数
                # 在 CombinedStrategySelector 中使用时，会传递全局的价格参数
                prices = calculate_price_suggestions(stock_code, date, data, {})
                details = {
                    'code': stock_code,
                    'score': None,  # 独立选择器不计算分数
                    'risk_reward_ratio': 0, # 独立选择器不计算
                    'prices': prices
                }
                selected_stocks.append(details)
        return selected_stocks


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
        "obv": OBVSelector,
        "long_term": LongTermValueSelector
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


class MLEnhancedCombinedSelector:
    """
    ML增强的组合选股器
    结合传统技术指标和机器学习预测
    """
    
    def __init__(self, base_selector_config: Dict[str, Any], ml_predictor_config: Dict[str, Any], 
                 ml_threshold: float = 0.6, ml_weight: float = 0.3):
        """
        初始化ML增强选股器
        
        Args:
            base_selector_config: 基础选股器配置
            ml_predictor_config: ML预测器配置
            ml_threshold: ML预测阈值
            ml_weight: ML权重
        """
        self.ml_threshold = ml_threshold
        self.ml_weight = ml_weight
        self.ml_predictor = None
        self.ml_predictor_config = ml_predictor_config
        
        # 初始化基础选股器
        base_class_name = base_selector_config.get("class")
        if base_class_name == "CombinedStrategySelector":
            self.base_selector = CombinedStrategySelector(**base_selector_config.get("params", {}))
        else:
            raise ValueError(f"不支持的基础选股器类型: {base_class_name}")
    
    def _load_ml_predictor(self, data):
        """加载或训练ML预测器"""
        if self.ml_predictor is not None:
            return
            
        try:
            from ml_predictor import LSTMStockPredictor
            
            model_path = self.ml_predictor_config.get("model_path", "./lstm_model.h5")
            
            self.ml_predictor = LSTMStockPredictor(
                sequence_length=self.ml_predictor_config.get("sequence_length", 20),
                epochs=self.ml_predictor_config.get("epochs", 30),
                batch_size=self.ml_predictor_config.get("batch_size", 32)
            )
            
            # 尝试加载现有模型
            try:
                import os
                if os.path.exists(model_path):
                    self.ml_predictor.load_model(model_path)
                    print(f"✅ 已加载现有ML模型: {model_path}")
                else:
                    print(f"⚠️  模型文件不存在: {model_path}")
                    print("🚀 开始训练新的ML模型...")
                    
                    # 训练新模型
                    self.ml_predictor.train(data, min_samples=500)
                    self.ml_predictor.save_model(model_path)
                    print(f"✅ 新模型已保存: {model_path}")
                    
            except Exception as e:
                print(f"⚠️  加载模型失败: {e}")
                print("🚀 开始训练新的ML模型...")
                
                # 训练新模型
                self.ml_predictor.train(data, min_samples=500)
                self.ml_predictor.save_model(model_path)
                print(f"✅ 新模型已保存: {model_path}")
                
        except ImportError:
            print("❌ 无法导入ML预测器，将使用基础选股器")
            self.ml_predictor = None
        except Exception as e:
            print(f"❌ ML预测器初始化失败: {e}")
            self.ml_predictor = None
    
    def select(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        ML增强选股方法
        
        Args:
            date: 选股日期
            data: 股票数据字典
            
        Returns:
            选中的股票列表
        """
        # 获取基础选股结果
        base_picks = self.base_selector.select(date, data)
        
        # 如果没有ML预测器，直接返回基础结果
        if self.ml_predictor is None:
            self._load_ml_predictor(data)
            if self.ml_predictor is None:
                return base_picks
        
        # 使用ML增强选股结果
        enhanced_picks = []
        ml_predictions = []
        
        print(f"🤖 开始ML增强选股分析 (基础候选: {len(base_picks)}只)")
        
        for pick in base_picks:
            stock_code = pick['code']
            if stock_code in data:
                try:
                    # 获取ML预测概率
                    hist_data = data[stock_code][data[stock_code]['date'] <= date]
                    ml_prob = self.ml_predictor.predict_probability(hist_data)
                    
                    # 只保留ML预测概率高于阈值的股票
                    if ml_prob >= self.ml_threshold:
                        original_score = pick.get('score', 0)
                        
                        # 计算综合评分：基础评分 * (1 + ML权重 * ML概率)
                        enhanced_score = original_score * (1 + self.ml_weight * ml_prob)
                        
                        # 额外的ML置信度加权
                        confidence_bonus = 0
                        if ml_prob >= 0.8:  # 高置信度
                            confidence_bonus = 0.5
                        elif ml_prob >= 0.7:  # 中高置信度
                            confidence_bonus = 0.3
                        elif ml_prob >= self.ml_threshold:  # 达到阈值
                            confidence_bonus = 0.1
                        
                        final_score = enhanced_score + confidence_bonus
                        
                        pick['score'] = final_score
                        pick['ml_probability'] = ml_prob
                        pick['confidence_level'] = self._get_confidence_level(ml_prob)
                        enhanced_picks.append(pick)
                        
                        ml_predictions.append({
                            'code': stock_code,
                            'prob': ml_prob,
                            'score': final_score
                        })
                        
                except Exception as e:
                    print(f"处理股票 {stock_code} 的ML预测时出错: {e}")
                    # 出错时不添加到最终结果
                    continue
        
        # 按照ML预测概率和综合评分排序
        enhanced_picks.sort(key=lambda x: (x.get('ml_probability', 0), x.get('score', 0)), reverse=True)
        
        # 打印ML预测统计
        if ml_predictions:
            high_confidence = sum(1 for p in ml_predictions if p['prob'] >= 0.8)
            medium_confidence = sum(1 for p in ml_predictions if 0.7 <= p['prob'] < 0.8)
            low_confidence = sum(1 for p in ml_predictions if self.ml_threshold <= p['prob'] < 0.7)
            
            print(f"📊 ML预测统计:")
            print(f"   高置信度 (≥80%): {high_confidence}只")
            print(f"   中置信度 (70-80%): {medium_confidence}只")
            print(f"   低置信度 ({self.ml_threshold*100:.0f}-70%): {low_confidence}只")
            print(f"   最终筛选: {len(enhanced_picks)}只")
        
        return enhanced_picks
    
    def _get_confidence_level(self, prob: float) -> str:
        """获取置信度等级"""
        if prob >= 0.8:
            return "高"
        elif prob >= 0.7:
            return "中高"
        elif prob >= self.ml_threshold:
            return "中"
        else:
            return "低"
