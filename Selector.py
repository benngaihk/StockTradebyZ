from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


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


class CombinedStrategySelector:
    """
    综合评分策略
    - BBIKDJSelector: +1分
    - MACDGoldenCrossSelector: +1分
    - RSIOversoldSelector: +1分
    - 总分 >= score_threshold
    """
    def __init__(self, score_threshold: int = 2, **kwargs):
        self.score_threshold = score_threshold
        self.s1 = BBIKDJSelector(**kwargs.get('bbikdj', {}))
        self.s2 = MACDGoldenCrossSelector(**kwargs.get('macd', {}))
        self.s3 = RSIOversoldSelector(**kwargs.get('rsi', {}))

    def select(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue

            score = 0
            # 为防止子选择器修改DataFrame，传入副本
            if self.s1._passes_filters(hist.copy()):
                score += 1
            if self.s2._passes_filters(hist.copy()):
                score += 1
            if self.s3._passes_filters(hist.copy()):
                score += 1
            
            if score >= self.score_threshold:
                picks.append(code)
        
        return picks
