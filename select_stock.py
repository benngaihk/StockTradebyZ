from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import numpy as np

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # 将日志写入文件
        logging.FileHandler("select_results.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("select")


# ---------- 价格建议计算 ----------

def calculate_price_suggestions(stock_code: str, trade_date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    计算股票的入场价、离场价、止损价建议
    
    Args:
        stock_code: 股票代码
        trade_date: 交易日期
        data: 股票数据字典
    
    Returns:
        包含entry_price, exit_price, stop_loss的字典
    """
    if stock_code not in data:
        return {"entry_price": 0.0, "exit_price": 0.0, "stop_loss": 0.0}
    
    df = data[stock_code].copy()
    df_sorted = df.sort_values('date')
    
    # 找到交易日期对应的数据，如果没有则找最接近的数据
    trade_date_mask = df_sorted['date'].dt.date == trade_date.date()
    if trade_date_mask.any():
        current_idx = df_sorted[trade_date_mask].index[0]
    else:
        # 找到最接近且不晚于指定日期的交易日
        before_dates = df_sorted[df_sorted['date'] <= trade_date]
        if before_dates.empty:
            return {"entry_price": 0.0, "exit_price": 0.0, "stop_loss": 0.0}
        current_idx = before_dates.index[-1]  # 最近的交易日
    
    current_data = df_sorted.loc[current_idx]
    current_close = current_data['close']
    current_low = current_data['low']
    current_high = current_data['high']
    
    # 获取最近20天的数据用于计算支撑阻力位
    end_idx = df_sorted.index.get_loc(current_idx)
    start_idx = max(0, end_idx - 19)
    recent_data = df_sorted.iloc[start_idx:end_idx+1]
    
    if len(recent_data) < 5:
        return {"entry_price": current_close, "exit_price": current_close * 1.05, "stop_loss": current_close * 0.95}
    
    # 计算支撑位和阻力位
    support_level = recent_data['low'].min()
    resistance_level = recent_data['high'].max()
    
    # 计算ATR（平均真实波幅）用于止损
    high_low = recent_data['high'] - recent_data['low']
    if len(recent_data) > 1:
        high_close = abs(recent_data['high'] - recent_data['close'].shift(1))
        low_close = abs(recent_data['low'] - recent_data['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.mean()
    else:
        atr = high_low.iloc[-1]
    
    # 计算价格建议
    # 入场价：当前收盘价附近，略低于收盘价以获得更好入场点
    entry_price = min(current_close * 0.99, (current_close + current_low) / 2)
    
    # 离场价：基于阻力位或10-12%收益目标
    resistance_target = min(resistance_level, current_close * 1.12)
    exit_price = max(current_close * 1.10, resistance_target)
    
    # 止损价：基于支撑位或ATR，取较高者以降低风险
    atr_stop = current_close - (atr * 1.5)
    support_stop = support_level * 0.98
    stop_loss = max(atr_stop, support_stop, current_close * 0.95)  # 最多5%止损
    
    return {
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2), 
        "stop_loss": round(stop_loss, 2),
        "actual_date": current_data['date'].strftime('%Y-%m-%d')
    }


def format_stock_with_prices(stock_code: str, trade_date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> str:
    """
    格式化股票信息，包含价格建议
    """
    prices = calculate_price_suggestions(stock_code, trade_date, data)
    
    if prices["entry_price"] == 0:
        return f"{stock_code} (无价格数据)"
    
    # 计算预期收益和风险比
    potential_return = (prices["exit_price"] - prices["entry_price"]) / prices["entry_price"] * 100
    potential_loss = (prices["entry_price"] - prices["stop_loss"]) / prices["entry_price"] * 100
    risk_reward_ratio = potential_return / potential_loss if potential_loss > 0 else 0
    
    return (f"{stock_code} | "
            f"基于日期: {prices['actual_date']} | "
            f"入场: ¥{prices['entry_price']} | "
            f"离场: ¥{prices['exit_price']} | " 
            f"止损: ¥{prices['stop_loss']} | "
            f"预期收益: {potential_return:+.1f}% | "
            f"风险: {potential_loss:.1f}% | "
            f"收益风险比: {risk_reward_ratio:.1f}")


# ---------- 工具 ----------

def load_data(data_dir: Path, codes: Iterable[str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for code in codes:
        fp = data_dir / f"{code}.csv"
        if not fp.exists():
            logger.warning("%s 不存在，跳过", fp.name)
            continue
        df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date")
        frames[code] = df
    return frames


def load_config(cfg_path: Path) -> List[Dict[str, Any]]:
    if not cfg_path.exists():
        logger.error("配置文件 %s 不存在", cfg_path)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        cfg_raw = json.load(f)

    # 兼容三种结构：单对象、对象数组、或带 selectors 键
    if isinstance(cfg_raw, list):
        cfgs = cfg_raw
    elif isinstance(cfg_raw, dict) and "selectors" in cfg_raw:
        cfgs = cfg_raw["selectors"]
    else:
        cfgs = [cfg_raw]

    if not cfgs:
        logger.error("configs.json 未定义任何 Selector")
        sys.exit(1)

    return cfgs


def instantiate_selector(cfg: Dict[str, Any]):
    """动态加载 Selector 类并实例化"""
    cls_name: str | None = cfg.get("class")
    if not cls_name:
        raise ValueError("缺少 class 字段")

    try:
        module = importlib.import_module("Selector")
        cls = getattr(module, cls_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"无法加载 Selector.{cls_name}: {e}") from e

    params = cfg.get("params", {})
    return cfg.get("alias", cls_name), cls(**params)


# ---------- 主函数 ----------

def main():
    p = argparse.ArgumentParser(description="Run selectors defined in configs.json")
    p.add_argument("--data-dir", default="./data", help="CSV 行情目录")
    p.add_argument("--config", default="./configs.json", help="Selector 配置文件")
    p.add_argument("--date", help="交易日 YYYY-MM-DD；缺省=数据最新日期")
    p.add_argument("--tickers", default="all", help="'all' 或逗号分隔股票代码列表")
    args = p.parse_args()

    # --- 加载行情 ---
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("数据目录 %s 不存在", data_dir)
        sys.exit(1)

    codes = (
        [f.stem for f in data_dir.glob("*.csv")]
        if args.tickers.lower() == "all"
        else [c.strip() for c in args.tickers.split(",") if c.strip()]
    )
    if not codes:
        logger.error("股票池为空！")
        sys.exit(1)

    data = load_data(data_dir, codes)
    if not data:
        logger.error("未能加载任何行情数据")
        sys.exit(1)

    trade_date = (
        pd.to_datetime(args.date)
        if args.date
        else max(df["date"].max() for df in data.values())
    )
    if not args.date:
        logger.info("未指定 --date，使用最近日期 %s", trade_date.date())

    # --- 加载 Selector 配置 ---
    selector_cfgs = load_config(Path(args.config))

    # --- 逐个 Selector 运行 ---
    for cfg in selector_cfgs:
        if cfg.get("activate", True) is False:
            continue
        try:
            alias, selector = instantiate_selector(cfg)
        except Exception as e:
            logger.error("跳过配置 %s：%s", cfg, e)
            continue

        picks = selector.select(trade_date, data)

        # 将结果写入日志，同时输出到控制台
        logger.info("")
        logger.info("============== 选股结果 [%s] ==============", alias)
        logger.info("交易日: %s", trade_date.date())
        logger.info("符合条件股票数: %d", len(picks))
        
        if picks:
            logger.info("📋 详细交易建议:")
            for stock in picks:
                stock_info = format_stock_with_prices(stock, trade_date, data)
                logger.info("   %s", stock_info)
            
        else:
            logger.info("无符合条件股票")


if __name__ == "__main__":
    main()
