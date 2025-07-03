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
# 此函数已移动到 Selector.py

def format_stock_with_prices(stock_info: Dict[str, Any]) -> str:
    """
    格式化股票信息，包含价格建议
    """
    stock_code = stock_info['code']
    score = stock_info.get('score')
    prices = stock_info['prices']
    risk_reward_ratio = stock_info.get('risk_reward_ratio', 0)
    
    if prices.get("entry_price", 0) == 0:
        return f"{stock_code} (无价格数据)"
    
    # 计算预期收益和风险比
    potential_return = (prices["exit_price"] - prices["entry_price"]) / prices["entry_price"] * 100
    potential_loss = (prices["entry_price"] - prices["stop_loss"]) / prices["entry_price"] * 100
    
    score_str = f"| 得分: {score:.2f} " if score is not None else ""

    return (f"{stock_code} {score_str}| "
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
        
        # 修复 'volume.1' -> 'amount' 的问题
        if 'volume.1' in df.columns:
            if 'amount' in df.columns:
                df = df.drop(columns=['amount'])
            df = df.rename(columns={'volume.1': 'amount'})
            
        # 防御性检查：确保 amount 列存在
        if 'amount' not in df.columns:
            df['amount'] = 0

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
    parser = argparse.ArgumentParser(description="Run selectors defined in configs.json")
    parser.add_argument("--data-dir", default="./data", help="CSV 行情目录")
    parser.add_argument(
        "--config", type=Path, default=Path("./configs.json"), help="策略配置文件"
    )
    parser.add_argument("--date", type=str, required=False, help="选股日期，格式 YYYY-MM-DD")
    parser.add_argument("--tickers", default="all", help="'all' 或逗号分隔股票代码列表")
    parser.add_argument("--log-file", type=str, default=None, help="指定日志输出文件路径")

    args = parser.parse_args()
    
    # --- 日志配置 ---
    # 移除所有现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 根据是否存在 --log-file 参数来决定输出目标
    if args.log_file:
        # 输出到文件
        handler = logging.FileHandler(args.log_file, mode='a', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(message)s')) # 回测时只记录核心信息
    else:
        # 输出到控制台
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    
    logger.addHandler(handler)
    logger.propagate = False

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
        logger.error("数据目录 %s 为空，请先运行 fetch_kline.py", data_dir)
        sys.exit(1)

    # --- 确定交易日 ---
    if args.date:
        try:
            trade_date = pd.to_datetime(args.date)
        except ValueError:
            logger.error("日期格式不正确，请使用 YYYY-MM-DD 格式")
            sys.exit(1)
    else:
        trade_date = max(df["date"].max() for df in data.values())
        if not args.log_file: # 只有在非回测模式下才打印
            logger.info("未指定 --date，使用最近日期 %s", trade_date.date())

    # --- 加载策略配置 ---
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        # 确保我们获取的是 'selectors' 键下的列表
        selector_cfgs = config_data.get("selectors", [])
        if not selector_cfgs:
            raise ValueError("配置文件中未找到 'selectors' 列表或列表为空")
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error("加载或解析配置文件 %s 时出错: %s", args.config, e)
        sys.exit(1)

    # --- 逐个 Selector 运行 ---
    for cfg in selector_cfgs:
        if not isinstance(cfg, dict) or cfg.get("activate", True) is False:
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
            for pick in picks:
                # Selector 已返回所有需要的信息
                log_line = format_stock_with_prices(pick)
                logger.info("   %s", log_line)
            
        else:
            logger.info("无符合条件股票")


if __name__ == "__main__":
    main()
