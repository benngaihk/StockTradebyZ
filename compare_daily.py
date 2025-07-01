#!/usr/bin/env python3
"""
每日选股结果比较工具
对比今日和昨日的选股结果，识别新机会和失效信号
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from select_stock import load_data, load_config, instantiate_selector

def get_previous_trading_date(current_date: datetime) -> datetime:
    """获取前一个交易日（简单实现：往前推1-3天）"""
    for i in range(1, 4):
        prev_date = current_date - timedelta(days=i)
        # 跳过周末
        if prev_date.weekday() < 5:  # 0-4是周一到周五
            return prev_date
    return current_date - timedelta(days=1)

def run_selector_for_date(data: Dict[str, pd.DataFrame], 
                         selector_configs: List, 
                         target_date: datetime) -> Dict[str, List[str]]:
    """为指定日期运行所有选股器"""
    results = {}
    
    for cfg in selector_configs:
        if cfg.get("activate", True) is False:
            continue
            
        try:
            alias, selector = instantiate_selector(cfg)
            picks = selector.select(target_date, data)
            results[alias] = picks
        except Exception as e:
            print(f"选股器 {cfg} 运行失败: {e}")
            results[cfg.get('alias', 'Unknown')] = []
    
    return results

def compare_results(today_results: Dict[str, List[str]], 
                   yesterday_results: Dict[str, List[str]]) -> None:
    """比较今日和昨日的选股结果"""
    
    print("=" * 60)
    print("📊 每日选股结果对比分析")
    print("=" * 60)
    
    for strategy in today_results:
        today_stocks = set(today_results[strategy])
        yesterday_stocks = set(yesterday_results.get(strategy, []))
        
        print(f"\n🎯 【{strategy}】")
        print(f"   今日选股: {len(today_stocks)}只 {list(today_stocks) if today_stocks else '无'}")
        print(f"   昨日选股: {len(yesterday_stocks)}只 {list(yesterday_stocks) if yesterday_stocks else '无'}")
        
        # 新增股票
        new_stocks = today_stocks - yesterday_stocks
        if new_stocks:
            print(f"   🆕 新增机会: {list(new_stocks)}")
        
        # 失效股票 
        lost_stocks = yesterday_stocks - today_stocks
        if lost_stocks:
            print(f"   ❌ 失效信号: {list(lost_stocks)}")
            
        # 持续股票
        continued_stocks = today_stocks & yesterday_stocks
        if continued_stocks:
            print(f"   🔄 持续信号: {list(continued_stocks)}")

def main():
    parser = argparse.ArgumentParser(description="比较每日选股结果")
    parser.add_argument("--data-dir", default="./data", help="CSV数据目录")
    parser.add_argument("--config", default="./configs.json", help="选股配置文件")
    parser.add_argument("--date", help="对比日期 YYYY-MM-DD，默认为今日")
    
    args = parser.parse_args()
    
    # 数据加载
    data_dir = Path(args.data_dir)
    codes = [f.stem for f in data_dir.glob("*.csv")]
    data = load_data(data_dir, codes)
    
    if not data:
        print("❌ 无法加载股票数据")
        return
    
    # 日期设置
    today = pd.to_datetime(args.date) if args.date else datetime.now()
    yesterday = get_previous_trading_date(today)
    
    print(f"📅 对比日期: {yesterday.date()} → {today.date()}")
    
    # 加载选股配置
    selector_configs = load_config(Path(args.config))
    
    # 运行选股
    print("\n🔄 正在运行选股分析...")
    today_results = run_selector_for_date(data, selector_configs, today)
    yesterday_results = run_selector_for_date(data, selector_configs, yesterday)
    
    # 比较结果
    compare_results(today_results, yesterday_results)
    
    print(f"\n💡 提示:")
    print(f"   - 关注'新增机会'股票，这些是今日新出现的买入信号")
    print(f"   - '失效信号'股票可能已经涨幅过大，不建议追高")
    print(f"   - '持续信号'股票需要判断是否还在合理买入区间")

if __name__ == "__main__":
    main() 