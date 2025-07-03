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

def validate_yesterday_performance(
    data: Dict[str, pd.DataFrame], 
    yesterday_results: Dict[str, List[str]],
    validation_date: datetime
) -> None:
    """验证昨日选股在今日的表现"""
    print("\n" + "=" * 60)
    print(f"📈 昨日({(validation_date - timedelta(days=1)).date()})选股在今日({validation_date.date()})表现")
    print("=" * 60)

    for strategy, stocks in yesterday_results.items():
        if not stocks:
            continue

        print(f"\n🎯 策略: 【{strategy}】")
        
        all_metrics = []
        for code in stocks:
            if code not in data or data[code].empty:
                print(f"   - {code}: 无法获取数据")
                continue
            
            day_data = data[code]
            
            try:
                # 修正：确保 '日期' 列是 datetime 类型
                if '日期' not in day_data.columns:
                    # 如果没有 '日期' 列，尝试使用 'trade_date' 或其他可能的列名
                    # 这里为了快速修复，我们先假设日期列就是索引
                    day_data.reset_index(inplace=True)
                    date_col = next((col for col in ['trade_date', 'date', '日期'] if col in day_data.columns), None)
                    if not date_col:
                        raise KeyError("无法在数据中找到日期列")
                else:
                    date_col = '日期'

                day_data[date_col] = pd.to_datetime(day_data[date_col])

                # 获取验证日和前一日的数据
                today_k = day_data[day_data[date_col].dt.strftime('%Y-%m-%d') == validation_date.strftime('%Y-%m-%d')]
                yesterday_k = day_data[day_data[date_col].dt.date < validation_date.date()].iloc[-1]

                if today_k.empty or yesterday_k.empty:
                    print(f"   - {code}: 缺少数据无法验证")
                    continue
                
                today_k = today_k.iloc[0]
                
                # 计算指标
                open_change = (today_k['open'] / yesterday_k['close'] - 1) * 100
                close_change = (today_k['close'] / yesterday_k['close'] - 1) * 100
                high_change = (today_k['high'] / yesterday_k['close'] - 1) * 100
                
                metrics = {
                    "代码": code,
                    "开盘涨幅(%)": f"{open_change:.2f}",
                    "收盘涨幅(%)": f"{close_change:.2f}",
                    "最高涨幅(%)": f"{high_change:.2f}",
                    "今日收盘": f"{today_k['close']:.2f}",
                    "昨日收盘": f"{yesterday_k['close']:.2f}",
                }
                all_metrics.append(metrics)

            except IndexError:
                print(f"   - {code}: 数据不足，无法找到前一日或当日K线")
            except Exception as e:
                print(f"   - {code}: 计算异常: {e}")
        
        if all_metrics:
            # 使用pandas美化输出
            df = pd.DataFrame(all_metrics)
            df.set_index('代码', inplace=True)
            print(df.to_string())

            # --- 计算平均回报 ---
            # 将百分比列转换为数值类型
            for col in ["开盘涨幅(%)", "收盘涨幅(%)", "最高涨幅(%)"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            avg_open_change = df["开盘涨幅(%)"].mean()
            avg_close_change = df["收盘涨幅(%)"].mean()
            avg_high_change = df["最高涨幅(%)"].mean()

            print("\n" + "-" * 25 + " 平均回报 " + "-" * 25)
            print(f"   策略【{strategy}】下共 {len(df)} 只股票的平均表现:")
            print(f"   - 平均开盘涨幅: {avg_open_change:.2f}%")
            print(f"   - 平均收盘涨幅: {avg_close_change:.2f}%")
            print(f"   - 平均最高涨幅: {avg_high_change:.2f}%")
            print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="比较每日选股结果")
    parser.add_argument("--data-dir", default="./data", help="CSV数据目录")
    parser.add_argument("--config", default="./configs.json", help="选股配置文件")
    parser.add_argument("--date", help="对比日期 YYYY-MM-DD，默认为今日")
    parser.add_argument("--validate", action="store_true", help="是否验证昨日选股表现")
    
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
    
    # 新增：验证昨日表现
    if args.validate:
        validate_yesterday_performance(data, yesterday_results, today)

    print(f"\n💡 提示:")
    print(f"   - 关注'新增机会'股票，这些是今日新出现的买入信号")
    print(f"   - '失效信号'股票可能已经涨幅过大，不建议追高")
    print(f"   - '持续信号'股票需要判断是否还在合理买入区间")

if __name__ == "__main__":
    main() 