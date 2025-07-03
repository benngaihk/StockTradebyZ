#!/usr/bin/env python3
"""
100日选股策略回测分析
评估各个战法的历史表现，为策略优化提供数据支持
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter

import pandas as pd

from Selector import calculate_price_suggestions
from select_stock import (
    load_data,
    load_config,
    instantiate_selector,
)

class StrategyBacktest:
    def __init__(self, data_dir: Path, config_path: Path):
        self.data_dir = data_dir
        self.config_path = config_path
        self.data = self._load_data()
        self.selector_configs = load_config(config_path)
        self.trading_calendar = self._initialize_trading_calendar()
        
        # 提取CombinedStrategySelector的price_params，用于回测
        self.price_params = {}
        for cfg in self.selector_configs:
            if cfg.get("class") == "CombinedStrategySelector" and "params" in cfg:
                self.price_params = cfg["params"].get("price_params", {})
                break
        
        # 回测结果存储
        self.daily_results = []  # 每日选股结果
        self.performance_stats = defaultdict(list)  # 策略表现统计
        
    def _initialize_trading_calendar(self) -> pd.DatetimeIndex:
        """
        基于所有数据，初始化一个全局的交易日历.
        此方法只在回测开始时调用一次，以保证准确性与性能.
        """
        if not self.data:
            return pd.DatetimeIndex([])
            
        print("🗓️ 正在初始化交易日历...")
        all_dates_series = pd.to_datetime(
            pd.concat([df["date"] for df in self.data.values()])
        ).unique()
        calendar = pd.DatetimeIndex(all_dates_series).sort_values()
        print("✅ 交易日历初始化完成！")
        return calendar

    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有股票数据"""
        codes = [f.stem for f in self.data_dir.glob("*.csv")]
        return load_data(self.data_dir, codes)
    
    def _get_trading_dates(self, end_date: datetime, days: int) -> List[datetime]:
        """获取过去N个交易日的日期列表, 基于预先计算好的交易日历"""
        # 筛选出在结束日期之前的日期
        trading_calendar = self.trading_calendar[self.trading_calendar <= end_date]
        
        if len(trading_calendar) < days:
            print(f"⚠️ 警告: 请求回测 {days} 天, 但可用交易日只有 {len(trading_calendar)} 天。")
            return trading_calendar.tolist()
            
        return trading_calendar[-days:].tolist()
    
    def _simulate_trade(
        self, stock_code: str, select_date: datetime, max_holding_days: int = 10
    ) -> Dict[str, Any]:
        """
        模拟单次交易，根据入场、离场、止损建议.

        Args:
            stock_code: 股票代码
            select_date: 选股日期
            max_holding_days: 最长持股天数

        Returns:
            一个包含交易详情的字典，如收益率、持股天数等.
        """
        # 1. 获取价格建议 (T日信号，T+1交易)
        # 注意：这里的 calculate_price_suggestions 已经包含了T+1逻辑
        price_suggestions = calculate_price_suggestions(
            stock_code, select_date, self.data, self.price_params
        )
        if price_suggestions.get("entry_price", 0) == 0 or price_suggestions.get("actual_date") == "N/A":
            return {"status": "no_suggestion", "return": 0.0}

        # T+1日的日期
        trade_date_str = price_suggestions["actual_date"]
        trade_date = pd.to_datetime(trade_date_str)
        
        exit_price_suggested = price_suggestions["exit_price"]
        stop_loss_suggested = price_suggestions["stop_loss"]

        # 2. 找到T+1日及之后的数据
        df = self.data.get(stock_code)
        if df is None:
            return {"status": "no_data", "return": 0.0}

        # 找到T+1日在数据中的位置
        trade_day_mask = df['date'] == trade_date
        if not trade_day_mask.any():
            return {"status": "no_trade_day_data", "return": 0.0}
        
        trade_day_index = df.index.get_loc(df[trade_day_mask].index[0])
        
        # 获取T+1日当天的数据
        entry_day_data = df.loc[df.index[trade_day_index]]

        # 3. 模拟T+1日入场
        # 我们直接使用T+1日的开盘价作为买入价，因为这是我们能采取的最早行动
        buy_price = entry_day_data['open']
        
        # 如果开盘价为0或无效，则无法交易
        if buy_price <= 0 or pd.isna(buy_price):
            return {"status": "invalid_buy_price", "return": 0.0}

        # 找到T+1日之后的数据进行监控
        holding_days_df = df.iloc[trade_day_index + 1 :]

        # 4. 寻找出场或止损机会
        holding_period = 0
        for _, day in holding_days_df.iterrows():
            holding_period += 1

            # 止盈条件：当日最高价 >= 建议离场价
            if day["high"] >= exit_price_suggested:
                sell_price = exit_price_suggested
                return {
                    "status": "exit_profit",
                    "return": (sell_price - buy_price) / buy_price * 100,
                    "holding_days": holding_period,
                    "entry_price": buy_price,
                    "exit_price": sell_price,
                }

            # 止损条件：当日最低价 <= 建议止损价
            if day["low"] <= stop_loss_suggested:
                sell_price = stop_loss_suggested
                return {
                    "status": "stop_loss",
                    "return": (sell_price - buy_price) / buy_price * 100,
                    "holding_days": holding_period,
                    "entry_price": buy_price,
                    "exit_price": sell_price,
                }

            # 达到最长持股天数
            if holding_period >= max_holding_days:
                sell_price = day["close"]  # 以当日收盘价卖出
                return {
                    "status": "timeout",
                    "return": (sell_price - buy_price) / buy_price * 100,
                    "holding_days": holding_period,
                    "entry_price": buy_price,
                    "exit_price": sell_price,
                }

        # 如果循环结束仍未卖出（数据不足），则以最后一天的收盘价卖出
        if not holding_days_df.empty:
            last_day = holding_days_df.iloc[-1]
            sell_price = last_day["close"]
            return {
                "status": "end_of_data",
                "return": (sell_price - buy_price) / buy_price * 100,
                "holding_days": holding_period + 1,
                "entry_price": buy_price,
                "exit_price": sell_price,
            }

        return {"status": "not_closed", "return": 0.0}
    
    def run_backtest(self, days: int = 100) -> None:
        """运行回测"""
        print(f"🚀 开始{days}日选股策略回测...")
        
        # 获取最新数据日期
        latest_date = max(df['date'].max() for df in self.data.values())
        trading_dates = self._get_trading_dates(latest_date, days)
        
        print(f"📅 回测期间: {trading_dates[0].date()} 到 {trading_dates[-1].date()}")
        
        # 逐日回测
        for i, trade_date in enumerate(trading_dates):
            print(f"\r📊 回测进度: {i+1}/{len(trading_dates)} ({trade_date.date()})", end="", flush=True)
            
            daily_result = {
                'date': trade_date,
                'strategies': {}
            }
            
            # 运行每个策略
            for cfg in self.selector_configs:
                if cfg.get("activate", True) is False:
                    continue
                    
                try:
                    alias, selector = instantiate_selector(cfg)
                    picks = selector.select(trade_date, self.data)
                    
                    # 模拟交易
                    trades = []
                    # 对于CombinedStrategySelector, picks是字典列表
                    if alias == "综合评分策略":
                        stock_codes = [p['code'] for p in picks]
                    else: # 其他selector返回字符串列表
                        stock_codes = picks

                    for stock in stock_codes:
                        trade_result = self._simulate_trade(stock, trade_date)
                        if trade_result["status"] not in [
                            "no_suggestion",
                            "no_data",
                            "invalid_buy_price",
                            "no_trade_day_data",
                            "not_closed"
                        ]:
                            trades.append(trade_result)
                    
                    returns = [t["return"] for t in trades]

                    daily_result['strategies'][alias] = {
                        'picks': stock_codes,
                        'count': len(stock_codes),
                        'trades': trades,
                        'returns': returns,
                        'avg_return': sum(returns) / len(returns) if returns else 0.0,
                        'win_rate': sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0
                    }
                    
                except Exception as e:
                    daily_result['strategies'][cfg.get('alias', 'Unknown')] = {
                        'picks': [],
                        'count': 0,
                        'trades': [],
                        'returns': [],
                        'avg_return': 0.0,
                        'win_rate': 0.0,
                        'error': str(e)
                    }
            
            self.daily_results.append(daily_result)
        
        print("\n✅ 回测完成！")
    
    @staticmethod
    def _create_strategy_stats():
        """Helper to initialize the stats dictionary for a strategy."""
        return {
            "total_picks": 0,
            "total_days": 0,
            "active_days": 0,
            "all_trades": [],
            "daily_counts": [],
            "stock_frequency": Counter(),
        }

    def analyze_results(self) -> None:
        """分析回测结果"""
        print("\n" + "="*80)
        print(f"📈 策略回测结果分析 - {len(self.daily_results)} 天")
        print("="*80)
        
        # 统计各策略表现
        strategy_stats = defaultdict(self._create_strategy_stats)
        
        for day_result in self.daily_results:
            for strategy, result in day_result['strategies'].items():
                stats = strategy_stats[strategy]
                stats['total_days'] += 1
                stats['total_picks'] += result['count']
                stats['daily_counts'].append(result['count'])
                
                if result.get("trades"):
                    stats['active_days'] += 1
                    stats['all_trades'].extend(result['trades'])
                    
                    # 统计股票出现频率
                    for stock in result['picks']:
                        stats['stock_frequency'][stock] += 1
        
        # 输出分析结果
        for strategy, stats in strategy_stats.items():
            print(f"\n🎯 【{strategy}】")
            print(f"   选股活跃度: {stats['active_days']}/{stats['total_days']} 天 ({stats['active_days']/stats['total_days']*100:.1f}%)")
            print(f"   总选股次数: {stats['total_picks']} 次")
            print(f"   平均每日选股: {stats['total_picks']/stats['total_days']:.2f} 只")
            
            all_trades = stats["all_trades"]
            if all_trades:
                all_returns = [t["return"] for t in all_trades]
                avg_return = sum(all_returns) / len(all_returns)
                win_rate = sum(1 for r in all_returns if r > 0) / len(all_returns)
                max_return = max(all_returns)
                min_return = min(all_returns)
                avg_holding_days = sum(t["holding_days"] for t in all_trades) / len(
                    all_trades
                )
                trade_outcomes = Counter(t["status"] for t in all_trades)

                print(f"   成功交易次数: {len(all_trades)} 次")
                print(f"   平均持股天数: {avg_holding_days:.1f} 天")
                print(f"   平均单笔收益: {avg_return:+.2f}%")
                print(f"   胜率: {win_rate*100:.1f}%")
                print(f"   最大收益: {max_return:+.2f}%")
                print(f"   最大亏损: {min_return:+.2f}%")
                outcomes_str = ", ".join(
                    [f"{k}: {v}" for k, v in trade_outcomes.items()]
                )
                print(f"   交易结果分布: {outcomes_str}")

                # 热门股票
                top_stocks = stats['stock_frequency'].most_common(5)
                if top_stocks:
                    print(f"   热门股票: {', '.join([f'{stock}({count}次)' for stock, count in top_stocks])}")
            else:
                print(f"   ❌ 期间内无成功交易记录")
        print("=" * 80)
    
    def generate_optimization_suggestions(self) -> None:
        """生成优化建议"""
        print(f"\n💡 策略优化建议")
        print("="*50)
        
        # 基于回测结果的分析
        strategy_performance = {}
        
        for day_result in self.daily_results:
            for strategy, result in day_result['strategies'].items():
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {
                        'returns': [],
                        'counts': [],
                        'win_rates': []
                    }
                
                if result['count'] > 0:
                    strategy_performance[strategy]['returns'].append(result['avg_return'])
                    strategy_performance[strategy]['counts'].append(result['count'])
                    strategy_performance[strategy]['win_rates'].append(result['win_rate'])
        
        # 生成建议
        recommendations = []
        
        for strategy, perf in strategy_performance.items():
            if not perf['returns']:
                recommendations.append(f"❌ {strategy}: 选股频率过低，建议放宽筛选条件")
                continue
                
            avg_return = sum(perf['returns']) / len(perf['returns'])
            avg_win_rate = sum(perf['win_rates']) / len(perf['win_rates'])
            avg_count = sum(perf['counts']) / len(perf['counts'])
            
            if avg_return < 0:
                recommendations.append(f"⚠️  {strategy}: 平均收益为负({avg_return:.2f}%)，建议调整参数或暂停使用")
            elif avg_return < 1:
                recommendations.append(f"🔧 {strategy}: 收益偏低({avg_return:.2f}%)，建议优化筛选条件")
            else:
                recommendations.append(f"✅ {strategy}: 表现良好(收益{avg_return:.2f}%，胜率{avg_win_rate*100:.1f}%)")
            
            if avg_count > 10:
                recommendations.append(f"📊 {strategy}: 选股数量较多({avg_count:.1f}只/天)，可考虑提高筛选标准")
        
        for rec in recommendations:
            print(f"   {rec}")
        
        print(f"\n📋 通用优化建议:")
        print(f"   1. 关注胜率>60%且平均收益>1%的策略")
        print(f"   2. 对于选股频率过低的策略，可以适当放宽J值或其他技术指标阈值")
        print(f"   3. 对于选股数量过多的策略，可以增加市值、成交量等过滤条件")
        print(f"   4. 建议结合多个策略组合使用，分散风险")

def main():
    parser = argparse.ArgumentParser(description="选股策略回测分析")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("./data"), help="股票数据目录"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./configs.json"),
        help="策略配置文件",
    )
    parser.add_argument(
        "--days", type=int, default=100, help="回测最近N个交易日"
    )
    args = parser.parse_args()

    backtester = StrategyBacktest(data_dir=args.data_dir, config_path=args.config)
    backtester.run_backtest(days=args.days)
    backtester.analyze_results()
    backtester.generate_optimization_suggestions()

if __name__ == "__main__":
    main() 