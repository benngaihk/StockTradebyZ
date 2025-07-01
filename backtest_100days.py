#!/usr/bin/env python3
"""
100日选股策略回测分析
评估各个战法的历史表现，为策略优化提供数据支持
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

import pandas as pd

from select_stock import load_data, load_config, instantiate_selector

class StrategyBacktest:
    def __init__(self, data_dir: Path, config_path: Path):
        self.data_dir = data_dir
        self.config_path = config_path
        self.data = self._load_data()
        self.selector_configs = load_config(config_path)
        
        # 回测结果存储
        self.daily_results = []  # 每日选股结果
        self.performance_stats = defaultdict(list)  # 策略表现统计
        
    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有股票数据"""
        codes = [f.stem for f in self.data_dir.glob("*.csv")]
        return load_data(self.data_dir, codes)
    
    def _get_trading_dates(self, end_date: datetime, days: int) -> List[datetime]:
        """获取过去N个交易日的日期列表"""
        dates = []
        current = end_date
        
        while len(dates) < days:
            # 检查是否有数据（简单判断：是否为工作日）
            if current.weekday() < 5:  # 周一到周五
                dates.append(current)
            current -= timedelta(days=1)
            
        return list(reversed(dates))
    
    def _calculate_next_day_return(self, stock_code: str, select_date: datetime) -> float:
        """计算选股后第二天的收益率"""
        if stock_code not in self.data:
            return 0.0
            
        df = self.data[stock_code]
        df_sorted = df.sort_values('date')
        
        # 找到选股日期的位置
        select_date_mask = df_sorted['date'].dt.date == select_date.date()
        if not select_date_mask.any():
            return 0.0
            
        select_idx = df_sorted[select_date_mask].index[0]
        
        # 找下一个交易日
        next_idx = None
        for i in range(select_idx + 1, len(df_sorted)):
            next_idx = df_sorted.index[i]
            break
            
        if next_idx is None:
            return 0.0
            
        select_close = df_sorted.loc[select_idx, 'close']
        next_close = df_sorted.loc[next_idx, 'close']
        
        return (next_close - select_close) / select_close * 100
    
    def run_backtest(self, days: int = 100) -> None:
        """运行回测"""
        print(f"🚀 开始{days}日选股策略回测...")
        
        # 获取最新数据日期
        latest_date = max(df['date'].max() for df in self.data.values())
        trading_dates = self._get_trading_dates(latest_date, days)
        
        print(f"📅 回测期间: {trading_dates[0].date()} 到 {trading_dates[-1].date()}")
        
        # 逐日回测
        for i, trade_date in enumerate(trading_dates):
            print(f"\r📊 回测进度: {i+1}/{days} ({trade_date.date()})", end="", flush=True)
            
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
                    
                    # 计算次日收益
                    returns = []
                    for stock in picks:
                        ret = self._calculate_next_day_return(stock, trade_date)
                        returns.append(ret)
                    
                    daily_result['strategies'][alias] = {
                        'picks': picks,
                        'count': len(picks),
                        'returns': returns,
                        'avg_return': sum(returns) / len(returns) if returns else 0.0,
                        'win_rate': sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0
                    }
                    
                except Exception as e:
                    daily_result['strategies'][cfg.get('alias', 'Unknown')] = {
                        'picks': [],
                        'count': 0,
                        'returns': [],
                        'avg_return': 0.0,
                        'win_rate': 0.0,
                        'error': str(e)
                    }
            
            self.daily_results.append(daily_result)
        
        print("\n✅ 回测完成！")
    
    def analyze_results(self) -> None:
        """分析回测结果"""
        print("\n" + "="*80)
        print("📈 策略回测结果分析")
        print("="*80)
        
        # 统计各策略表现
        strategy_stats = defaultdict(lambda: {
            'total_picks': 0,
            'total_days': 0,
            'active_days': 0,
            'all_returns': [],
            'daily_counts': [],
            'stock_frequency': Counter()
        })
        
        for day_result in self.daily_results:
            for strategy, result in day_result['strategies'].items():
                stats = strategy_stats[strategy]
                stats['total_days'] += 1
                stats['total_picks'] += result['count']
                stats['daily_counts'].append(result['count'])
                
                if result['count'] > 0:
                    stats['active_days'] += 1
                    stats['all_returns'].extend(result['returns'])
                    
                    # 统计股票出现频率
                    for stock in result['picks']:
                        stats['stock_frequency'][stock] += 1
        
        # 输出分析结果
        for strategy, stats in strategy_stats.items():
            print(f"\n🎯 【{strategy}】")
            print(f"   选股活跃度: {stats['active_days']}/{stats['total_days']} 天 ({stats['active_days']/stats['total_days']*100:.1f}%)")
            print(f"   总选股次数: {stats['total_picks']} 只")
            print(f"   平均每日选股: {stats['total_picks']/stats['total_days']:.2f} 只")
            
            if stats['all_returns']:
                avg_return = sum(stats['all_returns']) / len(stats['all_returns'])
                win_rate = sum(1 for r in stats['all_returns'] if r > 0) / len(stats['all_returns'])
                max_return = max(stats['all_returns'])
                min_return = min(stats['all_returns'])
                
                print(f"   平均次日收益: {avg_return:+.2f}%")
                print(f"   胜率: {win_rate*100:.1f}%")
                print(f"   最大收益: {max_return:+.2f}%")
                print(f"   最大亏损: {min_return:+.2f}%")
                
                # 热门股票
                top_stocks = stats['stock_frequency'].most_common(5)
                if top_stocks:
                    print(f"   热门股票: {', '.join([f'{stock}({count}次)' for stock, count in top_stocks])}")
            else:
                print(f"   ❌ 期间内无选股记录")
    
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
    parser = argparse.ArgumentParser(description="100日选股策略回测")
    parser.add_argument("--data-dir", default="./data", help="CSV数据目录")
    parser.add_argument("--config", default="./configs.json", help="选股配置文件")
    parser.add_argument("--days", type=int, default=100, help="回测天数")
    
    args = parser.parse_args()
    
    # 创建回测对象
    backtest = StrategyBacktest(Path(args.data_dir), Path(args.config))
    
    # 运行回测
    backtest.run_backtest(args.days)
    
    # 分析结果
    backtest.analyze_results()
    
    # 生成优化建议
    backtest.generate_optimization_suggestions()
    
    print(f"\n🎯 回测完成！建议根据以上分析结果调整策略参数。")

if __name__ == "__main__":
    main() 