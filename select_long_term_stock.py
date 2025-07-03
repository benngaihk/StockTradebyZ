import argparse
import re
from pathlib import Path
from typing import Dict, List, Iterable

import pandas as pd


def load_data(data_dir: Path, codes: Iterable[str]) -> Dict[str, pd.DataFrame]:
    """从CSV文件加载指定股票代码的历史数据"""
    frames: Dict[str, pd.DataFrame] = {}
    for code in codes:
        fp = data_dir / f"{code}.csv"
        if not fp.exists():
            print(f"警告: {fp.name} 的数据文件不存在，已跳过。")
            continue
        df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date")

        if 'volume.1' in df.columns:
            if 'amount' in df.columns:
                df = df.drop(columns=['amount'])
            df = df.rename(columns={'volume.1': 'amount'})

        if 'amount' not in df.columns:
            df['amount'] = 0

        frames[code] = df
    return frames


def parse_log(log_path: Path) -> List[Dict]:
    """解析日志文件，提取最新一次运行的选股结果。"""
    with log_path.open('r', encoding='utf-8') as f:
        content = f.read()

    # 按选股结果的标题分割，并获取最后一块内容
    blocks = content.split("============== 选股结果")
    if len(blocks) < 2:
        return []

    last_block = blocks[-1]
    results = []
    
    # 使用正则表达式从每行中提取所需信息
    line_pattern = re.compile(
        r"^\s*(?P<code>\d{6})\s*\|.*"
        r"\| 入场: ¥(?P<entry>[\d.]+)\s*"
        r"\| 离场: ¥(?P<exit>[\d.]+)\s*"
        r"\|.*\| 预期收益: \+(?P<return>[\d.]+)%.*$"
    )

    for line in last_block.strip().split('\n'):
        match = line_pattern.search(line.strip())
        if match:
            data = match.groupdict()
            results.append({
                'code': data['code'],
                'return_pct': float(data['return']),
                'entry_price': float(data['entry']),
                'exit_price': float(data['exit']),
            })
    return results


def calculate_5d_change(stock_data: pd.DataFrame) -> float:
    """计算最近5天的价格变化百分比。"""
    if len(stock_data) < 6:  # 需要至少6个数据点来计算5天的变化
        return 0.0
    
    # (T日收盘价 / T-5日收盘价) - 1
    price_change = (stock_data['close'].iloc[-1] / stock_data['close'].iloc[-6] - 1) * 100
    return price_change


def calculate_estimated_holding_days(stock_data: pd.DataFrame, target_return_pct: float, lookback_window: int = 20) -> float:
    """
    根据最近的日均涨幅估算达到目标收益所需的时间。
    :param stock_data: 股票历史数据
    :param target_return_pct: 目标收益率 (e.g., 20.0 for 20%)
    :param lookback_window: 用于计算平均涨幅的回看窗口期
    :return: 预估的持有天数
    """
    if len(stock_data) < lookback_window:
        return float('inf')  # 数据不足，返回无穷大以便排序

    # 计算每日收益率
    recent_data = stock_data.iloc[-lookback_window:]
    daily_returns = recent_data['close'].pct_change().dropna()

    # 只考虑上涨日的收益
    positive_returns = daily_returns[daily_returns > 0]

    if positive_returns.empty:
        return float('inf')  # 近期没有上涨日，无法估算

    # 计算平均每日涨幅
    avg_daily_positive_return = positive_returns.mean() * 100  # 转换为百分比

    if avg_daily_positive_return <= 0:
        return float('inf')

    # 预估天数 = 目标收益 / 日均涨幅
    estimated_days = target_return_pct / avg_daily_positive_return
    return estimated_days


def main():
    """主函数，执行分析流程。"""
    parser = argparse.ArgumentParser(description="分析选股结果，筛选高收益和高动量的股票。")
    parser.add_argument("--log-file", type=Path, default=Path("select_results.log"), help="选股结果日志文件路径。")
    parser.add_argument("--data-dir", default="./data", help="存放股票CSV数据的目录。")
    parser.add_argument("--min-return", type=float, default=20.0, help="筛选的最低预期收益率（百分比）。")
    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"错误: 日志文件 {args.log_file} 未找到。请先运行 select_stock.py --log-file {args.log_file}")
        return

    # 1. 解析日志并根据收益率筛选
    all_stocks = parse_log(args.log_file)
    filtered_stocks = [s for s in all_stocks if s['return_pct'] >= args.min_return]

    if not filtered_stocks:
        print(f"未找到预期收益 >= {args.min_return}% 的股票。")
        return

    # 2. 为筛选出的股票加载历史数据
    stock_codes = [s['code'] for s in filtered_stocks]
    all_data = load_data(Path(args.data_dir), stock_codes)

    # 3. 计算动量和预估持有时间
    stocks_with_metrics = []
    for stock in filtered_stocks:
        code = stock['code']
        if code in all_data:
            df = all_data[code]
            stock['momentum_5d_pct'] = calculate_5d_change(df)
            stock['holding_days'] = calculate_estimated_holding_days(df, stock['return_pct'])
            stocks_with_metrics.append(stock)

    # 4. 按预估持有时间从低到高排序
    sorted_stocks = sorted(stocks_with_metrics, key=lambda x: x['holding_days'])

    # 5. 打印最终结果
    print(f"--- 预期收益 > {args.min_return}% 的股票 (按预估持有天数排序) ---")
    header = f"{'代码':<8}{'买入价':<12}{'卖出价':<12}{'预期收益 (%)':<15}{'预估天数':<12}{'5日动量 (%)':<15}"
    print(header)
    print("-" * len(header))
    for stock in sorted_stocks:
        days_str = f"{stock['holding_days']:.1f}" if stock['holding_days'] != float('inf') else "N/A"
        line = (
            f"{stock['code']:<8}"
            f"¥{stock['entry_price']:<10.2f}"
            f"¥{stock['exit_price']:<10.2f}"
            f"{stock['return_pct']:<15.2f}"
            f"{days_str:<12}"
            f"{stock['momentum_5d_pct']:<15.2f}"
        )
        print(line)


if __name__ == "__main__":
    main() 