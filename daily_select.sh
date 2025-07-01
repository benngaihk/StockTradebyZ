#!/bin/bash

# 每日股票选股自动化脚本
# 使用方法：bash daily_select.sh

echo "🚀 开始每日选股流程..."
echo "📅 当前时间：$(date)"

# 激活虚拟环境（如果需要）
# source venv/bin/activate

echo ""
echo "📊 第一步：更新股票行情数据..."
python fetch_kline.py \
  --datasource mootdx \
  --frequency 4 \
  --exclude-gem True \
  --min-mktcap 5e9 \
  --max-mktcap 1e20 \
  --start today \
  --end today \
  --out ./data \
  --workers 10

echo ""
echo "🎯 第二步：运行选股策略..."
python select_stock.py \
  --data-dir ./data \
  --config ./configs.json

echo ""
echo "✅ 选股完成！请查看输出结果和 select_results.log 文件"
echo "📈 记得查看新选出的股票，并对比昨日结果" 