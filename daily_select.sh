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
echo "🎯 第二步：运行原版选股策略..."
python select_stock.py \
  --data-dir ./data \
  --config ./configs.json

echo ""
echo "🚀 第三步：运行优化版选股策略..."
python select_stock.py \
  --data-dir ./data \
  --config ./configs_optimized.json

echo ""
echo "📈 第四步：对比分析选股结果..."
python compare_daily.py

echo ""
echo "✅ 选股完成！"
echo "📋 总结："
echo "   - 查看 select_results.log 了解详细日志"
echo "   - 关注新选中的股票，避免追高已涨股票"
echo "   - 建议结合多个策略分散风险" 