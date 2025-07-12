# ML 增强选股器使用说明

## 概述

ML增强选股器结合了传统技术指标和机器学习预测，使用LSTM神经网络预测股票价格趋势，提高选股准确率。

## 功能特点

### 🧠 机器学习特性
- **LSTM神经网络**: 使用长短期记忆网络预测股票涨跌概率
- **多特征融合**: 结合价格、成交量、技术指标等16个特征
- **动态权重调整**: 根据ML预测概率调整传统选股器的评分

### 📊 技术指标
- **基础特征**: 开盘价、收盘价、最高价、最低价、成交量、成交额
- **收益率指标**: 1日、5日、10日收益率
- **均线指标**: 5日、10日、20日移动平均线
- **技术指标**: RSI、布林带、成交量比率、价格位置等

### 🎯 选股策略
- **传统策略**: BBI+KDJ、MACD金叉、RSI超卖、布林带、黄金交叉、OBV等
- **ML增强**: 对传统策略结果进行机器学习概率评估
- **智能过滤**: 根据ML预测概率过滤低质量信号

## 安装依赖

```bash
# 安装必要的Python包
pip3 install tensorflow scikit-learn pandas numpy
```

## 使用方法

### 1. 基本使用

```bash
# 使用ML增强选股器
python3 select_stock.py --config ./configs_ml_enhanced.json
```

### 2. 指定参数

```bash
# 指定日期和股票池
python3 select_stock.py \
  --config ./configs_ml_enhanced.json \
  --date 2024-01-15 \
  --tickers 000001,000002,600036
```

### 3. 训练自定义模型

```python
from ml_predictor import train_ml_model

# 训练模型
predictor = train_ml_model(
    data_dir="./data",
    model_path="./my_lstm_model.h5"
)
```

## 配置文件说明

### configs_ml_enhanced.json

```json
{
  "selectors": [
    {
      "class": "MLEnhancedCombinedSelector",
      "alias": "ML增强综合策略",
      "activate": true,
      "params": {
        "base_selector_config": {
          "class": "CombinedStrategySelector",
          "params": {
            "score_threshold": 1.0,
            "weights": {
              "bbikdj": 1.0,
              "macd": 0.8,
              "rsi": 0.6,
              "bollinger": 0.7,
              "goldencross": 1.2,
              "obv": 0.8
            }
          }
        },
        "ml_predictor_config": {
          "model_path": "./lstm_model.h5",
          "sequence_length": 20,
          "epochs": 30,
          "batch_size": 32
        },
        "ml_threshold": 0.55,
        "ml_weight": 0.4
      }
    }
  ]
}
```

### 主要参数说明

- **ml_threshold**: ML预测阈值（0.55表示上涨概率>55%才增强评分）
- **ml_weight**: ML权重系数（0.4表示ML预测可以增加40%的评分权重）
- **sequence_length**: LSTM输入序列长度（20个交易日）
- **epochs**: 训练轮数（30轮）
- **batch_size**: 批次大小（32）

## 输出格式

ML增强选股器的输出包含以下信息：

```
000001 | 得分: 2.45 | ML预测: 67.3% | 基于日期: 2024-01-15 | 入场: ¥14.50 | 离场: ¥16.24 | 止损: ¥13.78 | 预期收益: +12.0% | 风险: 5.0% | 收益风险比: 2.4
```

### 字段说明
- **得分**: 综合评分（传统策略+ML增强）
- **ML预测**: 机器学习预测的上涨概率
- **入场价**: 建议买入价格
- **离场价**: 建议卖出价格
- **止损价**: 建议止损价格
- **预期收益**: 预期收益率
- **风险**: 预期风险（止损幅度）
- **收益风险比**: 风险收益比

## 模型训练

### 自动训练
首次运行时，如果模型文件不存在，系统会自动训练新模型：

```bash
# 首次运行会自动训练模型
python3 select_stock.py --config ./configs_ml_enhanced.json
```

### 手动训练
也可以手动训练模型：

```bash
# 运行训练脚本
python3 -c "from ml_predictor import train_ml_model; train_ml_model()"
```

### 测试模型
运行测试脚本验证模型功能：

```bash
python3 test_ml_enhanced.py
```

## 性能优化建议

### 1. 数据质量
- 确保有足够的历史数据（建议至少1年以上）
- 定期更新股票数据
- 检查数据完整性

### 2. 模型参数调优
- 根据市场情况调整`ml_threshold`
- 根据风险偏好调整`ml_weight`
- 定期重新训练模型

### 3. 硬件建议
- 使用GPU加速训练（如果可用）
- 增加内存以处理更多股票数据
- 使用SSD提高数据读取速度

## 风险提示

⚠️ **重要提醒**：
1. 机器学习模型基于历史数据训练，不保证未来表现
2. 股票投资有风险，请根据个人风险承受能力操作
3. 建议结合其他分析方法和风险管理策略
4. 定期评估和调整模型参数

## 故障排除

### 常见问题

1. **模型训练失败**
   - 检查数据格式是否正确
   - 确保有足够的历史数据
   - 检查依赖包是否正确安装

2. **预测结果异常**
   - 检查输入数据质量
   - 重新训练模型
   - 调整模型参数

3. **性能问题**
   - 减少训练数据量
   - 降低模型复杂度
   - 使用更快的硬件

### 联系支持
如有问题，请检查日志文件或运行测试脚本诊断问题。

---

*最后更新：2024年* 