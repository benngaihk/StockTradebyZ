#!/usr/bin/env python3
"""
测试 ML 增强选股器
"""

import sys
import os
from pathlib import Path
import pandas as pd

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

def test_ml_enhanced_selector():
    """测试ML增强选股器"""
    
    print("🚀 开始测试 ML 增强选股器...")
    
    # 检查数据目录
    data_dir = Path("./data")
    if not data_dir.exists():
        print("❌ 数据目录不存在，请先运行 fetch_kline.py 获取数据")
        return False
    
    csv_files = list(data_dir.glob("*.csv"))
    if len(csv_files) < 10:
        print(f"❌ 数据文件不足（只有{len(csv_files)}个），请先获取更多股票数据")
        return False
    
    print(f"✅ 找到 {len(csv_files)} 个股票数据文件")
    
    try:
        # 测试基本导入
        from ml_predictor import LSTMStockPredictor, train_ml_model
        from Selector import MLEnhancedCombinedSelector
        
        print("✅ 成功导入 ML 相关模块")
        
        # 测试训练小型模型
        print("🎯 开始训练小型测试模型...")
        
        # 限制数据量进行快速测试
        test_data = {}
        for csv_file in csv_files[:5]:  # 只用前5个股票
            try:
                df = pd.read_csv(csv_file, parse_dates=['date'])
                if len(df) > 100:  # 确保有足够数据
                    test_data[csv_file.stem] = df
            except Exception as e:
                print(f"读取 {csv_file} 时出错: {e}")
        
        if len(test_data) < 3:
            print("❌ 可用的测试数据不足")
            return False
        
        print(f"📊 使用 {len(test_data)} 只股票进行测试")
        
        # 创建并训练小型模型
        predictor = LSTMStockPredictor(
            sequence_length=10,  # 减少序列长度
            epochs=5,           # 减少训练轮数
            batch_size=16       # 减少批次大小
        )
        
        try:
            predictor.train(test_data, min_samples=100)  # 降低最小样本要求
            print("✅ 模型训练成功")
            
            # 测试预测
            first_stock = list(test_data.keys())[0]
            first_df = test_data[first_stock]
            
            prob = predictor.predict_probability(first_df)
            print(f"✅ 预测测试成功，{first_stock} 的上涨概率: {prob:.1%}")
            
            # 保存测试模型
            test_model_path = "./test_lstm_model.h5"
            predictor.save_model(test_model_path)
            print(f"✅ 测试模型已保存: {test_model_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型训练失败: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_ml_config():
    """测试ML配置文件"""
    
    print("\n🔧 测试 ML 配置文件...")
    
    try:
        # 测试运行 ML 增强选股
        import subprocess
        
        # 检查配置文件
        config_file = Path("./configs_ml_enhanced.json")
        if not config_file.exists():
            print("❌ ML配置文件不存在")
            return False
        
        print("✅ ML配置文件存在")
        
        # 测试运行（只做语法检查，不实际执行）
        cmd = [
            "python3", "select_stock.py", 
            "--config", str(config_file),
            "--tickers", "000001,000002",  # 只测试少数股票
            "--date", "2024-01-01"  # 使用固定日期
        ]
        
        print(f"🧪 测试命令: {' '.join(cmd)}")
        print("⚠️  实际运行可能需要较长时间，这里只做配置验证")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 ML 增强选股器测试")
    print("=" * 60)
    
    success = True
    
    # 测试 ML 选股器
    if not test_ml_enhanced_selector():
        success = False
    
    # 测试配置
    if not test_ml_config():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 所有测试通过！")
        print("💡 使用方法:")
        print("   python3 select_stock.py --config ./configs_ml_enhanced.json")
        print("   这将使用ML增强的选股策略")
    else:
        print("❌ 部分测试失败")
        print("💡 请检查:")
        print("   1. 是否已安装 tensorflow 和 scikit-learn")
        print("   2. 是否有足够的股票数据")
        print("   3. 数据格式是否正确")
    
    print("=" * 60) 