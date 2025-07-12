"""
机器学习股票预测器
使用 LSTM 神经网络预测股票价格趋势
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class LSTMStockPredictor:
    """
    LSTM 股票预测器
    预测股票未来涨跌概率
    """
    
    def __init__(self, sequence_length=30, epochs=50, batch_size=32):
        """
        初始化预测器
        
        Args:
            sequence_length: 输入序列长度（天数）
            epochs: 训练轮数
            batch_size: 批次大小
        """
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        
    def _prepare_features(self, df):
        """
        准备特征数据
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            处理后的特征数据
        """
        data = df.copy()
        
        # 处理缺失的 amount 列
        if 'amount' not in data.columns or data['amount'].isna().all():
            # 如果没有成交额数据，用成交量 * 平均价格估算
            data['amount'] = data['volume'] * (data['high'] + data['low']) / 2
        
        # 填充缺失值
        data['amount'] = data['amount'].fillna(data['volume'] * data['close'])
        
        # 基础特征
        data['return_1d'] = data['close'].pct_change()
        data['return_5d'] = data['close'].pct_change(5)
        data['return_10d'] = data['close'].pct_change(10)
        
        # 技术指标 - 移动平均线
        data['ma_5'] = data['close'].rolling(window=5).mean()
        data['ma_10'] = data['close'].rolling(window=10).mean()
        data['ma_20'] = data['close'].rolling(window=20).mean()
        data['ma_60'] = data['close'].rolling(window=60).mean()
        data['ma_ratio_5_20'] = data['ma_5'] / data['ma_20']
        data['ma_ratio_10_20'] = data['ma_10'] / data['ma_20']
        data['ma_ratio_20_60'] = data['ma_20'] / data['ma_60']
        
        # 相对强弱指数 (RSI) - 多周期
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        data['rsi_6'] = calculate_rsi(data['close'], 6)
        data['rsi_14'] = calculate_rsi(data['close'], 14)
        data['rsi_21'] = calculate_rsi(data['close'], 21)
        
        # 布林带
        bb_window = 20
        data['bb_middle'] = data['close'].rolling(window=bb_window).mean()
        bb_std = data['close'].rolling(window=bb_window).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        data['macd_dif'] = ema_12 - ema_26
        data['macd_dea'] = data['macd_dif'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd_dif'] - data['macd_dea']
        data['macd_signal'] = (data['macd_dif'] > data['macd_dea']).astype(int)
        
        # KDJ指标
        low_min = data['low'].rolling(window=9).min()
        high_max = data['high'].rolling(window=9).max()
        rsv = (data['close'] - low_min) / (high_max - low_min) * 100
        data['kdj_k'] = rsv.ewm(alpha=1/3).mean()
        data['kdj_d'] = data['kdj_k'].ewm(alpha=1/3).mean()
        data['kdj_j'] = 3 * data['kdj_k'] - 2 * data['kdj_d']
        
        # 成交量指标
        data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
        data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio_5'] = data['volume'] / data['volume_ma_5']
        data['volume_ratio_20'] = data['volume'] / data['volume_ma_20']
        
        # 价格位置和动量
        data['price_position_20'] = (data['close'] - data['low'].rolling(window=20).min()) / \
                                   (data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min())
        data['price_position_60'] = (data['close'] - data['low'].rolling(window=60).min()) / \
                                   (data['high'].rolling(window=60).max() - data['low'].rolling(window=60).min())
        
        # 波动率
        data['volatility_5'] = data['return_1d'].rolling(window=5).std()
        data['volatility_20'] = data['return_1d'].rolling(window=20).std()
        
        # 高低价比例
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        # 成交额特征
        data['amount_ma_20'] = data['amount'].rolling(window=20).mean()
        data['amount_ratio'] = data['amount'] / data['amount_ma_20']
        
        # 选择特征列 (扩展到35个特征)
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount',
                       'return_1d', 'return_5d', 'return_10d',
                       'ma_5', 'ma_10', 'ma_20', 'ma_60', 'ma_ratio_5_20', 'ma_ratio_10_20', 'ma_ratio_20_60',
                       'rsi_6', 'rsi_14', 'rsi_21', 'bb_position', 'bb_width',
                       'macd_dif', 'macd_dea', 'macd_histogram', 'macd_signal',
                       'kdj_k', 'kdj_d', 'kdj_j',
                       'volume_ratio_5', 'volume_ratio_20',
                       'price_position_20', 'price_position_60',
                       'volatility_5', 'volatility_20',
                       'high_low_ratio', 'close_open_ratio', 'amount_ratio']
        
        # 删除缺失值和无穷值
        data = data[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
        
        return data
    
    def _create_sequences(self, data, target):
        """
        创建时间序列数据
        
        Args:
            data: 特征数据 (numpy array)
            target: 目标变量 (pandas Series)
            
        Returns:
            X, y: 序列数据和标签
        """
        X, y = [], []
        
        # 重置target的索引，使其从0开始
        target_values = target.values
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target_values[i])
            
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape):
        """
        构建增强版 LSTM 模型
        
        Args:
            input_shape: 输入形状
            
        Returns:
            编译后的模型
        """
        model = keras.Sequential([
            # 第一层LSTM，更多神经元
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # 第二层LSTM
            layers.LSTM(64, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # 第三层LSTM
            layers.LSTM(32),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # 全连接层
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(1, activation='sigmoid')  # 二分类：涨/跌
        ])
        
        # 使用更好的优化器和学习率调度
        optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train(self, stock_data_dict, min_samples=1000):
        """
        训练模型
        
        Args:
            stock_data_dict: 股票数据字典 {code: DataFrame}
            min_samples: 最小样本数量
        """
        print("🚀 开始训练 LSTM 股票预测模型...")
        
        all_X, all_y = [], []
        processed_count = 0
        
        # 处理每只股票的数据
        for code, df in stock_data_dict.items():
            if len(df) < self.sequence_length + 50:  # 确保有足够数据
                print(f"跳过 {code}: 数据不足 ({len(df)} 行)")
                continue
                
            try:
                # 准备特征
                features = self._prepare_features(df)
                if len(features) < self.sequence_length + 20:
                    print(f"跳过 {code}: 处理后数据不足 ({len(features)} 行)")
                    continue
                
                # 创建目标变量（未来1日涨跌）
                target = (features['close'].shift(-1) > features['close']).astype(int)
                target = target[:-1]  # 去掉最后一个NaN
                features = features[:-1]  # 对应去掉最后一行
                
                # 标准化特征
                features_scaled = self.scaler.fit_transform(features)
                
                # 创建序列
                X, y = self._create_sequences(features_scaled, target)
                
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
                    processed_count += 1
                    print(f"✅ 处理完成 {code}: {len(X)} 个样本")
                else:
                    print(f"跳过 {code}: 无法创建序列")
                    
            except Exception as e:
                print(f"处理股票 {code} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"📊 成功处理 {processed_count} 只股票")
        
        if not all_X:
            raise ValueError("没有足够的数据来训练模型")
        
        # 合并所有数据
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        
        print(f"📊 训练数据形状: X={X.shape}, y={y.shape}")
        print(f"📊 正样本比例: {y.mean():.2%}")
        
        if len(X) < min_samples:
            print(f"⚠️  样本数量 {len(X)} 小于最小要求 {min_samples}，但继续训练")
            # 不抛出异常，继续训练
        
        # 分割训练和测试数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 构建模型
        self.model = self._build_model((X.shape[1], X.shape[2]))
        
        # 训练模型
        print("🎯 开始训练...")
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        # 评估模型
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ 模型训练完成！")
        print(f"📈 测试集准确率: {accuracy:.2%}")
        print(f"📊 分类报告:")
        print(classification_report(y_test, y_pred))
        
        return history
    
    def predict_probability(self, df):
        """
        预测股票上涨概率
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            上涨概率 (0-1)
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train() 方法")
        
        try:
            # 准备特征
            features = self._prepare_features(df)
            if len(features) < self.sequence_length:
                return 0.5  # 数据不足，返回中性概率
            
            # 标准化
            features_scaled = self.scaler.transform(features)
            
            # 取最后一个序列
            X = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # 预测
            prob = self.model.predict(X, verbose=0)[0][0]
            
            return float(prob)
            
        except Exception as e:
            print(f"预测时出错: {e}")
            return 0.5  # 出错时返回中性概率
    
    def save_model(self, filepath):
        """保存模型"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"✅ 模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        self.model = keras.models.load_model(filepath)
        print(f"✅ 模型已从 {filepath} 加载")


class MLEnhancedSelector:
    """
    机器学习增强选股器
    结合传统技术指标和ML预测
    """
    
    def __init__(self, base_selector, ml_predictor, ml_threshold=0.6, ml_weight=0.3):
        """
        初始化ML增强选股器
        
        Args:
            base_selector: 基础选股器
            ml_predictor: ML预测器
            ml_threshold: ML预测阈值
            ml_weight: ML权重
        """
        self.base_selector = base_selector
        self.ml_predictor = ml_predictor
        self.ml_threshold = ml_threshold
        self.ml_weight = ml_weight
    
    def select(self, date, data):
        """
        选股方法
        
        Args:
            date: 选股日期
            data: 股票数据字典
            
        Returns:
            选中的股票列表
        """
        # 获取基础选股结果
        base_picks = self.base_selector.select(date, data)
        
        # 如果基础选股器返回字典列表（如CombinedStrategySelector）
        if base_picks and isinstance(base_picks[0], dict):
            enhanced_picks = []
            
            for pick in base_picks:
                stock_code = pick['code']
                if stock_code in data:
                    # 获取ML预测概率
                    hist_data = data[stock_code][data[stock_code]['date'] <= date]
                    ml_prob = self.ml_predictor.predict_probability(hist_data)
                    
                    # 结合ML预测调整评分
                    if ml_prob >= self.ml_threshold:
                        # ML预测为正，增加权重
                        original_score = pick.get('score', 0)
                        enhanced_score = original_score * (1 + self.ml_weight * ml_prob)
                        pick['score'] = enhanced_score
                        pick['ml_probability'] = ml_prob
                        enhanced_picks.append(pick)
                    elif ml_prob >= 0.4:  # 中性区间，保持原评分
                        pick['ml_probability'] = ml_prob
                        enhanced_picks.append(pick)
                    # ml_prob < 0.4 的股票被过滤掉
            
            # 重新排序
            enhanced_picks.sort(key=lambda x: x.get('score', 0), reverse=True)
            return enhanced_picks
        
        else:
            # 基础选股器返回字符串列表
            enhanced_picks = []
            
            for stock_code in base_picks:
                if stock_code in data:
                    hist_data = data[stock_code][data[stock_code]['date'] <= date]
                    ml_prob = self.ml_predictor.predict_probability(hist_data)
                    
                    if ml_prob >= self.ml_threshold:
                        enhanced_picks.append(stock_code)
            
            return enhanced_picks


def train_ml_model(data_dir="./data", model_path="./lstm_model.h5"):
    """
    训练ML模型的便捷函数
    
    Args:
        data_dir: 数据目录
        model_path: 模型保存路径
    """
    from pathlib import Path
    
    # 加载数据
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录 {data_dir} 不存在")
    
    # 读取所有股票数据
    stock_data = {}
    csv_files = list(data_path.glob("*.csv"))
    
    print(f"📁 找到 {len(csv_files)} 个CSV文件")
    
    for csv_file in csv_files[:50]:  # 限制数量以加快训练
        try:
            df = pd.read_csv(csv_file, parse_dates=['date'])
            stock_code = csv_file.stem
            stock_data[stock_code] = df
        except Exception as e:
            print(f"读取 {csv_file} 时出错: {e}")
    
    if not stock_data:
        raise ValueError("没有成功读取任何股票数据")
    
    print(f"📊 成功加载 {len(stock_data)} 只股票的数据")
    
    # 创建并训练模型
    predictor = LSTMStockPredictor(sequence_length=20, epochs=30)
    predictor.train(stock_data)
    
    # 保存模型
    predictor.save_model(model_path)
    
    return predictor


if __name__ == "__main__":
    # 训练模型示例
    try:
        predictor = train_ml_model()
        print("🎉 模型训练完成！")
    except Exception as e:
        print(f"❌ 训练失败: {e}") 