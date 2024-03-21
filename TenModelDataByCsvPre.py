import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

def load_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.drop('is_sure', axis=1)
    y = data['is_sure']
    return X, y

def preprocess_data(X_train, X_test, timesteps=1):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # 将数据调整为三维格式
    X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], timesteps, X_train_scaled.shape[1]))
    X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], timesteps, X_test_scaled.shape[1]))
    return X_train_3d, X_test_3d

if __name__ == "__main__":
    # CSV文件路径
    csv_file = 'model_data_1h_all.csv'

    # 加载数据
    X, y = load_data(csv_file)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000, shuffle=False)

    # 数据预处理
    X_train_3d, X_test_3d = preprocess_data(X_train, X_test, timesteps=1)

    # 加载模型
    model = load_model('rnn_attention_model.h5')

    # 对测试集进行预测
    y_pred_prob = model.predict(X_test_3d)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    # 输出测试结果
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
