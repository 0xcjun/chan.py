import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, SimpleRNN, Attention
import pickle

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

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = SimpleRNN(64, return_sequences=True)(inputs)
    # 将 SimpleRNN 层的输出同时作为 query 和 value 传递给 Attention 层
    attention = Attention()([x, x])
    outputs = Dense(1, activation='sigmoid')(attention)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def save_model(model, filepath):
    model.save(filepath)
    print("Model saved successfully!")

if __name__ == "__main__":
    # CSV文件路径
    csv_file = 'model_data_1h_all.csv'

    # 加载数据
    X, y = load_data(csv_file)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000, shuffle=False)

    # 数据预处理
    X_train_3d, X_test_3d = preprocess_data(X_train, X_test, timesteps=1)

    # 定义和训练模型
    input_shape = X_train_3d.shape[1:]
    model = build_model(input_shape)
    model.fit(X_train_3d, y_train, epochs=50, batch_size=32, validation_data=(X_test_3d, y_test), verbose=1)

    # 评估模型
    test_loss, test_accuracy = model.evaluate(X_test_3d, y_test)
    print("Test Accuracy:", test_accuracy)

    save_model(model,'rnn_attention_model.h5')