import pandas as pd 
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import pickle

if __name__ == "__main__":
    # CSV文件路径
    csv_file = 'model_data_1h_all.csv'

    # 从CSV文件中读取数据
    data = pd.read_csv(csv_file)

    my_df = pd.DataFrame(data)
  
    X = my_df.drop('is_sure', axis=1)
    y = my_df['is_sure']

    # 使用 SimpleImputer 对缺失值进行填充
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # 定义 MLP 模型
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', 
                          alpha=0.0001, batch_size='auto', learning_rate='constant', 
                          learning_rate_init=0.001, power_t=0.5, max_iter=200, 
                          shuffle=True, random_state=None, tol=0.0001, verbose=False, 
                          warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                          early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
                          beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, 
                          max_fun=15000)

    # 使用交叉验证进行模型评估
    scores = cross_val_score(model, X, y, cv=5, scoring='precision')

    # 输出交叉验证结果
    print("Precision:", scores.mean(), "+/-", scores.std())

    # 训练最终模型
    model.fit(X, y)

    # 保存模型
    with open('mlp_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # 输出分类报告
    y_pred = model.predict(X)
    print(classification_report(y, y_pred, digits=5))
