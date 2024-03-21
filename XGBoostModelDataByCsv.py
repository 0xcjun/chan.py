
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pickle


if __name__ == "__main__":
    # CSV文件路径
    csv_file = 'model_data_1h_all.csv'

    # 从CSV文件中读取数据
    data = pd.read_csv(csv_file)

    # 打印数据的前几行
    print(data.head())

    my_df = pd.DataFrame(data)
  
    X = my_df.drop('is_sure', axis=1)
    y = my_df['is_sure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, shuffle=False)

    # Initialize XGBoost model with GPU support
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'verbosity': 1,
        'tree_method': 'gpu_hist'  # Use GPU acceleration
    }
    model = xgb.XGBClassifier(**params)

    # Train the model
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=100)

    # Save the trained model
    with open('xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Evaluate model performance
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print("Test Accuracy:", test_accuracy)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=5))
