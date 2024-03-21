import pandas as pd 
from hypergbm.search_space import search_space_general
from hyperts import make_experiment
import pickle
from sklearn.model_selection import train_test_split
from hypergbm.search_space import search_space_general

if __name__ == "__main__":
    # CSV文件路径
    csv_file = 'model_data_1h_all.csv'

    # 从CSV文件中读取数据
    data = pd.read_csv(csv_file)

    my_df = pd.DataFrame(data)

    # 分割特征和目标列
    X = my_df.drop('is_sure', axis=1)  # 删除 'target' 列，其余列为特征
    y = my_df['is_sure']  # 选择 'target' 列作为目标

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=5, shuffle=False)
    # 定义超参数搜索空间
    search_space = search_space_general

    # 定义实验
    experiment =  make_experiment(my_df, 
                                    target='is_sure', 
                                    reward_metric='precision',#指定模型的评价指标
                                    search_space=search_space,
                                    max_trials=50,#最大搜索次数(max_trials) ，建议将最大搜索次数设置为30以上
                                    cv=True, num_folds=5 #cv设置为False时表示禁用交叉验证并使用经典的train_test_split方式进行模型训练；当cv设置为True时表示开启交叉验证，折数可通过参数num_folds设置（默认：3）
                                )

    # 运行实验并进行评估
    best_model = experiment.run()
    # 保存实验对象 以便下次继续训练
    # with open('hypergbm_experiment.pkl','wb') as f:
    #     pickle.dump(experiment, f)
    # 保存模型 以便使用
    with open('hypergbm_model_bp.pkl','wb') as f:
        pickle.dump(best_model, f)