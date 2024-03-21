
from typing import Dict, TypedDict

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from hypergbm.search_space import search_space_general
import pandas as pd 
from hyperts import make_experiment
from sklearn.metrics import classification_report
from hypergbm import make_experiment
import pickle

class T_SAMPLE_INFO(TypedDict):
    feature: CFeatures
    is_buy: bool
    open_time: CTime


def stragety_feature(last_klu):
    return {
        "open_klu_rate": (last_klu.close - last_klu.open)/last_klu.open,
    }


if __name__ == "__main__":
    """
    本demo主要演示如何记录策略产出的买卖点的特征
    然后将这些特征作为样本，训练一个模型(以XGB为demo)
    用于预测买卖点的准确性

    请注意，demo训练预测都用的是同一份数据，这是不合理的，仅仅是为了演示
    """
    code = "MANTAUSDT"
    begin_time = "2018-01-01"
    end_time = "2024-03-15"
    data_src = DATA_SRC.CSV
    lv_list = [KL_TYPE.K_60M]

    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "bi_strict": False,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": True,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
    })

    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.NONE,
    )

    bsp_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的bsp的特征

    # 跑策略，保存买卖点的特征
    for chan_snapshot in chan.step_load():
        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        cur_lv_chan = chan_snapshot[0]
       
        if last_bsp.klu.idx not in bsp_dict and cur_lv_chan[-2].idx == last_bsp.klu.klc.idx and last_bsp.is_buy:
            # 假如策略是：买卖点分形第三元素出现时交易
            bsp_dict[last_bsp.klu.idx] = {
                "feature": last_bsp.features,
                "is_buy": last_bsp.is_buy,
                "open_time": last_klu.time,
            }
            bsp_dict[last_bsp.klu.idx]['feature'].add_feat(stragety_feature(last_klu))  # 开仓K线特征
            print(last_bsp.klu.time, last_bsp.is_buy)

    # 生成libsvm样本特征
    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
    cur_feature_idx = 0
    data_list =[]
    features_name =[]
    for bsp_klu_idx, feature_info in bsp_dict.items():
        label = int(bsp_klu_idx in bsp_academy)  # 以买卖点识别是否准确为label
        features_v={}
        for feature_name, value in feature_info['feature'].items():
            if feature_name not in features_name:
                features_name.append(feature_name)
            features_v[feature_name]=value
            
        features_v['is_sure'] = label
        features_v['TimeStamp'] = pd.to_datetime(feature_info['open_time'].to_str())
        data_list.append(features_v)
    data = []
    for d in data_list :
        # for feature_name in features_name:
        #     if feature_name not in d.keys():
        #         d[feature_name] = '-'
        data.append(d)

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
    # 输出最佳模型的测试集准确率
    test_accuracy = accuracy_score(y_test, best_model.predict(X_test))
    print("Test Accuracy:", test_accuracy)

    y_pred=best_model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=5))