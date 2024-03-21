from typing import Dict, TypedDict
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
import pickle

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime

class T_SAMPLE_INFO(TypedDict):
    feature: CFeatures
    is_buy: bool
    open_time: CTime

def strategy_feature(last_klu):
    return {
        "open_klu_rate": (last_klu.close - last_klu.open)/last_klu.open,
    }

if __name__ == "__main__":
    code = "BTCUSDT5m"
    begin_time = "2018-01-01"
    end_time = "2024-03-15"
    data_src = DATA_SRC.CSV
    lv_list = [KL_TYPE.K_60M]

    config = CChanConfig({
        "trigger_step": True,
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

    bsp_dict: Dict[int, T_SAMPLE_INFO] = {}

    for chan_snapshot in chan.step_load():
        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        cur_lv_chan = chan_snapshot[0]
       
        if last_bsp.klu.idx not in bsp_dict and cur_lv_chan[-2].idx == last_bsp.klu.klc.idx and last_bsp.is_buy:
            bsp_dict[last_bsp.klu.idx] = {
                "feature": last_bsp.features,
                "is_buy": last_bsp.is_buy,
                "open_time": last_klu.time,
            }
            bsp_dict[last_bsp.klu.idx]['feature'].add_feat(strategy_feature(last_klu))

    data_list = []
    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
    for bsp_klu_idx, feature_info in bsp_dict.items():
        label = int(bsp_klu_idx in bsp_academy)
        features_v = {}
        for feature_name, value in feature_info['feature'].items():
            features_v[feature_name] = value
        features_v['is_sure'] = label
        # features_v['TimeStamp'] = pd.to_datetime(feature_info['open_time'].to_str())
        data_list.append(features_v)

    my_df = pd.DataFrame(data_list)
  
    X = my_df.drop('is_sure', axis=1)
    y = my_df['is_sure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=200, shuffle=False)

    #OrderedDict([('colsample_bylevel', 0.5), ('depth', 4), ('iterations', 87), ('l2_leaf_reg', 4), ('learning_rate', 0.06642787626013845), ('random_strength', 37)])
    # 使用CatBoost模型
    # model = CatBoostClassifier(iterations=87, 
    #                            colsample_bylevel=0.5,
    #                            learning_rate=0.06642787626013845, 
    #                            depth=4, l2_leaf_reg=4,
    #                            random_strength = 37,)
 # 使用CatBoost模型
    model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, loss_function='Logloss', verbose=100)

    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=100)

    with open('catboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print("Test Accuracy:", test_accuracy)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=5))