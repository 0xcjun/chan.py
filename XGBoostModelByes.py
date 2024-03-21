from typing import Dict, TypedDict
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from skopt import BayesSearchCV
import xgboost as xgb
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

    # Define the parameter search space
    param_space = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'max_depth': (1, 20),
        'min_child_weight': (1, 10),
        'subsample': (0.5, 1.0, 'uniform'),
        'colsample_bytree': (0.5, 1.0, 'uniform'),
        'gamma': (0, 5.0, 'uniform')
    }

    # Initialize XGBoost model
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', verbosity=1, n_estimators=100)

    # Perform Bayesian optimization
    opt = BayesSearchCV(model, param_space, n_iter=30, cv=3, scoring='accuracy', n_jobs=-1)
    opt.fit(X_train, y_train)

    # Save the trained model
    with open('xgboost_model.pkl', 'wb') as f:
        pickle.dump(opt.best_estimator_, f)

    # Evaluate model performance
    test_accuracy = accuracy_score(y_test, opt.predict(X_test))
    print("Test Accuracy:", test_accuracy)

    y_pred = opt.predict(X_test)
    print(classification_report(y_test, y_pred, digits=5))
