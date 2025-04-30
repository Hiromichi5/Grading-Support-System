# learning.py

# 機械学習によって、分類器を作成

import pandas as pd
import numpy as np
import glob
import sys
import os
import math
import matplotlib.pyplot as plt
import feature
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from joblib import dump
import pickle

# 現在のスクリプトのディレクトリを取得
current_directory = os.path.dirname(__file__)
# 親ディレクトリのパスを取得
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
# 親ディレクトリをモジュール検索パスに追加
sys.path.append(parent_directory)

start = []
end = []

# 設定した閾値による抽出すべき時間をまとめたdf_threを作成
df_thre = pd.read_csv(os.path.join(parent_directory,'Threshold_App','threshold.csv'))
print(df_thre)
for mark ,threshold in zip(df_thre['マーク'],df_thre['threshold']):
    if math.isnan(threshold):
        print(str(mark)+'のthresholdがNaNです。')
        start.append(np.nan)
        end.append(np.nan)
        # start, endカラムにnanを追加
    else:
        # start, endカラムに時間を追加
        file = os.path.join(parent_directory,'Threshold_App','static','img',str(mark),'threshold.csv')
        df_tmp = pd.read_csv(file)
        print(mark,",",threshold)
        s = df_tmp[df_tmp['threshold'] == threshold]['start'].values[0]
        e = df_tmp[df_tmp['threshold'] == threshold]['end'].values[0]
        start.append(s)
        end.append(e)
df_thre['start'] = start
df_thre['end'] = end
    

# 特徴量の抽出
df, interval_maru, interval_batu = feature.feature_create(df_thre)
# マルとバツのウィンドウサイズの平均値
print(interval_maru)
print(interval_batu)

# 交差検証
def hold_out(model):
    # 特徴量とクラスラベルの分離
    df.columns = df.columns.astype(str)
    X = df.drop(columns=["label","move"])  # 特徴量
    y = df["label"]  # クラスラベル
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1 Score:", f1)

def leave_one_out(model):
    # 特徴量とクラスラベルの分離
    df.columns = df.columns.astype(str)
    X = df.drop(columns=["label", "move"])  # 特徴量
    y = df["label"]  # クラスラベル

    loo = LeaveOneOut()
    accuracies = []
    f1_scores = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = model
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)
    # 平均AccuracyとF1スコアを計算
    mean_accuracy = sum(accuracies) / len(accuracies)
    mean_f1 = sum(f1_scores) / len(f1_scores)
    print("Mean Accuracy:", mean_accuracy)
    print("Mean F1 Score:", mean_f1)


def grid_search(X_train, y_train):
    # モデルのインスタンス化
    rf = RandomForestClassifier(random_state=42)

    # グリッドサーチで試すパラメータの設定
    param_grid = {
        'n_estimators': [50, 100, 200],  # 木の数
        'max_depth': [None, 10, 20, 30],  # 木の最大深さ
        'min_samples_split': [2, 4, 6],  # 分岐のために必要な最小サンプル数
        'min_samples_leaf': [1, 2, 4]  # 葉を形成するのに必要な最小サンプル数
    }

    # GridSearchCVのインスタンス化（交差検証でモデル選択）k=5分割交差検証
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)

    # グリッドサーチの実行
    grid_search.fit(X_train, y_train)
    # 最適なパラメータとそのスコアを表示
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

    # 最適なモデルを取得
    best_model = grid_search.best_estimator_

    return best_model

def model_save():
    # 特徴量とクラスラベルの分離
    X_train= df.drop(columns=["label","move"])  # 特徴量
    y_train= df["label"]  # クラスラベル

    # 標準化
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns, index=X_train.index)

    if not os.path.exists(os.path.join('model', 'Random_Forest')):
        os.makedirs(os.path.join('model', 'Random_Forest'))

    # スケーラーの保存
    dump(scaler, os.path.join('model', 'Random_Forest', 'scaler_RF.joblib'))

    # グリッドサーチでモデルの最適化
    model = grid_search(X_train_std, y_train)

    # モデルの保存
    with open(os.path.join('model', 'Random_Forest', 'model_RF.pkl'), 'wb') as model_file:
        pickle.dump(model, model_file)


if __name__ == "__main__":
    model_save()
    # leave_one_out() 
    # hold_out()