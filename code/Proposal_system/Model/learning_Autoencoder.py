import pandas as pd
import numpy as np
import glob
import sys
import os
import math
import matplotlib.pyplot as plt
import japanize_matplotlib
import feature
# from keras.models import load_model
from tensorflow.keras.models import save_model

# 現在のスクリプトのディレクトリを取得
current_directory = os.path.dirname(__file__)
# 親ディレクトリのパスを取得
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
# 親ディレクトリをモジュール検索パスに追加
sys.path.append(parent_directory)
from settings import STEP_SIZE, STOP_FRAME, MOVE_FRAME

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import joblib
from joblib import dump
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, Input, Reshape

if not os.path.exists(os.path.join('model', 'Autoencoder')):
    os.makedirs(os.path.join('model', 'Autoencoder'))

label = []
label_num = []
padding_size_maru, padding_size_batu = 0, 0
threshold_move = 0
label_maru, label_batu = [], []

# 設定した閾値による抽出すべき時間をまとめたdf_threを作成
def extract_acc():
    start = []
    end = []
    df_thre = pd.read_csv(os.path.join(parent_directory, 'Threshold_App', 'threshold.csv'))
    print(df_thre)
    for mark, threshold in zip(df_thre['マーク'], df_thre['threshold']):
        if math.isnan(threshold):
            print(str(mark) + 'のthresholdがNaNです。')
            start.append(np.nan)
            end.append(np.nan)
        else:
            file = os.path.join(parent_directory, 'Threshold_App', 'static', 'img', str(mark), 'threshold.csv')
            df_tmp = pd.read_csv(file)
            print(mark, ",", threshold)
            s = df_tmp[df_tmp['threshold'] == threshold]['start'].values[0]
            e = df_tmp[df_tmp['threshold'] == threshold]['end'].values[0]
            start.append(s)
            end.append(e)
    df_thre['start'] = start
    df_thre['end'] = end
    interval_maru, interval_batu = [], []
    return df_thre

def get_data(df_thre):
    # df_threをもとにデータを
    global label, label_num
    train_maru, train_batu = [], []
    list = ['マル1', 'マル2', 'マル3', 'マル4', 'マル5', 'マル6', 'マル7', 'マル8', 'マル9', 'マル10',
            'バツ1', 'バツ2', 'バツ3', 'バツ4', 'バツ5', 'バツ6', 'バツ7', 'バツ8', 'バツ9', 'バツ10']
    window_maru, window_batu = [], []
    merged_train = pd.DataFrame()
    for mark in list:
        file = os.path.join(parent_directory, 'acc_train', 'acc_data_training_' + mark + '.csv')
        df_tmp = pd.read_csv(file)
        df_tmp['Timestamp'] = pd.to_datetime(df_tmp['Timestamp'])
        start = df_thre[df_thre['マーク'] == mark]['start'].values[0]
        start = pd.to_datetime(start)
        end = df_thre[df_thre['マーク'] == mark]['end'].values[0]
        end = pd.to_datetime(end)
        df = pd.DataFrame()
        df = df_tmp[(df_tmp['Timestamp'] >= start) & (df_tmp['Timestamp'] <= end)][['X', 'Y', 'Z']]
        merged_train = pd.concat([merged_train, df], ignore_index=True)
        if 'マル' in mark:
            label_num.append(0)
            label.append("マル")
            train_maru.append(df)
            window_maru.append(len(df))
            label_maru.append(0)
        elif 'バツ' in mark:
            label_num.append(1)
            label.append("バツ")
            train_batu.append(df)
            window_batu.append(len(df))
            label_batu.append(1)
        else:
            label_num.append(np.nan)
            label.append(np.nan)
    global padding_size_maru, padding_size_batu
    padding_size_maru = int(np.median(window_maru))
    padding_size_batu = int(np.median(window_batu))
    print("マルwindow_size = ", padding_size_maru)
    print("バツwindow_size = ", padding_size_batu)
    # スケーリング
    scaler = StandardScaler()
    scaler.fit(merged_train.dropna()) 

    # スケーラーの保存
    dump(scaler, os.path.join('model', 'Autoencoder', 'scaler_autoencoder.joblib'))

    # ゼロパディング
    train_padding_maru, train_padding_batu = [], []
    for df in train_maru:
        df_std = pd.DataFrame(scaler.transform(df), columns=df.columns)  # スケーリング
        diff_size = padding_size_maru - len(df_std)
        if padding_size_maru >= len(df_std):
            padding_df = pd.DataFrame([[0, 0, 0]] * int(diff_size), columns=["X", "Y", "Z"])
            # データフレームを結合する
            merged_df = pd.concat([df_std.dropna(), padding_df.dropna()])
        else:
            merged_df = df_std[:int(diff_size)]

        train_padding_maru.append(merged_df)

    for df in train_batu:
        df_std = pd.DataFrame(scaler.transform(df), columns=df.columns)  # スケーリング
        diff_size = padding_size_batu - len(df_std)
        if padding_size_batu >= len(df_std):
            padding_df = pd.DataFrame([[0, 0, 0]] * int(diff_size), columns=["X", "Y", "Z"])
            merged_df = pd.concat([df_std.dropna(), padding_df.dropna()])
        else:
            merged_df = df_std[:int(diff_size)]
        train_padding_batu.append(merged_df)

    print("window_maru = ", window_maru)
    print(np.nanmedian(window_maru))
    print("window_batu = ", window_batu)
    print(np.nanmedian(window_batu))
    padding_size_maru = np.nanmedian(window_maru)
    padding_size_batu = np.nanmedian(window_batu)

    input_data_maru = np.array(train_padding_maru)
    input_data_batu = np.array(train_padding_batu)

    return input_data_maru, input_data_batu

def classification_machine(data, threshold_auto, threshold_move, autoencoder):  # dataはpadding_sizeの大きさであることを想定
    tmp = pd.DataFrame(data[0][0:STOP_FRAME][:], columns=['X', 'Y', 'Z'])
    # 加速度の絶対値を計算
    tmp['abs_acc'] = np.sqrt(tmp['X'] ** 2 + tmp['Y'] ** 2 + tmp['Z'] ** 2)
    # 加速度の変化を計算
    tmp['acc_change'] = abs(tmp['abs_acc'].diff())
    # 加速度変化に基づいて 'movement' フラグを設定する
    tmp['movement'] = tmp['acc_change'] > threshold_move
    if sum(tmp['movement']) > MOVE_FRAME:
        # 異常検知の場合、入力データをモデルに渡す前に異常か正常かを判定
        reconstruction_errors = np.mean(np.power(data - autoencoder.predict(data), 2), axis=1)
        anomalies = (reconstruction_errors > threshold_auto).astype(int)  # 異常検知 → 1, 正常 → 0
        if sum(anomalies[0]) == 0:  # 正常
            return 1
        else:  # 異常
            return 2
    else:
        return 0

def anomaly_detection(X_train, mark):
    print("--- anomaly_detection ---")
    input_dim = X_train.shape[1] * X_train.shape[2]  # 特徴量の数 * タイムステップ数
    encoding_dim = 32  # エンコーディングの次元

    # オートエンコーダーの定義
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    flattened = Flatten()(input_layer)  # 入力をフラット化
    encoder = Dense(encoding_dim, activation="relu")(flattened)
    decoder = Dense(input_dim, activation="sigmoid")(encoder)
    decoder = Reshape((X_train.shape[1], X_train.shape[2]))(decoder)  # 元の形状に戻す

    autoencoder = Model(inputs=input_layer, outputs=decoder)

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    autoencoder.summary()

    # データの整形
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])

    # オートエンコーダーの訓練
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True)

    # オートエンコーダーの保存
    save_model(autoencoder, os.path.join('model', 'Autoencoder', f'autoencoder_{mark}.h5'))

    # トレーニングデータに対する再構成誤差を計算
    X_train_pred = autoencoder.predict(X_train)
    train_mse_loss = np.mean(np.power(X_train - X_train_pred, 2), axis=1)

    # 閾値を設定（例えば再構成誤差の95パーセンタイルを使用）
    threshold = np.percentile(train_mse_loss, 95)

    return autoencoder, threshold

if __name__ == '__main__':
    df_thre = extract_acc() # 閾値とtrain加速度から抽出
    input_data_maru, input_data_batu = get_data(df_thre) # ○、×それぞれのデータのサイズを揃える
    autoencoder_maru, threshold_maru = anomaly_detection(input_data_maru, "maru") # 異常検知の○モデル生成
    autoencoder_batu, threshold_batu = anomaly_detection(input_data_batu, "batu") # 異常検知の×モデル生成
    
    # threshold_move(静止or動きの閾値)を計算
    tmp = pd.DataFrame(input_data_maru[0][:STOP_FRAME][:], columns=['X', 'Y', 'Z'])
    # 加速度の絶対値を計算
    tmp['abs_acc'] = np.sqrt(tmp['X'] ** 2 + tmp['Y'] ** 2 + tmp['Z'] ** 2)
    # 加速度の変化を計算
    tmp['acc_change'] = abs(tmp['abs_acc'].diff())
    # 加速度変化の95パーセンタイルをthreshold_moveとして設定
    threshold_move = np.percentile(tmp['acc_change'], 95)
    print("threshold_move:", threshold_move)
