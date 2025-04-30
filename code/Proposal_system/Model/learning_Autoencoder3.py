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
# from keras.saving import save_model
from tensorflow.keras.models import save_model
from sklearn.model_selection import GridSearchCV

# 現在のスクリプトのディレクトリを取得
current_directory = os.path.dirname(__file__)
# 親ディレクトリのパスを取得
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
# 親ディレクトリをモジュール検索パスに追加
sys.path.append(parent_directory)
from settings import STEP_SIZE,STOP_FRAME,MOVE_FRAME

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

if not os.path.exists(os.path.join('model','Autoencoder')):
    os.makedirs(os.path.join('model','Autoencoder'))

label = []
label_num = []
padding_size_maru ,padding_size_batu = 0, 0
threshold_move = 0
label_maru, label_batu = [], []

# 設定した閾値による抽出すべき時間をまとめたdf_threを作成
def extract_acc():
    start = []
    end = []
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
    interval_maru, interval_batu = [], []
    return df_thre

def get_data(df_thre):
    # df_threをもとにデータを
    global label, label_num
    train_maru, train_batu = [], []
    list = ['マル1','マル2','マル3','マル4','マル5','マル6','マル7','マル8','マル9','マル10',
            'バツ1','バツ2','バツ3','バツ4','バツ5','バツ6','バツ7','バツ8','バツ9','バツ10']
    window_maru, window_batu = [], []
    merged_train = pd.DataFrame()
    for mark in list:
        file = os.path.join(parent_directory,'acc_train','acc_data_training_'+mark+'.csv')
        df_tmp = pd.read_csv(file)
        df_tmp['Timestamp'] = pd.to_datetime(df_tmp['Timestamp'])
        start = df_thre[df_thre['マーク'] == mark]['start'].values[0]
        start = pd.to_datetime(start)
        end = df_thre[df_thre['マーク'] == mark]['end'].values[0]
        end = pd.to_datetime(end)
        df = pd.DataFrame()
        df = df_tmp[(df_tmp['Timestamp'] >= start) & (df_tmp['Timestamp'] <= end)][['X','Y','Z']]
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
    print("マルwindow_size = ",padding_size_maru)
    print("バツwindow_size = ",padding_size_batu)
    # スケーリング
    scaler = StandardScaler()
    scaler.fit(merged_train.dropna()) 
    
    # スケーラーの保存
    dump(scaler, os.path.join('model', 'Autoencoder','scaler_autoencoder.joblib'))

    # ゼロパディング
    train_padding_maru, train_padding_batu = [], []
    for df in train_maru:
        df_std = pd.DataFrame(scaler.transform(df),columns=df.columns) # スケーリング
        diff_size = padding_size_maru - len(df_std)
        if padding_size_maru >= len(df_std):
            padding_df = pd.DataFrame([[0, 0, 0]] * int(diff_size), columns=["X", "Y", "Z"])
            # データフレームを結合する
            merged_df = pd.concat([df_std.dropna(), padding_df.dropna()])
        else:
            merged_df = df_std[:int(diff_size)]

        train_padding_maru.append(merged_df)

    for df in train_batu:
        df_std = pd.DataFrame(scaler.transform(df),columns=df.columns) # スケーリング
        diff_size = padding_size_batu - len(df_std)
        if padding_size_batu >= len(df_std):
            padding_df = pd.DataFrame([[0, 0, 0]] * int(diff_size), columns=["X", "Y", "Z"])
            merged_df = pd.concat([df_std.dropna(), padding_df.dropna()])
        else:
            merged_df = df_std[:int(diff_size)]
        train_padding_batu.append(merged_df)

    print("window_maru = ",window_maru)
    print(np.nanmedian(window_maru))
    print("window_batu = ",window_batu)
    print(np.nanmedian(window_batu))
    padding_size_maru = np.nanmedian(window_maru)
    padding_size_batu = np.nanmedian(window_batu)

    input_data_maru = np.array(train_padding_maru)
    input_data_batu = np.array(train_padding_batu)

    return input_data_maru, input_data_batu

def classification_machine(data, threshold_auto, threshold_move, autoencoder): # dataはpadding_sizeの大きさであることを想定
    tmp = pd.DataFrame(data[0][0:STOP_FRAME][:],columns=['X','Y','Z'])
    # 加速度の絶対値を計算
    tmp['abs_acc'] = np.sqrt(tmp['X']**2 + tmp['Y']**2 + tmp['Z']**2)
    # 加速度の変化を計算
    tmp['acc_change'] = abs(tmp['abs_acc'].diff())
    # 加速度変化に基づいて 'movement' フラグを設定する
    tmp['movement'] = tmp['acc_change'] > threshold_move
    if sum(tmp['movement']) > MOVE_FRAME:
        # 異常検知の場合、入力データをモデルに渡す前に異常か正常かを判定
        reconstruction_errors = np.mean(np.power(data - autoencoder.predict(data), 2), axis=1)
        anomalies = (reconstruction_errors > threshold_auto).astype(int) #異常検知 → 1, 正常 → 0
        if sum(anomalies[0]) == 0: # 正常
            return 1
        else: # 異常
            return 2
    else:
        return 0

def anomaly_detection(X_train, X_test, mark):
    print("--- anomaly_detection ---")
    input_dim = X_train.shape[1] * X_train.shape[2]  # 特徴量の数 * タイムステップ数
    encoding_dim = 32  # エンコーディングの次元

    # オートエンコーダーの定義
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    flattened = Flatten()(input_layer)  # 入力を平滑化
    encoded = Dense(encoding_dim, activation='relu')(flattened)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    decoded = Reshape((X_train.shape[1], X_train.shape[2]))(decoded)  # 出力層の形状を元の入力データと同じにする
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # オートエンコーダーのトレーニング
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

    # 再構築誤差の計算
    reconstruction_error = np.mean(np.power(X_train - autoencoder.predict(X_train), 2), axis=1)
    print("X_train.shape")
    print(X_train.shape)
    # 閾値の設定(90%分位数）
    # threshold = np.quantile(reconstruction_error, 0.99)
    # print(threshold)
    # 平均値と標準偏差の計算
    mean_error = np.mean(reconstruction_error)
    std_dev = np.std(reconstruction_error) 
    # 平均値に標準偏差の何倍かを閾値として設定
    threshold = mean_error + 4 * std_dev  # 4倍の標準偏差を使用する例

    print(f"設定された閾値: {threshold}")

    # モデルの保存
    # with open(os.path.join('model', 'Autoencoder','autoencoder'+mark+'.pkl'), 'wb') as model_file:
    #     pickle.dump(autoencoder, model_file)
    
    # モデルを保存
    save_model(autoencoder,os.path.join('model', 'Autoencoder','autoencoder'+mark+'.keras'))

    return autoencoder, threshold

# 画像の保存
def save_img(pred,i,data,folder):
    # グラフの描画
    plt.figure(figsize=(10, 6))
    plt.plot(data['Elapsed'], data['X'], label='X軸')
    plt.plot(data['Elapsed'], data['Y'], label='Y軸')
    plt.plot(data['Elapsed'], data['Z'], label='Z軸')
    plt.axvspan(data['Elapsed'][i], data['Elapsed'][i+padding_size_maru], color='orange', alpha=0.3, label='ウィンドウ')
    plt.axvspan(data['Elapsed'][i], data['Elapsed'][i+padding_size_batu], color='green', alpha=0.3, label='ウィンドウ')
    title = str(data['Elapsed'][i])+'~'+str(data['Elapsed'][i+padding_size_maru])
    plt.title(str(pred)+' : '+title)
    plt.savefig(os.path.join('model','Autoencoder',folder,str(i)+'.png'))
    plt.close()

# グリッドサーチ
def grid_search_autoencoder(X_train, model, param_grid):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_result = grid_search.fit(X_train, X_train)
    
    print("Best Parameters:", grid_result.best_params_)
    print("Best Score (MSE):", grid_result.best_score_)
    
    return grid_result.best_estimator_


# パラメータグリッドの設定
param_grid_maru = {
    'epochs': [50, 100],
    'batch_size': [128, 256],
    'optimizer': ['adam', 'rmsprop'],
    'loss': ['binary_crossentropy', 'mean_squared_error'],
    'regularization': [None, 'l1', 'l2']
}

param_grid_batu = {
    'epochs': [50, 100],
    'batch_size': [128, 256],
    'optimizer': ['adam', 'rmsprop'],
    'loss': ['binary_crossentropy', 'mean_squared_error'],
    'regularization': [None, 'l1', 'l2']
}

if __name__ == "__main__":
    # データ取得
    df_thre = extract_acc()
    data_maru, data_batu = get_data(df_thre)

    X_train_maru, X_test_maru, y_train_maru, y_test_maru = train_test_split(data_maru, np.array(label_maru), test_size=0.2, random_state=42)
    X_train_batu, X_test_batu, y_train_batu, y_test_batu = train_test_split(data_batu, np.array(label_batu), test_size=0.2, random_state=42)
    # マルとバツのそれぞれのオートエンコーダでグリッドサーチを実行
    # best_autoencoder_maru = grid_search_autoencoder(X_train_maru, autoencoder_maru, param_grid_maru)
    # best_autoencoder_batu = grid_search_autoencoder(X_train_batu, autoencoder_batu, param_grid_batu)

    # 異常検出
    autoencoder_maru, threshold_auto_maru = anomaly_detection(X_train_maru, X_test_maru, 'マル')
    autoencoder_batu, threshold_auto_batu = anomaly_detection(X_train_batu, X_test_batu, 'バツ')
    threshold_move = np.nanmin(df_thre['threshold'])
    #閾値保存
    # データフレームを作成
    threshold = {'threshold_auto_maru': [threshold_auto_maru],
                 'threshold_auto_batu': [threshold_auto_batu],
                 'threshold_move': [threshold_move],
                 'padding_size_maru': [padding_size_maru],
                 'padding_size_batu': [padding_size_batu]
                 }
    threshold = pd.DataFrame(threshold)
    # データフレームをCSVファイルとして保存
    threshold.to_csv(os.path.join('model','Autoencoder','threshold.csv'),index=False)