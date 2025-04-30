import pandas as pd
import numpy as np
import glob
import sys
import os
import math
import matplotlib.pyplot as plt
import japanize_matplotlib

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Reshape

import tensorflow as tf
from joblib import dump

# 現在のスクリプトのディレクトリを取得
current_directory = os.path.dirname(__file__)
# 親ディレクトリのパスを取得
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
# 親ディレクトリをモジュール検索パスに追加
sys.path.append(parent_directory)
from settings import STEP_SIZE, STOP_FRAME, MOVE_FRAME

# 設定した閾値による抽出すべき時間をまとめたdf_threを作成
def extract_acc():
    start = []
    end = []
    df_thre = pd.read_csv(os.path.join(parent_directory, 'Threshold_App', 'threshold.csv'))
    print(df_thre)
    for mark, threshold in zip(df_thre['マーク'], df_thre['threshold']):
        if math.isnan(threshold):
            print(str(mark)+'のthresholdがNaNです。')
            start.append(np.nan)
            end.append(np.nan)
        else:
            file = os.path.join(parent_directory, 'Threshold_App', 'static', 'img', str(mark), 'threshold.csv')
            df_tmp = pd.read_csv(file)
            s = df_tmp[df_tmp['threshold'] == threshold]['start'].values[0]
            e = df_tmp[df_tmp['threshold'] == threshold]['end'].values[0]
            start.append(s)
            end.append(e)
    df_thre['start'] = start
    df_thre['end'] = end
    interval_maru, interval_batu = [], []
    return df_thre

# データ取得
def get_data(df_thre):
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
            train_maru.append(df)
            window_maru.append(len(df))
        elif 'バツ' in mark:
            train_batu.append(df)
            window_batu.append(len(df))
    
    padding_size_maru = int(np.median(window_maru))
    padding_size_batu = int(np.median(window_batu))
    print("マルウィンドウサイズ = ", padding_size_maru)
    print("バツウィンドウサイズ = ", padding_size_batu)
    
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
        padding_df = pd.DataFrame([[0, 0, 0]] * int(diff_size), columns=["X", "Y", "Z"])
        merged_df = pd.concat([df_std, padding_df]).values  # NumPy配列に変換
        train_padding_maru.append(merged_df)

    for df in train_batu:
        df_std = pd.DataFrame(scaler.transform(df), columns=df.columns)  # スケーリング
        diff_size = padding_size_batu - len(df_std)
        padding_df = pd.DataFrame([[0, 0, 0]] * int(diff_size), columns=["X", "Y", "Z"])
        merged_df = pd.concat([df_std, padding_df]).values  # NumPy配列に変換
        train_padding_batu.append(merged_df)

    # NumPy配列に変換
    input_data_maru = np.array(train_padding_maru)
    input_data_batu = np.array(train_padding_batu)

    return input_data_maru, input_data_batu, padding_size_maru, padding_size_batu

# オートエンコーダの定義
def build_autoencoder(input_shape=(None, 3), encoding_dim=3, optimizer='adam', regularization=None):
    # オートエンコーダの定義
    input_layer = Input(shape=input_shape[1:])  # バッチ次元を除いた形状を指定
    flattened = Flatten()(input_layer)  # 入力を平滑化
    encoded = Dense(encoding_dim, activation='relu', kernel_regularizer=regularization)(flattened)
    decoded = Dense(np.prod(input_shape[1:]), activation='sigmoid')(encoded)
    decoded = Reshape(input_shape[1:])(decoded)  # 出力層の形状を元の入力データと同じにする

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=optimizer, loss='mse')  # 平均二乗誤差を使用

    return autoencoder

# グリッドサーチの実行
def grid_search_autoencoder(X_train, input_shape, param_grid):
    model = KerasRegressor(build_fn=build_autoencoder, input_shape=input_shape, verbose=0)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid_search.fit(X_train, X_train)  # X_trainを入力として使用

    print("Best parameters found: ", grid_result.best_params_)
    best_model = grid_result.best_estimator_

    return best_model

if __name__ == "__main__":
    # データ取得
    df_thre = extract_acc()
    data_maru, data_batu, padding_size_maru, padding_size_batu = get_data(df_thre)

    # パラメータグリッドの設定
    param_grid_maru = {
        'epochs': [10, 50, 100],
        'batch_size': [32, 64, 128, 256],
        'optimizer': ['adam', 'rmsprop'],
        # 'regularization': [None, tf.keras.regularizers.l1(), tf.keras.regularizers.l2()]
    }

    param_grid_batu = {
        'epochs': [50, 100],
        'batch_size': [128, 256],
        'optimizer': ['adam', 'rmsprop'],
        # 'regularization': [None, tf.keras.regularizers.l1(), tf.keras.regularizers.l2()]
    }

    # マルとバツのそれぞれのオートエンコーダでグリッドサーチを実行
    best_autoencoder_maru = grid_search_autoencoder(data_maru, (padding_size_maru, 3), param_grid_maru)
    best_autoencoder_batu = grid_search_autoencoder(data_batu, (padding_size_batu, 3), param_grid_batu)
