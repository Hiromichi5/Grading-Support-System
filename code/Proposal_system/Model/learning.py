import pandas as pd
import numpy as np
import glob
import sys
import os
import math
import matplotlib.pyplot as plt
import japanize_matplotlib

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
from tensorflow.keras.models import Sequential, Model, save_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, Input, Reshape, UpSampling1D
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# 現在のスクリプトのディレクトリを取得
current_directory = os.path.dirname(__file__)
# 親ディレクトリのパスを取得
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
# 親ディレクトリをモジュール検索パスに追加
sys.path.append(parent_directory)
from settings import STEP_SIZE, STOP_FRAME, MOVE_FRAME

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
            # start, endカラムにnanを追加
        else:
            # start, endカラムに時間を追加
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
    tmp['abs_acc'] = (tmp['X'].abs() + tmp['Y'].abs() + tmp['Z'].abs())
    tmp = tmp.loc[(tmp['abs_acc'] >= threshold_move)]
    max_f = max(tmp.index)  # 閾値以上の加速度が現れた最後のフレーム

    if (max_f >= STOP_FRAME):
        data = data[:, 0:STOP_FRAME]
    else:
        data = data[:, 0:max_f + MOVE_FRAME]  # それ以降MOVER_FRAMEまでのデータを考慮する
    pred = autoencoder.predict(data)
    loss = np.mean(np.power(data - pred, 2), axis=1)
    print("loss:", loss)
    loss = np.mean(loss)
    print("loss:", loss)
    # 閾値を用いて丸かバツかを判断する
    if (loss >= threshold_auto):
        return "×"
    else:
        return "○"

def create_conv_autoencoder(input_shape, encoding_dim=32):
    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv1D(32, 3, activation='relu', padding='same')(input_layer)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(encoding_dim, activation='relu')(x)

    # Decoder
    x = Dense((input_shape[0] // 4) * 16, activation='relu')(encoded)
    x = Reshape((input_shape[0] // 4, 16))(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(input_shape[1], 3, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

# データの取得と準備
df_thre = extract_acc()
input_data_maru, input_data_batu = get_data(df_thre)
print(input_data_batu.shape)
print(input_data_maru.shape)
sys.exit()
# オートエンコーダのモデル作成
input_shape = input_data_maru.shape[1:]  # shapeを確認
autoencoder = create_conv_autoencoder(input_shape)
autoencoder.summary()

# データの結合とスプリット
X = np.concatenate((input_data_maru, input_data_batu), axis=0)
y = np.concatenate((np.zeros(input_data_maru.shape[0]), np.ones(input_data_batu.shape[0])), axis=0)

# トレーニングと検証用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# トレーニング
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

# 特徴抽出
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dense').output)
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# 分類器のトレーニング
clf = RandomForestClassifier()
clf.fit(X_train_encoded, y_train)

# 評価
accuracy = clf.score(X_test_encoded, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# モデルの保存
autoencoder.save(os.path.join('model', 'Autoencoder', 'autoencoder.h5'))
with open(os.path.join('model', 'Autoencoder', 'random_forest.pkl'), 'wb') as f:
    pickle.dump(clf, f)
