import pandas as pd
import numpy as np
import glob
import sys
import os
import math
import matplotlib.pyplot as plt
import japanize_matplotlib
import feature

# 現在のスクリプトのディレクトリを取得
current_directory = os.path.dirname(__file__)
# 親ディレクトリのパスを取得
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
# 親ディレクトリをモジュール検索パスに追加
sys.path.append(parent_directory)

from settings import STEP_SIZE

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

if not os.path.exists(os.path.join('model','1CNN')):
    os.makedirs(os.path.join('model','1CNN'))

label = []
label_num = []
padding_size = 0
threshold_move = 0

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
    # 特徴量
    global label, label_num
    train = []
    list = ['マル1','マル2','マル3','マル4','マル5','マル6','マル7','マル8','マル9','マル10',
            'バツ1','バツ2','バツ3','バツ4','バツ5','バツ6','バツ7','バツ8','バツ9','バツ10']
    window = []
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
        train.append(df)
        merged_train = pd.concat([merged_train, df], ignore_index=True)
        window.append(len(df))
        if 'マル' in mark:
            label_num.append(0)
            label.append("マル")
        elif 'バツ' in mark:
            label_num.append(1)
            label.append("バツ")
        else:
            label_num.append(np.nan)
            label.append(np.nan)
    global padding_size
    padding_size = np.max(window)

    # スケーリング
    scaler = StandardScaler()
    scaler.fit(merged_train.dropna()) 
    
    # スケーラーの保存
    dump(scaler, os.path.join('model','1CNN','scaler_1CNN.joblib'))

    # ゼロパディング
    train_padding = []
    for df in train:
        df_std = pd.DataFrame(scaler.transform(df),columns=df.columns) # スケーリング
        diff_size = padding_size - len(df_std)
        padding_df = pd.DataFrame([[0, 0, 0]] * diff_size, columns=["X", "Y", "Z"])
        # データフレームを結合する
        merged_df = pd.concat([df_std, padding_df], ignore_index=True)
        train_padding.append(merged_df)
    
    print("window = ",window)
    input_data = np.array(train_padding)
    print(input_data)
    print(input_data.shape)
    return input_data


def classification_machine(data, threshold_auto, threshold_move, model): # dataはpadding_sizeの大きさであることを想定
    tmp = pd.DataFrame(data[0][0:6][:],columns=['X','Y','Z'])
    # 加速度の絶対値を計算
    tmp['abs_acc'] = np.sqrt(tmp['X']**2 + tmp['Y']**2 + tmp['Z']**2)
    # 加速度の変化を計算
    tmp['acc_change'] = abs(tmp['abs_acc'].diff())
    # 加速度変化に基づいて 'movement' フラグを設定する
    tmp['movement'] = tmp['acc_change'] > threshold_move
    if sum(tmp['movement']) > 1:
        # 異常検知の場合、入力データをモデルに渡す前に異常か正常かを判定
        reconstruction_errors = np.mean(np.power(data - autoencoder.predict(data), 2), axis=1)
        anomalies = (reconstruction_errors > threshold_auto).astype(int) #異常検知 → 1, 正常 → 0
        print(anomalies)
        if sum(anomalies[0]) == 0: # 正常
            y_pred = model.predict(data)
            y_pred = np.round(y_pred).astype(int)  # 予測確率を0または1のクラスラベルに変換
            if y_pred[0][0]==0:
                print('マル')
                return 'マル'
            elif y_pred[0][0]==1:
                print('バツ')
                return 'バツ'
        else: # 異常
            print('異常検知')
            return '異常検知'
    else:
        print('閾値以下')
        return '閾値以下'

def anomaly_detection(X_train, X_test):
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
    threshold = mean_error + 5 * std_dev  # 2倍の標準偏差を使用する例

    print(f"設定された閾値: {threshold}")

    # モデルの保存
    with open(os.path.join('model', '1CNN', 'autoencoder.pkl'), 'wb') as model_file:
        pickle.dump(autoencoder, model_file)

    return autoencoder, threshold

def create_1CNNmodel(X_train, X_test):
    print("--- 1D CNN ---")
     # 1CNNモデルの定義
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(padding_size, 3)), # 畳み込み層（入力層含む）
        MaxPooling1D(pool_size=2), # プーリング層
        Dropout(0.5),
        Conv1D(filters=128, kernel_size=3, activation='relu'), # 2つ目の畳み込み層
        MaxPooling1D(pool_size=2),# プーリング層
        Dropout(0.5),
        Flatten(), # 平滑化層
        Dense(100, activation='relu'), # 全結合層
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 出力層: シグモイド活性化関数を使用
    ])

    # モデルのコンパイル: バイナリクロスエントロピーとAdamオプティマイザを使用
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # モデルのサマリー表示
    model.summary()

    # モデルのトレーニング
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    with open(os.path.join('model', '1CNN', '1CNN.pkl'), 'wb') as model_file:
        pickle.dump(autoencoder, model_file)
    return model

# 関数の保存
def save_img(pred,i,data,folder):
    # グラフの描画
    plt.figure(figsize=(10, 6))
    plt.plot(data['Elapsed'], data['X'], label='X軸')
    plt.plot(data['Elapsed'], data['Y'], label='Y軸')
    plt.plot(data['Elapsed'], data['Z'], label='Z軸')
    plt.axvspan(data['Elapsed'][i], data['Elapsed'][i+padding_size], color='orange', alpha=0.3, label='ウィンドウ')
    title = str(data['Elapsed'][i])+'~'+str(data['Elapsed'][i+padding_size])
    plt.title(str(pred)+' : '+title)
    plt.savefig(os.path.join('model','1CNN',folder,str(i)+'.png'))
    plt.close()

if __name__ == "__main__":
    # データ取得
    df_thre = extract_acc()
    data = get_data(df_thre)
    X_train, X_test, y_train, y_test = train_test_split(data, np.array(label_num), test_size=0.2, random_state=42)
    # 異常検出
    autoencoder, threshold_auto = anomaly_detection(X_train, X_test)
    threshold_move = np.nanmin(df_thre['threshold'])

    # 1D CNNモデルの作成
    model = create_1CNNmodel(X_train, X_test)

    # スケーラーのロード
    scaler = joblib.load(os.path.join('model','1CNN','scaler_1CNN.joblib'))
    folder = 'acc_data_check_マル5バツ5'

    if not os.path.exists(os.path.join('model','1CNN',folder)):
        os.makedirs(os.path.join('model','1CNN',folder))
    # df = pd.read_csv(os.path.join('check','acc_data_check_マル5バツ5.csv'))
    data = pd.read_csv(os.path.join('check',folder+'.csv'))
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce') # タイムスタンプ列をdatetimeオブジェクトに変換
    data = data.dropna(subset=['Timestamp']) # タイムスタンプでエラーが発生した行を削除
    start_time = data['Timestamp'].iloc[0] # 最初のタイムスタンプを基準として経過時間（秒）を計算
    data['Elapsed'] = (data['Timestamp'] - start_time).dt.total_seconds() # 0秒から始まるように変更
    df = data[['X','Y','Z']]
    result = []
    for i in range(0, len(df)-padding_size, 5):
        df_std = scaler.transform(df[i:i+padding_size])
        df_tmp = np.array([df_std])
        pred = classification_machine(df_tmp, threshold_auto, threshold_move,model)
        result.append(pred)
        save_img(pred,i,data,folder)
    print(result)
