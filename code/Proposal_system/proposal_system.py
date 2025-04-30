# proposal_system.py

from __future__ import print_function
from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *

from settings import NUM_SAMPLES,MAC_ADRESS,TEST_TIME,STOP_FRAME,MOVE_FRAME,STEP_SIZE,STEP_E,EFFICIENCY_STEP,EFFICIENCY_STEP_TIME,EFFICIENCY_TIME,EFFICIENCY_FRAME,EFFICIENCY_NEAR_FRAME
from time import sleep,time
import csv
from datetime import datetime, timedelta
import pytz
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import japanize_matplotlib
import sys
import pickle
from joblib import load
from keras.models import load_model
import concurrent.futures
import multiprocessing
from multiprocessing import Manager
import pygame
from playsound import playsound
import matplotlib.animation as animation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # WARNINGとINFOを非表示にする

if not os.path.exists(os.path.join('acc_test')):
    os.makedirs(os.path.join('acc_test'))
    
if not os.path.exists(os.path.join('Result')):
    os.makedirs(os.path.join('Result'))

counter_sliding = 0
count = 0
x, y, z, xyz = [], [], [], []
pred = []
result = []
color_list = []

def color_make(E,line_color):
    mean = np.mean(E) # 平均
    mean_near = np.mean(E[-EFFICIENCY_NEAR_FRAME:]) if len(E)>=EFFICIENCY_NEAR_FRAME else 1000 # 直近60sの平均
    E_current = E[-1] # 最新の作業効率
    if mean_near != 1000:
        if mean < E_current:
            if mean_near < E_current:
                line_color.append('r')
                color_list.append('r')
            else:
                line_color.append('y')
                color_list.append('y')
        else:
            if mean_near < E_current:
                line_color.append('g')
                color_list.append('g')
            else:
                line_color.append('b')
                color_list.append('b')
    else:
        line_color.append('k')
        color_list.append('k')

    # if len(line_color) == 300:
    #     line_color.pop(0)

def update_graph(E,fig,ax1,ax2,line_color):
    start = time()
    mean = np.mean(E) # 平均
    mean_near = np.mean(E[-EFFICIENCY_NEAR_FRAME:]) if len(E)>=EFFICIENCY_NEAR_FRAME else 1000 # 直近60sの平均
    E_current = E[-1] # 最新の作業効率

    # 右側
    ax2.clear() 
    ax2.text(0, 0.8, "評価：", size=30)
    
    if mean_near != 1000:
        if mean < E_current:
            if mean_near < E_current:
                ax2.text(0, 0.8, "　　　Excellent", size=30, color='r')
                line_color.append('r')
                color_list.append('r')
            else:
                ax2.text(0, 0.8, "　　　Good", size=30, color='y')
                line_color.append('y')
                color_list.append('y')
        else:
            if mean_near < E_current:
                ax2.text(0, 0.8, "　　　Poor", size=30, color='g')
                line_color.append('g')
                color_list.append('g')
            else:
                ax2.text(0, 0.8, "　　　Bad", size=30, color='b')
                line_color.append('b')
                color_list.append('b')
    else:
        ax2.text(0, 0.8, "　　　---", size=30)
        line_color.append('k')
        color_list.append('k')

    # if len(line_color) == 300:
    #     line_color.pop(0)

    ax2.text(0, 0.6, "作業効率：" + str(E_current), size=30)
    if mean_near != 1000:
        ax2.text(0, 0.4, "直近の平均：" + str(round(mean_near,2)), size=30)
    else:
        ax2.text(0, 0.4, "直近の平均：---", size=30)
    ax2.text(0, 0.2, "全体の平均：" + (str(round(mean, 2)) ), size=30)
    ax2.axis("off")

    start1 = time()
    # 左側
    width = min(len(E), 300)
    start_time = 30
    end_time = start_time + (len(E)-1)* 0.1
    step = 0.1  # ステップの幅
    end1 = time()
    # 配列の生成
    Time = [round(end_time - (width-1-i) * step,2) for i in range(width)]
    end2 = time()

    # グラフ作成
    ax1.clear()
    color_299 = line_color[-299:] # マジックナンバーだめ！
    for i in range(width-2):
        ax1.plot(Time[i:i+2], E[-width+i:-width+i+2], color=color_299[i])
    end3 = time()
    i = width-2
    ax1.plot(Time[i:i+2], E[-width+i:], color=color_299[i])
    ax1.axhline(y=np.mean(E), color='g', linestyle='--', label='平均')
    ax1.legend()
    ax1.set_title('作業効率の変化')
    ax1.set_xlabel('時間（秒）')
    end4 = time()
    fig.canvas.draw_idle()
    plt.pause(0.001)  # グラフを再描画
    end = time()

def count_clusters(arr,maru_frame,batu_frame):
    clusters = 0
    seq = 0
    current_cluster = [] # クラスターの最後のインデックス
    for i in range(len(arr)):
        if arr[i] in ['○', '×', '○×']:
            if not current_cluster: # 空であれば
                current_cluster.append(i)
                clusters += 1
                seq += 1
            elif i > current_cluster[-1] + 3: # 1個飛ばしを同じ塊とするなら2 n個飛ばすならn+1
                current_cluster.append(i)
                clusters += 1  
                if seq > maru_frame:
                    clusters = clusters + round(seq/maru_frame)
                seq = 0  
            else:
                current_cluster[-1] = i
                seq += 1
    # 最後の要素が連続していた場合の処理
    if seq >= maru_frame:
        clusters += seq // maru_frame
    return clusters

# テストデータ取得
def test(queue):
    # モデルのロード
    autoencoder_maru = load_model(os.path.join('Model','model','Autoencoder','autoencoderマル.keras'))
    autoencoder_batu = load_model(os.path.join('Model','model','Autoencoder','autoencoderバツ.keras'))

    # スケーラーをロード
    scaler = load(os.path.join('Model','model','Autoencoder','scaler_autoencoder.joblib'))
    df = pd.read_csv(os.path.join('Model','model','Autoencoder','threshold.csv'))
    threshold_auto_maru = df['threshold_auto_maru'][0]
    threshold_auto_batu = df['threshold_auto_batu'][0]
    threshold_move = df['threshold_move'][0]
    padding_size_maru = df['padding_size_maru'][0]
    padding_size_batu = df['padding_size_batu'][0]
    maru_frame = round(padding_size_maru/10)
    batu_frame = round(padding_size_batu/10)
    window_size = max([padding_size_maru,padding_size_batu])

    def classify(x,y,z,xyz):
        # 入力データ作成
        data = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
        df_std_maru = scaler.transform(data[0:int(padding_size_maru)])
        df_std_batu = scaler.transform(data[0:int(padding_size_batu)])
        df_tmp_maru = np.array([df_std_maru])
        df_tmp_batu = np.array([df_std_batu])
        # 静止検出
        move = []
        for i in range(STOP_FRAME):
            move.append(abs(xyz[i+1]-xyz[i]) > threshold_move)
        if sum(move) > MOVE_FRAME:
            # マル異常検知
            reconstruction_errors = np.mean(np.power(df_tmp_maru - autoencoder_maru.predict(df_tmp_maru,verbose=0), 2), axis=1)
            anomalies = (reconstruction_errors > threshold_auto_maru).astype(int) # 異常検知 → 1, 正常 → 0
            judge_maru = 0 if sum(anomalies[0]) == 0 else 1
            judge_maru = int(sum(anomalies[0]) != 0)
            # バツ異常検知
            reconstruction_errors = np.mean(np.power(df_tmp_batu - autoencoder_batu.predict(df_tmp_batu,verbose=0), 2), axis=1)
            anomalies = (reconstruction_errors > threshold_auto_batu).astype(int) # 異常検知 → 1, 正常 → 0
            judge_batu = int(sum(anomalies[0]) != 0)
            judge = judge_maru + judge_batu

            if judge == 0: # マルかつバツ
                # return 1
                return '○×' 
            elif judge == 1: # マルあるいはバツ
                if judge_maru == 1:
                    # return 1　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　
                    return '○'
                elif judge_batu ==1:
                    # return 1
                    return '×'
                else:
                    return 'Error1'
            elif judge == 2: # 異常検知
                # return 0
                return '異常'
            else:
                return 'Error2'
        else:
            # return 0
            return '静止'

    class State:
        def __init__(self, device, session):
            self.device = device
            self.samples = 0
            self.callback = FnVoid_VoidP_DataP(self.data_handler)
            self.session = session
            self.csv_writer = None

        def data_handler(self, ctx, data):
            # データハンドラー：データを表示し、サンプル数を更新
            acc_data = parse_value(data)
            # print("%s -> {x : %.3f, y : %.3f, z : %.3f}" % (self.device.address, acc_data.x, acc_data.y, acc_data.z))
            
            # 現在のUTC時間を取得
            current_time_utc = datetime.utcfromtimestamp(time()).replace(tzinfo=pytz.utc)

            # 日本時間に変換
            japan_tz = pytz.timezone('Asia/Tokyo')
            current_time = current_time_utc.astimezone(japan_tz)
            
            # CSV ファイルにデータを書き込む
            self.csv_writer.writerow([self.device.address, self.samples, current_time, acc_data.x, acc_data.y, acc_data.z])
            if len(x) > window_size:
                global counter_sliding,result,pred,count
                x.pop(0)
                y.pop(0)
                z.pop(0)
                xyz.pop(0)
                counter_sliding += 1
                if counter_sliding == STEP_SIZE: # ステップサイズごとに判定する 0.1s
                    result = classify(x,y,z,xyz)
                    print(result)
                    pred.append(result)
                    if len(pred) >= EFFICIENCY_FRAME:
                        # pred.pop(0)
                        queue.put(count_clusters(pred[-EFFICIENCY_FRAME:],maru_frame,batu_frame))

                        # count += 1
                        # if count == STEP_E: # 0.5sごとに作業効率を作成
                        #     # 共有変数の値を変更
                        #     queue.put(count_clusters(pred))
                        #     count = 0
                    counter_sliding = 0
            x.append(acc_data.x)
            y.append(acc_data.y)
            z.append(acc_data.z)
            xyz.append(np.sqrt(acc_data.x**2+acc_data.y**2+acc_data.z**2))
            
            self.samples += 1

    # 引数が指定されていない場合や、引数が 1 から 5 のいずれでもない場合はエラーメッセージを表示して終了
    valid_arguments = {'1', '2', '3', '4', '5'}
    if len(sys.argv) != 2 or sys.argv[1] not in valid_arguments:
        print("Usage: python script_name.py 1|2|3|4|5")
        sys.exit(1)

    mac = MAC_ADRESS[int(sys.argv[1]) - 1]
    print("MACアドレス : ", mac)

    session = 1  # 初回セッション
    states = []  # State インスタンスのリストを初期化
    start_sound = os.path.join("sound","「よぉーい…」.mp3")
    start_sound2 = os.path.join("sound","「スタート」.mp3")
    # 初期化
    pygame.init()
    pygame.mixer.init()

    # 必要ならディスプレイを設定してビデオシステムを初期化
    pygame.display.set_mode((1, 1))

    # サウンドファイルのロードと再生
    sound1 = pygame.mixer.Sound(start_sound)
    sound2 = pygame.mixer.Sound(start_sound2)
    try:
        # フォルダ名
        folder_name = "acc_test"
        # 接続
        d = MetaWear(mac)
        d.connect()
        print("Connected to " + d.address + " over " + ("USB" if d.usb.is_connected else "BLE"))
        states.append(State(d, session))  # ここで State インスタンスを追加

        print("Configuring device")
        # BLEの設定
        libmetawear.mbl_mw_settings_set_connection_parameters(states[0].device.board, 7.5, 7.5, 0, 6000)  # ここで states[0] を使う
        sleep(1.5)
        # 加速度センサの設定
        libmetawear.mbl_mw_acc_set_odr(states[0].device.board, 100.0)
        libmetawear.mbl_mw_acc_set_range(states[0].device.board, 16.0)
        libmetawear.mbl_mw_acc_write_acceleration_config(states[0].device.board)

        # セッションごとに繰り返し
        csv_filename = os.path.join(folder_name, f'acc_data_test.csv')

        with open(csv_filename, mode='w', newline='') as file:
            # CSV ファイルを開いてヘッダーを書き込む
            csv_header = ["Device Address", "Sample Number", "Timestamp", "X", "Y", "Z"]
            writer = csv.writer(file)
            writer.writerow(csv_header)
            states[0].samples = 0  # セッションごとにサンプル数をリセット
            states[0].csv_writer = writer
            sleep(0.5)
            key_pressed = 'y'
            if key_pressed == 'y':
                # 開始の合図
                
                print("よぉーい")
                pygame.mixer.Channel(1).play(sound1)
                sleep(1.5)
                print("スタート！！！")
                pygame.mixer.Channel(1).play(sound2)
                # 加速度の取得とサブスク
                signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(states[0].device.board)
                libmetawear.mbl_mw_datasignal_subscribe(signal, None, states[0].callback)
                # 加速度の開始
                libmetawear.mbl_mw_acc_enable_acceleration_sampling(states[0].device.board)
                libmetawear.mbl_mw_acc_start(states[0].device.board)

                # 待機
                sleep(TEST_TIME) 

                # 加速度の停止
                libmetawear.mbl_mw_acc_stop(states[0].device.board)
                libmetawear.mbl_mw_acc_disable_acceleration_sampling(states[0].device.board)

                print(f"Data saved to {csv_filename}")
                session += 1
                states[0].csv_writer = None  # ファイルをクローズ
    except KeyboardInterrupt:
        pass
    finally:
        # 全体の終了処理
        if states:
            if states[0].csv_writer:
                file.close()
            # 購読解除
            signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(states[0].device.board)
            libmetawear.mbl_mw_datasignal_unsubscribe(signal)
            # 切断
            libmetawear.mbl_mw_debug_disconnect(states[0].device.board)
        print("センサの切断完了")
        
        print("予測結果の保存")
        csv_pred = os.path.join('Result','pred.csv')
        # BGMタイミング保存
        with open(csv_pred, mode='w', newline='') as file:
            # CSV ファイルを開いてヘッダーを書き込む
            csv_header = ["Prediction"]
            writer = csv.writer(file)
            writer.writerow(csv_header)
            # 各要素をCSVに書き込む
            for p in pred:
                writer.writerow([p])

        # グラフ出力

# データの確認      
def check_data():
    # フォルダ名
    folder_name = "Train"
    # フォルダ内の全てのCSVファイルのリストを取得
    files = [file for file in os.listdir(folder_name) if file.endswith('.csv')]
    for file in files:
        # CSVファイルのフルパスを作成
        file_path = os.path.join(folder_name, file)
        # データを読み込み、リストに追加
        df = pd.read_csv(file_path)
        print(file,":",len(df))

def draw(queue, shared_line_color):
    try:
        start_support = os.path.join("sound","「起動しました」.mp3")
        pygame.init()
        pygame.mixer.init()
        pygame.display.set_mode((1, 1))
        support = pygame.mixer.Sound(start_support)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1]})
        ax1.text(0.5, 0.5, "準備中", size=60, ha='center', va='center')
        ax2.text(0.5, 0.5, "準備中", size=60, ha='center', va='center')
        ax1.axis("off")
        ax2.axis("off")
        fig.canvas.draw_idle()
        plt.pause(0.01) 
        plt.show(block=False) 
        count_graph = 0
        count_voice = 0
        E = []
        while True:
            if not queue.empty():
                if count_voice == 0:
                    pygame.mixer.Channel(1).play(support)
                    count_voice = 1
                value = queue.get()
                E.append(value)
                count_graph += 1
                if count_graph == 4: # 3 →0.3sに１回グラフが更新
                    update_graph(E,fig,ax1,ax2,shared_line_color)
                    count_graph = 0
                else:
                    color_make(E,shared_line_color)
    except KeyboardInterrupt:
        pass
    finally:
        print("作業効率保存")
        csv_Eff = os.path.join('Result','eff_result.csv')
        # BGMタイミング保存
        with open(csv_Eff, mode='w', newline='') as file:
            # CSV ファイルを開いてヘッダーを書き込む
            csv_header = ["Efficiency","Color"]
            writer = csv.writer(file)
            writer.writerow(csv_header)
            # 各要素をCSVに書き込む
            for e,c in zip(E,color_list):
                writer.writerow([e,c])

def sound(shared_line_color):
    try:
        BGM = os.path.join("sound", "Fancy_Pop.mp3")
        Excellent = os.path.join("sound","「エクセレント」.mp3")
        Good = os.path.join("sound","「グッド」.mp3")
        Poor = os.path.join("sound","「がんばりましょう」.mp3")
        Bad = os.path.join("sound","警告.mp3")
        # 初期化
        pygame.init()
        pygame.mixer.init()
        # 必要ならディスプレイを設定してビデオシステムを初期化
        pygame.display.set_mode((1, 1))

        # サウンドファイルのロードと再生
        pygame.mixer.music.load(BGM)
        pygame.mixer.music.set_volume(0.5)  # ここで音量を0.2（20%）に設定
        pygame.mixer.music.play(-1)

        pre_value = 0
        sound_T, sound_type =[], []
        while True:
            if shared_line_color:
                color = shared_line_color[-1]
                # 現在のUTC時間を取得
                current_time_utc = datetime.utcfromtimestamp(time()).replace(tzinfo=pytz.utc)
                # 日本時間に変換
                japan_time = pytz.timezone('Asia/Tokyo')
                sound_time = current_time_utc.astimezone(japan_time)
                if color == 'r':
                    value = pygame.mixer.Sound(Excellent)
                    if color != pre_value:
                        pygame.mixer.Channel(1).play(value)
                        sound_T.append(sound_time)
                        sound_type.append('Excellent')
                elif color == 'y':
                    value = pygame.mixer.Sound(Good)
                    if color != pre_value:
                        pygame.mixer.Channel(1).play(value)
                        sound_T.append(sound_time)
                        sound_type.append('Good')
                elif color == 'g':
                    value = pygame.mixer.Sound(Poor)
                    if color != pre_value:
                        pygame.mixer.Channel(1).play(value)
                        sound_T.append(sound_time)
                        sound_type.append('Poor')
                elif color == 'b':
                    value = pygame.mixer.Sound(Bad)
                    pygame.mixer.Channel(1).play(value)
                    sound_T.append(sound_time)
                    sound_type.append('Bad')
                sleep(1)  # 1秒待機してから再度チェック
                pre_value = color
    except KeyboardInterrupt:
        pass
    finally:
        print("BGM切断")
        pygame.mixer.music.stop() #BGM切断
        csv_sound = os.path.join('Result','sound_result.csv')
        # BGMタイミング保存
        with open(csv_sound, mode='w', newline='') as file:
            # CSV ファイルを開いてヘッダーを書き込む
            csv_header = ["Timestamp", "Sound"]
            writer = csv.writer(file)
            writer.writerow(csv_header)
            # 各要素をCSVに書き込む
            for t, s in zip(sound_T, sound_type):
                writer.writerow([t, s])

if __name__ == "__main__":
    # マネージャーを作成して共有リストを作成
    manager = Manager()
    shared_line_color = manager.list()

    # 共有キューを作成
    queue = multiprocessing.Queue()
    fig = plt.figure()

    # プロセスを開始し、共有変数を渡す
    process1 = multiprocessing.Process(target=test, args=(queue,))
    process2 = multiprocessing.Process(target=draw, args=(queue, shared_line_color))
    process3 = multiprocessing.Process(target=sound, args=(shared_line_color,))

    process1.start()
    process3.start()
    sleep(30)  # process2はどうせ30s機能しない
    process2.start()

    # プロセスの終了を待つ
    process1.join()
    process2.join()
    process3.join()