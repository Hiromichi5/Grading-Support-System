from settings import NUM_SAMPLES,MAC_ADRESS,TEST_TIME,STOP_FRAME,MOVE_FRAME,STEP_SIZE,STEP_E,EFFICIENCY_STEP,EFFICIENCY_STEP_TIME,EFFICIENCY_TIME,EFFICIENCY_FRAME,EFFICIENCY_NEAR_FRAME
import cv2
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Manager

from keras.models import load_model
from PIL import Image
from time import sleep,time
# import csv
# from datetime import datetime, timedelta
# import pytz
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import japanize_matplotlib
import pygame
import sys
import os

# 閾値設定
brightness_threshold = 150  # 明度が200以上で「白っぽい」と判定
saturation_threshold = 60  # 彩度が60以下で「白っぽい」と判定
white_ratio_threshold = 0.7  # 白っぽいピクセルが画像の8割を超える場合

# 丸判定用オートエンコーダの読み込み
model_save_path_maru = 'autoencoder_model_maru.keras'
autoencoder_maru = load_model(model_save_path_maru)
print(f'{model_save_path_maru} をロードしました。')

# 罰判定用オートエンコーダの読み込み
model_save_path_batsu = 'autoencoder_model_batsu.keras'
autoencoder_batsu = load_model(model_save_path_batsu)
print(f'{model_save_path_batsu} をロードしました。')

# 閾値設定
threshold_maru = 0.2
threshold_batsu = 0.1

img_maru_path = os.path.join("img","正解_枠.png")
img_batu_path = os.path.join("img","不正解_枠.png")
img_anknown_path = os.path.join("img","？_枠.png")

img_maru = Image.open(img_maru_path)  # 上に重ねる画像B
img_batu = Image.open(img_batu_path)  # 上に重ねる画像C
img_ankown = Image.open(img_anknown_path)  # 上に重ねる画像C

# フォルダ作成
if not os.path.exists('test'):
    os.mkdir('test')
if not os.path.exists('test/color'):
    os.mkdir('test/color')


color_list = []
p = 64 # いっぺんのピクセルサイズ

# 異常判定関数
def detect_anomaly(original_image, autoencoder, threshold):
    original_image = cv2.resize(original_image, (p, p)).astype('float32') / 255.0  # 正規化
    original_image = np.expand_dims(original_image, axis=-1)  # チャンネル次元を追加
    reconstructed_image = autoencoder.predict(np.expand_dims(original_image, axis=0))
    mse = np.mean(np.power(original_image - reconstructed_image[0], 2))
    # print(f"再構成誤差: {mse}, 閾値: {threshold}, 判定: {'正常' if mse < threshold else '異常'}")  # デバッグ用出力
    return mse, mse < threshold

# image_queue から result_queue
def combined_anomaly_detection(image_queue, result_queue):
    while True:
        image_data = image_queue.get()
        if image_data is None:
            break
        test_img, x_center, y_center, frame_count = image_data

        # 丸・バツそれぞれの再構成誤差と異常判定を取得
        mse_maru, is_anomaly_maru = detect_anomaly(test_img, autoencoder_maru, threshold_maru)
        mse_batsu, is_anomaly_batsu = detect_anomaly(test_img, autoencoder_batsu, threshold_batsu)
        
        # 判定ロジックの分岐
        if is_anomaly_maru and not is_anomaly_batsu:
            final_result = "正解"
        elif not is_anomaly_maru and is_anomaly_batsu:
            final_result = "不正解"
        elif not is_anomaly_maru and not is_anomaly_batsu:
            final_result = "異常"
        else:
            final_result = "異常"

        # 各判定の結果を表示
        # print(f'丸判定 - 再構成誤差: {mse_maru:.4f}, 判定: {"正常" if is_anomaly_maru else "異常"}')
        # print(f'罰判定 - 再構成誤差: {mse_batsu:.4f}, 判定: {"正常" if is_anomaly_batsu else "異常"}')
        # print(f'最終判定: {final_result}')

        result_queue.put((final_result, x_center, y_center, frame_count))
        
        # return final_result

# 赤色の範囲を指定
lower_red1 = np.array([0, 50, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 100])
upper_red2 = np.array([180, 255, 255])

# フレームから画像を処理
def save_images_from_frame(frame_queue, image_queue):
    while True:
        frame_data = frame_queue.get()
        if frame_data is None:
            break
        
        frame, frame_count, timestamp = frame_data
        frame_number = f"{frame_count}_{int(timestamp)}"

        # フレームをBGRからHSVに変換
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 赤色のマスクを作成（2つの範囲をマスクで結合）
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask = mask1 + mask2

        # モルフォロジー演算でマスクを膨張処理
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)  # まず膨張させる

        # 赤色部分の輪郭を検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 順に処理
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 100:
                # 外接矩形を取得
                x, y, w, h = cv2.boundingRect(contour)

                # 正方形のサイズを計算
                side = max(w, h)

                # 画像外にはみ出ないように正方形領域を調整
                x_center, y_center = x + w // 2, y + h // 2
                x_square = max(0, x_center - side // 2)
                y_square = max(0, y_center - side // 2)
                x_square_end = min(mask.shape[1], x_square + side)
                y_square_end = min(mask.shape[0], y_square + side)

                # 正方形領域を抽出
                square_region = mask[y_square:y_square_end, x_square:x_square_end]

                # カラー画像から対応する領域を抽出
                color_region = frame[y_square:y_square_end, x_square:x_square_end]

                # エッジ検出の追加
                edges = cv2.Canny(square_region, 100, 200)
                edge_count = cv2.countNonZero(edges)

                # エッジが少ない場合はスキップ

                if edge_count < 50:  # この閾値は調整可能
                    print(f"エッジが少ないためスキップ: test{frame_number}_{i+1}")
                    continue

                # 元の正方形のサイズ
                original_size = side

                # 拡大縮小率を計算
                scale_factor =  p / original_size

                # 拡大縮小しすぎを排除
                if 0.3 <= scale_factor <= 2.5:
                    # リサイズ前の白い点（ピクセル値が255）の数を数える
                    white_pixel_count = cv2.countNonZero(square_region)
                    if white_pixel_count > 200:
                        # 正方形領域をp x pにリサイズ
                        resized_binary_square = cv2.resize(square_region, (p, p))
                        resized_color_square = cv2.resize(color_region, (p, p))
                        # HSV空間に変換
                        hsv_square = cv2.cvtColor(resized_color_square, cv2.COLOR_BGR2HSV)

                        # 白っぽいピクセルのマスクを生成
                        white_mask = (hsv_square[:, :, 2] > brightness_threshold) & (hsv_square[:, :, 1] < saturation_threshold)

                        # 白っぽいピクセルの割合を計算
                        white_pixel_ratio = np.sum(white_mask) / (p * p)

                        # 白っぽいピクセルが8割以上なら次の処理へ
                        if white_pixel_ratio > white_ratio_threshold:
                            # ファイルに保存（左から右の順に番号付け）
                            output_filename = f'test/test{frame_number}_{i+1}.jpg'
                            cv2.imwrite(output_filename, resized_binary_square)

                            # カラー画像を保存
                            output_filename_color = f'test/color/test{frame_number}_{i+1}.jpg'
                            cv2.imwrite(output_filename_color, resized_color_square)
                            # print(f'【test{frame_number}_{i+1}.jpg】面積:{area},エッジ:{edge_count},拡大率:{scale_factor},白:{white_pixel_count}')

                            image_queue.put((resized_binary_square, x_center, y_center, frame_count)) 


        print("保存完了")

def img_print(x, y, result,base):
    if result == "正解":
        tenpu = Image.open(img_maru)
    elif result == "不正解":
        tenpu = Image.open(img_batu)
    elif result == "異常":
        tenpu = Image.open(img_anknown)
    """maru.pngの中心を指定したx, y座標と一致させる"""
    # maru.pngのサイズ取得
    width, height = tenpu.size

    # 座標を調整（画像の中心がx, yに一致するようにする）
    adjusted_x = x - width // 2
    adjusted_y = y - height // 2

    # 画像を重ねる
    return base.paste(tenpu, (adjusted_x, adjusted_y), tenpu)

def count_result(result_queue,E_queue):
    identifier = 0.1
    counter = 0
    while True:
        result_data = result_queue.get()
        if result_data is None:
            break
        final_result, x_center, y_center, frame_count = result_data
        if identifier == frame_count:
            counter += 1
        else:
            identifier = frame_count
            number = counter
            counter = 0
            E_queue.put(number)
            print(number)
    # frame = img_print(x_center,y_center,final_result,frame)

def a(E_queue):
    while True:
        a_data = E_queue.get()
        if a_data is None:
            break
        print(a_data)

def draw(E_queue, shared_line_color):
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
        E = [0]
        while True:
            if not E_queue.empty():
                if count_voice == 0:
                    pygame.mixer.Channel(1).play(support)
                    count_voice = 1
                value = E_queue.get()
                E.append(value)
            update_graph(E,fig,ax1,ax2,shared_line_color)
                # else:
                #     color_make(E,shared_line_color)
    except KeyboardInterrupt:
        pass
    # finally:
    #     print("作業効率保存")
    #     csv_Eff = os.path.join('Result','eff_result.csv')
    #     # BGMタイミング保存
    #     with open(csv_Eff, mode='w', newline='') as file:
    #         # CSV ファイルを開いてヘッダーを書き込む
    #         csv_header = ["Efficiency","Color"]
    #         writer = csv.writer(file)
    #         writer.writerow(csv_header)
    #         # 各要素をCSVに書き込む
    #         for e,c in zip(E,color_list):
    #             writer.writerow([e,c])

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

    ax2.text(0, 0.6, "作業効率：" + str(E_current), size=30)
    if mean_near != 1000:
        ax2.text(0, 0.4, "直近の平均：" + str(round(mean_near,2)), size=30)
    else:
        ax2.text(0, 0.4, "直近の平均：---", size=30)
    ax2.text(0, 0.2, "全体の平均：" + (str(round(mean, 2)) ), size=30)
    ax2.axis("off")

    # 左側
    width = min(len(E), 300)
    start_time = 30
    end_time = start_time + (len(E)-1)* 0.1
    step = 0.1  # ステップの幅
    # 配列の生成
    Time = [round(end_time - (width-1-i) * step,2) for i in range(width)]

    # グラフ作成
    ax1.clear()
    color_299 = line_color[-299:] # マジックナンバーだめ！
    for i in range(width-2):
        ax1.plot(Time[i:i+2], E[-width+i:-width+i+2], color=color_299[i])

    i = width-2
    ax1.plot(Time[i:i+2], E[-width+i:], color=color_299[i])
    ax1.axhline(y=np.mean(E), color='g', linestyle='--', label='平均')
    ax1.legend()
    ax1.set_title('作業効率の変化')
    ax1.set_xlabel('時間（秒）')

    fig.canvas.draw_idle()
    plt.pause(0.001)  # グラフを再描画

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

# メイン処理
def main():
    # マネージャーを作成して共有リストを作成
    manager = Manager()
    shared_line_color = manager.list()

    frame_queue = multiprocessing.Queue()
    image_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    E_queue = multiprocessing.Queue()

    # 別プロセスで1秒ごとのフレーム保存・処理
    draw_process = multiprocessing.Process(target=draw, args=(E_queue,shared_line_color))
    save_process = multiprocessing.Process(target=save_images_from_frame, args=(frame_queue,image_queue))
    detect_process = multiprocessing.Process(target=combined_anomaly_detection, args=(image_queue,result_queue))
    count_process = multiprocessing.Process(target=count_result, args=(result_queue,E_queue))


    save_process.start()
    detect_process.start()
    count_process.start()
    draw_process.start()

    cap = cv2.VideoCapture(1)
    frame_count = 0
    last_save_time = time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("カメラ映像の読み込みに失敗しました")
            break
        
        # フレームを90度時計回りに回転
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        current_time = time()
        if current_time - last_save_time >= 1:
            frame_queue.put((frame, frame_count, current_time))
            last_save_time = current_time
            
        # result_data = result_queue.get()
        # if result_data is None:
        #     break
        # final_result, x_center, y_center, frame_count = result_data
        # frame = img_print(x_center,y_center,final_result,frame)
        cv2.imshow('Camera Output', frame)
        frame_count += 1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    frame_queue.put(None)  # 終了信号を送信
    save_process.join()
    detect_process.join()
    count_process.join()
    draw_process.join()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
