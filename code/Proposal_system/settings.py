# settings.py
# 定数設定ファイル

# サンプル数 マル5,バツ5の場合10
NUM_SAMPLES = 20

# MAC アドレス
MAC_ADRESS = ["EF:7D:EE:22:15:8B", # 1
              "F9:41:F3:3B:21:24", # 2
              "ED:60:85:42:09:A9", # 3
              "EB:7A:92:A7:79:D0", # 4
              "EB:0E:8D:A0:F6:8C"] # 5
SENSOR_FREQUENCY = 100

# 閾値の範囲 (mG)
MIN_THRE = 0.01
MAX_THRE = 0.2
STEP_THRE = 0.02

# 待機時間 (s)
WAITING_TIME = 2.0
TEST_TIME = 3000

# スライディングウィンドウのステップ数　(frame)
STEP_SIZE = 10
# 動き検出を行うフレーム数　ウィンドウの先頭から何フレーム目までで検出を行うか
STOP_FRAME = 5
# ↑のうち何フレームがTrueとなったときに動きありと判定するか
MOVE_FRAME = 1 

# 作業効率の時間感覚 (s)
EFFICIENCY_TIME = 30
EFFICIENCY_FRAME = int(EFFICIENCY_TIME*SENSOR_FREQUENCY/STEP_SIZE) # 300

# 作業効率の計算感覚 (frame)
EFFICIENCY_STEP = 20
EFFICIENCY_STEP_TIME = EFFICIENCY_STEP/SENSOR_FREQUENCY # 20フレームにかかる時間
FRAME_PER_S = int(1/EFFICIENCY_STEP_TIME) # 1sあたりのフレーム数
# 作業効率のマーク感覚 (frame) 連続何個の未記入が続くと別のマークとみなすか
MARK_STEP = 5

# 直近の作業効率 (s)
EFFICIENCY_NEAR = 60
EFFICIENCY_NEAR_FRAME = EFFICIENCY_NEAR*FRAME_PER_S

# ファイル構成

# acc_train - acc_data_training_マル1.csv
#           - acc_data_training_マル2.csv

# acc_train_extract - acc_data_training_ex_マル1.csv
#                   - acc_data_training_ex_マル2.csv

# static - img - マル1 - threshold.csv
#                     - マル1_0.5.png
#                     - マル1_1.0.png
#              - マル2 - threshold.csv
#                     - マル2_0.5.png
#                     - マル2_1.0.png
if __name__ == "__main__":
    # print(EFFICIENCY_NEAR_FRAME)
    # print("Hellow")