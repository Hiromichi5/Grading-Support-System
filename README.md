# 📘 テスト採点支援システム(grading-aid-system) – README

## 🧾 概要

本プロジェクトは、**加速度センサを搭載したペン**を利用して、テスト採点作業をリアルタイムに解析・支援するシステムです。  
採点者の「○」「×」などの動作を自動で検出し、**作業効率の可視化と警告通知**を通じて、集中力の維持や作業負荷の軽減を目的としています。

---

## 🚀 特徴

- ✍️ **加速度センサデータの解析による動作認識**
- 📊 **採点効率の数値化・可視化（ヒストグラム生成）**
- 🔔 **集中力の低下を検出し、視覚・聴覚でフィードバック**

---

## 📁 ディレクトリ構成
    grading-aid-system/ 
    ├── README.md                         # 本ドキュメント 
    ├── code/                             # コード一式 
    │ ├── Graph/                          # 作業効率のヒストグラム生成 
    │ │ ├── histgram.py                   # 効率グラフ生成スクリプト 
    │ │ └── myenv/                        # Python仮想環境（venv等） 
    │ └── Proposal_system/                # 支援システム本体（リアルタイム処理） 
    ├── result/                           # ヒストグラム画像などの出力結果 
    │ ├── result_A.jpg 
    │ ├── ... 
    │ └── subject_O/ 
    ├── row_data/                         # 各被験者の加速度データ 
    │ ├── subject_A/ 
    │ │ ├──row_data.csv                   # 生データ（加速度）
    │ │ ├──refficiency_data_support.csv   # 作業効率ヒストグラム（支援あり）
    │ │ └──refficiency_data_nosupport.csv # 作業効率ヒストグラム（支援なし）
    │ ├── ... 
    │ └── subject_O/ 
    └── requirements.txt                  # 必要なPythonライブラリ一覧


---

## 📄 フォルダ・ファイルの説明

### `code/Graph/`
- `histgram.py`: 被験者ごとの作業効率をヒストグラムとして可視化するスクリプトです。
- `myenv/`: 実行環境用の仮想環境（venv）フォルダ。

### `code/Proposal_system/`
- 加速度センサデータをリアルタイムで処理し、採点動作を自動判定。
- 作業効率を定量化し、視覚/聴覚でフィードバックを提供。

### `result/`
- `histgram.py` 実行後に出力されるヒストグラム画像を格納します。
- 各被験者の結果画像（例：`result_A.jpg` など）が保存されます。

### `row_data/`
- 被験者の生データ（加速度データCSVなど）を格納しています。
- `subject_A/` ～ `subject_O/` の各フォルダに分かれています。

### `requirements.txt`
- 本プロジェクトで必要なPythonライブラリの一覧です。
- `pip install -r requirements.txt` で一括インストールが可能です。

---

## 🚀 実行手順（それぞれのフォルダ内のREADMEファイルにて説明あり）

### 1. 🛠 仮想環境のセットアップ

1. 仮想環境を作成します（初回のみ）：
    ```bash
    python -m venv myenv
    ```

2. 仮想環境を有効にします：

    - Mac/Linux：
      ```bash
      source myenv/bin/activate
      ```

    - Windows：
      ```bash
      myenv\Scripts\activate
      ```

3. 依存ライブラリをインストールします：
    ```bash
    pip install -r requirements.txt
    ```

> ※ Python 3.8 〜 3.10 あたりを推奨します。

[Proposal_system（提案システム）の説明はこちら](./code/Proposal_system/README.md)

[Graph（結果の出力）の説明はこちら](./code/Graph/README.md)

[row_dataの説明はこちら](./row_data/README.md)
