# 

Mnist を学習させるサンプル実装の TensorFlow と PyTorch の比較

## 実行環境

- Python: 3.10
- TensorFlow: 2.12.0
- PyTorch: 2.0.0

以下 venv か Docker のどちらかに従う
venv の場合 Python はバージョン指定せずホストのものを使用する（細かく合わせたい場合は pyenv の使用をおすすめ）

## venv（pipパッケージをプロジェクトごとに分ける）

### 環境構築

```sh
python3 -m venv venv
source venv/bin/activate
# 正しく仮想環境に入ると、シェルの先頭に `(venv)` が表示される
pip3 install -r requirements.txt
```

### 2度目以降の実行前準備

```sh
source venv/bin/activate
```

### 実行

```sh
python3 train_tensorflow.py
python3 train_pytorch.py
```

### 終了

```sh
deactivate  # 仮想環境から抜ける
```

## Docker

### 環境構築

1. `.env.template` をコピーし、`.env` ファイルを作成する
2. `USERNAME` にユーザー名、`USER_UID` に uid、`USER_GID` に gid を記述する（`id` コマンドで確認できる）
3. `docker compose up -d --build` コマンドでビルド＆コンテナ立ち上げ

### 2度目以降の実行前準備

```sh
docker compose start
```

### 実行

```sh
docker compose exec python python3 train_tensorflow.py
docker compose exec python python3 train_pytorch.py
```

### 終了

```sh
docker compose stop 
```

## TensorFlow と PyTorch の比較

- 画像のデータ構造
  - TensorFlow: [Batch, Height, Width, Channel] (channel last)
  - PyTorch: [Batch, Channel, Height, Width] (channel first)
- CPU/GPU の指定
  - TensorFlow: 通常、明示的に指定しない
  - PyTorch: `.to('cpu')`, `.to('cuda')` で明示的に指定
- データセットの管理（バッチ単位でのイテレータ化）
  - TensorFlow: `tensorflow.data.Dataset`
  - PyTorch: `torch.utils.data.DataLoader`
- モデル定義
  - TensorFlow: `tensorflow.keras.Model` クラスを継承するほか、Sequential API や Functional API の利用も多い
  - PyTorch: `torch.nn.Module` クラスを継承することが多い
- 学習経過の出力
  - TensorFlow: `fit()` メソッドを呼び出すと自動的に学習経過を表示
  - PyTorch: `print()` 等を用いて全て自力で書かないと何も表示されない
- その他参考サイト
  - https://logmi.jp/tech/articles/325685
