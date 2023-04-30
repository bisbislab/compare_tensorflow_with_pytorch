import os

import tensorflow as tf
from tensorflow.keras.layers import Dense, Rescaling
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

# (A) パラメータ
NUM_EPOCHS = 10
NUM_BATCH = 100
LEARNING_RATE = 0.001
IMAGE_SIZE = 28 * 28


# (B) データに対する前処理
def transform(image):
    normalization = Rescaling(1.0 / 255)

    image = tf.reshape(image, (-1, IMAGE_SIZE))
    image = normalization(image)
    return image


# (C) MNIST データセットの取得
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# (D) データローダーの作成
# 変数名が紛らわしいが PyTorch 実装での train_dataloader/test_dataloader に対応
train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(len(X_train))
    .batch(NUM_BATCH)
    .map(lambda x, y: (transform(x), y))
)
test_dataset = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(NUM_BATCH)
    .map(lambda x, y: (transform(x), y))
)


# (E) モデル
class Net(tf.keras.Model):
    def __init__(self, input_size, output_size):
        """コンストラクタ

        Args:
            input_size (int): 入力サイズ
            output_size (int): 出力サイズ
        """
        super().__init__()

        self.fc1 = Dense(100, activation="sigmoid", input_shape=(input_size,))
        self.fc2 = Dense(output_size, activation="softmax")

    def call(self, x):
        """順伝搬

        Args:
            x (Tensor): 入力データ

        Returns:
            Tensor: 出力データ
        """
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# (F) モデルのインスタンス生成
model = Net(IMAGE_SIZE, 10)

# (G) 損失関数
criterion = SparseCategoricalCrossentropy()

# (H) 最適化手法
optimizer = Adam(learning_rate=LEARNING_RATE)

# コンパイル
model.compile(optimizer=optimizer, loss=criterion, metrics="acc")

# (J) モデルの重みの保存
# callback で毎エポックの最後に呼び出す
save_dir = "model_weights/tensorflow"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_weights = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_dir + "/model_weights_epoch_{epoch:04d}.ckpt", save_weights_only=True
)

# (I) 学習
model.fit(train_dataset, epochs=NUM_EPOCHS, batch_size=NUM_BATCH, callbacks=[save_weights])

# (K) テスト
model.evaluate(test_dataset)
