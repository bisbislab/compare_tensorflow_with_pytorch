# 参考：https://imagingsolution.net/deep-learning/pytorch/pytorch_mnist_sample_program/

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# (A) パラメータ
NUM_EPOCHS = 10
NUM_BATCH = 100
LEARNING_RATE = 0.001
IMAGE_SIZE = 28 * 28

# GPU が使えるかどうか
# GPU が使える場合、後ほど to("cuda") でデータやモデルを GPU に転送する
device = "cuda" if torch.cuda.is_available() else "cpu"

# (B) データに対する前処理
# transforms.ToTensor() は PIL Image 型から PyTorch で扱える Tensor 型への変換を行い、スケールを [0, 255] から [0, 1] に変換する
# Python では、画像は PIL Image 型か OpenCV(Numpy) 型で扱うことが多い
# その他、リサイズを行う transforms.Resize()、ランダムに左右回転を行う transforms.RandomHorizontalFlip() など色々用意されている
transform = transforms.Compose([transforms.ToTensor()])

# (C) MNIST データセットの取得
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, transform=transform)

# (D) データローダーの作成
# これを for 文などに渡すことで、バッチ単位でデータを取得できるようになる
train_dataloader = DataLoader(train_dataset, batch_size=NUM_BATCH, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=NUM_BATCH)


# (E) モデル
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        """コンストラクタ

        Args:
            input_size (int): 入力サイズ
            output_size (int): 出力サイズ
        """
        super().__init__()

        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        """順伝搬

        Args:
            x (Tensor): 入力データ

        Returns:
            Tensor: 出力データ
        """
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# (F) モデルのインスタンス生成（＆GPU に転送）
model = Net(IMAGE_SIZE, 10).to(device)

# (G) 損失関数
criterion = nn.CrossEntropyLoss()

# (H) 最適化手法
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# モデルを訓練モードにする
# Dropout の有効/無効が切り替わったりする
model.train()

# (I) 学習
for epoch in range(NUM_EPOCHS):  # 設定エポック数だけ学習
    loss_sum = 0  # 1エポック分の Loss を蓄積

    for inputs, labels in train_dataloader:  # ミニバッチ単位で学習
        # データを（GPU が使えたら）GPU に転送
        inputs = inputs.to(device)
        labels = labels.to(device)

        # optimizer を初期化
        optimizer.zero_grad()

        inputs = inputs.view(-1, IMAGE_SIZE)  # 今回は画像を1次元データに変換
        outputs = model(inputs)  # 順伝搬

        # Loss の計算
        loss = criterion(outputs, labels)
        loss_sum += loss

        # 勾配の計算
        loss.backward()

        # 重みの更新
        optimizer.step()

    # 学習状況の表示
    print(f"Epoch: {epoch+1}/{NUM_EPOCHS}, Loss: {loss_sum.item() / len(train_dataloader)}")

    # (J) モデルの重みの保存
    save_dir = "model_weights/pytorch"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), f"{save_dir}/model_weights_epoch_{epoch+1:04d}.pth")

# モデルを評価モードにする
model.eval()

loss_sum = 0
correct = 0

# (K) テスト
with torch.no_grad():  # 逆伝搬しないので勾配計算を行わない
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        inputs = inputs.view(-1, IMAGE_SIZE)
        outputs = model(inputs)

        loss_sum += criterion(outputs, labels)

        pred = outputs.argmax(1)  # 予測確率の最も高いインデックスを取得
        correct += pred.eq(labels.view_as(pred)).sum().item()  # 正解数をカウント

# 結果の表示
print(
    f"Loss: {loss_sum.item() / len(test_dataloader)}, "
    f"Accuracy: {100*correct/len(test_dataset)}% ({correct}/{len(test_dataset)})"
)
