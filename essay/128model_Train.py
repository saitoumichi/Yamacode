# ============================================================
# 128model_Train.py
# ------------------------------------------------------------
# このスクリプトは，3次元胸部CT画像を用いて VoxelMorph 系の
# 位置合わせモデルを学習するためのコードです。
# 
# 大きく 2 段階の学習を行っています。
# 1. 人工的に作った変形場を使った事前学習
#    - 正解の変形場が分かる状態で，変形ベクトル場の予測を学ぶ
# 2. 実際の画像ペアを用いた追加学習
#    - 画像同士が似るように位置合わせを学ぶ
# 
# 主な流れは次の通りです。
# 1. CT データを読み込んでサイズをそろえる
# 2. moving / fixed 画像ペアを作るジェネレータを準備する
# 3. 事前学習で人工変形場を使って学習する
# 4. 学習済み重みを初期値にして追加学習する
# 5. 学習済みモデルを保存する
# 
# このコードは「学習用」のスクリプトであり，評価や可視化だけを
# 行うテストコードとは役割が異なります。
# ============================================================

# 標準ライブラリ
import os
import os, sys
# 数値計算・深層学習関連
import numpy as np  # 配列処理や数値計算に使う
import torch  # PyTorch 本体
import torch.nn as nn  # ニューラルネットワークの層やモジュール
import torch.nn.functional as F  # 活性化関数や畳み込み演算など
from torch.nn import Sequential  # 層を順番に積み重ねるクラス
import torch.optim as optim  # 最適化アルゴリズム
# VoxelMorph / 医用画像処理関連
import voxelmorph as vxm  # VoxelMorph のライブラリ
import neurite as ne  # 医用画像処理関連の補助ライブラリ
import scipy.ndimage  # 画像の補間や平滑化に使う

# VoxelMorph のバックエンドとして PyTorch を使うように指定する
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ.get('VXM_BACKEND')

# GPU が使えれば GPU を，使えなければ CPU を使う
# この device を使って，モデルやテンソルを同じ計算環境に載せる
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 実際にどの計算環境が選ばれたかを表示する
print(device)

# =====================
# 学習データの読み込みと前処理
# =====================
# 学習用の CT ボリュームを .npz ファイルから読み込む
x_train = np.load('D:\Saito\Data\TrainData_NoBed.npz')['Train']

# 軸の並びを入れ替えて，先頭次元が「症例番号」になる形にそろえる
# これにより x_train[0] が 1症例分の 3D 画像になる
x_train = np.transpose(x_train, (3, 0, 1, 2))

# 元の 3D 画像は大きいため，各軸を半分にしてメモリ負荷を下げる
# ここで縮小後のサイズを計算する
new_shape = tuple([dim // 2 for dim in x_train.shape[1:]])

# 全症例を順番に補間して縮小する
x_train_resized = np.zeros((x_train.shape[0], *new_shape))  # 新しい形に合わせて初期化

for i in range(x_train.shape[0]):
    # 1症例ずつ 3次元補間で縮小する
    x_train_resized[i] = scipy.ndimage.zoom(x_train[i], (0.5, 0.5, 0.5), order=3)
# 縮小後の 1症例あたりのサイズと，全体のデータ数を確認する
print('Resized train vol_shape:', x_train_resized.shape[1:])
print('Resized train shape:', x_train_resized.shape)

# 縮小後の画像を以後の学習データとして使う
x_train = x_train_resized

# =====================
# 学習用データジェネレータ
# =====================
# 学習時に moving / fixed の画像ペアをランダムに作る関数
def vxm_data_generator(x_data, batch_size):
    vol_shape = x_data.shape[1:]  # データ形状を取得
    # 1症例あたりの 3D ボリュームサイズを表す
    ndims = len(vol_shape)
    
    # 変形なしを表すゼロの変形場を用意する
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # バッチサイズ分だけランダムに症例を選び，moving 画像を作る
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        # 別にランダム抽出して fixed 画像を作る
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]

        # PyTorch で扱いやすい (B, C, D, H, W) 形式に並べ替える
        moving_images = torch.tensor(moving_images).permute(0, 4, 1, 2, 3).float()
        fixed_images = torch.tensor(fixed_images).permute(0, 4, 1, 2, 3).float()
        # モデル入力と，学習時に参照する出力側のターゲットを作る
        inputs = [moving_images, fixed_images]
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)

# 実際に 1バッチ取り出して，入力と出力の形が想定通りか確認する
train_generator = vxm_data_generator(x_train, batch_size=2)
in_sample, out_sample = next(train_generator)

# in_sampleとout_sampleの内容を確認する
print("入力サンプルの形状:")
print("Moving画像の形状:", in_sample[0].shape)
print("Fixed画像の形状:", in_sample[1].shape)

print("\n出力サンプルの形状:")
print("変形後画像（学習ターゲット）の形状:", out_sample[0].shape)
print("ゼロ変形場の形状:", out_sample[1].shape)

# =====================
# 損失関数
# =====================
# 画像同士の差と，変形の滑らかさに関する損失を定義する
mse_loss = vxm.losses.MSE().loss
# Grad 損失は，変形場が急激に変わりすぎないようにするための正則化
grad_loss = vxm.losses.Grad('l2').loss

# MSE と Grad を合わせた総損失
def total_loss(y_true, y_pred):
    mse = mse_loss(y_true, y_pred)
    grad = grad_loss(y_true, y_pred)
    return mse + 0.01 * grad, mse, grad

# MSE だけを取り出して使いたいときの補助関数
def MSE_Loss(y_true, y_pred):
    mse = mse_loss(y_true, y_pred)
    return mse

# 局所正規化相互相関（LNCC）の損失関数
# このコードでは最終的に使っていないが，候補として残されている
def lncc_loss(I, J, window=9, eps=1e-5):
    # I, J は (B, 1, D, H, W) 形状の 3次元画像バッチを想定する
    padding = window // 2
    weight = torch.ones(1, 1, window, window, window, device=I.device)

    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, weight, padding=padding)
    J_sum = F.conv3d(J, weight, padding=padding)
    I2_sum = F.conv3d(I2, weight, padding=padding)
    J2_sum = F.conv3d(J2, weight, padding=padding)
    IJ_sum = F.conv3d(IJ, weight, padding=padding)

    win_size = window ** 3
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    lncc = cross * cross / (I_var * J_var + eps)
    return -torch.mean(lncc)  # LNCC を大きくしたいので，損失としてはマイナスを付けて最小化する

# =====================
# モデル設定
# =====================
# 3次元画像を扱うので次元数は 3
# moving と fixed を合わせて扱う位置合わせモデルを構成する
ndim = 3
unet_input_features = 2

# U-Net の encoder / decoder で使うチャネル数
nb_features = [
    [32, 64, 64, 64, 64],
    [64, 64, 64, 64, 64, 32, 16, 16]
]

# VoxelMorph 系の 3D モデルを作成する
# 入力サイズは (64, 128, 128)
model3D = vxm.networks.VxmDense1((64, 128, 128), nb_features, int_steps=0)
model3D.to(device)
optimizer = optim.Adam(model3D.parameters(), lr=1e-4)

# 画像再現損失と変形場損失の重み
A = 100
B = 0.01

# 学習の進み具合を notebook 上で見やすくするためのライブラリ
from tqdm.notebook import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt

# Spatial Transformer は，変形場を使って画像を実際にワープする層
transformer = vxm.layers.SpatialTransformer((64, 128, 128)).to(device)

# 人工的に作った変形場を滑らかにして，不自然なギザギザを減らす関数
def gaussian_smooth_3d(tensor, kernel_size=5, sigma=1.0):
    """3次元ガウシアンフィルタで変形場をなめらかにする。"""
    # scipy のガウシアンフィルタを使って平滑化する
    from scipy.ndimage import gaussian_filter
    tensor_np = tensor.cpu().numpy()
    # バッチ方向・チャネル方向にはフィルタをかけない
    smoothed_np = gaussian_filter(tensor_np, sigma=[0, 0, sigma, sigma, sigma])
    return torch.tensor(smoothed_np, dtype=torch.float32, device=tensor.device)

# =====================
# 第1段階: 人工変形場を使った事前学習
# =====================
epochs = 40000
# 最初は小さい変形だけを学ばせ，あとで徐々に大きな変形にする
shift_range = 1

# 損失の推移をあとで確認できるように記録しておく
losses = []
loss_vecs = []
loss_images = []
loss_hightVecs = []

# 事前学習のメインループ
for epoch in tqdm(range(epochs)):
    # 一定エポックごとに人工変形の大きさを増やして，学習難易度を上げる
    if epoch % 1000 == 0 and epoch > 0:
        print(f"Epoch {epoch}: 人工変形の大きさを ±{shift_range} ボクセルに増やします。")
        shift_range += 1

    # 取り出した moving 画像を Tensor にして device に載せる
    train_batch, _ = next(train_generator)
    moving_images = torch.tensor(train_batch[0], dtype=torch.float32).to(device)

    # ひとつ前の更新でたまった勾配をリセットする
    optimizer.zero_grad()

    # まず粗い解像度でランダムな変形場を作る
    B, D, H, W = 2, 8, 16, 16  # バッチサイズと画像の次元

    # x, y, z の3方向にランダムな変位を与える
    displacement_field = (torch.rand((B, 3, D, H, W), dtype=torch.float32) * 2 - 1) * shift_range
    displacement_field = displacement_field.to(device)

    # 変形場を滑らかにしてから，画像サイズまで拡大する
    displacement_field = gaussian_smooth_3d(displacement_field, sigma=2.0)
    displacement_field = torch.nn.functional.interpolate(displacement_field, size=(64,128,128), mode='trilinear', align_corners=False)

    # 正解となる人工変形画像を作る
    moving_images2 = transformer(moving_images, displacement_field)

    # モデルに元画像と人工変形後画像を入力し，位置合わせ結果と変形場を予測させる
    transformed_image, Vec = model3D(moving_images, moving_images2)

    # 予測した変形場が，人工的に作った正解変形場に近いかを評価する
    loss_vec = MSE_Loss(displacement_field, Vec) * B
    # 変形後画像が，人工的に作った正解画像に近いかを評価する
    loss_image = MSE_Loss(moving_images2, transformed_image) * A

    # 2つの損失を合わせて最終的な学習目標にする
    loss = loss_vec + loss_image

    # 逆伝播してパラメータを更新する
    loss.backward()
    optimizer.step()

    # 一定間隔で学習済み重みを保存する
    if (epoch + 1) % 100 == 0:
        torch.save(
            model3D.state_dict(),
            f'a.pth'
        )
        
    # 可視化用に損失を保存する
    losses.append(loss.cpu().item())
    loss_vecs.append(loss_vec.cpu().item())
    loss_images.append(loss_image.cpu().item())

    # notebook 上で損失の推移を確認する
    if epoch % 10 == 0:
        clear_output(wait=True)  # 出力をリフレッシュ
        plt.figure(figsize=(10,5))
        plt.plot(losses, label='総損失')
        plt.plot(loss_vecs, label='変形場損失')
        plt.plot(loss_images, label='画像損失')
        plt.xlabel("エポック")
        plt.ylabel("損失")
        plt.title("学習中の損失の推移")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # エポックごとのロスの表示
    print(f"Epoch {epoch+1}/{epochs}, 総損失: {loss:.4f}, 変形場損失: {loss_vec:.4f}, 画像損失: {loss_image:.4f}, 変形幅: ±{shift_range} ボクセル")

# =====================
# 第2段階: 実画像ペアを使った追加学習
# =====================
# 第1段階で保存した重みを読み込んで初期値にする
model3D = vxm.networks.VxmDense1((64, 128, 128), nb_features, int_steps=0)
model3D.load_state_dict(torch.load('a.pth', map_location=device))
model3D.to(device)
optimizer = optim.Adam(model3D.parameters(), lr=1e-4)

# 第2段階では，人工変形を使わずに画像類似度ベースで学習する
epochs = 10000

# 第2段階でも損失の推移を記録する
losses = []

# 追加学習のメインループ
for epoch in tqdm(range(epochs)):

    # moving / fixed を Tensor にして device に載せる
    train_batch, _ = next(train_generator)
    moving_images = torch.tensor(train_batch[0], dtype=torch.float32).to(device)
    fixed_images = torch.tensor(train_batch[1], dtype=torch.float32).to(device)

    # 勾配を初期化する
    optimizer.zero_grad()

    # moving を fixed に合わせるように推論する
    transformed_image, Vec = model3D(moving_images, fixed_images)
    # 変形後画像と fixed 画像の差を損失として計算する
    loss = MSE_Loss(fixed_images, transformed_image)

    # 逆伝播して重みを更新する
    loss.backward()
    optimizer.step()

    # 現時点のモデル重みを保存する
    torch.save(model3D.state_dict(), 'a2.pth')

    # 損失を記録して，学習の進み具合を確認できるようにする
    losses.append(loss.cpu().item())
    
    # エポックごとのロスの表示
    print(f"Epoch {epoch+1}/{epochs}, 損失: {loss:.4f}")
