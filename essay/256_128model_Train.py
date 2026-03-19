# ============================================================
# 256_128model_Train.py
# ------------------------------------------------------------
# このスクリプトは，3次元胸部CT画像を用いて VoxelMorph 系モデルを
# 学習するためのコードです。
# 
# この版では，入力画像に 3次元 Haar wavelet 変換を適用し，
# 8個の周波数成分に分解してから変形場を推定します。
# その後，各成分を個別にワープし，逆 wavelet 変換によって
# 元の画像空間へ戻して学習を進めます。
# 
# 大きく 2 段階の学習を行っています。
# 1. 人工的に作った変形場を使った事前学習
#    - 正解の変形場が分かる状態で，変形場の予測を学ぶ
# 2. 実際の画像ペアを用いた追加学習
#    - 画像同士が似るように位置合わせを学ぶ
# 
# 主な流れは次の通りです。
# 1. CT データを読み込む
# 2. moving / fixed の画像ペアを作る
# 3. Haar wavelet で 8成分に分解する
# 4. 事前学習で人工変形画像を使って学習する
# 5. 逆 wavelet 変換で再構成した画像に対して損失を計算する
# 6. 追加学習で実画像ペアに合わせる
# ============================================================

# 標準ライブラリ
import os
import os, sys
# 数値計算・深層学習関連
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
import torch.optim as optim
# VoxelMorph / 医用画像処理関連
import voxelmorph as vxm
import neurite as ne
import scipy.ndimage

# VoxelMorph のバックエンドとして PyTorch を使うように指定する
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ.get('VXM_BACKEND')

# GPU が使えれば GPU を，使えなければ CPU を使う
# この device を使って，モデルやテンソルを同じ計算環境に載せる
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 実際にどの計算環境が選ばれたかを表示する
print(device)

# =====================
# 学習データの読み込み
# =====================
# 学習用の CT ボリュームを .npz ファイルから読み込む
x_train = np.load('CT_Train_NoBed.npz')['Train']

# 軸の並びを入れ替えて，先頭次元が「症例番号」になる形にそろえる
# これにより x_train[0] が 1症例分の 3D 画像になる
x_train = np.transpose(x_train, (3, 0, 1, 2))

# 1症例あたりのサイズと，全体のデータ数を確認する
print('Resized train vol_shape:', x_train.shape[1:])
print('Resized train shape:', x_train.shape)

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

        # モデル入力と，学習時に参照する出力側ターゲットを作る
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
# 画像同士の差や変形の滑らかさを評価するための損失を定義する
mse_loss = vxm.losses.MSE().loss

# Grad 損失は，変形場が急激に変わりすぎないようにするための正則化
grad_loss = vxm.losses.Grad('l2').loss

# MSE と Grad を合わせた総損失
def total_loss(y_true, y_pred):
    mse = mse_loss(y_true, y_pred)
    grad = grad_loss(y_true, y_pred)
    return mse + 0.01 * grad, mse, grad
#     return mse_loss(y_true, y_pred)

# MSE だけを取り出して使いたいときの補助関数
def MSE_Loss(y_true, y_pred):
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    mse = mse_loss(y_true, y_pred)
    return mse

# 局所正規化相互相関（LNCC）の損失関数
# このコードでは候補として残されている
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

# Wavelet 変換後の解像度に対応した VoxelMorph 系モデルを作成する
# 入力サイズは (128, 256, 256)
model3D = vxm.networks.VxmDense_128_256_128((128, 256, 256), nb_features, int_steps=0)
model3D.to(device)
optimizer = optim.Adam(model3D.parameters(), lr=1e-4)

# Spatial Transformer は，変形場を使って画像をワープする層
# wavelet 後サイズ用と元画像サイズ用の 2種類を用意している
transformer = vxm.layers.SpatialTransformer((64, 128, 128)).to(device)
transformer256 = vxm.layers.SpatialTransformer((128, 256, 256)).to(device)

# =====================
# 3次元 Haar wavelet 変換
# =====================
# 入力画像を 8個の周波数成分に分解する
def haar_wavelet_3d(x):
    """
    x : torch.Tensor, 形状は (B, 1, D, H, W)
    戻り値 : torch.Tensor, 形状は (B, 8, D/2, H/2, W/2)
    """

    # 偶数・奇数インデックス
    x000 = x[:, :, 0::2, 0::2, 0::2]
    x001 = x[:, :, 0::2, 0::2, 1::2]
    x010 = x[:, :, 0::2, 1::2, 0::2]
    x011 = x[:, :, 0::2, 1::2, 1::2]
    x100 = x[:, :, 1::2, 0::2, 0::2]
    x101 = x[:, :, 1::2, 0::2, 1::2]
    x110 = x[:, :, 1::2, 1::2, 0::2]
    x111 = x[:, :, 1::2, 1::2, 1::2]

    # Haar 係数（正規化つき）
    LLL = (x000 + x001 + x010 + x011 + x100 + x101 + x110 + x111) / 8
    LLH = (x000 - x001 + x010 - x011 + x100 - x101 + x110 - x111) / 8
    LHL = (x000 + x001 - x010 - x011 + x100 + x101 - x110 - x111) / 8
    LHH = (x000 - x001 - x010 + x011 + x100 - x101 - x110 + x111) / 8
    HLL = (x000 + x001 + x010 + x011 - x100 - x101 - x110 - x111) / 8
    HLH = (x000 - x001 + x010 - x011 - x100 + x101 - x110 + x111) / 8
    HHL = (x000 + x001 - x010 - x011 - x100 - x101 + x110 + x111) / 8
    HHH = (x000 - x001 - x010 + x011 - x100 + x101 + x110 - x111) / 8

    return torch.cat(
        [LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH],
        dim=1
    )

# 8個の wavelet 成分から元の 3次元画像を再構成する
def inverse_haar_wavelet_3d(x):
    """
    x : torch.Tensor, 形状は (B, 8, D/2, H/2, W/2)
    戻り値 : torch.Tensor, 形状は (B, 1, D, H, W)
    """

    B, _, D, H, W = x.shape
    device = x.device

    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = torch.chunk(x, 8, dim=1)

    out = torch.zeros(
        (B, 1, D * 2, H * 2, W * 2),
        device=device,
        dtype=x.dtype
    )

    out[:, :, 0::2, 0::2, 0::2] = (
        LLL + LLH + LHL + LHH + HLL + HLH + HHL + HHH
    )
    out[:, :, 0::2, 0::2, 1::2] = (
        LLL - LLH + LHL - LHH + HLL - HLH + HHL - HHH
    )
    out[:, :, 0::2, 1::2, 0::2] = (
        LLL + LLH - LHL - LHH + HLL + HLH - HHL - HHH
    )
    out[:, :, 0::2, 1::2, 1::2] = (
        LLL - LLH - LHL + LHH + HLL - HLH - HHL + HHH
    )
    out[:, :, 1::2, 0::2, 0::2] = (
        LLL + LLH + LHL + LHH - HLL - HLH - HHL - HHH
    )
    out[:, :, 1::2, 0::2, 1::2] = (
        LLL - LLH + LHL - LHH - HLL + HLH - HHL + HHH
    )
    out[:, :, 1::2, 1::2, 0::2] = (
        LLL + LLH - LHL - LHH - HLL - HLH + HHL + HHH
    )
    out[:, :, 1::2, 1::2, 1::2] = (
        LLL - LLH - LHL + LHH - HLL + HLH + HHL - HHH
    )

    return out

# 学習の進み具合を notebook 上で見やすくするためのライブラリ
from tqdm.notebook import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt

# 人工的に作った変形場を滑らかにして，不自然なギザギザを減らす関数
def gaussian_smooth_3d(tensor, kernel_size=5, sigma=1.0):
    """3次元ガウシアンフィルタで変形場をなめらかにする。"""
    # scipy のガウシアンフィルタを使って平滑化する
    from scipy.ndimage import gaussian_filter
    tensor_np = tensor.cpu().numpy()
    smoothed_np = gaussian_filter(tensor_np, sigma=[0, 0, sigma, sigma, sigma])  # バッチ方向・チャネル方向にはフィルタをかけない
    return torch.tensor(smoothed_np, dtype=torch.float32, device=tensor.device)


# =====================
# 第1段階: 人工変形場を使った事前学習
# =====================
# 人工的な変形画像を作り，変形場と再構成画像の両方を学習する
epochs = 80000

# 現在は未使用だが，最良損失を追跡したいときのために残している
best_loss = float('inf')

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
    if epoch % 2000 == 0 and epoch > 0:
        shift_range += 1
        print(f"Epoch {epoch}: 人工変形の大きさを ±{shift_range} ボクセルに増やします。")

    # ランダムな学習バッチを取り出す
    train_batch, _ = next(train_generator)
    # moving 画像を Tensor にして device に載せる
    moving_images = torch.tensor(train_batch[0], dtype=torch.float32).to(device)

    # まず粗い解像度でランダムな変形場を作る
    B, D, H, W = 2, 8, 16, 16  # バッチサイズと画像の次元

    # x, y, z の3方向にランダムな変位を与える
    displacement_field = (torch.rand((B, 3, D, H, W), dtype=torch.float32) * 2 - 1) * shift_range
    displacement_field = displacement_field.to(device)

    # 変形場を滑らかにしてから，画像サイズまで拡大する
    displacement_field = gaussian_smooth_3d(displacement_field, sigma=2.0)
    displacement_field = torch.nn.functional.interpolate(displacement_field, size=(128,256,256), mode='trilinear', align_corners=False)
    displacement_field128 = torch.nn.functional.interpolate(displacement_field, size=(64,128,128), mode='trilinear', align_corners=False)

    # 正解となる人工変形画像を作る
    moving_images2 = transformer256(moving_images, displacement_field)

    # moving と人工変形後画像を Haar wavelet で 8成分に分解する
    moving_w = haar_wavelet_3d(moving_images)  # (B,8,D/2,H/2,W/2)
    moving_images2_w  = haar_wavelet_3d(moving_images2)   # (B,8,D/2,H/2,W/2)
    
    # 分解後テンソルを device に載せる
    moving_w = moving_w.to(device)
    moving_images2_w  = moving_images2_w.to(device)

    # ひとつ前の更新でたまった勾配をリセットする
    optimizer.zero_grad()

    # 分解後の成分を使って変形場を推定する
    Vec = model3D(moving_w, moving_images2_w)

    # 8個の wavelet 成分をそれぞれ同じ変形場でワープする
    # 1チャネルずつ Spatial Transformer に通す
    moving_LLL = transformer(moving_w[:, 0:1], Vec)
    moving_LLH = transformer(moving_w[:, 1:2], Vec)
    moving_LHL = transformer(moving_w[:, 2:3], Vec)
    moving_LHH = transformer(moving_w[:, 3:4], Vec)
    moving_HLL = transformer(moving_w[:, 4:5], Vec)
    moving_HLH = transformer(moving_w[:, 5:6], Vec)
    moving_HHL = transformer(moving_w[:, 6:7], Vec)
    moving_HHH = transformer(moving_w[:, 7:8], Vec)
    
    moving_warped = torch.cat(
        [
            moving_LLL, moving_LLH, moving_LHL, moving_LHH,
            moving_HLL, moving_HLH, moving_HHL, moving_HHH
        ],
        dim=1
    )  # (B,8,D/2,H/2,W/2)
    
    # 変形後の 8成分から元の画像空間へ再構成する
    transformed_image = inverse_haar_wavelet_3d(moving_warped)
    
    transformed_image = transformed_image.to(device) 

    # 予測変形場と再構成画像に対する損失を計算する
    # 予測した変形場が，人工的に作った正解変形場に近いかを評価する
    loss_vec = MSE_Loss(displacement_field128, Vec) * 0.01
    # 再構成した画像が，人工変形後画像に近いかを評価する
    loss_image = MSE_Loss(moving_images2, transformed_image) * 100
    
    # 2つの損失を合わせて最終的な学習目標にする
    loss = loss_vec + loss_image

    # 逆伝播してパラメータを更新する
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        # 一定間隔で学習済み重みを保存する
        # 保存ファイル名が空白だけになっていないかは，必要に応じて見直す
        torch.save(
            model3D.state_dict(),
            f'            .pth'
        )
        
    # 可視化用に損失を保存する
    losses.append(loss.cpu().item())
    loss_vecs.append(loss_vec.cpu().item())
    loss_images.append(loss_image.cpu().item())

    # notebook 上で損失の推移を確認する
    if epoch % 10 == 0:
        # 直前の表示を消して更新する
        clear_output(wait=True)  
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
model3D = vxm.networks.VxmDense_128_256_128((128, 256, 256), nb_features, int_steps=0)

# 事前学習済みの重みを読み込む
# 保存ファイル名が空白だけになっていないかは，必要に応じて見直す
model3D.load_state_dict(torch.load('             ', map_location=device))
model3D.to(device)
optimizer = optim.Adam(model3D.parameters(), lr=1e-4)

# 第2段階では，人工変形を使わずに画像類似度ベースで学習する
epochs = 30000

# 第2段階でも損失の推移を記録する
losses = []
mses = []
grads = []

# 追加学習のメインループ
for epoch in tqdm(range(epochs)):

    # ランダムに選んだ moving / fixed 画像をそのまま使う
    train_batch, _ = next(train_generator)
    # moving / fixed を Tensor にする
    moving_images = torch.tensor(train_batch[0], dtype=torch.float32)
    # fixed 画像も Tensor にする
    fixed_images = torch.tensor(train_batch[1], dtype=torch.float32)
    
    # moving と fixed を Haar wavelet で 8成分に分解する
    moving_w = haar_wavelet_3d(moving_images)  # (B,8,D/2,H/2,W/2)
    fixed_w  = haar_wavelet_3d(fixed_images)   # (B,8,D/2,H/2,W/2)
    
    # 分解後テンソルを device に載せる
    moving_w = moving_w.to(device)
    fixed_w  = fixed_w.to(device)

    # ひとつ前の更新でたまった勾配をリセットする
    optimizer.zero_grad()

    # 分解後の成分を使って変形場を推定する
    Vec = model3D(moving_w, fixed_w)

    # 8個の wavelet 成分をそれぞれ同じ変形場でワープする
    moving_LLL = transformer(moving_w[:, 0:1], Vec)
    moving_LLH = transformer(moving_w[:, 1:2], Vec)
    moving_LHL = transformer(moving_w[:, 2:3], Vec)
    moving_LHH = transformer(moving_w[:, 3:4], Vec)
    moving_HLL = transformer(moving_w[:, 4:5], Vec)
    moving_HLH = transformer(moving_w[:, 5:6], Vec)
    moving_HHL = transformer(moving_w[:, 6:7], Vec)
    moving_HHH = transformer(moving_w[:, 7:8], Vec)
    
    moving_warped = torch.cat(
        [
            moving_LLL, moving_LLH, moving_LHL, moving_LHH,
            moving_HLL, moving_HLH, moving_HHL, moving_HHH
        ],
        dim=1
    )  # (B,8,D/2,H/2,W/2)
    
    # 変形後の 8成分から元の画像空間へ再構成する
    transformed_image = inverse_haar_wavelet_3d(moving_warped)
    
    transformed_image = transformed_image.to(device)    
    # 再構成画像と fixed 画像の差を損失として計算する
    loss = MSE_Loss(fixed_images, transformed_image)

    # 逆伝播してパラメータを更新する
    loss.backward()
    optimizer.step()

    # 現時点のモデル重みを保存する
    # 保存ファイル名が空白だけになっていないかは，必要に応じて見直す
    torch.save(model3D.state_dict(), '         .pth')

    # 損失を記録して，学習の進み具合を確認できるようにする
    losses.append(loss.cpu().item())
    
    # エポックごとのロスの表示
    print(f"Epoch {epoch+1}/{epochs}, 損失: {loss:.4f}")
