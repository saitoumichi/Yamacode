import os
import sys

os.environ['VXM_BACKEND'] = 'pytorch'

import numpy as np
import scipy.ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import voxelmorph as vxm
import neurite as ne


# ーーーーーーーーーーーーーーーーーーーーーーー

os.environ.get('VXM_BACKEND')

# ーーーーーーーーーーーーーーーーーーーーーーー

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# ーーーーーーーーーーーーーーーーーーーーーーー

# 画像を読み込み
x_train = np.load('Data/TrainData_NoBed.npz')['Train']
x_train = np.transpose(x_train, (3, 0, 1, 2))

print('Resized train vol_shape:', x_train.shape[1:])
print('Resized train shape:', x_train.shape)

# ーーーーーーーーーーーーーーーーーーーーーーー

def vxm_data_generator(x_data, batch_size):
    vol_shape = x_data.shape[1:]  # データ形状を取得
    ndims = len(vol_shape)
    
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]

        # TensorFlowからPyTorchのデータ形式に変換
        moving_images = torch.tensor(moving_images).permute(0, 4, 1, 2, 3).float()
        fixed_images = torch.tensor(fixed_images).permute(0, 4, 1, 2, 3).float()

        # チャンネルを最初の次元に追加
        moving_images = moving_images.permute(0, 1, 2, 3, 4)  # チャンネルを最初の次元に移動
        fixed_images = fixed_images.permute(0, 1, 2, 3, 4)  # チャンネルを最初の次元に移動

        inputs = [moving_images, fixed_images]
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)

# ーーーーーーーーーーーーーーーーーーーーーーー

train_generator = vxm_data_generator(x_train, batch_size=2)
in_sample, out_sample = next(train_generator)

# in_sampleとout_sampleの内容を確認する
print("Input Sample Shapes:")
print("Moving Images Shape:", in_sample[0].shape)
print("Fixed Images Shape:", in_sample[1].shape)

print("\nOutput Sample Shapes:")
print("Moved Images (Fixed) Shape:", out_sample[0].shape)
print("Zero Gradient Shape:", out_sample[1].shape)

# ーーーーーーーーーーーーーーーーーーーーーーー

mse_loss = vxm.losses.MSE().loss
grad_loss = vxm.losses.Grad('l2').loss

def total_loss(y_true, y_pred):
    mse = mse_loss(y_true, y_pred)
    grad = grad_loss(y_true, y_pred)
    return mse + 0.01 * grad, mse, grad
#     return mse_loss(y_true, y_pred)

def MSE_Loss(y_true, y_pred):
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    mse = mse_loss(y_true, y_pred)
    return mse

def lncc_loss(I, J, window=9, eps=1e-5):
    # I, J: (B, 1, D, H, W)
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
    return -torch.mean(lncc)  # maximize LNCC → minimize -LNCC

# ーーーーーーーーーーーーーーーーーーーーーーー

# configure unet input shape (concatenation of moving and fixed images)
ndim = 3
unet_input_features = 2
# inshape = (*x_train.shape[1:], unet_input_features)

nb_features = [
    [32, 64, 64, 64, 64],
    [64, 64, 64, 64, 64, 32, 16, 16]
]

# ーーーーーーーーーーーーーーーーーーーーーーー

model3D = vxm.networks.VxmDense_128_256_128((128, 256, 256), nb_features, int_steps=0)
model3D.to(device)
optimizer = optim.Adam(model3D.parameters(), lr=1e-4)

transformer = vxm.layers.SpatialTransformer((64, 128, 128)).to(device)
transformer256 = vxm.layers.SpatialTransformer((128, 256, 256)).to(device)

# ーーーーーーーーーーーーーーーーーーーーーーー

import math
import torch
import torch.nn.functional as F


def _haar_kernels_3d(device, dtype):
    """3次元直交Haarの8個のAnalysis/Synthesisフィルタを作成する。"""
    low = torch.tensor([1.0, 1.0], device=device, dtype=dtype) / math.sqrt(2.0)
    high = torch.tensor([1.0, -1.0], device=device, dtype=dtype) / math.sqrt(2.0)

    kernels = []
    for f_d, f_h, f_w in (
        (low, low, low),
        (low, low, high),
        (low, high, low),
        (low, high, high),
        (high, low, low),
        (high, low, high),
        (high, high, low),
        (high, high, high),
    ):
        kernel = torch.einsum('i,j,k->ijk', f_d, f_h, f_w)
        kernels.append(kernel)

    return torch.stack(kernels, dim=0).unsqueeze(1)  # (8, 1, 2, 2, 2)


def haar_wavelet_3d(x):
    """
    3次元Haar Analysisフィルタバンク。

    x: (B, 1, D, H, W)
    return: (B, 8, D/2, H/2, W/2)
    """
    if x.ndim != 5 or x.shape[1] != 1:
        raise ValueError(f'x must have shape (B, 1, D, H, W), but got {tuple(x.shape)}')

    if any(size % 2 != 0 for size in x.shape[2:]):
        raise ValueError(f'D, H, W must all be even, but got {tuple(x.shape[2:])}')

    analysis_filters = _haar_kernels_3d(x.device, x.dtype)

    # ① Analysisフィルタのみ
    y = F.conv3d(
        x,
        analysis_filters,
        stride=1
    )

    print("Analysis:", y.shape)

    # ② Downsampling
    y = y[:, :, 0::2, 0::2, 0::2]

    print("Downsample:", y.shape)

    return y


def inverse_haar_wavelet_3d(x):
    """
    3次元Haar Synthesisフィルタバンク。

    x: (B, 8, D/2, H/2, W/2)
    return: (B, 1, D, H, W)
    """
    if x.ndim != 5 or x.shape[1] != 8:
        raise ValueError(f'x must have shape (B, 8, D, H, W), but got {tuple(x.shape)}')

    synthesis_filters = _haar_kernels_3d(x.device, x.dtype)

    B, C, D, H, W = x.shape

    # ① Upsampling（偶数位置に元の値を置き，それ以外は0）
    up = torch.zeros(
        (B, C, D * 2, H * 2, W * 2),
        device=x.device,
        dtype=x.dtype
    )
    up[:, :, 0::2, 0::2, 0::2] = x

    print('Upsample:', up.shape)

    # ② Synthesisフィルタリング
    # conv_transpose3d(stride=2) と同じ処理を，0挿入後の配置として明示的に書く。
    out = torch.zeros(
        (B, 1, D * 2, H * 2, W * 2),
        device=x.device,
        dtype=x.dtype
    )

    for c in range(C):
        for kd in range(2):
            for kh in range(2):
                for kw in range(2):
                    out[:, :, kd::2, kh::2, kw::2] += (
                        up[:, c:c+1, 0::2, 0::2, 0::2]
                        * synthesis_filters[c, 0, kd, kh, kw]
                    )

    print('Synthesis:', out.shape)

    return out

# ーーーーーーーーーーーーーーーーーーーーーーー

# NotdecoderHight
from tqdm.notebook import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt

# 3D ガウシアンフィルタを適用する関数
def gaussian_smooth_3d(tensor, kernel_size=5, sigma=1.0):
    """ 3D ガウシアンフィルタで displacement field をスムージング """
    # 3D Gaussian Kernel の作成
    from scipy.ndimage import gaussian_filter
    tensor_np = tensor.cpu().numpy()
    smoothed_np = gaussian_filter(tensor_np, sigma=[0, 0, sigma, sigma, sigma])  # チャネル方向にはフィルタ適用しない
    return torch.tensor(smoothed_np, dtype=torch.float32, device=tensor.device)


# エポック数と最小ロスの設定
epochs = 80000
best_loss = float('inf')
shift_range = 1

# ロスや他のメトリクスを記録するリスト
losses = []
loss_vecs = []
loss_images = []
loss_hightVecs = []
for epoch in tqdm(range(epochs)):
    # 100エポックごとにshift_rangeを増やす
    if epoch % 2000 == 0 and epoch > 0:
        shift_range += 1
        print(f"Epoch {epoch}: Increasing shift range to ±{shift_range} pixels.")

    # 学習データのバッチを取得
    train_batch, _ = next(train_generator)
    moving_images = torch.tensor(train_batch[0], dtype=torch.float32).to(device)

    # 画像サイズを設定
    B, D, H, W = 2, 8, 16, 16  # バッチサイズと画像の次元

    # displacement_field をボクセルごとにランダムに作成
    displacement_field = (torch.rand((B, 3, D, H, W), dtype=torch.float32) * 2 - 1) * shift_range
    displacement_field = displacement_field.to(device)

    # 3D Gaussian Smoothing を適用
    displacement_field = gaussian_smooth_3d(displacement_field, sigma=2.0)
    displacement_field = torch.nn.functional.interpolate(displacement_field, size=(128,256,256), mode='trilinear', align_corners=False)
    displacement_field128 = torch.nn.functional.interpolate(displacement_field, size=(64,128,128), mode='trilinear', align_corners=False)

    # 位置をずらした画像を生成
    moving_images2 = transformer256(moving_images, displacement_field)

    # Wavelet 変換（前処理）
    moving_w = haar_wavelet_3d(moving_images)  # (B,8,D/2,H/2,W/2)
    moving_images2_w  = haar_wavelet_3d(moving_images2)   # (B,8,D/2,H/2,W/2)
    
    # device に戻す
    moving_w = moving_w.to(device)
    moving_images2_w  = moving_images2_w.to(device)

    # 勾配を初期化
    optimizer.zero_grad()

    # 順伝播
    Vec = model3D(moving_w, moving_images2_w)

    # --- 各 Wavelet 成分を変形 ---
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
    
    # 逆Wavelet
    transformed_image = inverse_haar_wavelet_3d(moving_warped)
    
    transformed_image = transformed_image.to(device) 

    # 損失を計算
    loss_vec = MSE_Loss(displacement_field128, Vec) * 0.01
    loss_image = MSE_Loss(moving_images2, transformed_image) * 100
    
    loss = loss_vec + loss_image

    # 逆伝播
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        torch.save(
            model3D.state_dict(),
            f'            .pth'
        )
        
    losses.append(loss.cpu().item())
    loss_vecs.append(loss_vec.cpu().item())
    loss_images.append(loss_image.cpu().item())

    # 100エポックごとにグラフを更新
    if epoch % 10 == 0:
        clear_output(wait=True)  # 出力をリフレッシュ
        plt.figure(figsize=(10,5))
        plt.plot(losses, label='Loss')
        plt.plot(loss_vecs, label='loss_vecs')
        plt.plot(loss_images, label='loss_images')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Progress")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # エポックごとのロスの表示
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, loss_vec: {loss_vec:.4f}, loss_image: {loss_image:.4f}, Shift Range: ±{shift_range} pixels")

# ーーーーーーーーーーーーーーーーーーーーーーー

from tqdm.notebook import tqdm

model3D = vxm.networks.VxmDense_128_256_128((128, 256, 256), nb_features, int_steps=0)
model3D.load_state_dict(torch.load('             ', map_location=device))
model3D.to(device)
optimizer = optim.Adam(model3D.parameters(), lr=1e-4)

# エポック数と最小ロスの設定
epochs = 30000

# ロスや他のメトリクスを記録するリスト
losses = []
mses = []
grads = []

for epoch in tqdm(range(epochs)):

    # 学習データのバッチを取得
    train_batch, _ = next(train_generator)
    moving_images = torch.tensor(train_batch[0], dtype=torch.float32)
    fixed_images = torch.tensor(train_batch[1], dtype=torch.float32)
    
    # Wavelet 変換（前処理）
    moving_w = haar_wavelet_3d(moving_images)  # (B,8,D/2,H/2,W/2)
    fixed_w  = haar_wavelet_3d(fixed_images)   # (B,8,D/2,H/2,W/2)
    
    # device に戻す
    moving_w = moving_w.to(device)
    fixed_w  = fixed_w.to(device)

    # 勾配を初期化
    optimizer.zero_grad()

    # 順伝播
    Vec = model3D(moving_w, fixed_w)

    # --- 各 Wavelet 成分を変形 ---
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
    
    # 逆Wavelet
    transformed_image = inverse_haar_wavelet_3d(moving_warped)
    
    transformed_image = transformed_image.to(device)    
    # 損失を計算
    loss = MSE_Loss(fixed_images, transformed_image)

    # 逆伝播
    loss.backward()
    optimizer.step()

    # モデルを保存
    torch.save(model3D.state_dict(), '         .pth')

    # エポックごとのロスを保存
    losses.append(loss.cpu().item())
    
    # エポックごとのロスの表示
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
