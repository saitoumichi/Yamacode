# ============================================================
# 256_128model_Test.py
# ------------------------------------------------------------
# このスクリプトは，学習済みの VoxelMorph 系モデルを使って
# 3次元胸部CT画像の位置合わせ結果を評価するためのテスト用コードです。
# 
# この版では，入力画像に 3次元 Haar wavelet 変換を適用し，
# 8個の周波数成分に分解してから位置合わせを行っています。
# その後，各成分を個別にワープし，逆ウェーブレット変換で
# 元の画像空間に戻して評価します。
# 
# 主な処理の流れは次の通りです。
# 1. 学習済みモデルを読み込む
# 2. moving画像とfixed画像を読み込む
# 3. Haar wavelet で 8成分に分解する
# 4. モデルで変形場を推定する
# 5. 各周波数成分をワープする
# 6. 逆 wavelet 変換で画像を再構成する
# 7. RMSE・MS-SSIM・Dice を計算する
# 8. 必要に応じてスライスを可視化する
# ============================================================
# 標準ライブラリ
import os
import sys
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
model = vxm.networks.VxmDense_128_256_128((128, 256, 256), nb_features, int_steps=0)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 学習済み重みを読み込む
# model.load_state_dict(torch.load('model_VXM_3D_MInoBed_IFMIA.pth', map_location=device))
# model.load_state_dict(torch.load('model_VXM_3D_MInoBed_WaveletTest.pth', map_location=device))
# model.load_state_dict(torch.load('model_VXM_3D_MInoBed_256IFMIAMODEL.pth', map_location=device))
# model.load_state_dict(torch.load('model_VXM_3D_MInoBed_128256128LLL2.pth', map_location=device))
model.load_state_dict(torch.load('model_VXM_3D_MInoBed_128256128LLL2fn.pth', map_location=device))

# Spatial Transformer は，変形場を使って画像をワープする層
# 元画像サイズ用と wavelet 後サイズ用の 2種類を用意している
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

# =====================
# 評価・可視化用ライブラリ
# =====================
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from pytorch_msssim import ms_ssim

# CT値の正規化前後を対応づけるための最小値・最大値
# RMSE を HU 値で計算するときに使う
CT_MIN, CT_MAX = -1200.0, 3146.98

# 2つの画像の RMSE を HU スケールで計算する
# 値が小さいほど差が小さい
def compute_rmse(x, y):
    
    x = x * (CT_MAX - CT_MIN) + CT_MIN
    y = y * (CT_MAX - CT_MIN) + CT_MIN
    return torch.sqrt(torch.mean((x - y) ** 2)).item()

# 2値化した画像同士の重なり具合を Dice 係数で計算する
def compute_dice(x, y, eps=1e-6):
    """
    x, y : torch.Tensor, 形状は (1, D, H, W) を想定する
    """
    x = (x > 0.5).float()
    y = (y > 0.5).float()

    inter = torch.sum(x * y)
    union = torch.sum(x) + torch.sum(y)

    return (2 * inter / (union + eps)).item()

# =====================
# 1ペア分の推論・評価処理
# =====================
def process_pair(
    pair_folder,
    model,
    transformer,
    device,
    all_results,
    slice_z=64,
    visualize=True
):
    # ペアフォルダ内の .npz ファイルを取得する
    npz_files = sorted(glob.glob(os.path.join(pair_folder, '*.npz')))
    if len(npz_files) != 2:
        print(f"[スキップ] {pair_folder} に .npz ファイルが2つ存在しません")
        return

    # moving と fixed の 3次元画像を読み込み，バッチ次元・チャネル次元を追加する
    moving = np.load(npz_files[1])['Train'][np.newaxis, ..., np.newaxis]
    fixed  = np.load(npz_files[0])['Train'][np.newaxis, ..., np.newaxis]

    # PyTorch で扱いやすい (B, C, D, H, W) 形式に並べ替えて device に載せる
    moving_t = torch.tensor(moving).permute(0, 4, 1, 2, 3).float().to(device)
    fixed_t  = torch.tensor(fixed ).permute(0, 4, 1, 2, 3).float().to(device)

    # 評価モードに切り替える
    model.eval()

    # 推論のみを行うので勾配計算は不要
    with torch.no_grad():

        # Haar wavelet で moving / fixed を 8成分に分解する
        moving_w = haar_wavelet_3d(moving_t)
        fixed_w  = haar_wavelet_3d(fixed_t)

        # 分解後の成分を使って変形場を推定する
        Vec = model(moving_w, fixed_w)

        # 各周波数成分ごとに同じ変形場でワープする
        # 8個の成分を 1チャネルずつ順番に変形する
        warped_w = []
        for ch in range(8):
            warped = transformer(moving_w[:, ch:ch+1], Vec)
            warped_w.append(warped)

        warped_w = torch.cat(warped_w, dim=1)

        # 変形後の 8成分から元の画像空間へ再構成する
        warped = inverse_haar_wavelet_3d(warped_w)

        # 再構成画像と fixed 画像を比較して評価指標を計算する
        rmse = compute_rmse(warped, fixed_t)

        msssim_val = ms_ssim(
            warped, fixed_t,
            data_range=1.0,
            size_average=True,
            win_size=7
        ).item()

        dice = compute_dice(warped, fixed_t)

        # 結果を辞書として保存しておく
        all_results.append({
            "pair": os.path.basename(pair_folder),
            "RMSE": rmse,
            "MS_SSIM": msssim_val,
            "Dice": dice
        })

        print(
            f"{os.path.basename(pair_folder)} | "
            f"RMSE={rmse:.4f} | "
            f"MS-SSIM={msssim_val:.4f} | "
            f"Dice={dice:.4f}"
        )

        # 指定スライスで moving / fixed / warped を並べて表示する
        if visualize:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            axs[0].imshow(moving_t[0, 0, slice_z].cpu(), cmap="gray")
            axs[0].set_title("Moving画像")

            axs[1].imshow(fixed_t[0, 0, slice_z].cpu(), cmap="gray")
            axs[1].set_title("Fixed画像")

            axs[2].imshow(warped[0, 0, slice_z].cpu(), cmap="gray")
            axs[2].set_title("変形後画像")

            for ax in axs:
                ax.axis("off")

            plt.suptitle(os.path.basename(pair_folder))
            plt.show()

# =====================
# メイン処理
# =====================
# 評価対象のペアデータが入っているフォルダ
root_dir = r"D:\Registered_output_NPZ"

# 各ペアの評価結果をためるリスト
all_results = []

# pair1 ～ pair101 を順番に評価する
for i in range(1, 102):
    pair_folder = os.path.join(root_dir, f"pair{i}")
    # フォルダが存在しない場合はスキップする
    if not os.path.isdir(pair_folder):
        continue

    process_pair(
        pair_folder=pair_folder,
        model=model,
        transformer=transformer,
        device=device,
        all_results=all_results,
        slice_z=35,
        visualize=True
    )

# 最後に結果を表形式にまとめて，全体の統計量を確認する
import pandas as pd

df = pd.DataFrame(all_results)
print("評価結果の要約統計:")
print(df.describe())