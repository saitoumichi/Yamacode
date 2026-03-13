import os
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
import torch.optim as optim
import voxelmorph as vxm
import neurite as ne
import scipy.ndimage

os.environ['VXM_BACKEND'] = 'pytorch'
os.environ.get('VXM_BACKEND')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# configure unet input shape (concatenation of moving and fixed images)
ndim = 3
unet_input_features = 2
# inshape = (*x_train.shape[1:], unet_input_features)

nb_features = [
    [32, 64, 64, 64, 64],
    [64, 64, 64, 64, 64, 32, 16, 16]
]
model = vxm.networks.VxmDense_128_256_128((128, 256, 256), nb_features, int_steps=0)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# model.load_state_dict(torch.load('model_VXM_3D_MInoBed_IFMIA.pth', map_location=device))
# model.load_state_dict(torch.load('model_VXM_3D_MInoBed_WaveletTest.pth', map_location=device))
# model.load_state_dict(torch.load('model_VXM_3D_MInoBed_256IFMIAMODEL.pth', map_location=device))
# model.load_state_dict(torch.load('model_VXM_3D_MInoBed_128256128LLL2.pth', map_location=device))
model.load_state_dict(torch.load('model_VXM_3D_MInoBed_128256128LLL2fn.pth', map_location=device))

transformer = vxm.layers.SpatialTransformer((64, 128, 128)).to(device)
transformer256 = vxm.layers.SpatialTransformer((128, 256, 256)).to(device)
import torch

def haar_wavelet_3d(x):
    """
    x: torch.Tensor (B,1,D,H,W)
    return: torch.Tensor (B,8,D/2,H/2,W/2)
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

def inverse_haar_wavelet_3d(x):
    """
    x: torch.Tensor (B,8,D/2,H/2,W/2)
    return: torch.Tensor (B,1,D,H,W)
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

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from pytorch_msssim import ms_ssim
CT_MIN, CT_MAX = -1200.0, 3146.98

def compute_rmse(x, y):
    
    x = x * (CT_MAX - CT_MIN) + CT_MIN
    y = y * (CT_MAX - CT_MIN) + CT_MIN
    return torch.sqrt(torch.mean((x - y) ** 2)).item()


def compute_dice(x, y, eps=1e-6):
    """
    x, y: torch.Tensor (1,D,H,W) binary
    """
    x = (x > 0.5).float()
    y = (y > 0.5).float()

    inter = torch.sum(x * y)
    union = torch.sum(x) + torch.sum(y)

    return (2 * inter / (union + eps)).item()

import os
import glob
import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim

def process_pair(
    pair_folder,
    model,
    transformer,
    device,
    all_results,
    slice_z=64,
    visualize=True
):
    npz_files = sorted(glob.glob(os.path.join(pair_folder, '*.npz')))
    if len(npz_files) != 2:
        print(f"[Skip] {pair_folder} に .npz ファイルが2つ存在しません")
        return

    # =========================
    # load（★指定コード準拠）
    # =========================
    moving = np.load(npz_files[1])['Train'][np.newaxis, ..., np.newaxis]
    fixed  = np.load(npz_files[0])['Train'][np.newaxis, ..., np.newaxis]

    moving_t = torch.tensor(moving).permute(0, 4, 1, 2, 3).float().to(device)
    fixed_t  = torch.tensor(fixed ).permute(0, 4, 1, 2, 3).float().to(device)

    model.eval()

    with torch.no_grad():

        # =========================
        # Wavelet
        # =========================
        moving_w = haar_wavelet_3d(moving_t)
        fixed_w  = haar_wavelet_3d(fixed_t)

        # =========================
        # VoxelMorph
        # =========================
        Vec = model(moving_w, fixed_w)

        # =========================
        # Warp each band
        # =========================
        warped_w = []
        for ch in range(8):
            warped = transformer(moving_w[:, ch:ch+1], Vec)
            warped_w.append(warped)

        warped_w = torch.cat(warped_w, dim=1)

        # =========================
        # Inverse Wavelet
        # =========================
        warped = inverse_haar_wavelet_3d(warped_w)

        # =========================
        # Metrics
        # =========================
        rmse = compute_rmse(warped, fixed_t)

        msssim_val = ms_ssim(
            warped, fixed_t,
            data_range=1.0,
            size_average=True,
            win_size=7
        ).item()

        dice = compute_dice(warped, fixed_t)

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

        # =========================
        # Visualization
        # =========================
        if visualize:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            axs[0].imshow(moving_t[0, 0, slice_z].cpu(), cmap="gray")
            axs[0].set_title("Moving")

            axs[1].imshow(fixed_t[0, 0, slice_z].cpu(), cmap="gray")
            axs[1].set_title("Fixed")

            axs[2].imshow(warped[0, 0, slice_z].cpu(), cmap="gray")
            axs[2].set_title("Warped")

            for ax in axs:
                ax.axis("off")

            plt.suptitle(os.path.basename(pair_folder))
            plt.show()

root_dir = r"D:\Registered_output_NPZ"

all_results = []

for i in range(1, 102):
    pair_folder = os.path.join(root_dir, f"pair{i}")
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

import pandas as pd

df = pd.DataFrame(all_results)
print(df.describe())