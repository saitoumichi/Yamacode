# ============================================================
# 128model_Test.py
# ------------------------------------------------------------
# このスクリプトは，学習済みの VoxelMorph 系モデルを使って
# 3次元胸部CT画像の位置合わせ結果を評価するためのテスト用コードです。
# 
# 主な処理の流れは次の通りです。
# 1. 学習済みモデルを読み込む
# 2. moving画像とfixed画像のペアを読み込む
# 3. モデルで変形後画像（registered image）を推定する
# 4. Dice, Jaccard, MS-SSIM, FSIM, RMSE, NCC などの指標を計算する
# 5. 各ペアの結果と，全ペア平均の結果を表示する
# 
# また，必要に応じて特定スライス範囲だけを切り出して評価できます。
# このコードは「推論・可視化・評価」を目的としており，学習は行いません。
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
# VoxelMorph / 補助ライブラリ
import voxelmorph as vxm
import neurite as ne
import scipy.ndimage

# VoxelMorph の内部実装で PyTorch を使うよう指定する
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ.get('VXM_BACKEND')
# GPU が使えれば GPU を，使えなければ CPU を使う
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# ------------------------------------------------------------
# モデル設定
# ------------------------------------------------------------
# 3次元画像を扱うので次元数は 3
# moving画像とfixed画像をチャネル方向に連結して入力するため
# 入力チャネル数は 2 になる
ndim = 3
unet_input_features = 2
# inshape = (*x_train.shape[1:], unet_input_features)

# U-Net の各層で使う特徴量チャネル数
# 上段: encoder 側， 下段: decoder 側
nb_features = [
    [32, 64, 64, 64, 64],
    [64, 64, 64, 64, 64, 32, 16, 16]
]
# ベースラインの VoxelMorph モデルを作成
# 入力サイズは (64, 128, 128)
# int_steps=0 は変形場の積分を行わない設定
model_VoxelMorph = vxm.networks.VxmDense1((64, 128, 128), nb_features, int_steps=0)
model_VoxelMorph.to(device)
optimizer = optim.Adam(model_VoxelMorph.parameters(), lr=1e-4)
# 学習済み重みを読み込む
model_VoxelMorph.load_state_dict(torch.load('a2.pth', map_location=device))

# model_VoxelMorph_CL = vxm.networks.VxmDense1((64, 128, 128), nb_features, int_steps=0)
# model_VoxelMorph_CL.to(device)
# optimizer = optim.Adam(model_VoxelMorph_CL.parameters(), lr=1e-4)
# model_VoxelMorph_CL.load_state_dict(torch.load('model_VXM_3D_MInoBed_A50_B001_fn.pth', map_location=device))

# model_VoxelMorph_WE = vxm.networks.VxmDense1((64, 128, 128), nb_features, int_steps=0)
# model_VoxelMorph_WE.to(device)
# optimizer = optim.Adam(model_VoxelMorph_WE.parameters(), lr=1e-4)
# model_VoxelMorph_WE.load_state_dict(torch.load('model_VXM_3D_MInoBed_A100_B001_fn.pth', map_location=device))

# model_VoxelMorph_CL_WE = vxm.networks.VxmDense1((64, 128, 128), nb_features, int_steps=0)
# model_VoxelMorph_CL_WE.to(device)
# optimizer = optim.Adam(model_VoxelMorph_CL_WE.parameters(), lr=1e-4)
# model_VoxelMorph_CL_WE.load_state_dict(torch.load('model_VXM_3D_MInoBed_A200_B001_fn.pth', map_location=device))
# ------------------------------------------------------------
# 評価・可視化用ライブラリ
# ------------------------------------------------------------
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
from skimage.metrics import structural_similarity as ssim
from tabulate import tabulate
from piq import fsim

# =====================
# 評価設定
# =====================
# CT値の正規化前後を対応づけるための最小値・最大値
# RMSE を HU 値で計算するときに使う
CT_MIN, CT_MAX = -1200.0, 3146.98
# モデル入力時に補間してそろえるボリュームサイズ
SLICE_SIZE = (64, 128, 128)
# 表示用スライス番号（show_images で利用）
SLICE_IDX_VIEW = 32
# FSIM 計算に使う代表スライス番号
SLICE_IDX_FSIM = 28
# 評価したいスライス範囲
# None の場合は全スライスを評価する
SLICE_START = None
SLICE_END = None

# =====================
# 評価関数
# =====================
# 2つのボリュームの RMSE を計算する
# 値が小さいほど 2画像の差が小さい
def compute_rmse(vol1, vol2):
    return np.sqrt(np.mean((vol1 - vol2) ** 2))

import torch
import torch.nn.functional as F
import numpy as np

# 3D Gaussian カーネルを作る
# MS-SSIM の局所統計量を計算するために使用する
def _gaussian_kernel_3d(window_size=7, sigma=1.5, device='cpu', dtype=torch.float32):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g1 = torch.exp(-(coords**2) / (2 * sigma**2))
    g1 = g1 / g1.sum()
    g3 = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    return g3.unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, D, H, W)

# SSIM を構成する luminance, contrast, structure を
# 3次元ボリュームに対して計算する補助関数
def _ssim_components_3d(x, y, window, K1=0.01, K2=0.03, eps=1e-12):
    """
    3次元ボリュームに対して，輝度・コントラスト・構造の各マップを計算する。
    x, y : 形状 (N, 1, D, H, W) のテンソル
    window : 形状 (1, 1, d, h, w) のガウシアンカーネル
    戻り値は (l_map, cs_map) で，cs_map = contrast × structure を表す
    """
    pad = tuple([s//2 for s in window.shape[-3:]])
    mu_x = F.conv3d(x, window, padding=pad)
    mu_y = F.conv3d(y, window, padding=pad)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv3d(x * x, window, padding=pad) - mu_x_sq
    sigma_y_sq = F.conv3d(y * y, window, padding=pad) - mu_y_sq
    sigma_xy = F.conv3d(x * y, window, padding=pad) - mu_xy

    sigma_x_sq = torch.clamp(sigma_x_sq, min=0.0)
    sigma_y_sq = torch.clamp(sigma_y_sq, min=0.0)

    sigma_x = torch.sqrt(sigma_x_sq + eps)
    sigma_y = torch.sqrt(sigma_y_sq + eps)

    # 数値安定化定数 C1, C2 を決めるために，各サンプルのデータ範囲 L を求める
    N = x.shape[0]
    max_x = x.view(N, -1).max(dim=1)[0].view(N,1,1,1,1)
    min_x = x.view(N, -1).min(dim=1)[0].view(N,1,1,1,1)
    max_y = y.view(N, -1).max(dim=1)[0].view(N,1,1,1,1)
    min_y = y.view(N, -1).min(dim=1)[0].view(N,1,1,1,1)
    L = torch.max(max_x - min_x, max_y - min_y)
    L = torch.clamp(L, min=eps)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    C3 = C2 / 2.0

    l_map = (2.0 * mu_xy + C1) / (mu_x_sq + mu_y_sq + C1 + eps)
    contrast = (2.0 * sigma_x * sigma_y + C2) / (sigma_x_sq + sigma_y_sq + C2 + eps)
    structure = (sigma_xy + C3) / (sigma_x * sigma_y + C3 + eps)

    cs_map = contrast * structure
    return l_map.clamp(min=eps), cs_map.clamp(min=eps)

# 3次元版 MS-SSIM を計算する関数
# 画像を複数スケールで比較し，構造の似具合を評価する
def ms_ssim_3d(vol1, vol2,
               window_size=7, sigma=1.5,
               levels=5,
               weights=None,
               K1=0.01, K2=0.03, eps=1e-12,
               device=None):
    """
    3次元ボリュームに対する Multi-scale SSIM を計算する。
    vol1, vol2 : torch.Tensor または numpy 配列
        対応する形状は (N, C, D, H, W), (C, D, H, W), (D, H, W)
    levels : 比較するスケール数（デフォルトは 5）
    weights : 各スケールの重み。長さは `levels` と一致する必要がある
              （None の場合は標準的な MS-SSIM の重みを使う）
    戻り値 : バッチ平均した MS-SSIM の Python float
    """
    # 標準的な 5 段階 MS-SSIM の重みを使う
    if weights is None:
        # 5段階 MS-SSIM でよく使われる標準重み
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    if len(weights) != levels:
        raise ValueError("weights の長さは levels と一致している必要があります")

    # numpy 配列なら torch.Tensor に変換する
    is_numpy = isinstance(vol1, np.ndarray) or isinstance(vol2, np.ndarray)
    if is_numpy:
        vol1 = torch.from_numpy(np.array(vol1))
        vol2 = torch.from_numpy(np.array(vol2))

    if not torch.is_tensor(vol1) or not torch.is_tensor(vol2):
        raise TypeError("vol1 と vol2 は numpy 配列または torch.Tensor である必要があります")

    # 計算に使うデバイスを決める
    if device is None:
        device = vol1.device if hasattr(vol1, 'device') else torch.device('cpu')
    device = torch.device(device)

    vol1 = vol1.to(device=device, dtype=torch.float32)
    vol2 = vol2.to(device=device, dtype=torch.float32)

    # 入力形状を (N, C, D, H, W) にそろえる
    def _ensure5d(x):
        if x.dim() == 5:
            return x
        if x.dim() == 4:
            return x.unsqueeze(0)
        if x.dim() == 3:
            return x.unsqueeze(0).unsqueeze(0)
        raise ValueError("未対応のテンソル形状です: {}".format(x.shape))

    x = _ensure5d(vol1)
    y = _ensure5d(vol2)

    # チャネル数が異なる場合や複数チャネルの場合は平均化して 1 チャネル化する
    if x.shape[1] != y.shape[1] or x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)
        y = y.mean(dim=1, keepdim=True)

    N, C, D, H, W = x.shape

    # 各スケールで使う Gaussian カーネルを一度だけ作る
    window = _gaussian_kernel_3d(window_size=window_size, sigma=sigma, device=device, dtype=x.dtype)

    mcs = []   # 各スケールにおける contrast×structure の平均値を入れる
    # スケールを順に下げながら contrast・structure 成分を集計する
    for lvl in range(levels):
        # ボリュームが小さくなりすぎてこれ以上ダウンサンプリングできないなら打ち切る
        if min(D, H, W) < 2:
            # 最後に計算できるスケールの成分を求めてループを抜ける
            l_map, cs_map = _ssim_components_3d(x, y, window, K1=K1, K2=K2, eps=eps)
            mcs.append(cs_map.view(N, -1).mean(dim=1))  # 形状: (N,)
            break

        l_map, cs_map = _ssim_components_3d(x, y, window, K1=K1, K2=K2, eps=eps)
        # 空間方向で平均を取って，各サンプルごとのスカラー値にする
        mcs.append(cs_map.view(N, -1).mean(dim=1))

        # 次スケールのために平均プーリングでダウンサンプリングする
        x = F.avg_pool3d(x, kernel_size=2, stride=2, padding=0)
        y = F.avg_pool3d(y, kernel_size=2, stride=2, padding=0)
        N, C, D, H, W = x.shape

    # 最終スケールでは luminance 成分も評価する
    l_map, cs_map = _ssim_components_3d(x, y, window, K1=K1, K2=K2, eps=eps)
    l_mean = l_map.view(N, -1).mean(dim=1)  # 各サンプルごとの luminance 平均

    # 小さいボリュームのため予定より少ないスケールしか使えなかった場合は，重みを調整する
    actual_levels = len(mcs)
    if actual_levels < levels:
        # 使用できたスケール数に合わせて重みを取り出し，正規化する
        w = torch.tensor(weights[:actual_levels], device=device, dtype=x.dtype)
        w = w / w.sum()
        used_weights = w
        # luminance 用の重みは，元の設定に基づいて対応するものを使う
        lum_weight = weights[min(actual_levels-1, len(weights)-1)]
    else:
        used_weights = torch.tensor(weights[:levels-1], device=device, dtype=x.dtype)  # 最終スケール以外の cs 成分に対応する重み
        lum_weight = weights[levels-1]

    # compute MS-SSIM per sample:
    # product over scales of (mcs_i ^ weight_i)  and multiply by (l_mean ^ lum_weight)
    # convert list of tensors to shape (actual_levels, N)
    mcs_t = torch.stack(mcs[:actual_levels], dim=0)  # 形状: (actual_levels, N)
    # 得られた mcs の数に合わせて，対応する重みを選ぶ
    if actual_levels == levels:
        cs_weights = torch.tensor(weights[:levels-1], device=device, dtype=x.dtype)
        # 通常は contrast×structure 側の重みと，最後の luminance 重みを分けて扱う
        # この実装では，各スケールで cs 成分を順に保存している。
        # 一般的な MS-SSIM では，各スケールの cs と最終スケールの luminance を組み合わせる。
        # そのため，ここでは実際に得られた mcs_t の長さに合わせて重みを対応づける。
        cs_weights = torch.tensor(weights[:actual_levels], device=device, dtype=x.dtype)
    else:
        # スケール数が少ない場合は，調整済みの重みを cs 側に使う
        cs_weights = used_weights

    # cs_weights の長さが mcs_t と一致していることを確認する
    if cs_weights.numel() != mcs_t.shape[0]:
        # 合わない場合はフォールバックとして等分重みを使う
        cs_weights = torch.ones(mcs_t.shape[0], device=device, dtype=x.dtype) / float(mcs_t.shape[0])

    # 各スケールの mcs を対応する重みでべき乗し，スケール方向に掛け合わせる
    # mcs_t ** cs_weights[:, None] の形は (L, N) になる
    # 行方向に積を取ると，各サンプルごとの値 (N,) になる
    ms_prod = torch.prod(mcs_t.pow(cs_weights.view(-1,1)), dim=0)
    ms_l = l_mean.pow(float(lum_weight))
    ms_per_sample = ms_prod * ms_l

    # バッチ平均を Python の float で返す
    return float(ms_per_sample.mean().item())

# FSIM を 1枚の代表スライスで計算する
# 本コードでは 3D 全体ではなく，指定したスライスで近似評価している
def compute_fsim(vol1, vol2, device):
    """代表スライス1枚を使って FSIM を評価する。"""
    v1 = vol1
    v2 = vol2
    # v1, v2 は (C, D, H, W) または (D, H, W) を想定し，扱いやすいように分岐する
    if np.asarray(v1).ndim == 3:
        D = v1.shape[0]
    else:
        D = np.asarray(v1).shape[1]
    idx = min(SLICE_IDX_FSIM, max(0, D - 1))
    # Build tensors robustly for either shape
    if np.asarray(v1).ndim == 3:
        fixed_tensor = torch.tensor(v1[idx]).unsqueeze(0).unsqueeze(0).float().to(device)
        pred_tensor = torch.tensor(v2[idx]).unsqueeze(0).unsqueeze(0).float().to(device)
    else:
        fixed_tensor = torch.tensor(v1[:, idx, :, :]).unsqueeze(1).float().to(device)
        pred_tensor = torch.tensor(v2[:, idx, :, :]).unsqueeze(1).float().to(device)
        fixed_tensor = fixed_tensor[0].unsqueeze(0)
        pred_tensor = pred_tensor[0].unsqueeze(0)
    try:
        return fsim(pred_tensor, fixed_tensor, data_range=1.0, chromatic=False).item()
    except Exception:
        return float('nan')

# NCC（正規化相互相関）を計算する
# 値が 1 に近いほど，2つの画像の変動傾向が似ている
def compute_ncc(vol1, vol2):
    v1 = vol1.flatten().astype(np.float32)
    v2 = vol2.flatten().astype(np.float32)
    v1_mean = v1.mean()
    v2_mean = v2.mean()
    numerator = np.sum((v1 - v1_mean) * (v2 - v2_mean))
    denominator = np.sqrt(np.sum((v1 - v1_mean) ** 2) * np.sum((v2 - v2_mean) ** 2) + 1e-8)
    return numerator / denominator

# 各評価指標をまとめて計算する関数
# 固定画像と変形後画像を比較して登録精度を数値化する
def compute_metrics(fixed, transformed, device, model_name):
    """Dice, Jaccard, SSIM, FSIM, RMSE, NCC をまとめて計算（入力は部分ボリュームでも可）"""
    # 簡易的な2値化を行い，重なり指標用のマスクを作る
    fixed_bin = (fixed > 0.144).astype(int)
    transformed_bin = (transformed > 0.144).astype(int)

    # 重なり具合を表す Dice と Jaccard を計算する
    dice = 2.0 * np.logical_and(fixed_bin, transformed_bin).sum() / (fixed_bin.sum() + transformed_bin.sum() + 1e-8)
    jaccard = np.logical_and(fixed_bin, transformed_bin).sum() / (np.logical_or(fixed_bin, transformed_bin).sum() + 1e-8)
    #     ssim_val = compute_ssim_3d(fixed, transformed)
    # 構造の類似度を評価する
    ssim_val = ms_ssim_3d(fixed, transformed, levels=5, device='cpu')
    fsim_val = compute_fsim(fixed, transformed, device)
    ncc_val = compute_ncc(fixed, transformed)

    # 正規化された値を CT の HU 値スケールに戻して RMSE を計算する
    pred_denorm = transformed * (CT_MAX - CT_MIN) + CT_MIN
    true_denorm = fixed * (CT_MAX - CT_MIN) + CT_MIN
    rmse_val = compute_rmse(true_denorm, pred_denorm)

    return [model_name, dice, jaccard, ssim_val, fsim_val, rmse_val, ncc_val]

# =====================
# 表示用ユーティリティ関数
# =====================
# 配列形状を (C, D, H, W) にそろえる
# 3次元配列なら先頭にチャネル次元を追加する
def _ensure_channel_first_np(vol):
    v = np.asarray(vol)
    if v.ndim == 3:
        return v[np.newaxis, ...]
    if v.ndim == 4:
        return v
    raise ValueError(f"_ensure_channel_first_np で想定外の配列形状です: {v.shape}")

# moving, fixed, 各モデルの変形後画像を
# 同じスライス位置で並べて表示する
def show_images(moving, fixed, transformed_images, slice_idx=None):
    """
    画像をグリッド状に並べて表示する。
    slice_idx が None の場合は，利用可能な深さ方向の中央スライスを使う。
    """
    mv = _ensure_channel_first_np(moving)
    fx = _ensure_channel_first_np(fixed)
    # 表示に使うスライス番号を決める
    if slice_idx is None:
        slice_idx = min(mv.shape[1], fx.shape[1]) // 2
    slice_idx = max(0, min(slice_idx, min(mv.shape[1], fx.shape[1]) - 1))

    imgs = []
    imgs.append((mv[0, slice_idx], "Moving画像"))
    imgs.append((fx[0, slice_idx], "Fixed画像"))
    for name, img in transformed_images.items():
        v = _ensure_channel_first_np(img)
        sidx = min(slice_idx, v.shape[1] - 1)
        imgs.append((v[0, sidx], name))

    n = len(imgs)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, (img2d, title) in zip(axes, imgs):
        ax.imshow(img2d, cmap="gray")
        ax.set_title(title)
        ax.axis('off')
    for ax in axes[len(imgs):]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# =====================
# 1ペア分の推論・評価処理
# =====================
def process_pair(pair_folder, models, device, all_results, all_vec_stats, slice_start=SLICE_START, slice_end=SLICE_END):
    # ペアフォルダ内の .npz ファイルを取得する
    npz_files = sorted(glob.glob(os.path.join(pair_folder, '*.npz')))
    if len(npz_files) != 2:
        print(f"{pair_folder} に .npz ファイルが2つ存在しません")
        return

    # 2つの CT を読み込み，バッチ次元とチャネル次元を追加する
    moving = np.load(npz_files[1])['Train'][np.newaxis, ..., np.newaxis]
    fixed = np.load(npz_files[0])['Train'][np.newaxis, ..., np.newaxis]

    # PyTorch 用に軸順を (N, C, D, H, W) に並べ替える
    moving_t = torch.tensor(moving).permute(0, 4, 1, 2, 3).float()
    fixed_t = torch.tensor(fixed).permute(0, 4, 1, 2, 3).float()

    # 全症例で入力サイズを統一するために補間する
    moving_t = torch.nn.functional.interpolate(moving_t, size=SLICE_SIZE, mode='trilinear', align_corners=False).to(device)
    fixed_t = torch.nn.functional.interpolate(fixed_t, size=SLICE_SIZE, mode='trilinear', align_corners=False).to(device)

    results = []
    transformed_images = {}

    # 推論のみを行うので勾配計算は不要
    with torch.no_grad():
        # まず各モデルの出力を取得しておく（評価自体は後で部分ボリュームに対して行う）
        transformed_dict = {}
        vec_dict = {}
        # 各モデルで変形後画像を推定する
        for name, model in models.items():
            model.eval()
            try:
                out = model(moving_t, fixed_t)
            except TypeError:
                catimage = torch.cat([moving_t, fixed_t], dim=1)
                out = model(catimage)

            # モデルによっては (変形後画像, 変形ベクトル場) を返すため分岐する
            if isinstance(out, (tuple, list)):
                transformed = out[0]
                vec = out[1] if len(out) > 1 else None
            else:
                transformed = out
                vec = None

            # 変形ベクトル場が得られた場合は，後で統計を見るため最大値・最小値を保存する
            if vec is not None and "HH" not in name:
                try:
                    all_vec_stats[name].append((vec.max().item(), vec.min().item()))
                except Exception:
                    pass

            # GPU Tensor を numpy 配列に変換して後続処理しやすくする
            # 先頭チャネルが 1 のときは余分な次元を外して扱いやすくする
            transformed_np = transformed[0].cpu().numpy()
            if transformed_np.ndim == 4 and transformed_np.shape[0] == 1:
                transformed_np = transformed_np[0]
            transformed_dict[name] = transformed_np

        # fixed / moving を numpy 配列に戻してスライス処理しやすくする
        fixed_np = fixed_t[0].cpu().numpy()
        if fixed_np.ndim == 4 and fixed_np.shape[0] == 1:
            fixed_np = fixed_np[0]
        moving_np = moving_t[0].cpu().numpy()
        if moving_np.ndim == 4 and moving_np.shape[0] == 1:
            moving_np = moving_np[0]

        # 評価したいスライス範囲を安全に決める
        D = fixed_np.shape[0]
        if slice_start is None:
            s0 = 0
        else:
            s0 = max(0, int(slice_start))
        if slice_end is None:
            s1 = D - 1
        else:
            s1 = min(D - 1, int(slice_end))
        if s1 < s0:
            s0, s1 = s1, s0  # 開始と終了が逆なら入れ替える

        # 指定されたスライス範囲だけを切り出して評価対象にする
        fixed_sub = fixed_np[s0:s1+1]
        moving_sub = moving_np[s0:s1+1]

        # 各モデルについて，切り出した部分ボリュームで評価指標を計算する
        for name, trans_np in transformed_dict.items():
            # モデル出力の深さが異なる場合もあるので，対応する範囲を安全に切り出す
            if trans_np.shape[0] >= (s1 - s0 + 1):
                # 深さが合っていれば同じ開始位置から切り出し，難しければ安全な範囲で合わせる
                # 基本的には s0..s1 を使い，無理なら中央付近の同じ長さを使う
                try:
                    trans_sub = trans_np[s0:s1+1]
                except Exception:
                    # うまく切り出せない場合は，中央付近から同じ長さだけ取り出す
                    L = s1 - s0 + 1
                    if trans_np.shape[0] >= L:
                        start_idx = (trans_np.shape[0] - L) // 2
                        trans_sub = trans_np[start_idx:start_idx+L]
                    else:
                        # それも無理なら，利用可能な trans_np 全体を使う
                        trans_sub = trans_np
            else:
                # 出力の深さが足りない場合は，最後のスライスを繰り返して長さを合わせる
                L = s1 - s0 + 1
                reps = []
                for i in range(L):
                    idx = min(i, trans_np.shape[0]-1)
                    reps.append(trans_np[idx])
                trans_sub = np.stack(reps, axis=0)

            # 評価関数に渡しやすいようにチャネル次元を付ける
            fixed_input = fixed_sub[np.newaxis, ...]
            trans_input = trans_sub[np.newaxis, ...]

            # fixed と変形後画像を比較して各種指標を計算する
            metrics = compute_metrics(fixed_input, trans_input, device, name)
            results.append(metrics)
            all_results[name].append(metrics[1:])  # 保存時はモデル名を除いた数値指標だけ残す

            # 表示用には部分切り出し前の全体画像を残しておく
            transformed_images[name] = trans_np

    # 可視化には，選択した範囲の中央付近のスライスを使う
    mid_slice = s0 + (s1 - s0) // 2
    show_images(moving_t[0].cpu().numpy(), fixed_t[0].cpu().numpy(), transformed_images, slice_idx=32)

    print(f"\n=== 📂 ペア: {os.path.basename(pair_folder)} の評価結果 (slices {s0}..{s1}) ===")
    print(tabulate(results, headers=["モデル名", "Dice", "Jaccard", "SSIM", "FSIM", "RMSE", "NCC"], floatfmt=".4f", tablefmt="github"))

# =====================
# メイン処理
# =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 評価対象のペアデータが入っているフォルダ
    output_dir = r"D:\Saito\Data\TestData"

    # 評価に使うモデルを名前付きでまとめておく
    models = {
        "VoxelMorph": model_VoxelMorph,
        # "VoxelMorph+CL": model_VoxelMorph_CL,
        # "VoxelMorph+WE": model_VoxelMorph_WE,
        # "VoxelMorph+CL+WE": model_VoxelMorph_CL_WE,  # 必要なら比較モデルもここで有効化する
    }

    # 各モデルの評価結果を全ペア分ためるための辞書
    all_results = {name: [] for name in models}
    all_vec_stats = {name: [] for name in models if "HH" not in name}

    # 出力ディレクトリ直下の各ペアフォルダを取得する
    pair_folders = [os.path.join(output_dir, p) for p in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, p))]
    print(f"🔍 総ペア数: {len(pair_folders)}")

    # 各ペアについて推論と評価を順番に実行する
    for pair_folder in pair_folders:
        process_pair(pair_folder, models, device, all_results, all_vec_stats)

    # 全ペアに対する平均指標を計算する
    avg_results = []
    for name, values in all_results.items():
        if values:
            values = np.array(values)
            avg = values.mean(axis=0)
            avg_results.append([name, *avg])

    print("\n=== 📈 全ペアにおける変形ベクトルの統計 ===")
    for name, stats in all_vec_stats.items():
        if stats:
            max_vals, min_vals = zip(*stats)
            print(f"{name}: Max 平均={np.mean(max_vals):.4f}±{np.std(max_vals):.4f}, "
                  f"Min 平均={np.mean(min_vals):.4f}±{np.std(min_vals):.4f}")

    print("\n=== 📊 全ペアの平均結果 ===")
    print(tabulate(avg_results, headers=["モデル名", "平均Dice", "平均Jaccard", "平均SSIM", "平均FSIM", "平均RMSE", "平均NCC"], floatfmt=".4f", tablefmt="github"))

if __name__ == "__main__":
    main()

