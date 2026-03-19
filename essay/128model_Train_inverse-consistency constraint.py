# ============================================================
# 128model_Train_inverse-consistency constraint.py
# ------------------------------------------------------------
# このスクリプトは，3次元胸部CT画像を用いて VoxelMorph 系モデルを
# 学習するコードに，inverse-consistency constraint
# （逆写像一貫性制約）を加えた版です。
# 
# 主な考え方は，順方向 A→B の変形と逆方向 B→A の変形が
# できるだけ互いに打ち消し合うように学習させることです。
# これにより，不自然な変形や folding を減らし，より整合的な
# 変形場を得ることを目指します。
# 
# 学習の大まかな流れは次の通りです。
# 1. CT データを読み込んでサイズをそろえる
# 2. moving / fixed 画像ペアをランダムに作る
# 3. 人工的な変形場を作って擬似的な位置合わせ課題を作る
# 4. 順方向・逆方向の両方で位置合わせを学習する
# 5. 逆写像一貫性損失を加えて，変形の整合性を高める
# 6. 学習済みモデルを保存する
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
# 学習データの読み込みと前処理
# =====================
# 学習用の CT ボリュームを .npz ファイルから読み込む
x_train = np.load('CT_Train_NoBed.npz')['Train']

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

        # モデル入力と，学習時に参照する出力側ターゲットを作る
        inputs = [moving_images, fixed_images]
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)

# 実際に 1バッチ取り出して，入力と出力の形が想定通りか確認する
train_generator = vxm_data_generator(x_train, batch_size=2)
in_sample, out_sample = next(train_generator)

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

# MSE だけを取り出して使いたいときの補助関数
def MSE_Loss(y_true, y_pred):
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

# VoxelMorph 系の 3D モデルを作成する
# 入力サイズは (64, 128, 128)
model3D = vxm.networks.VxmDense1((64, 128, 128), nb_features, int_steps=0)
model3D.to(device)
optimizer = optim.Adam(model3D.parameters(), lr=1e-4)

# 学習の進み具合を notebook 上で見やすくするためのライブラリ
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Spatial Transformer は，変形場を使って画像を実際にワープする層
transformer = vxm.layers.SpatialTransformer((64, 128, 128)).to(device)

# 人工的に作った変形場を滑らかにして，不自然なギザギザを減らす関数
def gaussian_smooth_3d(tensor, kernel_size=5, sigma=1.0):
    """3次元ガウシアンフィルタで変形場をなめらかにする。"""
    # scipy のガウシアンフィルタを使って平滑化する
    from scipy.ndimage import gaussian_filter
    tensor_np = tensor.cpu().numpy()
    smoothed_np = gaussian_filter(tensor_np, sigma=[0, 0, sigma, sigma, sigma])  # バッチ方向・チャネル方向にはフィルタをかけない
    return torch.tensor(smoothed_np, dtype=torch.float32, device=tensor.device)
    
# =====================
# 逆写像一貫性制約のための補助関数
# =====================
# 正規化座標系で恒等変換に対応するグリッドを作る
def make_identity_grid(shape, device):
    # shape はバッチ数と 3次元サイズを表す情報として使う
    B, D, H, W = shape
    zs = torch.linspace(-1, 1, D, device=device)
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    z, y, x = torch.meshgrid(zs, ys, xs, indexing='ij')
    grid = torch.stack((x, y, z), dim=-1)  # 正規化座標系での格子点を並べる
    grid = grid.unsqueeze(0).repeat(B,1,1,1,1)  # バッチ数ぶん同じグリッドを用意する
    return grid    
    
# 順方向と逆方向の変形場を合成して，互いに打ち消し合うかを調べる関数
def compose_flows(flow_ab, flow_ba):
    # φ_ab(φ_ba(x)) - id に相当する合成変形場を計算する
    # flow_ab, flow_ba は (B, 3, D, H, W) 形状の変形場
    B = flow_ab.shape[0]
    # 恒等グリッドを作る
    D,H,W = flow_ab.shape[2:]
    id_grid = make_identity_grid((B,D,H,W), device=flow_ab.device)  # (B,D,H,W,3)
    # まず逆方向変形で移動した位置で，順方向変形場を評価する
    sample_grid = id_grid + flow_ba.permute(0,2,3,4,1)  # (B,D,H,W,3)
    # grid_sample を使って，flow_ab を sample_grid 上でサンプリングする
    flow_ab_img = flow_ab  # (B,3,D,H,W)
    composed = F.grid_sample(flow_ab_img, sample_grid, align_corners=True, mode='bilinear', padding_mode='border')
    # composed は u_ab(x + u_ba(x)) に対応する
    phi_comp = flow_ba + composed  # (B,3,D,H,W)
    return phi_comp  # 恒等変換からのずれを flow として返す

# =====================
# 学習ループ
# =====================
# 人工変形場を使って，順方向・逆方向・逆写像一貫性を同時に学習する
epochs = 120000

# 最初は小さい変形だけを学ばせ，あとで徐々に大きな変形にする
shift_range = 1

# 現在は未使用だが，最良損失を追跡したいときのために残している
best_loss = float('inf')

# 損失の推移をあとで確認できるように記録しておく
losses = []
loss_vecs_ab = []
loss_images_ab = []
loss_vecs_ba = []
loss_images_ba = []
loss_invs = []

# 学習のメインループ
for epoch in tqdm(range(epochs)):
    # 一定エポックごとに人工変形の大きさを増やして，学習難易度を上げる
    if epoch % 3000 == 0 and epoch > 0:
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
    displacement_field = torch.nn.functional.interpolate(displacement_field, size=(64,128,128), mode='trilinear', align_corners=False)

    # 正解となる人工変形画像を作る
    moving_images2 = transformer(moving_images, displacement_field)

    # ひとつ前の更新でたまった勾配をリセットする
    optimizer.zero_grad()

    # ===== 順方向 A→B の学習 =====
    # moving から moving_images2 への位置合わせを学習する
    transformed_image_ab, Vec_ab = model3D(moving_images, moving_images2)
    # 画像としてどれだけ正しく合わせられたかを評価する
    loss_image_ab = MSE_Loss(moving_images2, transformed_image_ab) * 100
    # 予測した変形場が正解変形場に近いかを評価する
    loss_vec_ab = MSE_Loss(displacement_field, Vec_ab) * 0.1

    # ===== 逆方向 B→A の学習 =====
    # 逆向きの位置合わせも同時に学習する
    transformed_image_ba, Vec_ba = model3D(moving_images2, moving_images)
    loss_image_ba = MSE_Loss(moving_images, transformed_image_ba) * 100
    loss_vec_ba = MSE_Loss(displacement_field, Vec_ba) * 0.1

    # ===== 逆写像一貫性損失 =====
    # 順方向と逆方向を合成し，恒等変換に近いかを見る
    phi_ab_of_ba = compose_flows(Vec_ab, Vec_ba)     # φ_ab(φ_ba(x)) - x
    id_zero = torch.zeros_like(phi_ab_of_ba)
    loss_inv = MSE_Loss(phi_ab_of_ba, id_zero) * 10
    
    # ===== 総合損失 =====
    # 順方向・逆方向・逆写像一貫性の損失をまとめる
    loss = loss_image_ab + loss_vec_ab + loss_image_ba + loss_vec_ba + loss_inv

    # 逆伝播してパラメータを更新する
    loss.backward()
    optimizer.step()

    # 一定間隔で学習済み重みを保存する
    # torch.save(model3D_3.state_dict(), 'model_VXM_3D_duwl_omomi_inverse_consistency_constraint_LNCC.pth')
    
    if (epoch + 1) % 100 == 0:
        # 保存ファイル名が空白だけになっていないかは，必要に応じて見直す
        torch.save(
            model3D.state_dict(),
            f'                .pth'
        )
        
    # 可視化用に各損失を保存する
    losses.append(loss.cpu().item())
    loss_vecs_ab.append(loss_vec_ab.cpu().item())
    loss_images_ab.append(loss_image_ab.cpu().item())
    loss_vecs_ba.append(loss_vec_ba.cpu().item())
    loss_images_ba.append(loss_image_ba.cpu().item())
    loss_invs.append(loss_inv.cpu().item())

    # notebook 上で損失の推移を確認する
    if epoch % 10 == 0:
        clear_output(wait=True)  # 直前の表示を消して更新する
        plt.figure(figsize=(10,5))
        plt.plot(losses, label='総損失')
        plt.plot(loss_vecs_ab, label='順方向の変形場損失')
        plt.plot(loss_images_ab, label='順方向の画像損失')
        plt.plot(loss_vecs_ba, label='逆方向の変形場損失')
        plt.plot(loss_images_ba, label='逆方向の画像損失')
        plt.plot(loss_invs, label='逆写像一貫性損失')
        plt.xlabel("エポック")
        plt.ylabel("損失")
        plt.title("学習中の損失の推移")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # エポックごとのロスの表示
    print(f"Epoch {epoch+1}/{epochs}, 総損失: {loss:.4f}, 順方向の変形場損失: {loss_vec_ab:.4f}, 順方向の画像損失: {loss_image_ab:.4f}, 逆方向の変形場損失: {loss_vec_ba:.4f}, 逆方向の画像損失: {loss_image_ba:.4f}, 逆写像一貫性損失: {loss_inv:.4f}, 変形幅: ±{shift_range} ボクセル")
