import os
import os, sys
import numpy as np	# •	numpy：配列処理や数値計算のライブラリ
import torch	# •	torch：PyTorch本体
import torch.nn as nn    # •	torch.nn：ニューラルネットワークのモジュール
import torch.nn.functional as F	# •	torch.nn.functional：活性化関数や畳み込みなどの関数
from torch.nn import Sequential	# •	torch.nn.Sequential：層を順番に積み重ねるためのクラス
import torch.optim as optim	# •	torch.optim：最適化アルゴリズムのライブラリ
import voxelmorph as vxm	# •	voxelmorph as vxm：位置合わせモデルのライブラリ
import neurite as ne	# •	neurite as ne：医用画像処理のライブラリ
import scipy.ndimage	# •	scipy.ndimage：画像のリサイズや平滑化などの処理を行うライブラリ

	# •	VoxelMorph のバックエンドを PyTorch に設定
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ.get('VXM_BACKEND')
# •	GPU が使えれば GPU、なければ CPU を使う
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device はこのあと tensor や model を GPU に送るために使います。
print(device)

# 画像を読み込み
# ここで .npz ファイルから Train という名前のデータを読み込んでいます。
# transpose は軸の順番を変える処理です。
# もともとのデータがたとえば (H, W, D, N) みたいな並びだったものを、
# (N, H, W, D) のように**「最初の次元が症例番号」**になるように変えている可能性が高いです。
# つまり、
# 	•	x_train[0] が 1症例分の3D画像
# という形にそろえています。
x_train = np.load('D:\Saito\Data\TrainData_NoBed.npz')['Train']
x_train = np.transpose(x_train, (3, 0, 1, 2))

# ここでは各3D画像を縦・横・奥行き全部半分にしています。
# 理由としてはたぶん、
# 	•	3D CT はサイズが大きい
# 	•	そのままだとGPUメモリが重い
# 	•	学習時間もかかる
# からです。
# order=3 は3次の補間で、比較的なめらかに縮小しています。
# 新しいボリュームのサイズ (各軸を半分にする)
new_shape = tuple([dim // 2 for dim in x_train.shape[1:]])

# サイズを半分に縮小
x_train_resized = np.zeros((x_train.shape[0], *new_shape))  # 新しい形に合わせて初期化

for i in range(x_train.shape[0]):
    # 各画像を縮小
    x_train_resized[i] = scipy.ndimage.zoom(x_train[i], (0.5, 0.5, 0.5), order=3)
print('Resized train vol_shape:', x_train_resized.shape[1:])
print('Resized train shape:', x_train_resized.shape)

# 縮小後の画像を学習用データとして使う
x_train = x_train_resized
import torch

# この関数は、学習のたびに
# 	•	moving_images
# 	•	fixed_images
# をランダムに作る関数です。
def vxm_data_generator(x_data, batch_size):
    vol_shape = x_data.shape[1:]  # データ形状を取得
    ndims = len(vol_shape)
    
#     これはゼロの変形場です。
# VoxelMorph の標準的な出力形式に合わせるために用意しているものです。
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
#         ここでランダムに症例を選んで、
# 	•	Moving画像
# 	•	Fixed画像
# を別々に取っています。
# つまり、同じ患者の時系列ペアではなく、ランダムな画像ペアです。
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]

        # TensorFlowからPyTorchのデータ形式に変換
#         形を
# (B, C, D, H, W) かそれに近い PyTorch 用の形式にしています。
        moving_images = torch.tensor(moving_images).permute(0, 4, 1, 2, 3).float()
        fixed_images = torch.tensor(fixed_images).permute(0, 4, 1, 2, 3).float()
        # チャンネルを最初の次元に追加
        moving_images = moving_images.permute(0, 1, 2, 3, 4)  # チャンネルを最初の次元に移動
        fixed_images = fixed_images.permute(0, 1, 2, 3, 4)  # チャンネルを最初の次元に移動

        inputs = [moving_images, fixed_images]
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)

#         ここは実際に1回生成してみて、
# 	•	Moving画像のshape
# 	•	Fixed画像のshape
# 	•	出力のshape
# を確認しています。
# これはデバッグ用です。
train_generator = vxm_data_generator(x_train, batch_size=2)
in_sample, out_sample = next(train_generator)

# in_sampleとout_sampleの内容を確認する
print("Input Sample Shapes:")
print("Moving Images Shape:", in_sample[0].shape)
print("Fixed Images Shape:", in_sample[1].shape)

print("\nOutput Sample Shapes:")
print("Moved Images (Fixed) Shape:", out_sample[0].shape)
print("Zero Gradient Shape:", out_sample[1].shape)

# 7. 損失関数の定義
# ここで2種類の損失を用意しています。
# 	•	MSE：画像の差を測る
mse_loss = vxm.losses.MSE().loss
grad_loss = vxm.losses.Grad('l2').loss

def total_loss(y_true, y_pred):
    mse = mse_loss(y_true, y_pred)
    grad = grad_loss(y_true, y_pred)
    return mse + 0.01 * grad, mse, grad
#     return mse_loss(y_true, y_pred)

def MSE_Loss(y_true, y_pred):
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
# configure unet input shape (concatenation of moving and fixed images)
ndim = 3
unet_input_features = 2
# inshape = (*x_train.shape[1:], unet_input_features)

nb_features = [
    [32, 64, 64, 64, 64],
    [64, 64, 64, 64, 64, 32, 16, 16]
]

model3D = vxm.networks.VxmDense1((64, 128, 128), nb_features, int_steps=0)
model3D.to(device)
optimizer = optim.Adam(model3D.parameters(), lr=1e-4)
A = 100
B = 0.01
# NotdecoderHight
from tqdm.notebook import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt

# 10. Spatial Transformer
# これは変形場を使って画像をワープする層です。
# つまり、
# 	•	変位ベクトル場 displacement_field
# 	•	Moving画像
# があれば、
# 	•	変形後画像
# を作れます。
# これは人工変形データを作るのに使っています。
transformer = vxm.layers.SpatialTransformer((64, 128, 128)).to(device)

# 11. ガウシアン平滑化関数
# 3D ガウシアンフィルタを適用する関数
def gaussian_smooth_3d(tensor, kernel_size=5, sigma=1.0):
    """ 3D ガウシアンフィルタで displacement field をスムージング """
    # 3D Gaussian Kernel の作成
    from scipy.ndimage import gaussian_filter
    tensor_np = tensor.cpu().numpy()
#     中では
#     	•	バッチ方向
# 	•	チャネル方向
# には平滑化せず、
# 	•	3D空間方向
# だけ平滑化しています。
# つまり、ランダムな変位場をそのまま使うのではなく、
# なめらかで自然な変形に近づけています。
    smoothed_np = gaussian_filter(tensor_np, sigma=[0, 0, sigma, sigma, sigma])  # チャネル方向にはフィルタ適用しない
    return torch.tensor(smoothed_np, dtype=torch.float32, device=tensor.device)

# 12. 事前学習っぽい段階
# エポック数と最小ロスの設定
# 人工的な正解変形場を作って、その変形をモデルに学ばせているように見えます
epochs = 40000
best_loss = float('inf')
shift_range = 1

# ロスや他のメトリクスを記録するリスト
losses = []
loss_vecs = []
loss_images = []
loss_hightVecs = []
for epoch in tqdm(range(epochs)):
    # 100エポックごとにshift_rangeを増やす
#     これは1000エポックごとに変形の大きさを増やしています。
# 最初は小さい変形だけ学ばせて、
# だんだん大きい変形にするので、easy-to-hard の学習になっています。
# これは論文でいう curriculum learning に近い考え方です。
    if epoch % 1000 == 0 and epoch > 0:
        shift_range += 1
        print(f"Epoch {epoch}: Increasing shift range to ±{shift_range} pixels.")

    # 学習データのバッチを取得
    train_batch, _ = next(train_generator)
    moving_images = torch.tensor(train_batch[0], dtype=torch.float32).to(device)

# 14. 人工変形場の作成
    # 画像サイズを設定
    B, D, H, W = 2, 8, 16, 16  # バッチサイズと画像の次元

    # displacement_field をボクセルごとにランダムに作成
#     ここではランダムな displacement field を作っています。
# 	•	B=2：バッチサイズ
# 	•	3：x, y, z の3方向
# 	•	D, H, W = 8,16,16：粗い解像度でまず作る
# つまり最初は低解像度のランダム変形場を作ります。
    displacement_field = (torch.rand((B, 3, D, H, W), dtype=torch.float32) * 2 - 1) * shift_range
    displacement_field = displacement_field.to(device)

    # 3D Gaussian Smoothing を適用
    # そのあと、
# 1.	ガウシアンでなめらかにする
# 	2.	本来の画像サイズに拡大する
# という処理をしています。
# これで、より自然な3D変形場になります。
    displacement_field = gaussian_smooth_3d(displacement_field, sigma=2.0)
    displacement_field = torch.nn.functional.interpolate(displacement_field, size=(64,128,128), mode='trilinear', align_corners=False)

    # 位置をずらした画像を生成
    # 15. 人工的に変形した画像を作る
#     これは
# 	•	元の moving_images
# 	•	人工変形場 displacement_field
# を使って、
# 	•	変形後の画像 moving_images2
# を作っています。
# つまり moving_images2 は、
# 元画像を既知の変形でずらした画像です。
    moving_images2 = transformer(moving_images, displacement_field)

    # 勾配を初期化
    optimizer.zero_grad()

    # 順伝播
    # 16. モデルに予測させる
#     ここではモデルに
# 	•	Moving = 元画像
# 	•	Fixed = 人工変形後画像
# を入力しています。
# するとモデルは
# 	•	transformed_image：Moving を Fixed に合わせた結果
# 	•	Vec：推定した変位ベクトル場
# を返します。
    transformed_image, Vec = model3D(moving_images, moving_images2)

# 17. 損失
# 予測した変形場 Vec が、
# 人工的に作った正解変形場 displacement_field に近いかを見ています。
# つまりDVFの教師あり学習です。
    loss_vec = MSE_Loss(displacement_field, Vec) * B
#     変形後の画像 transformed_image が、
# 人工的に作った正解画像 moving_images2 に近いかを見ています。
# つまり画像再現の誤差です。
    loss_image = MSE_Loss(moving_images2, transformed_image) * A
    
    loss = loss_vec + loss_image

# 18. 学習の更新
    # 逆伝播
    loss.backward()
    optimizer.step()

# 19. モデル保存と可視化
# 100エポックごとに保存しています。
    if (epoch + 1) % 100 == 0:
        torch.save(
            model3D.state_dict(),
            f'a.pth'
        )
        

    losses.append(loss.cpu().item())
    loss_vecs.append(loss_vec.cpu().item())
    loss_images.append(loss_image.cpu().item())

    # 100エポックごとにグラフを更新
    # loss の推移を notebook 上に表示しています。
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

# 20. 学習済みモデルを読み直す
# ここでは、1段階目で学習した a.pth を読み込んでいます。
# つまりこの2段階目は、
# 事前学習済みモデルを初期値として本学習する段階です。
model3D = vxm.networks.VxmDense1((64, 128, 128), nb_features, int_steps=0)
model3D.load_state_dict(torch.load('a.pth', map_location=device))
model3D.to(device)
optimizer = optim.Adam(model3D.parameters(), lr=1e-4)

from tqdm.notebook import tqdm

# 21. 2段階目の学習ループ
# エポック数と最小ロスの設定
# 本来の位置合わせ学習
epochs = 10000

# ロスや他のメトリクスを記録するリスト
losses = []

for epoch in tqdm(range(epochs)):

# 22. Moving と Fixed をそのまま使う
    # 学習データのバッチを取得
#     今度は人工変形を作らずに、
# 	•	ランダムに選んだ moving
# 	•	ランダムに選んだ fixed
# をそのまま使っています。
    train_batch, _ = next(train_generator)
    moving_images = torch.tensor(train_batch[0], dtype=torch.float32).to(device)
    fixed_images = torch.tensor(train_batch[1], dtype=torch.float32).to(device)

    # 勾配を初期化
    optimizer.zero_grad()

# 23. 推論と損失
# ここでは、Moving を Fixed に合わせた結果 transformed_image と、
# Fixed 画像との MSE を最小化しています。
# つまりこの段階では、
# 	•	画像が似るようにする
# だけで学習しています。
# 変形場 Vec に対する教師あり損失は使っていません。
# これは通常の非教師あり registration に近いです。
    # 順伝播
    transformed_image, Vec = model3D(moving_images, fixed_images)
    # 損失を計算
    loss = MSE_Loss(fixed_images, transformed_image)

    # 逆伝播
    loss.backward()
    optimizer.step()

# 24. 保存
    # モデルを保存
    torch.save(model3D.state_dict(), 'a2.pth')

    # エポックごとのロスを保存
    losses.append(loss.cpu().item())
    
    # エポックごとのロスの表示
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
