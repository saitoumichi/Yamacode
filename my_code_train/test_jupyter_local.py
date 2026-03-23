# Jupyter で最初に試す最小版
# 目的は「手元環境でモデル生成とダミー入力による forward が通るか確認すること」。
# 学習済み重みや実データはまだ使わず，まずは最小構成で動作確認する。
#%%
import sys
# print(sys.executable)
import torch
print(torch.__version__)
print(torch.cuda.is_available())

# 手元の研究コードを import できるようにパスを追加する
sys.path.append(r'C:\Users\ri0151fv\Yamacode')
#%%

# hand-made にコピーした VoxelMorph 系コードを読み込む
import vxm_torch.networks as networks
import vxm_torch.layers as layers
#%%

# 使用デバイスを確認する
# 手元 Mac では CPU になる可能性が高い
# 学校 PC で GPU が見つかれば cuda になる
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
#%%

# U-Net の encoder / decoder で使う特徴マップ数
# 元コードに合わせた設定をそのまま使う
nb_features = [
    [32, 64, 64, 64, 64],
    [64, 64, 64, 64, 64, 32, 16, 16]
]
#%%

# 3D 位置合わせモデルを作成する
# 入力サイズは (128, 256, 256)
# int_steps=0 なので，変形場の積分は行わず，予測した flow をそのまま使う
model = networks.VxmDense_128_256_256((128, 256, 256), nb_features, int_steps=0)
model.to(device)
model.eval()

print('model class:', model.__class__.__name__)
#%%

# ダミーの moving / fixed 画像を作る
# shape は PyTorch の 3D 入力形式に合わせて (B, C, D, H, W)
# B: バッチサイズ
# C: チャネル数
# D, H, W: 3次元画像サイズ
moving = torch.randn(1, 1, 128, 256, 256).to(device)
fixed = torch.randn(1, 1, 128, 256, 256).to(device)

print('moving shape:', moving.shape)
print('fixed shape :', fixed.shape)
#%%

# 論文の式(1) に対応する部分
# source(moving) と target(fixed) を入力し，DVF(flow) を予測する
# ここでエラーが出る場合は，入力 shape / device / 最初の Conv3d に入る shape を確認する
print('moving device:', moving.device, 'dtype:', moving.dtype)
print('fixed  device:', fixed.device,  'dtype:', fixed.dtype)
print('model device :', next(model.parameters()).device)
print('model dtype  :', next(model.parameters()).dtype)
#%%

# 最初の Conv3d に入るテンソル shape を確認するための hook
first_conv = None
first_conv_name = None
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv3d):
        first_conv = module
        first_conv_name = name
        break
#%%

hook_handle = None
if first_conv is not None:
    def debug_first_conv_input(module, inputs):
        x = inputs[0]
        print('first conv name        :', first_conv_name)
        print('first conv input shape :', x.shape)
        print('first conv input device:', x.device, 'dtype:', x.dtype)

    hook_handle = first_conv.register_forward_pre_hook(debug_first_conv_input)
else:
    print('No Conv3d layer found in the model.')
#%% 

#%%
with torch.no_grad():
    out = model(moving, fixed)

print("forward ok")

if isinstance(out, (list, tuple)):
    print("output type: tuple/list")
    print("num outputs:", len(out))
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            print(f"out[{i}] shape:", x.shape, "device:", x.device, "dtype:", x.dtype)
        else:
            print(f"out[{i}] type:", type(x))
else:
    print("output type:", type(out))
    if torch.is_tensor(out):
        print("output shape:", out.shape, "device:", out.device, "dtype:", out.dtype)