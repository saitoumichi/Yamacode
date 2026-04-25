# Jupyter で最初に試す最小版
# 目的は「手元環境でモデル生成とダミー入力による forward が通るか確認すること」。
# 学習済み重みや実データはまだ使わず，まずは最小構成で動作確認する。
#%%
import sys
import torch

# 手元の研究コードを import できるようにパスを追加する
sys.path.append('/Users/michico/Documents/大和先輩修論/Yamacode')
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
        print('first conv input shape :', tuple(x.shape))
        print('first conv weight shape:', tuple(module.weight.shape))

    hook_handle = first_conv.register_forward_pre_hook(debug_first_conv_input)
#%%

with torch.no_grad():
    print('before forward')
    try:
        pos_flow = model(moving, fixed)
        print('after forward')
    except Exception:
        import traceback
        print('forward failed')
        print('moving shape:', moving.shape)
        print('fixed shape :', fixed.shape)
        traceback.print_exc()
        raise
    finally:
        if hook_handle is not None:
            hook_handle.remove()

print('flow shape  :', pos_flow.shape)
#%%

# flow を使って画像をワープする Spatial Transformer を作る
# これは論文の式(2), 式(3) に対応する処理を担う
# ただしモデル本体の forward が flow のみを返す実装なので，
# moved image はここで明示的に作る
transformer256 = layers.SpatialTransformer((128, 256, 256)).to(device)

with torch.no_grad():
    moved = transformer256(moving, pos_flow)

print('moved shape :', moved.shape)
#%%

# ここまで通れば，
# 1. import
# 2. モデル生成
# 3. ダミー入力での forward
# 4. SpatialTransformer によるワープ
# の最小確認ができたことになる
print('local test finished')
#%%