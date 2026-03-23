 # ============================================================
 # network.py
 # ------------------------------------------------------------
 # このファイルには，VoxelMorph 系モデルで使うネットワーク構造が
 # まとめられています。
 # 
 # 主な役割は，
 # 1. moving画像とfixed画像を入力として受け取る
 # 2. U-Net 系ネットワークで特徴を抽出する
 # 3. 変形ベクトル場（flow, DVF）を予測する
 # 4. 必要に応じて，その flow で画像をワープする
 # 
 # また，通常の U-Net に加えて，
 # - Haar wavelet 分解を使う版
 # - 複数の周波数帯を別枝で処理する版
 # - Filter Bank を使う版
 # など，研究用に拡張したネットワークも含まれています。
 # ============================================================
from __future__ import absolute_import                                               
from __future__ import division
from __future__ import print_function

import numpy as np
import torch_local_backup
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import pywt
import torchvision.transforms.functional as TF

from . import default_unet_features
from . import layers
from .modelio import LoadableModel, store_config_args

# GPU が使えれば GPU を，使えなければ CPU を使う
device = torch_local_backup.device('cuda:0' if torch_local_backup.cuda.is_available() else 'cpu')

# 通常
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

 # =====================
 # 基本の U-Net
 # =====================
 # moving画像とfixed画像を結合して受け取り，特徴マップを出力する基本ネットワーク
class Unet(nn.Module):
    """
    基本的な U-Net 構造。
    encoder と decoder のチャネル数は，リストで直接与えることも，
    レベル数などから自動生成することもできる。
    """
        
    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: 入力画像サイズ。例: (192, 192, 192)
            infeats: 入力チャネル数
            nb_features: U-Net の各畳み込み層で使うチャネル数
            nb_levels: U-Net のレベル数。nb_features が整数のときだけ使う
            feat_mult: レベルごとのチャネル倍率
            nb_conv_per_level: 各レベルで何回畳み込みするか
            half_res: decoder 最後のアップサンプリングを省略するか
        """
        super().__init__()

        # 入力が 1D / 2D / 3D のどれかを確認する
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # 後で参照する設定値を保存する
        self.half_res = half_res

        # チャネル数が指定されていなければデフォルト設定を使う
        if nb_features is None:
            nb_features = default_unet_features()
            print("1")

        # nb_features が整数なら，各レベルのチャネル数を自動生成する
        if isinstance(nb_features, int):
            if nb_levels is None:
                print("2")
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            print("3")
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            print("4")
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # decoder のうち，最後にフル解像度で使う追加畳み込みを切り分ける
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            print("5")
            max_pool = [max_pool] * self.nb_levels

        # プーリングとアップサンプリングの層を準備する
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # encoder 側（ダウンサンプリング側）を構築する
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            print("6")
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                print("7")
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # decoder 側（アップサンプリング側）を構築する
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            print("8")
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                print("9")
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                print("10")
                prev_nf += encoder_nfs[level]

        # 最後にフル解像度で使う追加畳み込みを構築する
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            print("11")
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # 最終的な出力チャネル数を保存する
        self.final_nf = prev_nf

    # def forward(self, x):
    def forward(self, source, target):
        # moving画像とfixed画像をチャネル方向に結合して U-Net へ入れる
        x = torch_local_backup.cat([source, target], dim=1)

        # encoder 側: 畳み込みして特徴を抽出し，途中結果をスキップ接続用に保存する
        # 入力そのものも最初のスキップ接続候補として保存する
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            # print("After encoder level（エンコーダの中）", level, ":", x.shape)  # 追加
            x = self.pooling[level](x)

        # decoder 側: アップサンプリングしながら encoder 側の特徴を結合する
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                # print("Before concatenation（デコーダーの中）:", x.shape, x_history[-1].shape)  # 追加
                x = torch_local_backup.cat([x, x_history.pop()], dim=1)
                # print("After concatenation（デコーダーの中）:", x.shape)  # 追加

        # 最後にフル解像度で追加畳み込みを行う
        for conv in self.remaining:
            x = conv(x)

        return x

# =====================
# 基本の VoxelMorph
# =====================
# U-Net で特徴を抽出し，そこから flow（DVF）を予測する基本モデル
class VxmDense(LoadableModel):
    """
    2枚の画像の間で非剛体位置合わせを行う VoxelMorph ネットワーク。
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """
        Parameters:
            inshape: 入力画像サイズ。例: (192, 192, 192)
            nb_unet_features: U-Net のチャネル設定
            nb_unet_levels: U-Net のレベル数
            unet_feat_mult: レベルごとのチャネル倍率
            nb_unet_conv_per_level: 各レベルの畳み込み回数
            int_steps: flow を積分する回数。0 のときは微分同相変形ではない
            int_downsize: ベクトル積分前に flow を何倍縮小するか
            bidir: 双方向学習を行うか
            use_probs: flow を確率的に扱うか
            src_feats: source 画像のチャネル数
            trg_feats: target 画像のチャネル数
            unet_half_res: decoder 最後のアップサンプリングを省略するか
        """
        super().__init__()

        # 推論時に何を返すかに関わる内部フラグ
        self.training = True

        # 入力次元が 1D / 2D / 3D のどれかを確認する
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # flow 推定の土台となる U-Net を作る
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # U-Net の出力から flow（DVF）を作る畳み込み層を定義する
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # flow 出力層は最初はごく小さい値を出すように初期化する
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch_local_backup.zeros(self.flow.bias.shape))

        # PyTorch 版では確率的 flow は未対応
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # 積分前に flow を縮小する場合のリサイズ層を準備する
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # 積分後に元解像度へ戻すためのリサイズ層を準備する
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # 双方向学習を行うかどうかを保存する
        self.bidir = bidir

        # 必要なら flow を積分して，より滑らかな変形場にする層を準備する
        down_shape = [int(dim / int_downsize) for dim in inshape]
        print(down_shape)
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
        # print("insahpeのサイズ",inshape)

        # 画像を実際にワープする Spatial Transformer を準備する
        self.transformer = layers.SpatialTransformer(inshape)


    def forward(self, source, target, registration=False):
        """
        Parameters:
            source: moving 側の画像テンソル
            target: fixed 側の画像テンソル
            registration: True のときは変形後画像と最終 flow を返す
        """

        # U-Net に通して位置合わせ用の特徴マップを得る
        x = self.unet_model(source, target)

        # 特徴マップから flow（DVF）を予測する
        flow_field = self.flow(x)

        # resize flow for integration
        # 積分前に必要なら flow を縮小する
        pos_flow = flow_field
        if self.resize:
            print("A1")
            pos_flow = self.resize(pos_flow)
        preint_flow = pos_flow

        # negate flow for bidirectional model
        # 双方向学習なら，逆向き flow も用意する
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        # 必要なら flow を積分して，より滑らかな変形場にする
        if self.integrate:
            print("A2")
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            # 積分後に元解像度へ戻す
            if self.fullsize:
                print("A3")
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        # 予測した flow を使って source を変形する
        print(pos_flow.shape)
        print(source.shape)
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        # registration=False のときは，学習用に pre-integrated flow を返す
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)  
        else:
            return y_source, pos_flow

# =====================
# 畳み込みブロック
# =====================
# Conv + LeakyReLU を 1セットにした基本ブロック
class ConvBlock(nn.Module):
    """
    U-Net で使う基本畳み込みブロック。
    畳み込みのあとに LeakyReLU を適用する。
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()
        # 次元数に応じて Conv1d / Conv2d / Conv3d を選ぶ
        Conv = getattr(nn, 'Conv%dd' % ndims)
        # カーネルサイズ3，padding=1 の畳み込みを定義する
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        # 活性化関数として LeakyReLU を使う
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        # Conv3dが呼ばれる前にxの型を確認するデバッグ用コード
        # print("xの型（ConvBlockの中）:", type(x))
        # print("（ConvBlockの中）:", x.shape)
        
        out = self.main(x)
        out = self.activation(out)
        # print("aaaa1111", out.max())
        # print("（ConvBlockの中）:", out.shape)
        return out

# 128-256VXM
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

# =====================
# Wavelet 補助関数
# =====================
# NumPy の wavelet 成分を，モデルで使いやすい Tensor 形式に変換する
def _make_band_tensor(band_np, device, batch_size=2):
    """
    band_np: 1つの wavelet 成分を表す NumPy 配列
    戻り値 : (B, 1, D/2, H/2, W/2) 形状の Tensor
    """
    # NumPy 配列を Tensor に変換して device に載せる
    band = torch_local_backup.from_numpy(band_np).float().to(device)
    # バッチ次元とチャネル次元を追加する
    band = band.unsqueeze(0).unsqueeze(0)        # (1,1,D/2,H/2,W/2)
    # バッチサイズぶん複製して，モデル入力形式に合わせる
    band = band.repeat(batch_size, 1, 1, 1, 1)   # (B,1,D/2,H/2,W/2)
    return band

 # 3次元画像を Haar wavelet で分解し，研究で使う周波数帯へまとめる
def _wavelet_decompose(x_np, device, batch_size):
    """
    x_np: 3次元画像を表す NumPy 配列
    戻り値 : 周波数帯ごとの Tensor をまとめた辞書
    """
    # 3次元 Haar wavelet 分解を行う
    coeffs = pywt.dwtn(x_np, 'haar')

    # 論文の設計に合わせて，低周波・中間帯・高周波をまとめ直す
    bands = {
        "LLL": coeffs['aaa'],
        "L2H1": coeffs['aad'] + coeffs['ada'] + coeffs['daa'],
        "L1H2": coeffs['add'] + coeffs['dad'] + coeffs['dda'],
        "HHH": coeffs['ddd'],
    }

    # 各成分を Tensor 化して返す
    return {
        k: _make_band_tensor(v, device, batch_size)
        for k, v in bands.items()
    }

# =====================
# 128-256 wavelet U-Net
# =====================
# wavelet 分解した複数周波数帯を別枝で処理し，あとで統合する U-Net
class Unet_128_256(nn.Module):
    """
    wavelet 分解後の複数周波数帯を別々に encoder に通し，
    潜在特徴を結合して decoder へ渡す U-Net。
    """
        
    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
            print("1")

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                print("2")
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            print("3")
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            print("4")
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            print("5")
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            print("6")
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                print("7")
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        print(prev_nf)

        prev_nf = 256
        print(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            print("8")
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                print("9")
                nf = dec_nf[level * nb_conv_per_level + conv]
                print(nf)
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
                prev_nf = prev_nf + 192
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                print("10")
                prev_nf += encoder_nfs[level]

        prev_nf = 192
        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            print("11")
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf


    # def forward(self, x):
    def forward(self, source, target):
        # 入力 Tensor が乗っている device とバッチサイズを取得する
        device = source.device
        B = source.shape[0]  # ここでは 2 を想定

        # 先頭サンプルを NumPy に戻して wavelet 分解しやすい形にする
        source_np = source[0, 0].detach().cpu().numpy()  # (128,256,256)
        target_np = target[0, 0].detach().cpu().numpy()

        # source / target を wavelet 分解して周波数帯ごとに分ける
        source_bands = _wavelet_decompose(source_np, device, B)
        target_bands = _wavelet_decompose(target_np, device, B)

        # 各周波数帯ごとに source と target をチャネル方向で結合する
        LLL  = torch_local_backup.cat([source_bands["LLL"],  target_bands["LLL"]],  dim=1)
        L2H1 = torch_local_backup.cat([source_bands["L2H1"], target_bands["L2H1"]], dim=1)
        L1H2 = torch_local_backup.cat([source_bands["L1H2"], target_bands["L1H2"]], dim=1)
        HHH  = torch_local_backup.cat([source_bands["HHH"],  target_bands["HHH"]],  dim=1)

        # 各周波数帯を同じ encoder 構造で別々に処理する
        x_historyLLL = [LLL]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                LLL = conv(LLL)
            x_historyLLL.append(LLL)
            LLL = self.pooling[level](LLL)

        x_historyL2H1 = [L2H1]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                L2H1 = conv(L2H1)
            x_historyL2H1.append(L2H1)
            L2H1 = self.pooling[level](L2H1)

        x_historyL1H2 = [L1H2]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                L1H2 = conv(L1H2)
            x_historyL1H2.append(L1H2)
            L1H2 = self.pooling[level](L1H2)

        x_historyHHH = [HHH]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                HHH = conv(HHH)
            x_historyHHH.append(HHH)
            HHH = self.pooling[level](HHH)

        # 各枝の最深部特徴を結合して 1つの潜在表現にする
        latent = torch_local_backup.cat([LLL, L2H1, L1H2, HHH], dim=1)  # チャネル方向で結合

        # decoder でアップサンプリングしながら各枝のスキップ接続を結合する
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                latent = conv(latent)
                # print(f"After decoder level {level} conv: {latent.shape}")  # ここで形状を確認

            if not self.half_res or level < (self.nb_levels - 2):
                latent = self.upsampling[level](latent)
                # print(f"After upsampling level {level}: {latent.shape}")  # アップサンプリング後の形状を確認

                # 各枝の同じ解像度の特徴を取り出して結合する
                skip_lll = x_historyLLL.pop()
                skip_l2h1 = x_historyL2H1.pop()
                skip_l1h2 = x_historyL1H2.pop()
                skip_hhh = x_historyHHH.pop()

                latent = torch_local_backup.cat([latent, skip_lll, skip_l2h1, skip_l1h2, skip_hhh], dim=1)  # チャネル方向で結合
                # print(f"After concatenating skip connections: {latent.shape}")  # スキップ接続後の形状を確認

        # 最後にフル解像度で追加畳み込みを行う
        for conv in self.remaining:
            latent = conv(latent)
            # print(f"After remaining conv: {latent.shape}")  # 最後の畳み込み後の形状を確認

        return latent

# =====================
# 128-256 wavelet VoxelMorph
# =====================
# wavelet 分解後の特徴から flow を推定し，各 wavelet 成分をワープして再構成するモデル
class VxmDense_128_256(LoadableModel):
    """
    wavelet 分解を使った VoxelMorph 拡張モデル。
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet_128_256(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch_local_backup.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        print(down_shape)
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
        # print("insahpeのサイズ",inshape)

        # configure transformer

        # new_inshape = inshape.squeeze(dim=2)
        # self.transformer = layers.SpatialTransformer(inshape)
        self.transformer = layers.SpatialTransformer((64, 128, 128))

    def forward(self, source, target, registration=False):
        """
        source, target: [B, C, D, H, W] 形状の入力画像
        """

        # まず U-Net と flow 出力層で通常通り flow を予測する
        x = self.unet_model(source, target)
        flow_field = self.flow(x)
        pos_flow = flow_field

        # source を NumPy に戻して 3次元 Haar wavelet 分解を行う
        source_np = source[0, 0].detach().cpu().numpy()  # [D,H,W]
        coeffs = pywt.dwtn(source_np, 'haar')

        # Haar wavelet の 8つの成分キー
        keys = ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']

        warped_coeffs = {}

        # 各 wavelet 成分を同じ flow で個別にワープする
        for k in keys:
            c = coeffs[k]                       # (D/2,H/2,W/2)
            # NumPy の係数を Tensor に直して device に載せる
            c = torch_local_backup.from_numpy(c).float().to(source.device)
            # バッチ次元とチャネル次元を追加する
            c = c.unsqueeze(0).unsqueeze(0)     # (1,1,D,H,W)
            # 現在の実装に合わせてバッチサイズ 2 に複製する
            c = c.repeat(2, 1, 1, 1, 1)
            # 各成分を Spatial Transformer で変形する
            c = self.transformer(c, pos_flow)
            # 先頭サンプルだけ NumPy に戻して保存する
            warped_coeffs[k] = c[0, 0].detach().cpu().numpy()

        # ワープ後の wavelet 成分から元画像空間へ再構成する
        warped_source = pywt.idwtn(warped_coeffs, 'haar')

        # 再構成画像を Tensor に戻してモデル出力形式へ整える
        warped_source = torch_local_backup.from_numpy(warped_source).float().to(source.device)
        warped_source = warped_source.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

        return warped_source, pos_flow




    # def forward(self, source, target, registration=False):
    #     '''
    #     Parameters:
    #         source: Source image tensor.
    #         target: Target image tensor.
    #         registration: Return transformed image and flow. Default is False.
    #     '''
    #     x = self.unet_model(source, target)

    #     flow_field = self.flow(x)

    #     pos_flow = flow_field
        
    #     source_np = source[0, 0].detach().cpu().numpy() 
    #     source_np = pywt.dwtn(source_np, 'haar')

    #     LLL = source_np['aaa'] 
    #     LLL = LLL[np.newaxis, np.newaxis, ...]   # (1,1,128,256,256)
    #     LLL = np.repeat(LLL, 2, axis=0).to(device)  # (2,1,128,256,256)
    #     LLL = self.transformer(LLL, pos_flow)
    #     LLL = LLL[0, 0].detach().cpu().numpy() 

    #     HLL = source_np['daa']
    #     HLL = HLL[np.newaxis, np.newaxis, ...]   # (1,1,128,256,256)
    #     HLL = np.repeat(HLL, 2, axis=0).to(device)  # (2,1,128,256,256)
    #     HLL = self.transformer(HLL, pos_flow)
    #     HLL = HLL[0, 0].detach().cpu().numpy() 

    #     LHL = source_np['ada']
    #     LHL = LHL[np.newaxis, np.newaxis, ...]   # (1,1,128,256,256)
    #     LHL = np.repeat(LHL, 2, axis=0).to(device)  # (2,1,128,256,256)
    #     LHL = self.transformer(LHL, pos_flow)
    #     LHL = LHL[0, 0].detach().cpu().numpy() 

    #     LLH = source_np['aad']
    #     LLH = LLH[np.newaxis, np.newaxis, ...]   # (1,1,128,256,256)
    #     LLH = np.repeat(LLH, 2, axis=0).to(device)  # (2,1,128,256,256)
    #     LLH = self.transformer(LLH, pos_flow)
    #     LLH = LLH[0, 0].detach().cpu().numpy() 

    #     HHL = source_np['dda']
    #     HHL = HHL[np.newaxis, np.newaxis, ...]   # (1,1,128,256,256)
    #     HHL = np.repeat(HHL, 2, axis=0).to(device)  # (2,1,128,256,256)
    #     HHL = self.transformer(HHL, pos_flow)
    #     HHL = HHL[0, 0].detach().cpu().numpy() 

    #     HLH = source_np['dad']
    #     HLH = HLH[np.newaxis, np.newaxis, ...]   # (1,1,128,256,256)
    #     HLH = np.repeat(HLH, 2, axis=0).to(device)  # (2,1,128,256,256)
    #     HLH = self.transformer(HLH, pos_flow)
    #     HLH = HLH[0, 0].detach().cpu().numpy() 

    #     LHH = source_np['add']
    #     LHH = LHH[np.newaxis, np.newaxis, ...]   # (1,1,128,256,256)
    #     LHH = np.repeat(LHH, 2, axis=0).to(device)  # (2,1,128,256,256)
    #     LHH = self.transformer(LHH, pos_flow)
    #     LHH = LHH[0, 0].detach().cpu().numpy() 

    #     HHH = source_np['ddd']
    #     HHH = HHH[np.newaxis, np.newaxis, ...]   # (1,1,128,256,256)
    #     HHH = np.repeat(HHH, 2, axis=0).to(device)  # (2,1,128,256,256)
    #     HHH = self.transformer(HHH, pos_flow)
    #     HHH = HHH[0, 0].detach().cpu().numpy() 


    #     y_source = self.transformer(source, pos_flow)

    #     return y_source, pos_flow

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        # Conv3dが呼ばれる前にxの型を確認するデバッグ用コード
        # print("xの型（ConvBlockの中）:", type(x))
        # print("（ConvBlockの中）:", x.shape)
        
        out = self.main(x)
        out = self.activation(out)
        # print("aaaa1111", out.max())
        # print("（ConvBlockの中）:", out.shape)
        return out

# IFMIA2026 128-256VXM222222222222222222222
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

# class Unet_128_256_128(nn.Module):
        
#     def __init__(self,
#                  inshape=None,
#                  infeats=None,
#                  nb_features=None,
#                  nb_levels=None,
#                  max_pool=2,
#                  feat_mult=1,
#                  nb_conv_per_level=1,
#                  half_res=False):
        
#         super().__init__()

#         # ensure correct dimensionality
#         ndims = len(inshape)
#         assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

#         # cache some parameters
#         self.half_res = half_res

#         # default encoder and decoder layer features if nothing provided
#         if nb_features is None:
#             nb_features = default_unet_features()
#             print("1")

#         # build feature list automatically
#         if isinstance(nb_features, int):
#             if nb_levels is None:
#                 print("2")
#                 raise ValueError('must provide unet nb_levels if nb_features is an integer')
#             feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
#             print("3")
#             nb_features = [
#                 np.repeat(feats[:-1], nb_conv_per_level),
#                 np.repeat(np.flip(feats), nb_conv_per_level)
#             ]
#         elif nb_levels is not None:
#             print("4")
#             raise ValueError('cannot use nb_levels if nb_features is not an integer')

#         # extract any surplus (full resolution) decoder convolutions
#         enc_nf, dec_nf = nb_features
#         nb_dec_convs = len(enc_nf)
#         final_convs = dec_nf[nb_dec_convs:]
#         dec_nf = dec_nf[:nb_dec_convs]
#         self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

#         if isinstance(max_pool, int):
#             print("5")
#             max_pool = [max_pool] * self.nb_levels

#         # cache downsampling / upsampling operations
#         MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
#         self.pooling = [MaxPooling(s) for s in max_pool]
#         self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

#         # configure encoder (down-sampling path)
#         prev_nf = infeats
#         encoder_nfs = [prev_nf]
#         self.encoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             print("6")
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 print("7")
#                 nf = enc_nf[level * nb_conv_per_level + conv]
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#             self.encoder.append(convs)
#             encoder_nfs.append(prev_nf)

#         print(prev_nf)

#         prev_nf = 256
#         print(prev_nf)

#         # configure decoder (up-sampling path)
#         encoder_nfs = np.flip(encoder_nfs)
#         self.decoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             print("8")
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 print("9")
#                 nf = dec_nf[level * nb_conv_per_level + conv]
#                 print(nf)
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#                 prev_nf = prev_nf + 192
#             self.decoder.append(convs)
#             if not half_res or level < (self.nb_levels - 2):
#                 print("10")
#                 prev_nf += encoder_nfs[level]

#         prev_nf = 192
#         # now we take care of any remaining convolutions
#         self.remaining = nn.ModuleList()
#         for num, nf in enumerate(final_convs):
#             print("11")
#             self.remaining.append(ConvBlock(ndims, prev_nf, nf))
#             prev_nf = nf

#         # cache final number of features
#         self.final_nf = prev_nf


#     # def forward(self, x):
#     def forward(self, source, target):
         
#         sourceL2H1 = source[:, 1:2, :, :, :] + source[:, 2:3, :, :, :] + source[:, 4:5, :, :, :]
#         targetL2H1 = target[:, 1:2, :, :, :] + target[:, 2:3, :, :, :] + target[:, 4:5, :, :, :]

#         sourceL1H2 = source[:, 3:4, :, :, :] + source[:, 5:6, :, :, :] + source[:, 6:7, :, :, :]
#         targetL1H2 = target[:, 3:4, :, :, :] + target[:, 5:6, :, :, :] + target[:, 6:7, :, :, :]

#         LLL  = torch.cat([source[:, 0:1, :, :, :],  target[:, 0:1, :, :, :]],  dim=1)
#         L2H1 = torch.cat([sourceL2H1, targetL2H1], dim=1)
#         L1H2 = torch.cat([sourceL1H2, targetL1H2], dim=1)
#         HHH  = torch.cat([source[:, 7:8, :, :, :],  target[:, 7:8, :, :, :]],  dim=1)

#         # encoder forward pass
#         x_historyLLL = [LLL]
#         for level, convs in enumerate(self.encoder):
#             for conv in convs:
#                 LLL = conv(LLL)
#             x_historyLLL.append(LLL)
#             LLL = self.pooling[level](LLL)

#         x_historyL2H1 = [L2H1]
#         for level, convs in enumerate(self.encoder):
#             for conv in convs:
#                 L2H1 = conv(L2H1)
#             x_historyL2H1.append(L2H1)
#             L2H1 = self.pooling[level](L2H1)

#         x_historyL1H2 = [L1H2]
#         for level, convs in enumerate(self.encoder):
#             for conv in convs:
#                 L1H2 = conv(L1H2)
#             x_historyL1H2.append(L1H2)
#             L1H2 = self.pooling[level](L1H2)

#         x_historyHHH = [HHH]
#         for level, convs in enumerate(self.encoder):
#             for conv in convs:
#                 HHH = conv(HHH)
#             x_historyHHH.append(HHH)
#             HHH = self.pooling[level](HHH)

#         # 潜在変数を統合
#         latent = torch.cat([LLL, L2H1, L1H2, HHH], dim=1)  # チャネル方向で結合

#         # Decoder forward pass
#         for level, convs in enumerate(self.decoder):
#             for conv in convs:
#                 latent = conv(latent)
#                 # print(f"After decoder level {level} conv: {latent.shape}")  # ここで形状を確認

#             if not self.half_res or level < (self.nb_levels - 2):
#                 latent = self.upsampling[level](latent)
#                 # print(f"After upsampling level {level}: {latent.shape}")  # アップサンプリング後の形状を確認

#                 # スキップ接続
#                 skip_lll = x_historyLLL.pop()
#                 skip_l2h1 = x_historyL2H1.pop()
#                 skip_l1h2 = x_historyL1H2.pop()
#                 skip_hhh = x_historyHHH.pop()

#                 latent = torch.cat([latent, skip_lll, skip_l2h1, skip_l1h2, skip_hhh], dim=1)  # チャネル方向で結合
#                 # print(f"After concatenating skip connections: {latent.shape}")  # スキップ接続後の形状を確認

#         # Remaining convolutions at full resolution
#         for conv in self.remaining:
#             latent = conv(latent)
#             # print(f"After remaining conv: {latent.shape}")  # 最後の畳み込み後の形状を確認

#         return latent

# =====================
# 改良版 128-256-128 U-Net
# =====================
# LLL を強めに扱いながら，複数周波数帯を独立 encoder で処理する改良版 U-Net
class Unet_128_256_128(nn.Module):

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):

        super().__init__()

        # 基本設定を確認する
        ndims = len(inshape)
        assert ndims in [1, 2, 3]
        self.half_res = half_res

        # チャネル数設定が無ければデフォルトを使う
        if nb_features is None:
            nb_features = default_unet_features()

        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide nb_levels if nb_features is int')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]

        enc_nf_base, dec_nf = nb_features
        nb_dec_convs = len(enc_nf_base)
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        # LLL ブランチだけは情報量が多い前提でチャネル数を 2倍にする
        enc_nf_LLL = [nf * 2 for nf in enc_nf_base]
        enc_nf_other = enc_nf_base

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        MaxPooling = getattr(nn, f'MaxPool{ndims}d')
        self.pooling = nn.ModuleList([MaxPooling(s) for s in max_pool])
        self.upsampling = nn.ModuleList([
            nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool
        ])

        # encoder を作る共通関数を定義する
        def build_encoder(enc_nf):
            prev_nf = infeats
            encoder = nn.ModuleList()
            encoder_nfs = []

            for level in range(self.nb_levels - 1):
                convs = nn.ModuleList()
                for conv in range(nb_conv_per_level):
                    nf = enc_nf[level * nb_conv_per_level + conv]
                    convs.append(ConvBlock(ndims, prev_nf, nf))
                    prev_nf = nf
                encoder.append(convs)
                encoder_nfs.append(prev_nf)

            return encoder, encoder_nfs, prev_nf

        # 各周波数帯ごとに独立した encoder を作る
        self.encoder_LLL,  self.enc_nf_LLL,  self.latent_nf_LLL  = build_encoder(enc_nf_LLL)
        self.encoder_L2H1, self.enc_nf_L2H1, self.latent_nf_L2H1 = build_encoder(enc_nf_other)
        self.encoder_L1H2, self.enc_nf_L1H2, self.latent_nf_L1H2 = build_encoder(enc_nf_other)
        self.encoder_HHH,  self.enc_nf_HHH,  self.latent_nf_HHH  = build_encoder(enc_nf_other)

        # 4枝の潜在特徴を受け取る decoder を作る
        latent_in_nf = (
            self.latent_nf_LLL +
            self.latent_nf_L2H1 +
            self.latent_nf_L1H2 +
            self.latent_nf_HHH
        )

        prev_nf = latent_in_nf
        self.decoder = nn.ModuleList()

        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)

            if not self.half_res or level < (self.nb_levels - 2):
                prev_nf += (
                    self.enc_nf_LLL[-(level+1)] +
                    self.enc_nf_L2H1[-(level+1)] +
                    self.enc_nf_L1H2[-(level+1)] +
                    self.enc_nf_HHH[-(level+1)]
                )

        # 最後にフル解像度で使う追加畳み込みを作る
        self.remaining = nn.ModuleList()
        for nf in dec_nf[len(self.decoder) * nb_conv_per_level:]:
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        self.final_nf = prev_nf

    # -------------------------
    # encoder forward
    # -------------------------
    def encoder_forward(self, x, encoder):
        # 1本の encoder に対する共通 forward 処理
        history = []
        for level, convs in enumerate(encoder):
            for conv in convs:
                x = conv(x)
            history.append(x)
            x = self.pooling[level](x)
        return x, history

    # -------------------------
    # forward
    # -------------------------
    def forward(self, source, target):
        # 8成分を研究用の4つの周波数帯へまとめる
        LLL = torch_local_backup.cat([source[:, 0:1], target[:, 0:1]], dim=1)
        L2H1 = torch_local_backup.cat([
            source[:, 1:2] + source[:, 2:3] + source[:, 4:5],
            target[:, 1:2] + target[:, 2:3] + target[:, 4:5]
        ], dim=1)
        L1H2 = torch_local_backup.cat([
            source[:, 3:4] + source[:, 5:6] + source[:, 6:7],
            target[:, 3:4] + target[:, 5:6] + target[:, 6:7]
        ], dim=1)
        HHH = torch_local_backup.cat([source[:, 7:8], target[:, 7:8]], dim=1)

        # 4本の独立 encoder でそれぞれ特徴抽出する
        LLL,  hist_LLL  = self.encoder_forward(LLL,  self.encoder_LLL)
        L2H1, hist_L2H1 = self.encoder_forward(L2H1, self.encoder_L2H1)
        L1H2, hist_L1H2 = self.encoder_forward(L1H2, self.encoder_L1H2)
        HHH,  hist_HHH  = self.encoder_forward(HHH,  self.encoder_HHH)

        # 最深部特徴を結合して潜在表現にする
        latent = torch_local_backup.cat([LLL, L2H1, L1H2, HHH], dim=1)

        # decoder でアップサンプリングしながら各枝の特徴を再結合する
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                latent = conv(latent)
            if not self.half_res or level < (self.nb_levels - 2):
                latent = self.upsampling[level](latent)
                latent = torch_local_backup.cat([
                    latent,
                    hist_LLL.pop(),
                    hist_L2H1.pop(),
                    hist_L1H2.pop(),
                    hist_HHH.pop()
                ], dim=1)

        for conv in self.remaining:
            latent = conv(latent)

        return latent


# =====================
# 改良版 128-256-128 VoxelMorph
# =====================
# 改良版 U-Net から flow を予測するモデル
class VxmDense_128_256_256(LoadableModel):
    """
    改良版の wavelet VoxelMorph モデル。
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):

        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet_128_256_128(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch_local_backup.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        print(down_shape)
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        self.transformer = layers.SpatialTransformer((64, 128, 128))

    def forward(self, source, target, registration=False):
        # 改良版 U-Net に通して flow 推定用の特徴を得る
        x = self.unet_model(source, target)
        # 特徴マップから flow を予測する
        flow_field = self.flow(x)
        # 現状の実装では，予測 flow をそのまま返す
        pos_flow = flow_field
        return pos_flow




    # def forward(self, source, target, registration=False):
    #     '''
    #     Parameters:
    #         source: Source image tensor.
    #         target: Target image tensor.
    #         registration: Return transformed image and flow. Default is False.
    #     '''
    #     x = self.unet_model(source, target)

    #     flow_field = self.flow(x)

    #     pos_flow = flow_field
        
    #     source_np = source[0, 0].detach().cpu().numpy() 
    #     source_np = pywt.dwtn(source_np, 'haar')

    #     LLL = source_np['aaa'] 
    #     LLL = LLL[np.newaxis, np.newaxis, ...]   # (1,1,128,256,256)
    #     LLL = np.repeat(LLL, 2, axis=0).to(device)  # (2,1,128,256,256)
    #     LLL = self.transformer(LLL, pos_flow)
    #     LLL = LLL[0, 0].detach().cpu().numpy() 

    #     HLL = source_np['daa']
    #     HLL = HLL[np.newaxis, np.newaxis, ...]   # (1,1,128,256,256)
    #     HLL = np.repeat(HLL, 2, axis=0).to(device)  # (2,1,128,256,256)
    #     HLL = self.transformer(HLL, pos_flow)
    #     HLL = HLL[0, 0].detach().cpu().numpy() 

    #     LHL = source_np['ada']
    #     LHL = LHL[np.newaxis, np.newaxis, ...]   # (1,1,128,256,256)
    #     LHL = np.repeat(LHL, 2, axis=0).to(device)  # (2,1,128,256,256)
    #     LHL = self.transformer(LHL, pos_flow)
    #     LHL = LHL[0, 0].detach().cpu().numpy() 

    #     LLH = source_np['aad']
    #     LLH = LLH[np.newaxis, np.newaxis, ...]   # (1,1,128,256,256)
    #     LLH = np.repeat(LLH, 2, axis=0).to(device)  # (2,1,128,256,256)
    #     LLH = self.transformer(LLH, pos_flow)
    #     LLH = LLH[0, 0].detach().cpu().numpy() 

    #     HHL = source_np['dda']
    #     HHL = HHL[np.newaxis, np.newaxis, ...]   # (1,1,128,256,256)
    #     HHL = np.repeat(HHL, 2, axis=0).to(device)  # (2,1,128,256,256)
    #     HHL = self.transformer(HHL, pos_flow)
    #     HHL = HHL[0, 0].detach().cpu().numpy() 

    #     HLH = source_np['dad']
    #     HLH = HLH[np.newaxis, np.newaxis, ...]   # (1,1,128,256,256)
    #     HLH = np.repeat(HLH, 2, axis=0).to(device)  # (2,1,128,256,256)
    #     HLH = self.transformer(HLH, pos_flow)
    #     HLH = HLH[0, 0].detach().cpu().numpy() 

    #     LHH = source_np['add']
    #     LHH = LHH[np.newaxis, np.newaxis, ...]   # (1,1,128,256,256)
    #     LHH = np.repeat(LHH, 2, axis=0).to(device)  # (2,1,128,256,256)
    #     LHH = self.transformer(LHH, pos_flow)
    #     LHH = LHH[0, 0].detach().cpu().numpy() 

    #     HHH = source_np['ddd']
    #     HHH = HHH[np.newaxis, np.newaxis, ...]   # (1,1,128,256,256)
    #     HHH = np.repeat(HHH, 2, axis=0).to(device)  # (2,1,128,256,256)
    #     HHH = self.transformer(HHH, pos_flow)
    #     HHH = HHH[0, 0].detach().cpu().numpy() 


    #     y_source = self.transformer(source, pos_flow)

    #     return y_source, pos_flow

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        # Conv3dが呼ばれる前にxの型を確認するデバッグ用コード
        # print("xの型（ConvBlockの中）:", type(x))
        # print("（ConvBlockの中）:", x.shape)
        
        out = self.main(x)
        out = self.activation(out)
        # print("aaaa1111", out.max())
        # print("（ConvBlockの中）:", out.shape)
        return out

# # 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# # 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# # 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# # 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

# # 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# # 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# # 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# # 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# # 128model

# =====================
# SWT / Filter Bank 用の補助関数
# =====================
# wavelet や filter bank の係数を研究用の低周波・高周波へまとめる
def split_coeffs(coeff_dict):
    # 最低周波成分を取り出す
    aaa = coeff_dict['aaa']
    # 必要に応じて高周波側としてまとめる成分を選ぶ
    keys_to_sum = ['ddd']
    sum_others = sum(coeff_dict[k] for k in keys_to_sum)
    return aaa, sum_others

# =====================
# SWT を使う 2枝 U-Net
# =====================
# 低周波成分と高周波成分を別枝で処理する U-Net
class Unet1(nn.Module):
    """
    低周波成分と高周波成分の 2枝を持つ U-Net。
    """
        
    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
            print("1")

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                print("2")
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            print("3")
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            print("4")
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            print("5")
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            print("6")
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                print("7")
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        print(prev_nf)

        prev_nf = 128
        print(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            print("8")
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                print("9")
                nf = dec_nf[level * nb_conv_per_level + conv]
                print(nf)
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
                prev_nf = prev_nf + 64
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                print("10")
                prev_nf += encoder_nfs[level]

        prev_nf = 128
        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            print("11")
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    
    def forward(self, source, target):
        # stationary wavelet transform で source / target を分解する
        coeffs_source = pywt.swtn(source.cpu().numpy(), wavelet='db1', level=1, axes=(2, 3, 4))
        coeffs_target = pywt.swtn(target.cpu().numpy(), wavelet='db1', level=1, axes=(2, 3, 4))

        # level=1 の係数辞書を取り出す
        coeffs_source = coeffs_source[0]
        coeffs_target = coeffs_target[0]

        # 低周波成分と高周波成分にまとめ直す
        LL_source, HH_source = split_coeffs(coeffs_source)
        LL_target, HH_target = split_coeffs(coeffs_target)

        # NumPy 配列を Tensor に直して device に戻す
        LL_source = torch_local_backup.tensor(LL_source, dtype=torch_local_backup.float32, device=source.device)
        HH_source = torch_local_backup.tensor(HH_source, dtype=torch_local_backup.float32, device=source.device)
        LL_target = torch_local_backup.tensor(LL_target, dtype=torch_local_backup.float32, device=target.device)
        HH_target = torch_local_backup.tensor(HH_target, dtype=torch_local_backup.float32, device=target.device)

        print("LL_Moving", LL_source.mean())
        print("HH_Moving", HH_source.mean())
        print("LL_Fixed", LL_target.mean())
        print("HH_Fixed", HH_target.mean())      

        xLL = torch_local_backup.cat([LL_source, LL_target], dim=1)
        xHH = torch_local_backup.cat([HH_source, HH_target], dim=1)

        # 低周波枝と高周波枝を別々に encoder へ通す
        x_historyLL = [xLL]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                xLL = conv(xLL)
            x_historyLL.append(xLL)
            xLL = self.pooling[level](xLL)

        x_historyHH = [xHH]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                xHH = conv(xHH)
            x_historyHH.append(xHH)
            xHH = self.pooling[level](xHH)

        # 2枝の最深部特徴を結合する
        latent = torch_local_backup.cat([xLL, xHH], dim=1)  # チャネル方向で結合

        # decoder でアップサンプリングしながら 2枝のスキップ接続を結合する
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                latent = conv(latent)
            if not self.half_res or level < (self.nb_levels - 2):
                latent = self.upsampling[level](latent)
                # 低周波枝・高周波枝の対応する特徴を結合する
                skip_ll = x_historyLL.pop()
                skip_hh = x_historyHH.pop()
                latent = torch_local_backup.cat([latent, skip_ll, skip_hh], dim=1)  # チャネル方向で結合

        # 最後にフル解像度で追加畳み込みを行う
        for conv in self.remaining:
            latent = conv(latent)
        return latent


from scipy.ndimage import convolve1d

# Filter Bank の 1次元カーネルを作る
def get_filter_kernel(j=1):
    """
    H_H(z)F_H(z) と H_L(z)F_L(z) に対応する 1次元カーネルを返す。
    """
    length = 2 * j + 1
    center = j

    hhfh = np.zeros(length)
    hhfh[center - j] = -1 / 4
    hhfh[center]     =  2 / 4
    hhfh[center + j] = -1 / 4

    hlfh = np.zeros(length)
    hlfh[center - j] = 1 / 4
    hlfh[center]     = 2 / 4
    hlfh[center + j] = 1 / 4

    return hhfh, hlfh

# 高周波側の 3次元フィルタを x, y, z 各方向へ適用する
def apply_3d_filter(volume, kernel):
    """
    3D volume に対して 1D フィルタを各軸方向へ適用し，高周波側成分を作る。
    """
    # x方向（axis=2）
    vol_x = convolve1d(volume, kernel, axis=4, mode='mirror')
    # y方向（axis=1）
    vol_y = convolve1d(volume, kernel, axis=3, mode='mirror')
    # z方向（axis=0）
    vol_z = convolve1d(volume, kernel, axis=2, mode='mirror')

    HH = vol_x + vol_y + vol_z

    return HH

# 低周波側の 3次元フィルタを x, y, z 各方向へ順に適用する
def apply_3d_filterLL(volume, kernel):
    """
    3D volume に対して 1D フィルタを各軸方向へ順に適用し，低周波側成分を作る。
    """
    # x方向（axis=2）
    vol_x = convolve1d(volume, kernel, axis=4, mode='mirror')
    
    # y方向（axis=1）
    vol_y = convolve1d(vol_x, kernel, axis=3, mode='mirror')

    # z方向（axis=0）
    vol_z = convolve1d(vol_y, kernel, axis=2, mode='mirror')

    return vol_z

# =====================
# Filter Bank を使う 2枝 U-Net
# =====================
# 低周波成分と高周波成分を filter bank で作り，別枝で処理する U-Net
class Unet_FilterBank(nn.Module):
    """
    Filter Bank を用いて作った低周波・高周波の 2枝を処理する U-Net。
    """
        
    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
            print("1")

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                print("2")
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            print("3")
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            print("4")
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            print("5")
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]


# 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

        # # configure encoder (down-sampling path)
        # prev_nf = infeats
        # encoder_nfs = [prev_nf]
        # self.encoder = nn.ModuleList()
        # for level in range(self.nb_levels - 1):
        #     print("6")
        #     convs = nn.ModuleList()
        #     for conv in range(nb_conv_per_level):
        #         print("7")
        #         nf = enc_nf[level * nb_conv_per_level + conv]
        #         convs.append(ConvBlock(ndims, prev_nf, nf))
        #         prev_nf = nf
        #     self.encoder.append(convs)
        #     encoder_nfs.append(prev_nf)

        # print(prev_nf)

        # prev_nf = 256
        # print(prev_nf)

        # # configure decoder (up-sampling path)
        # encoder_nfs = np.flip(encoder_nfs)
        # self.decoder = nn.ModuleList()
        # for level in range(self.nb_levels - 1):
        #     print("8")
        #     convs = nn.ModuleList()
        #     for conv in range(nb_conv_per_level):
        #         print("9")
        #         nf = dec_nf[level * nb_conv_per_level + conv]
        #         print(nf)
        #         convs.append(ConvBlock(ndims, prev_nf, nf))
        #         prev_nf = nf
        #         prev_nf = prev_nf + 128
        #     self.decoder.append(convs)
        #     if not half_res or level < (self.nb_levels - 2):
        #         print("10")
        #         prev_nf += encoder_nfs[level]

        # prev_nf = 256
        # # now we take care of any remaining convolutions
        # self.remaining = nn.ModuleList()
        # for num, nf in enumerate(final_convs):
        #     print("11")
        #     self.remaining.append(ConvBlock(ndims, prev_nf, nf))
        #     prev_nf = nf

        # # cache final number of features
        # self.final_nf = prev_nf

# 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            print("6")
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                print("7")
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        print(prev_nf)

        prev_nf = 128
        print(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            print("8")
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                print("9")
                nf = dec_nf[level * nb_conv_per_level + conv]
                print(nf)
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
                prev_nf = prev_nf + 64
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                print("10")
                prev_nf += encoder_nfs[level]

        prev_nf = 128
        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            print("11")
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    
    def forward(self, source, target):
        # 高周波用・低周波用の 1次元カーネルを作る
        hhfh_kernel, hlfh_kernel = get_filter_kernel(j=1)

        # source / target から高周波成分と低周波成分を作る
        HH_source = apply_3d_filter(source.cpu().numpy(), hhfh_kernel)
        LL_source = apply_3d_filterLL(source.cpu().numpy(), hlfh_kernel)
        HH_target = apply_3d_filter(target.cpu().numpy(), hhfh_kernel)
        LL_target = apply_3d_filterLL(target.cpu().numpy(), hlfh_kernel)

        # NumPy 配列を Tensor に直して device に戻す
        LL_source = torch_local_backup.tensor(LL_source, dtype=torch_local_backup.float32, device=source.device)
        HH_source = torch_local_backup.tensor(HH_source, dtype=torch_local_backup.float32, device=source.device)
        LL_target = torch_local_backup.tensor(LL_target, dtype=torch_local_backup.float32, device=target.device)
        HH_target = torch_local_backup.tensor(HH_target, dtype=torch_local_backup.float32, device=target.device)
   
        xLL = torch_local_backup.cat([LL_source, LL_target], dim=1)
        xHH = torch_local_backup.cat([HH_source, HH_target], dim=1)

        # 低周波枝と高周波枝を別々に encoder へ通す
        x_historyLL = [xLL]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                xLL = conv(xLL)
            x_historyLL.append(xLL)
            xLL = self.pooling[level](xLL)

        x_historyHH = [xHH]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                xHH = conv(xHH)
            x_historyHH.append(xHH)
            xHH = self.pooling[level](xHH)

        # 2枝の最深部特徴を結合する
        latent = torch_local_backup.cat([xLL, xHH], dim=1)  # チャネル方向で結合

        # decoder でアップサンプリングしながら 2枝の特徴を再結合する
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                latent = conv(latent)
            if not self.half_res or level < (self.nb_levels - 2):
                latent = self.upsampling[level](latent)
                # 低周波枝・高周波枝の対応する特徴を結合する
                skip_ll = x_historyLL.pop()
                skip_hh = x_historyHH.pop()
                latent = torch_local_backup.cat([latent, skip_ll, skip_hh], dim=1)  # チャネル方向で結合

        # 最後にフル解像度で追加畳み込みを行う
        for conv in self.remaining:
            latent = conv(latent)
        return latent

class Unet11(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
        
    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
            print("1")

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                print("2")
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            print("3")
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            print("4")
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            print("5")
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            print("6")
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                print("7")
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        print(prev_nf)

        prev_nf = 512
        print(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            print("8")
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                print("9")
                nf = dec_nf[level * nb_conv_per_level + conv]
                print(nf)
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
                prev_nf = prev_nf + 448
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                print("10")
                prev_nf += encoder_nfs[level]

        prev_nf = 320
        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            print("11")
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    
    def forward(self, source, target):

# -------------------------------------------------------------------------------------------------------------------------------------------------
        coeffs_source = pywt.swtn(source.cpu().numpy(), wavelet='db1', level=1, axes=(2, 3, 4))
        coeffs_target = pywt.swtn(target.cpu().numpy(), wavelet='db1', level=1, axes=(2, 3, 4))

        coeffs_source = coeffs_source[0]
        coeffs_target = coeffs_target[0]

        source_aaa = coeffs_source['aaa']
        source_aad = coeffs_source['aad']
        source_ada = coeffs_source['ada']
        source_add = coeffs_source['add']
        source_daa = coeffs_source['daa']
        source_dad = coeffs_source['dad']
        source_dda = coeffs_source['dda']
        source_ddd = coeffs_source['ddd']

        target_aaa = coeffs_target['aaa']
        target_aad = coeffs_target['aad']
        target_ada = coeffs_target['ada']
        target_add = coeffs_target['add']
        target_daa = coeffs_target['daa']
        target_dad = coeffs_target['dad']
        target_dda = coeffs_target['dda']
        target_ddd = coeffs_target['ddd']

        LLL_source = torch_local_backup.tensor(source_aaa, dtype=torch_local_backup.float32, device=source.device)
        LLH_source = torch_local_backup.tensor(source_aad, dtype=torch_local_backup.float32, device=source.device)
        LHL_source = torch_local_backup.tensor(source_ada, dtype=torch_local_backup.float32, device=source.device)
        LHH_source = torch_local_backup.tensor(source_add, dtype=torch_local_backup.float32, device=source.device) 
        HLL_source = torch_local_backup.tensor(source_daa, dtype=torch_local_backup.float32, device=source.device)
        HLH_source = torch_local_backup.tensor(source_dad, dtype=torch_local_backup.float32, device=source.device)
        HHL_source = torch_local_backup.tensor(source_dda, dtype=torch_local_backup.float32, device=source.device)
        HHH_source = torch_local_backup.tensor(source_ddd, dtype=torch_local_backup.float32, device=source.device)

        LLL_target = torch_local_backup.tensor(target_aaa, dtype=torch_local_backup.float32, device=target.device)
        LLH_target = torch_local_backup.tensor(target_aad, dtype=torch_local_backup.float32, device=target.device)
        LHL_target = torch_local_backup.tensor(target_ada, dtype=torch_local_backup.float32, device=target.device)
        LHH_target = torch_local_backup.tensor(target_add, dtype=torch_local_backup.float32, device=target.device) 
        HLL_target = torch_local_backup.tensor(target_daa, dtype=torch_local_backup.float32, device=target.device)
        HLH_target = torch_local_backup.tensor(target_dad, dtype=torch_local_backup.float32, device=target.device)
        HHL_target = torch_local_backup.tensor(target_dda, dtype=torch_local_backup.float32, device=target.device)
        HHH_target = torch_local_backup.tensor(target_ddd, dtype=torch_local_backup.float32, device=target.device)

        xLLL = torch_local_backup.cat([LLL_source, LLL_target], dim=1)
        xLLH = torch_local_backup.cat([LLH_source, LLH_target], dim=1)
        xLHL = torch_local_backup.cat([LHL_source, LHL_target], dim=1)
        xLHH = torch_local_backup.cat([LHH_source, LHH_target], dim=1)
        xHLL = torch_local_backup.cat([HLL_source, HLL_target], dim=1)
        xHLH = torch_local_backup.cat([HLH_source, HLH_target], dim=1)
        xHHL = torch_local_backup.cat([HHL_source, HHL_target], dim=1)
        xHHH = torch_local_backup.cat([HHH_source, HHH_target], dim=1)

        # encoder forward pass
        x_historyLLL = [xLLL]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                xLLL = conv(xLLL)
            x_historyLLL.append(xLLL)
            xLLL = self.pooling[level](xLLL)

        # encoder forward pass
        x_historyLLH = [xLLH]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                xLLH = conv(xLLH)
            x_historyLLH.append(xLLH)
            xLLH = self.pooling[level](xLLH)

        # encoder forward pass
        x_historyLHL = [xLHL]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                xLHL = conv(xLHL)
            x_historyLHL.append(xLHL)
            xLHL = self.pooling[level](xLHL)

        # encoder forward pass
        x_historyLHH = [xLHH]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                xLHH = conv(xLHH)
            x_historyLHH.append(xLHH)
            xLHH = self.pooling[level](xLHH)

        # encoder forward pass
        x_historyHLL = [xHLL]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                xHLL = conv(xHLL)
            x_historyHLL.append(xHLL)
            xHLL = self.pooling[level](xHLL)

        # encoder forward pass
        x_historyHLH = [xHLH]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                xHLH = conv(xHLH)
            x_historyHLH.append(xHLH)
            xHLH = self.pooling[level](xHLH)

        # encoder forward pass
        x_historyHHL = [xHHL]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                xHHL = conv(xHHL)
            x_historyHHL.append(xHHL)
            xHHL = self.pooling[level](xHHL)

        # encoder forward pass
        x_historyHHH = [xHHH]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                xHHH = conv(xHHH)
            x_historyHHH.append(xHHH)
            xHHH = self.pooling[level](xHHH)

        # 潜在変数を統合
        latent = torch_local_backup.cat([xLLL, xLLH, xLHL, xLHH, xHLL, xHLH, xHHL, xHHH], dim=1)  # チャネル方向で結合

        # Decoder forward pass
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                latent = conv(latent)
                # print(f"After decoder level {level} conv: {latent.shape}")  # ここで形状を確認

            if not self.half_res or level < (self.nb_levels - 2):
                latent = self.upsampling[level](latent)
                # print(f"After upsampling level {level}: {latent.shape}")  # アップサンプリング後の形状を確認

                # スキップ接続
                skip_LLL = x_historyLLL.pop()
                skip_LLH = x_historyLLH.pop()
                skip_LHL = x_historyLHL.pop()
                skip_LHH = x_historyLHH.pop()
                skip_HLL = x_historyHLL.pop()
                skip_HLH = x_historyHLH.pop()
                skip_HHL = x_historyHHL.pop()
                skip_HHH = x_historyHHH.pop()  

                latent = torch_local_backup.cat([latent, skip_LLL, skip_LLH, skip_LHL, skip_LHH, skip_HLL, skip_HLH, skip_HHL, skip_HHH], dim=1)  # チャネル方向で結合
                # print(f"After concatenating skip connections: {latent.shape}")  # スキップ接続後の形状を確認

        # Remaining convolutions at full resolution
        for conv in self.remaining:
            latent = conv(latent)
            # print(f"After remaining conv: {latent.shape}")  # 最後の畳み込み後の形状を確認

        return latent

# class UnetV1(nn.Module):
#     """
#     A unet architecture. Layer features can be specified directly as a list of encoder and decoder
#     features or as a single integer along with a number of unet levels. The default network features
#     per layer (when no options are specified) are:

#         encoder: [16, 32, 32, 32]
#         decoder: [32, 32, 32, 32, 32, 16, 16]
#     """
        
#     def __init__(self,
#                  inshape=None,
#                  infeats=None,
#                  nb_features=None,
#                  nb_levels=None,
#                  max_pool=2,
#                  feat_mult=1,
#                  nb_conv_per_level=1,
#                  half_res=False):
        
#         """
#         Parameters:
#             inshape: Input shape. e.g. (192, 192, 192)
#             infeats: Number of input features.
#             nb_features: Unet convolutional features. Can be specified via a list of lists with
#                 the form [[encoder feats], [decoder feats]], or as a single integer. 
#                 If None (default), the unet features are defined by the default config described in 
#                 the class documentation.
#             nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
#                 Default is None.
#             feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
#                 Default is 1.
#             nb_conv_per_level: Number of convolutions per unet level. Default is 1.
#             half_res: Skip the last decoder upsampling. Default is False.
#         """
#         super().__init__()

#         # ensure correct dimensionality
#         ndims = len(inshape)
#         assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

#         # cache some parameters
#         self.half_res = half_res

#         # default encoder and decoder layer features if nothing provided
#         if nb_features is None:
#             nb_features = default_unet_features()
#             print("1")

#         # build feature list automatically
#         if isinstance(nb_features, int):
#             if nb_levels is None:
#                 print("2")
#                 raise ValueError('must provide unet nb_levels if nb_features is an integer')
#             feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
#             print("3")
#             nb_features = [
#                 np.repeat(feats[:-1], nb_conv_per_level),
#                 np.repeat(np.flip(feats), nb_conv_per_level)
#             ]
#         elif nb_levels is not None:
#             print("4")
#             raise ValueError('cannot use nb_levels if nb_features is not an integer')

#         # extract any surplus (full resolution) decoder convolutions
#         enc_nf, dec_nf = nb_features
#         nb_dec_convs = len(enc_nf)
#         final_convs = dec_nf[nb_dec_convs:]
#         dec_nf = dec_nf[:nb_dec_convs]
#         self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

#         if isinstance(max_pool, int):
#             print("5")
#             max_pool = [max_pool] * self.nb_levels

#         # cache downsampling / upsampling operations
#         MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
#         self.pooling = [MaxPooling(s) for s in max_pool]
#         self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

#         # configure encoder (down-sampling path)
#         prev_nf = infeats
#         encoder_nfs = [prev_nf]
#         self.encoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             print("6")
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 print("7")
#                 nf = enc_nf[level * nb_conv_per_level + conv]
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#             self.encoder.append(convs)
#             encoder_nfs.append(prev_nf)

#         print(prev_nf)

#         prev_nf = 128
#         print(prev_nf)

#         # configure decoder (up-sampling path)
#         encoder_nfs = np.flip(encoder_nfs)
#         self.decoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             print("8")
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 print("9")
#                 nf = dec_nf[level * nb_conv_per_level + conv]
#                 print(nf)
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#                 prev_nf = prev_nf + 64
#             self.decoder.append(convs)
#             if not half_res or level < (self.nb_levels - 2):
#                 print("10")
#                 prev_nf += encoder_nfs[level]

#         prev_nf = 128
#         # now we take care of any remaining convolutions
#         self.remaining = nn.ModuleList()
#         for num, nf in enumerate(final_convs):
#             print("11")
#             self.remaining.append(ConvBlock(ndims, prev_nf, nf))
#             prev_nf = nf

#         # cache final number of features
#         self.final_nf = prev_nf

#         self.attn_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))  # X, Y, Z 各軸の重み（初期値は等しい）
#         self.softmax = nn.Softmax(dim=0)  # 重みの正規化
#     def forward(self, source, target):
#         import pywt
#         import numpy as np
#         import torch

#         level = 1
#         wavelet = 'db1'  # 使用するウェーブレット

#         # NumPy に変換
#         source_np = source.cpu().numpy()
#         target_np = target.cpu().numpy()

#         # === X軸方向の SWT ===
#         coeffs_x_src = [pywt.swt(source_np[i, :, :], wavelet, level) for i in range(source_np.shape[0])]
#         LL_x_src = np.stack([c[0][0] for c in coeffs_x_src], axis=0)
#         HH_x_src = np.stack([c[0][1] for c in coeffs_x_src], axis=0)

#         coeffs_x_tgt = [pywt.swt(target_np[i, :, :], wavelet, level) for i in range(target_np.shape[0])]
#         LL_x_tgt = np.stack([c[0][0] for c in coeffs_x_tgt], axis=0)
#         HH_x_tgt = np.stack([c[0][1] for c in coeffs_x_tgt], axis=0)

#         # === Y軸方向の SWT ===
#         coeffs_y_src = [pywt.swt(source_np[:, i, :], wavelet, level) for i in range(source_np.shape[1])]
#         LL_y_src = np.stack([c[0][0] for c in coeffs_y_src], axis=1)
#         HH_y_src = np.stack([c[0][1] for c in coeffs_y_src], axis=1)

#         coeffs_y_tgt = [pywt.swt(target_np[:, i, :], wavelet, level) for i in range(target_np.shape[1])]
#         LL_y_tgt = np.stack([c[0][0] for c in coeffs_y_tgt], axis=1)
#         HH_y_tgt = np.stack([c[0][1] for c in coeffs_y_tgt], axis=1)

#         # === Z軸方向の SWT ===
#         coeffs_z_src = [pywt.swt(source_np[:, :, i], wavelet, level) for i in range(source_np.shape[2])]
#         LL_z_src = np.stack([c[0][0] for c in coeffs_z_src], axis=2)
#         HH_z_src = np.stack([c[0][1] for c in coeffs_z_src], axis=2)

#         coeffs_z_tgt = [pywt.swt(target_np[:, :, i], wavelet, level) for i in range(target_np.shape[2])]
#         LL_z_tgt = np.stack([c[0][0] for c in coeffs_z_tgt], axis=2)
#         HH_z_tgt = np.stack([c[0][1] for c in coeffs_z_tgt], axis=2)

#         # === 平均による LL 成分統合 ===
#         LL_source = (LL_x_src + LL_y_src + LL_z_src) / 3
#         LL_target = (LL_x_tgt + LL_y_tgt + LL_z_tgt) / 3

#         # === NumPy → Tensor変換（全て）===
#         device = source.device

#         LL_source = torch.tensor(LL_source, dtype=torch.float32).to(device)
#         LL_target = torch.tensor(LL_target, dtype=torch.float32).to(device)

#         HH_x_src = torch.tensor(HH_x_src, dtype=torch.float32).to(device)
#         HH_y_src = torch.tensor(HH_y_src, dtype=torch.float32).to(device)
#         HH_z_src = torch.tensor(HH_z_src, dtype=torch.float32).to(device)

#         HH_x_tgt = torch.tensor(HH_x_tgt, dtype=torch.float32).to(device)
#         HH_y_tgt = torch.tensor(HH_y_tgt, dtype=torch.float32).to(device)
#         HH_z_tgt = torch.tensor(HH_z_tgt, dtype=torch.float32).to(device)

#         # === 注意重みによる加重平均 ===
#         attn = self.softmax(self.attn_weights)  # torch.Size([3])

#         HH_source = attn[0] * HH_x_src + attn[1] * HH_y_src + attn[2] * HH_z_src
#         HH_target = attn[0] * HH_x_tgt + attn[1] * HH_y_tgt + attn[2] * HH_z_tgt

#         # === ログ確認 ===
#         print("LL_Moving", LL_source.mean().item())
#         print("HH_Moving", HH_source.mean().item())
#         print("LL_Fixed", LL_target.mean().item())
#         print("HH_Fixed", HH_target.mean().item())

#         # === 統合入力の作成（concat）===
#         xLL = torch.cat([LL_source, LL_target], dim=1)  # dim=1はチャンネル方向想定
#         xHH = torch.cat([HH_source, HH_target], dim=1)

#     # def forward(self, source, target):
        
#     #     level = 1
#     #     wavelet = 'db1'  # 使用するウェーブレット

#     #     source_np = source.cpu().numpy()
#     #     target_np = target.cpu().numpy()
#     #         # X軸方向の SWT

#     #     coeffs_x_src = [pywt.swt(source_np[i, :, :], wavelet, level) for i in range(source_np.shape[0])]
#     #     LL_x_src = np.stack([c[0][0] for c in coeffs_x_src], axis=0)  # 軸を明示
#     #     HH_x_src = np.stack([c[0][1] for c in coeffs_x_src], axis=0)  # 軸を明示

#     #     coeffs_x_tgt = [pywt.swt(target_np[i, :, :], wavelet, level) for i in range(target_np.shape[0])]
#     #     LL_x_tgt = np.stack([c[0][0] for c in coeffs_x_tgt], axis=0)
#     #     HH_x_tgt = np.stack([c[0][1] for c in coeffs_x_tgt], axis=0)

#     #     # Y軸方向の SWT
#     #     coeffs_y_src = [pywt.swt(source_np[:, i, :], wavelet, level) for i in range(source_np.shape[1])]

#     #     # 正しくリストから NumPy 配列へ変換
#     #     LL_y_src = np.stack([c[0][0] for c in coeffs_y_src], axis=1)  # 軸を明示
#     #     HH_y_src = np.stack([c[0][1] for c in coeffs_y_src], axis=1)  # 軸を明示

#     #     coeffs_y_tgt = [pywt.swt(target_np[:, i, :], wavelet, level) for i in range(target_np.shape[1])]
#     #     LL_y_tgt = np.stack([c[0][0] for c in coeffs_y_tgt], axis=1)
#     #     HH_y_tgt = np.stack([c[0][1] for c in coeffs_y_tgt], axis=1)

#     #     # Z軸方向の SWT
#     #     coeffs_z_src = [pywt.swt(source_np[:, :, i], wavelet, level) for i in range(source_np.shape[2])]
#     #     LL_z_src = np.stack([c[0][0] for c in coeffs_z_src], axis=2)
#     #     HH_z_src = np.stack([c[0][1] for c in coeffs_z_src], axis=2)

#     #     coeffs_z_tgt = [pywt.swt(target_np[:, :, i], wavelet, level) for i in range(target_np.shape[2])]
#     #     LL_z_tgt = np.stack([c[0][0] for c in coeffs_z_tgt], axis=2)
#     #     HH_z_tgt = np.stack([c[0][1] for c in coeffs_z_tgt], axis=2)

#     #     # 3軸の結果を統合（平均）
#     #     LL_source = (LL_x_src + LL_y_src + LL_z_src) / 3
#     #     LL_target = (LL_x_tgt + LL_y_tgt + LL_z_tgt) / 3

#     #     # 高周波（HH）成分の注意重み
#     #     attn = self.softmax(self.attn_weights)

#     #     # HH_source = X, Y, Z 方向ごとの HH を加重平均
#     #     HH_source = attn[0] * HH_x_src + attn[1] * HH_y_src + attn[2] * HH_z_src
#     #     HH_target = attn[0] * HH_x_tgt + attn[1] * HH_y_tgt + attn[2] * HH_z_tgt

#     #     # Tensor に変換して GPU に戻す
#     #     LL_source = torch.tensor(LL_source, dtype=torch.float32, device=source.device)
#     #     HH_source = torch.tensor(HH_source, dtype=torch.float32, device=source.device)

#     #     LL_target = torch.tensor(LL_target, dtype=torch.float32, device=target.device)
#     #     HH_target = torch.tensor(HH_target, dtype=torch.float32, device=target.device)

#     #     # HH_source = torch.abs(HH_source)  # 高周波成分の絶対値
#     #     # HH_target = torch.abs(HH_target)  # 高周波成分の絶対値

#     #     print("LL_Moving", LL_source.mean())
#     #     print("HH_Moving", HH_source.mean())
#     #     print("LL_Fixed", LL_target.mean())
#     #     print("HH_Fixed", HH_target.mean())      

#     #     xLL = torch.cat([LL_source, LL_target], dim=1)
#     #     xHH = torch.cat([HH_source, HH_target], dim=1)

#         # encoder forward pass
#         x_historyLL = [xLL]
#         for level, convs in enumerate(self.encoder):
#             for conv in convs:
#                 xLL = conv(xLL)
#                 # print(f"After encoder level {level} conv: {xLL.shape}")  # ここで形状を確認
#             x_historyLL.append(xLL)
#             xLL = self.pooling[level](xLL)
#             # print(f"After pooling level {level}: {xLL.shape}")  # プーリング後の形状を確認

#         # encoder forward pass
#         x_historyHH = [xHH]
#         for level, convs in enumerate(self.encoder):
#             for conv in convs:
#                 xHH = conv(xHH)
#                 # print(f"After encoder level {level} conv: {xHH.shape}")  # ここで形状を確認
#             x_historyHH.append(xHH)
#             xHH = self.pooling[level](xHH)
#             # print(f"After pooling level {level}: {xHH.shape}")  # プーリング後の形状を確認

#         # 潜在変数を統合
#         latent = torch.cat([xLL, xHH], dim=1)  # チャネル方向で結合

#         # Decoder forward pass
#         for level, convs in enumerate(self.decoder):
#             for conv in convs:
#                 latent = conv(latent)
#                 # print(f"After decoder level {level} conv: {latent.shape}")  # ここで形状を確認

#             if not self.half_res or level < (self.nb_levels - 2):
#                 latent = self.upsampling[level](latent)
#                 # print(f"After upsampling level {level}: {latent.shape}")  # アップサンプリング後の形状を確認

#                 # スキップ接続
#                 skip_ll = x_historyLL.pop()
#                 skip_hh = x_historyHH.pop()

#                 latent = torch.cat([latent, skip_ll, skip_hh], dim=1)  # チャネル方向で結合
#                 # print(f"After concatenating skip connections: {latent.shape}")  # スキップ接続後の形状を確認

#         # Remaining convolutions at full resolution
#         for conv in self.remaining:
#             latent = conv(latent)
#             # print(f"After remaining conv: {latent.shape}")  # 最後の畳み込み後の形状を確認

#         return latent

class VxmDense1(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet_FilterBank(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch_local_backup.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        print(down_shape)
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
        # print("insahpeのサイズ",inshape)

        # configure transformer

        # new_inshape = inshape.squeeze(dim=2)
        self.transformer = layers.SpatialTransformer(inshape)


    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        # print("（読み込んだ画像サイズ）:", source.shape, target.shape)
        # x = torch.cat([source, target], dim=1)
        # x = x.unsqueeze(-1)
        # print("（Catした画像サイズ）:", x.shape)
        x = self.unet_model(source, target)
        # print("aaaa2222", x.max())
        # print("（Xサイズ）:", x.shape)

        # transform into flow field
        flow_field = self.flow(x)
        # print("aaaa3333", flow_field.max())
        # print("（flow_fieldサイズ）:", flow_field.shape)



#-------------------------------------------------Kihon----------------------------------------------------------------------

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            print("A1")
            pos_flow = self.resize(pos_flow)
        preint_flow = pos_flow


        # pos_flowは変形ベクトル

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            print("A2")
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                print("A3")
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        # preint_flow = torch.ones_like(pos_flow)*(50)
        # print(pos_flow.mean())


        print(pos_flow.shape)
        print(source.shape)


        y_source = self.transformer(source, pos_flow)

        # y_source = self.transformer(source, preint_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            # print("aaaa", preint_flow.max())
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)  

                                                
        
        else:
            # pos_flow = torch.ones_like(pos_flow)*10
            # print("bbbb", pos_flow.mean())
            return y_source, pos_flow

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        # Conv3dが呼ばれる前にxの型を確認するデバッグ用コード
        # print("xの型（ConvBlockの中）:", type(x))
        # print("（ConvBlockの中）:", x.shape)
        
        out = self.main(x)
        out = self.activation(out)
        # print("aaaa1111", out.max())
        # print("（ConvBlockの中）:", out.shape)
        return out

# 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# Xねっと高周波

class Unet_FilterBank2(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
        
    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
            print("1")

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                print("2")
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            print("3")
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            print("4")
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            print("5")
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            print("6")
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                print("7")
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        print(prev_nf)

        prev_nf = 128
        print(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            print("8")
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                print("9")
                nf = dec_nf[level * nb_conv_per_level + conv]
                print(nf)
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
                prev_nf = prev_nf + 64
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                print("10")
                prev_nf += encoder_nfs[level]

        prev_nf = 128
        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            print("11")
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        prev_nf = 64
        self.decoder_HH = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            print("8")
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                print("9")
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder_HH.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                print("10")
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining_HH  = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            print("11")
            self.remaining_HH .append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    
    def forward(self, source, target):

        hhfh_kernel, hlfh_kernel = get_filter_kernel(j=1)

        HH_source = apply_3d_filter(source.cpu().numpy(), hhfh_kernel)
        LL_source = apply_3d_filterLL(source.cpu().numpy(), hlfh_kernel)

        HH_target = apply_3d_filter(target.cpu().numpy(), hhfh_kernel)
        LL_target = apply_3d_filterLL(target.cpu().numpy(), hlfh_kernel)

        # Tensor に変換して GPU に戻す
        LL_source = torch_local_backup.tensor(LL_source, dtype=torch_local_backup.float32, device=source.device)
        HH_source = torch_local_backup.tensor(HH_source, dtype=torch_local_backup.float32, device=source.device)

        LL_target = torch_local_backup.tensor(LL_target, dtype=torch_local_backup.float32, device=target.device)
        HH_target = torch_local_backup.tensor(HH_target, dtype=torch_local_backup.float32, device=target.device)
   

        xLL = torch_local_backup.cat([LL_source, LL_target], dim=1)
        xHH = torch_local_backup.cat([HH_source, HH_target], dim=1)

        # encoder forward pass
        x_historyLL = [xLL]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                xLL = conv(xLL)
                # print(f"After encoder level {level} conv: {xLL.shape}")  # ここで形状を確認
            x_historyLL.append(xLL)
            xLL = self.pooling[level](xLL)
            # print(f"After pooling level {level}: {xLL.shape}")  # プーリング後の形状を確認

        # encoder forward pass
        x_historyHH = [xHH]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                xHH = conv(xHH)
                # print(f"After encoder level {level} conv: {xHH.shape}")  # ここで形状を確認
            x_historyHH.append(xHH)
            xHH = self.pooling[level](xHH)
            # print(f"After pooling level {level}: {xHH.shape}")  # プーリング後の形状を確認

        # 潜在変数を統合
        latent = torch_local_backup.cat([xLL, xHH], dim=1)  # チャネル方向で結合
        # x_historyHH2 = x_historyHH
        import copy
        x_historyHH2 = [x.detach().clone() for x in x_historyHH]

        # Decoder forward pass
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                latent = conv(latent)
                # print(f"After decoder level {level} conv: {latent.shape}")  # ここで形状を確認

            if not self.half_res or level < (self.nb_levels - 2):
                latent = self.upsampling[level](latent)
                # print(f"After upsampling level {level}: {latent.shape}")  # アップサンプリング後の形状を確認

                # スキップ接続
                skip_ll = x_historyLL.pop()
                skip_hh = x_historyHH.pop()
                # print(f"xHH sdsdsdssdhape: {latent.shape}")
                # print(f"x_histdsdsdsdoryHH[-1] shape: {skip_hh.shape}")
                latent = torch_local_backup.cat([latent, skip_ll, skip_hh], dim=1)  # チャネル方向で結合
                # print(f"After concatenating skip connections: {latent.shape}")  # スキップ接続後の形状を確認

        # Remaining convolutions at full resolution
        for conv in self.remaining:
            latent = conv(latent)
            # print(f"After remaining conv: {latent.shape}")  # 最後の畳み込み後の形状を確認

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder_HH):
            for conv in convs:
                xHH = conv(xHH)

            if not self.half_res or level < (self.nb_levels - 2):

                xHH = self.upsampling[level](xHH)

                skip_hh = x_historyHH2.pop()

                # print("Before concatenation（デコーダーの中）:", x.shape, x_history[-1].shape)  # 追加
                xHH = torch_local_backup.cat([xHH, skip_hh], dim=1)
                # print("After concatenation（デコーダーの中）:", x.shape)  # 追加

        # remaining convs at full resolution
        for conv in self.remaining_HH :
            xHH = conv(xHH)


        return latent, xHH, HH_source



class VxmDense2(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):

        super().__init__()

        self.training = True

        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        self.unet_model = Unet_FilterBank2(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch_local_backup.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        print(down_shape)
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        self.transformer = layers.SpatialTransformer(inshape)


    def forward(self, source, target, registration=False):

        x, xHH, HH_source = self.unet_model(source, target)
        flow_field = self.flow(x)
        flow_fieldHH = self.flow(xHH)

        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)
        preint_flow = pos_flow
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        y_source = self.transformer(source, pos_flow)

        # y_source = self.transformer(source, preint_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            # print("aaaa", preint_flow.max())
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow, flow_fieldHH, HH_source)                                    
        
        else:
            return y_source, pos_flow, flow_fieldHH, HH_source

class ConvBlock(nn.Module):

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):     
        out = self.main(x)
        out = self.activation(out)
        return out

    
    # # 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# # 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# # 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# # 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# # Xねっとkaiiiiiiii通常
# class Unet(nn.Module):
#     """
#     A unet architecture. Layer features can be specified directly as a list of encoder and decoder
#     features or as a single integer along with a number of unet levels. The default network features
#     per layer (when no options are specified) are:

#         encoder: [16, 32, 32, 32]
#         decoder: [32, 32, 32, 32, 32, 16, 16]
#     """
        
#     def __init__(self,
#                  inshape=None,
#                  infeats=None,
#                  nb_features=None,
#                  nb_levels=None,
#                  max_pool=2,
#                  feat_mult=1,
#                  nb_conv_per_level=1,
#                  half_res=False):
        
#         """
#         Parameters:
#             inshape: Input shape. e.g. (192, 192, 192)
#             infeats: Number of input features.
#             nb_features: Unet convolutional features. Can be specified via a list of lists with
#                 the form [[encoder feats], [decoder feats]], or as a single integer. 
#                 If None (default), the unet features are defined by the default config described in 
#                 the class documentation.
#             nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
#                 Default is None.
#             feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
#                 Default is 1.
#             nb_conv_per_level: Number of convolutions per unet level. Default is 1.
#             half_res: Skip the last decoder upsampling. Default is False.
#         """
#         super().__init__()

#         # ensure correct dimensionality
#         ndims = len(inshape)
#         assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

#         # cache some parameters
#         self.half_res = half_res

#         # default encoder and decoder layer features if nothing provided
#         if nb_features is None:
#             nb_features = default_unet_features()
#             print("1")

#         # build feature list automatically
#         if isinstance(nb_features, int):
#             if nb_levels is None:
#                 print("2")
#                 raise ValueError('must provide unet nb_levels if nb_features is an integer')
#             feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
#             print("3")
#             nb_features = [
#                 np.repeat(feats[:-1], nb_conv_per_level),
#                 np.repeat(np.flip(feats), nb_conv_per_level)
#             ]
#         elif nb_levels is not None:
#             print("4")
#             raise ValueError('cannot use nb_levels if nb_features is not an integer')

#         # extract any surplus (full resolution) decoder convolutions
#         enc_nf, dec_nf = nb_features
#         nb_dec_convs = len(enc_nf)
#         final_convs = dec_nf[nb_dec_convs:]
#         dec_nf = dec_nf[:nb_dec_convs]
#         self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

#         if isinstance(max_pool, int):
#             print("5")
#             max_pool = [max_pool] * self.nb_levels

#         # cache downsampling / upsampling operations
#         MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
#         self.pooling = [MaxPooling(s) for s in max_pool]
#         self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

#         # configure encoder (down-sampling path)
#         prev_nf = infeats
#         encoder_nfs = [prev_nf]
#         self.encoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             print("6")
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 print("7")
#                 nf = enc_nf[level * nb_conv_per_level + conv]
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#             self.encoder.append(convs)
#             encoder_nfs.append(prev_nf)

#         print(prev_nf)

#         prev_nf = 128
#         print(prev_nf)

#         # configure decoder (up-sampling path)
#         encoder_nfs = np.flip(encoder_nfs)
#         self.decoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             print("8")
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 print("9")
#                 nf = dec_nf[level * nb_conv_per_level + conv]
#                 print(nf)
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#                 prev_nf = prev_nf + 64
#             self.decoder.append(convs)
#             if not half_res or level < (self.nb_levels - 2):
#                 print("10")
#                 prev_nf += encoder_nfs[level]

#         prev_nf = 128
#         # now we take care of any remaining convolutions
#         self.remaining = nn.ModuleList()
#         for num, nf in enumerate(final_convs):
#             print("11")
#             self.remaining.append(ConvBlock(ndims, prev_nf, nf))
#             prev_nf = nf

#         # cache final number of features
#         self.final_nf = prev_nf

#     def forward(self, LL1, LL2, HH1, HH2):
        
#         xLL = torch.cat([LL1, LL2], dim=1)
#         xHH = torch.cat([HH1, HH2], dim=1)

#         # encoder forward pass
#         x_historyLL = [xLL]
#         for level, convs in enumerate(self.encoder):
#             for conv in convs:
#                 xLL = conv(xLL)
#                 # print(f"After encoder level {level} conv: {xLL.shape}")  # ここで形状を確認
#             x_historyLL.append(xLL)
#             xLL = self.pooling[level](xLL)
#             # print(f"After pooling level {level}: {xLL.shape}")  # プーリング後の形状を確認

#         # encoder forward pass
#         x_historyHH = [xHH]
#         for level, convs in enumerate(self.encoder):
#             for conv in convs:
#                 xHH = conv(xHH)
#                 # print(f"After encoder level {level} conv: {xHH.shape}")  # ここで形状を確認
#             x_historyHH.append(xHH)
#             xHH = self.pooling[level](xHH)
#             # print(f"After pooling level {level}: {xHH.shape}")  # プーリング後の形状を確認

#         # 潜在変数を統合
#         latent = torch.cat([xLL, xHH], dim=1)  # チャネル方向で結合

#         # Decoder forward pass
#         for level, convs in enumerate(self.decoder):
#             for conv in convs:
#                 latent = conv(latent)
#                 # print(f"After decoder level {level} conv: {latent.shape}")  # ここで形状を確認

#             if not self.half_res or level < (self.nb_levels - 2):
#                 latent = self.upsampling[level](latent)
#                 # print(f"After upsampling level {level}: {latent.shape}")  # アップサンプリング後の形状を確認

#                 # スキップ接続
#                 skip_ll = x_historyLL.pop()
#                 skip_hh = x_historyHH.pop()

#                 latent = torch.cat([latent, skip_ll, skip_hh], dim=1)  # チャネル方向で結合
#                 # print(f"After concatenating skip connections: {latent.shape}")  # スキップ接続後の形状を確認

#         latent = F.interpolate(latent, scale_factor=2, mode='trilinear', align_corners=False)

#         # Remaining convolutions at full resolution
#         for conv in self.remaining:
#             latent = conv(latent)
#             print(f"After remaining conv: {latent.shape}")  # 最後の畳み込み後の形状を確認

#         return latent

# class VxmDense(LoadableModel):
#     """
#     VoxelMorph network for (unsupervised) nonlinear registration between two images.
#     """

#     @store_config_args
#     def __init__(self,
#                  inshape,
#                  nb_unet_features=None,
#                  nb_unet_levels=None,
#                  unet_feat_mult=1,
#                  nb_unet_conv_per_level=1,
#                  int_steps=7,
#                  int_downsize=2,
#                  bidir=False,
#                  use_probs=False,
#                  src_feats=1,
#                  trg_feats=1,
#                  unet_half_res=False):
#         """ 
#         Parameters:
#             inshape: Input shape. e.g. (192, 192, 192)
#             nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
#                 the form [[encoder feats], [decoder feats]], or as a single integer. 
#                 If None (default), the unet features are defined by the default config described in 
#                 the unet class documentation.
#             nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
#                 Default is None.
#             unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
#                 Default is 1.
#             nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
#             int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
#                 value is 0.
#             int_downsize: Integer specifying the flow downsample factor for vector integration. 
#                 The flow field is not downsampled when this value is 1.
#             bidir: Enable bidirectional cost function. Default is False.
#             use_probs: Use probabilities in flow field. Default is False.
#             src_feats: Number of source image features. Default is 1.
#             trg_feats: Number of target image features. Default is 1.
#             unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
#                 Default is False.
#         """
#         super().__init__()

#         # internal flag indicating whether to return flow or integrated warp during inference
#         self.training = True

#         # ensure correct dimensionality
#         ndims = len(inshape)
#         assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

#         # configure core unet model
#         self.unet_model = Unet(
#             inshape,
#             infeats=(src_feats + trg_feats),
#             nb_features=nb_unet_features,
#             nb_levels=nb_unet_levels,
#             feat_mult=unet_feat_mult,
#             nb_conv_per_level=nb_unet_conv_per_level,
#             half_res=unet_half_res,
#         )

#         # configure unet to flow field layer
#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

#         # init flow layer with small weights and bias
#         self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
#         self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

#         # probabilities are not supported in pytorch
#         if use_probs:
#             raise NotImplementedError(
#                 'Flow variance has not been implemented in pytorch - set use_probs to False')

#         # configure optional resize layers (downsize)
#         if not unet_half_res and int_steps > 0 and int_downsize > 1:
#             self.resize = layers.ResizeTransform(int_downsize, ndims)
#         else:
#             self.resize = None

#         # resize to full res
#         if int_steps > 0 and int_downsize > 1:
#             self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
#         else:
#             self.fullsize = None

#         # configure bidirectional training
#         self.bidir = bidir

#         # configure optional integration layer for diffeomorphic warp
#         down_shape = [int(dim / int_downsize) for dim in inshape]
#         print(down_shape)
#         self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
#         # print("insahpeのサイズ",inshape)

#         # configure transformer

#         # new_inshape = inshape.squeeze(dim=2)
#         self.transformer = layers.SpatialTransformer(inshape)


#     def forward(self, LL1, LL2, HH1, HH2, source, registration=False):
#         '''
#         Parameters:
#             source: Source image tensor.
#             target: Target image tensor.
#             registration: Return transformed image and flow. Default is False.
#         '''

#         # concatenate inputs and propagate unet
#         # print("（読み込んだ画像サイズ）:", source.shape, target.shape)
#         # x = torch.cat([source, target], dim=1)
#         # x = x.unsqueeze(-1)
#         # print("（Catした画像サイズ）:", x.shape)
#         x = self.unet_model(LL1, LL2, HH1, HH2, )
#         # print("aaaa2222", x.max())
#         # print("（Xサイズ）:", x.shape)

#         # transform into flow field
#         flow_field = self.flow(x)
#         # print("aaaa3333", flow_field.max())
#         # print("（flow_fieldサイズ）:", flow_field.shape)



# #-------------------------------------------------Kihon----------------------------------------------------------------------

#         # resize flow for integration
#         pos_flow = flow_field
#         if self.resize:
#             print("A1")
#             pos_flow = self.resize(pos_flow)
#         preint_flow = pos_flow


#         # pos_flowは変形ベクトル

#         # negate flow for bidirectional model
#         neg_flow = -pos_flow if self.bidir else None

#         # integrate to produce diffeomorphic warp
#         if self.integrate:
#             print("A2")
#             pos_flow = self.integrate(pos_flow)
#             neg_flow = self.integrate(neg_flow) if self.bidir else None

#             # resize to final resolution
#             if self.fullsize:
#                 print("A3")
#                 pos_flow = self.fullsize(pos_flow)
#                 neg_flow = self.fullsize(neg_flow) if self.bidir else None

#         # warp image with flow field
#         # preint_flow = torch.ones_like(pos_flow)*(50)
#         # print(pos_flow.mean())


#         print(pos_flow.shape)
#         # print(source.shape)


#         y_source = self.transformer(source, pos_flow)

#         # y_source = self.transformer(source, preint_flow)
#         y_target = self.transformer(target, neg_flow) if self.bidir else None

#         # return non-integrated flow field if training
#         if not registration:
#             # print("aaaa", preint_flow.max())
#             return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)  

                                                
        
#         else:
#             # pos_flow = torch.ones_like(pos_flow)*10
#             # print("bbbb", pos_flow.mean())
#             return y_source, pos_flow

# class ConvBlock(nn.Module):
#     """
#     Specific convolutional block followed by leakyrelu for unet.
#     """

#     def __init__(self, ndims, in_channels, out_channels, stride=1):
#         super().__init__()

#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.main = Conv(in_channels, out_channels, 3, stride, 1)
#         self.activation = nn.LeakyReLU(0.2)

#     def forward(self, x):
#         # Conv3dが呼ばれる前にxの型を確認するデバッグ用コード
#         # print("xの型（ConvBlockの中）:", type(x))
#         # print("（ConvBlockの中）:", x.shape)
        
#         out = self.main(x)
#         out = self.activation(out)
#         # print("aaaa1111", out.max())
#         # print("（ConvBlockの中）:", out.shape)
#         return out
# 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000


# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions.normal import Normal

# from .. import default_unet_features
# from . import layers
# from .modelio import LoadableModel, store_config_args


# class Unet(nn.Module):
#     """
#     A unet architecture. Layer features can be specified directly as a list of encoder and decoder
#     features or as a single integer along with a number of unet levels. The default network features
#     per layer (when no options are specified) are:

#         encoder: [16, 32, 32, 32]
#         decoder: [32, 32, 32, 32, 32, 16, 16]
#     """
        
#     def __init__(self,
#                  inshape=None,
#                  infeats=None,
#                  nb_features=None,
#                  nb_levels=None,
#                  max_pool=2,
#                  feat_mult=1,
#                  nb_conv_per_level=1,
#                  half_res=False):
        
#         """
#         Parameters:
#             inshape: Input shape. e.g. (192, 192, 192)
#             infeats: Number of input features.
#             nb_features: Unet convolutional features. Can be specified via a list of lists with
#                 the form [[encoder feats], [decoder feats]], or as a single integer. 
#                 If None (default), the unet features are defined by the default config described in 
#                 the class documentation.
#             nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
#                 Default is None.
#             feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
#                 Default is 1.
#             nb_conv_per_level: Number of convolutions per unet level. Default is 1.
#             half_res: Skip the last decoder upsampling. Default is False.
#         """
#         super().__init__()

#         # ensure correct dimensionality
#         ndims = len(inshape)
#         assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

#         # cache some parameters
#         self.half_res = half_res

#         # default encoder and decoder layer features if nothing provided
#         if nb_features is None:
#             nb_features = default_unet_features()
#             print("1")

#         # build feature list automatically
#         if isinstance(nb_features, int):
#             if nb_levels is None:
#                 print("2")
#                 raise ValueError('must provide unet nb_levels if nb_features is an integer')
#             feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
#             print("3")
#             nb_features = [
#                 np.repeat(feats[:-1], nb_conv_per_level),
#                 np.repeat(np.flip(feats), nb_conv_per_level)
#             ]
#         elif nb_levels is not None:
#             print("4")
#             raise ValueError('cannot use nb_levels if nb_features is not an integer')

#         # extract any surplus (full resolution) decoder convolutions
#         enc_nf, dec_nf = nb_features
#         nb_dec_convs = len(enc_nf)
#         final_convs = dec_nf[nb_dec_convs:]
#         dec_nf = dec_nf[:nb_dec_convs]
#         self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

#         if isinstance(max_pool, int):
#             print("5")
#             max_pool = [max_pool] * self.nb_levels

#         # cache downsampling / upsampling operations
#         MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
#         self.pooling = [MaxPooling(s) for s in max_pool]
#         self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

#         # configure encoder (down-sampling path)
#         prev_nf = infeats
#         encoder_nfs = [prev_nf]
#         self.encoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             print("6")
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 print("7")
#                 nf = enc_nf[level * nb_conv_per_level + conv]
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#             self.encoder.append(convs)
#             encoder_nfs.append(prev_nf)

#         # configure decoder (up-sampling path)
#         encoder_nfs = np.flip(encoder_nfs)
#         self.decoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             print("8")
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 print("9")
#                 nf = dec_nf[level * nb_conv_per_level + conv]
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#             self.decoder.append(convs)
#             if not half_res or level < (self.nb_levels - 2):
#                 print("10")
#                 prev_nf += encoder_nfs[level]

#         # now we take care of any remaining convolutions
#         self.remaining = nn.ModuleList()
#         for num, nf in enumerate(final_convs):
#             print("11")
#             self.remaining.append(ConvBlock(ndims, prev_nf, nf))
#             prev_nf = nf

#         # cache final number of features
#         self.final_nf = prev_nf

#     def forward(self, x):

#         # encoder forward pass
#         x_history = [x]
#         for level, convs in enumerate(self.encoder):
#             for conv in convs:
#                 x = conv(x)
#             x_history.append(x)
#             # print("After encoder level（エンコーダの中）", level, ":", x.shape)  # 追加
#             x = self.pooling[level](x)

#         # decoder forward pass with upsampling and concatenation
#         for level, convs in enumerate(self.decoder):
#             for conv in convs:
#                 x = conv(x)
#             if not self.half_res or level < (self.nb_levels - 2):
#                 x = self.upsampling[level](x)
#                 # print("Before concatenation（デコーダーの中）:", x.shape, x_history[-1].shape)  # 追加
#                 x = torch.cat([x, x_history.pop()], dim=1)
#                 # print("After concatenation（デコーダーの中）:", x.shape)  # 追加

#         # remaining convs at full resolution
#         for conv in self.remaining:
#             x = conv(x)

#         return x


# class VxmDense(LoadableModel):
#     """
#     VoxelMorph network for (unsupervised) nonlinear registration between two images.
#     """

#     @store_config_args
#     def __init__(self,
#                  inshape,
#                  nb_unet_features=None,
#                  nb_unet_levels=None,
#                  unet_feat_mult=1,
#                  nb_unet_conv_per_level=1,
#                  int_steps=7,
#                  int_downsize=2,
#                  bidir=False,
#                  use_probs=False,
#                  src_feats=1,
#                  trg_feats=1,
#                  unet_half_res=False):
#         """ 
#         Parameters:
#             inshape: Input shape. e.g. (192, 192, 192)
#             nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
#                 the form [[encoder feats], [decoder feats]], or as a single integer. 
#                 If None (default), the unet features are defined by the default config described in 
#                 the unet class documentation.
#             nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
#                 Default is None.
#             unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
#                 Default is 1.
#             nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
#             int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
#                 value is 0.
#             int_downsize: Integer specifying the flow downsample factor for vector integration. 
#                 The flow field is not downsampled when this value is 1.
#             bidir: Enable bidirectional cost function. Default is False.
#             use_probs: Use probabilities in flow field. Default is False.
#             src_feats: Number of source image features. Default is 1.
#             trg_feats: Number of target image features. Default is 1.
#             unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
#                 Default is False.
#         """
#         super().__init__()

#         # internal flag indicating whether to return flow or integrated warp during inference
#         self.training = True

#         # ensure correct dimensionality
#         ndims = len(inshape)
#         assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

#         # # configure core unet model
#         # self.unet_model = Unet(
#         #     inshape,
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )

# #------------------------------------------------------------------------------------------------------------------------
        
#         self.unet_model_16 = Unet(
#             (16,16,16),
#             infeats=(src_feats + trg_feats),
#             nb_features=nb_unet_features,
#             nb_levels=nb_unet_levels,
#             feat_mult=unet_feat_mult,
#             nb_conv_per_level=nb_unet_conv_per_level,
#             half_res=unet_half_res,
#         )

#         self.unet_model_32 = Unet(
#             (32,32,32),
#             infeats=(src_feats + trg_feats),
#             nb_features=nb_unet_features,
#             nb_levels=nb_unet_levels,
#             feat_mult=unet_feat_mult,
#             nb_conv_per_level=nb_unet_conv_per_level,
#             half_res=unet_half_res,
#         )

#         self.unet_model_64 = Unet(
#             (64,64,64),
#             infeats=(src_feats + trg_feats),
#             nb_features=nb_unet_features,
#             nb_levels=nb_unet_levels,
#             feat_mult=unet_feat_mult,
#             nb_conv_per_level=nb_unet_conv_per_level,
#             half_res=unet_half_res,
#         )

#         self.unet_model_128 = Unet(
#             (128,128,128),
#             infeats=(src_feats + trg_feats),
#             nb_features=nb_unet_features,
#             nb_levels=nb_unet_levels,
#             feat_mult=unet_feat_mult,
#             nb_conv_per_level=nb_unet_conv_per_level,
#             half_res=unet_half_res,
#         )        
 
#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.flow_16 = Conv(self.unet_model_16.final_nf, ndims, kernel_size=3, padding=1)
#         self.flow_16.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_16.weight.shape))
#         self.flow_16.bias = nn.Parameter(torch.zeros(self.flow_16.bias.shape))

#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.flow_32 = Conv(self.unet_model_32.final_nf, ndims, kernel_size=3, padding=1)
#         self.flow_32.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_32.weight.shape))
#         self.flow_32.bias = nn.Parameter(torch.zeros(self.flow_32.bias.shape))

#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.flow_64 = Conv(self.unet_model_64.final_nf, ndims, kernel_size=3, padding=1)
#         self.flow_64.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_64.weight.shape))
#         self.flow_64.bias = nn.Parameter(torch.zeros(self.flow_64.bias.shape))

#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.flow_128 = Conv(self.unet_model_128.final_nf, ndims, kernel_size=3, padding=1)
#         self.flow_128.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128.weight.shape))
#         self.flow_128.bias = nn.Parameter(torch.zeros(self.flow_128.bias.shape))
# #------------------------------------------------------------------------------------------------------------------------

#         # # configure unet to flow field layer
#         # Conv = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

#         # # init flow layer with small weights and bias
#         # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
#         # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))





                
#         # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
#         # self.flow.bias = nn.Parameter(torch.ones(self.flow.bias.shape))

#         # probabilities are not supported in pytorch
#         if use_probs:
#             raise NotImplementedError(
#                 'Flow variance has not been implemented in pytorch - set use_probs to False')

#         # configure optional resize layers (downsize)
#         if not unet_half_res and int_steps > 0 and int_downsize > 1:
#             self.resize = layers.ResizeTransform(int_downsize, ndims)
#         else:
#             self.resize = None

#         # resize to full res
#         if int_steps > 0 and int_downsize > 1:
#             self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
#         else:
#             self.fullsize = None

#         # configure bidirectional training
#         self.bidir = bidir

#         # configure optional integration layer for diffeomorphic warp
#         down_shape = [int(dim / int_downsize) for dim in inshape]
#         print(down_shape)
#         self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
#         # print("insahpeのサイズ",inshape)

#         # configure transformer

#         # new_inshape = inshape.squeeze(dim=2)
#         self.transformer = layers.SpatialTransformer(inshape)

# #------------------------------------------------------------------------------------------------------------------------

#         self.transformer_16 = layers.SpatialTransformer((16,16,16))
#         self.transformer_32 = layers.SpatialTransformer((32,32,32))
#         self.transformer_64 = layers.SpatialTransformer((64,64,64))
#         self.transformer_128 = layers.SpatialTransformer((128,128,128))
       
# #------------------------------------------------------------------------------------------------------------------------


#     def forward(self, source, target, registration=False):
#         '''
#         Parameters:
#             source: Source image tensor.
#             target: Target image tensor.
#             registration: Return transformed image and flow. Default is False.
#         '''

#         # # concatenate inputs and propagate unet
#         # # print("（読み込んだ画像サイズ）:", source.shape, target.shape)
#         # x = torch.cat([source, target], dim=1)
#         # # x = x.unsqueeze(-1)
#         # print("（Catした画像サイズ）:", x.shape)
#         # x = self.unet_model(x)
#         # print("aaaa2222", x.max())
#         # print("（Xサイズ）:", x.shape)

#         # # transform into flow field
#         # flow_field = self.flow(x)
#         # print("aaaa3333", flow_field.max())
#         # print("（flow_fieldサイズ）:", flow_field.shape)

# #---------------------------------------------------------きほん-------------------------------------------------------------

#         # x = torch.cat([source, target], dim=1)

#         # source_16 = torch.nn.functional.interpolate(source, size=(16,16,16), mode='trilinear', align_corners=False)
#         # target_16 = torch.nn.functional.interpolate(target, size=(16,16,16), mode='trilinear', align_corners=False)
#         # x_16 = torch.cat([source_16, target_16], dim=1)
#         # x_16 = self.unet_model_16(x_16)
#         # flow_field_16 = self.flow_16(x_16)

#         # source_32 = torch.nn.functional.interpolate(source, size=(32,32,32), mode='trilinear', align_corners=False)
#         # target_32 = torch.nn.functional.interpolate(target, size=(32,32,32), mode='trilinear', align_corners=False)
#         # x_32 = torch.cat([source_32, target_32], dim=1)        
#         # x_32 = self.unet_model_32(x_32)
#         # flow_field_32 = self.flow_32(x_32)
#         # flow_field_16_resize = torch.nn.functional.interpolate(flow_field_16, size=(32,32,32), mode='trilinear', align_corners=False)
#         # # flow_field_32_new = (flow_field_32 + flow_field_16_resize)/2  
#         # flow_field_32_new = (flow_field_32 + flow_field_16_resize)  


#         # source_64 = torch.nn.functional.interpolate(source, size=(64,64,64), mode='trilinear', align_corners=False)
#         # target_64 = torch.nn.functional.interpolate(target, size=(64,64,64), mode='trilinear', align_corners=False)
#         # x_64 = torch.cat([source_64, target_64], dim=1)         
#         # x_64 = self.unet_model_64(x_64)
#         # flow_field_64 = self.flow_64(x_64)
#         # flow_field_32_resize = torch.nn.functional.interpolate(flow_field_32_new, size=(64,64,64), mode='trilinear', align_corners=False)
#         # # flow_field_64_new = (flow_field_64 + flow_field_32_resize)/2
#         # flow_field_64_new = (flow_field_64 + flow_field_32_resize)


#         # x = torch.cat([source, target], dim=1)
#         # x_128 = self.unet_model_128(x)
#         # flow_field_128 = self.flow_128(x_128)
#         # flow_field_64_resize = torch.nn.functional.interpolate(flow_field_64_new, size=(128,128,128), mode='trilinear', align_corners=False)
#         # # flow_field_128_new = (flow_field_128 + flow_field_64_resize)/2 
#         # flow_field_128_new = (flow_field_128 + flow_field_64_resize)

#         # flow_field = flow_field_128_new       

# #------------------------------------------------------------------------------------------------------------------------
 
# #----------------------------------------------------------おうよう------------------------------------------------------------

#         # x = torch.cat([source, target], dim=1)

#         # source_16 = torch.nn.functional.interpolate(source, size=(16,16,16), mode='trilinear', align_corners=False)
#         # target_16 = torch.nn.functional.interpolate(target, size=(16,16,16), mode='trilinear', align_corners=False)
#         # source_32 = torch.nn.functional.interpolate(source, size=(32,32,32), mode='trilinear', align_corners=False)
#         # target_32 = torch.nn.functional.interpolate(target, size=(32,32,32), mode='trilinear', align_corners=False)
#         # source_64 = torch.nn.functional.interpolate(source, size=(64,64,64), mode='trilinear', align_corners=False)
#         # target_64 = torch.nn.functional.interpolate(target, size=(64,64,64), mode='trilinear', align_corners=False)
#         # source_128 = source
#         # target_128 = target

#         # x_16 = torch.cat([source_16, target_16], dim=1)
#         # x_16 = self.unet_model_16(x_16)
#         # flow_field_16 = self.flow_16(x_16)
#         # flow_field_16_resize = torch.nn.functional.interpolate(flow_field_16, size=(32,32,32), mode='trilinear', align_corners=False)
#         # moved_32 = self.transformer_32(source_32, flow_field_16_resize)

#         # x_32 = torch.cat([moved_32, target_32], dim=1)        
#         # x_32 = self.unet_model_32(x_32)
#         # flow_field_32 = self.flow_32(x_32)
#         # flow_field_32_resize = torch.nn.functional.interpolate(flow_field_32, size=(64,64,64), mode='trilinear', align_corners=False)
#         # moved_64 = self.transformer_64(source_64, flow_field_32_resize)

#         # x_64 = torch.cat([moved_64, target_64], dim=1)        
#         # x_64 = self.unet_model_64(x_64)
#         # flow_field_64 = self.flow_64(x_64)
#         # flow_field_64_resize = torch.nn.functional.interpolate(flow_field_64, size=(128,128,128), mode='trilinear', align_corners=False)
#         # moved_128 = self.transformer_128(source_128, flow_field_64_resize)

#         # x_128 = torch.cat([moved_128, target_128], dim=1)        
#         # x_128 = self.unet_model_128(x_128)
#         # flow_field_128 = self.flow_128(x_128)

#         # flow_field = flow_field_128

      
# #------------------------------------------------------------------------------------------------------------------------
# #----------------------------------------------------------おうようベクトル２倍------------------------------------------------------------

#         # x = torch.cat([source, target], dim=1)

#         # source_16 = torch.nn.functional.interpolate(source, size=(16,16,16), mode='trilinear', align_corners=False)
#         # target_16 = torch.nn.functional.interpolate(target, size=(16,16,16), mode='trilinear', align_corners=False)
#         # source_32 = torch.nn.functional.interpolate(source, size=(32,32,32), mode='trilinear', align_corners=False)
#         # target_32 = torch.nn.functional.interpolate(target, size=(32,32,32), mode='trilinear', align_corners=False)
#         # source_64 = torch.nn.functional.interpolate(source, size=(64,64,64), mode='trilinear', align_corners=False)
#         # target_64 = torch.nn.functional.interpolate(target, size=(64,64,64), mode='trilinear', align_corners=False)
#         # source_128 = source
#         # target_128 = target

#         # x_16 = torch.cat([source_16, target_16], dim=1)
#         # x_16 = self.unet_model_16(x_16)
#         # flow_field_16 = self.flow_16(x_16)
#         # flow_field_16_resize = torch.nn.functional.interpolate(flow_field_16, size=(32,32,32), mode='trilinear', align_corners=False)
#         # flow_field_16_resize = flow_field_16_resize*2
#         # moved_32 = self.transformer_32(source_32, flow_field_16_resize)

#         # x_32 = torch.cat([moved_32, target_32], dim=1)        
#         # x_32 = self.unet_model_32(x_32)
#         # flow_field_32 = self.flow_32(x_32)
#         # flow_field_32_resize = torch.nn.functional.interpolate(flow_field_32, size=(64,64,64), mode='trilinear', align_corners=False)
#         # flow_field_32_resize = flow_field_32_resize*2
#         # moved_64 = self.transformer_64(source_64, flow_field_32_resize)

#         # x_64 = torch.cat([moved_64, target_64], dim=1)        
#         # x_64 = self.unet_model_64(x_64)
#         # flow_field_64 = self.flow_64(x_64)
#         # flow_field_64_resize = torch.nn.functional.interpolate(flow_field_64, size=(128,128,128), mode='trilinear', align_corners=False)
#         # flow_field_64_resize = flow_field_64_resize*2
#         # moved_128 = self.transformer_128(source_128, flow_field_64_resize)

#         # x_128 = torch.cat([moved_128, target_128], dim=1)        
#         # x_128 = self.unet_model_128(x_128)
#         # flow_field_128 = self.flow_128(x_128)

#         # flow_field = flow_field_128

# #------------------------------------------------------------------------------------------------------------------------

# #----------------------------------------------------------おうよう 64と128のみ------------------------------------------------------------

#         # x = torch.cat([source, target], dim=1)

#         # source_64 = torch.nn.functional.interpolate(source, size=(64,64,64), mode='trilinear', align_corners=False)
#         # target_64 = torch.nn.functional.interpolate(target, size=(64,64,64), mode='trilinear', align_corners=False)
#         # source_128 = source
#         # target_128 = target

#         # x_64 = torch.cat([source_64, target_64], dim=1)        
#         # x_64 = self.unet_model_64(x_64)
#         # flow_field_64 = self.flow_64(x_64)
#         # flow_field_64_resize = torch.nn.functional.interpolate(flow_field_64, size=(128,128,128), mode='trilinear', align_corners=False)
#         # flow_field_64_resize = flow_field_64_resize*2
#         # moved_128 = self.transformer_128(source_128, flow_field_64_resize)

#         # x_128 = torch.cat([moved_128, target_128], dim=1)        
#         # x_128 = self.unet_model_128(x_128)
#         # flow_field_128 = self.flow_128(x_128)

#         # flow_field = flow_field_128

# #------------------------------------------------------------------------------------------------------------------------
# #----------------------------------------------------------おうよう改定晩------------------------------------------------------------

#         # x = torch.cat([source, target], dim=1)

#         # source_16 = torch.nn.functional.interpolate(source, size=(16,16,16), mode='trilinear', align_corners=False)
#         # target_16 = torch.nn.functional.interpolate(target, size=(16,16,16), mode='trilinear', align_corners=False)
#         # source_32 = torch.nn.functional.interpolate(source, size=(32,32,32), mode='trilinear', align_corners=False)
#         # target_32 = torch.nn.functional.interpolate(target, size=(32,32,32), mode='trilinear', align_corners=False)
#         # source_64 = torch.nn.functional.interpolate(source, size=(64,64,64), mode='trilinear', align_corners=False)
#         # target_64 = torch.nn.functional.interpolate(target, size=(64,64,64), mode='trilinear', align_corners=False)
#         # source_128 = source
#         # target_128 = target

#         # x_16 = torch.cat([source_16, target_16], dim=1)
#         # x_16 = self.unet_model_16(x_16)
#         # flow_field_16 = self.flow_16(x_16)

#         # flow_field_16_32_resize = torch.nn.functional.interpolate(flow_field_16, size=(32,32,32), mode='trilinear', align_corners=False)
#         # flow_field_16_32_resize = flow_field_16_32_resize*2
#         # moved_32 = self.transformer_32(source_32, flow_field_16_32_resize)

#         # x_32 = torch.cat([moved_32, target_32], dim=1)        
#         # x_32 = self.unet_model_32(x_32)
#         # flow_field_32 = self.flow_32(x_32)

#         # flow_field_16_64_resize = torch.nn.functional.interpolate(flow_field_16, size=(64,64,64), mode='trilinear', align_corners=False)
#         # flow_field_32_64_resize = torch.nn.functional.interpolate(flow_field_32, size=(64,64,64), mode='trilinear', align_corners=False)
#         # flow_field_16_64_resize = flow_field_16_64_resize*4
#         # flow_field_32_64_resize = flow_field_32_64_resize*2

#         # flow_field_16_Plus_32_resize = flow_field_16_64_resize + flow_field_32_64_resize
#         # moved_64 = self.transformer_64(source_64, flow_field_16_Plus_32_resize)

#         # x_64 = torch.cat([moved_64, target_64], dim=1)        
#         # x_64 = self.unet_model_64(x_64)
#         # flow_field_64 = self.flow_64(x_64)

#         # flow_field_16_128_resize = torch.nn.functional.interpolate(flow_field_16, size=(128,128,128), mode='trilinear', align_corners=False)
#         # flow_field_32_128_resize = torch.nn.functional.interpolate(flow_field_32, size=(128,128,128), mode='trilinear', align_corners=False)
#         # flow_field_64_128_resize = torch.nn.functional.interpolate(flow_field_64, size=(128,128,128), mode='trilinear', align_corners=False) 
#         # flow_field_16_128_resize = flow_field_16_128_resize*8
#         # flow_field_32_128_resize = flow_field_32_128_resize*4
#         # flow_field_64_128_resize = flow_field_64_128_resize*2

#         # flow_field_16_Plus_32_Plus_64_resize = flow_field_16_128_resize + flow_field_32_128_resize + flow_field_64_128_resize

#         # moved_128 = self.transformer_128(source_128, flow_field_16_Plus_32_Plus_64_resize)

#         # x_128 = torch.cat([moved_128, target_128], dim=1)        
#         # x_128 = self.unet_model_128(x_128)
#         # flow_field_128 = self.flow_128(x_128)

#         # flow_field = flow_field_128

# #------------------------------------------------------------------------------------------------------------------------

# #----------------------------------------------------------おうよう改定晩2------------------------------------------------------------

#         x = torch.cat([source, target], dim=1)

#         source_16 = torch.nn.functional.interpolate(source, size=(16,16,16), mode='trilinear', align_corners=False)
#         target_16 = torch.nn.functional.interpolate(target, size=(16,16,16), mode='trilinear', align_corners=False)
#         source_32 = torch.nn.functional.interpolate(source, size=(32,32,32), mode='trilinear', align_corners=False)
#         target_32 = torch.nn.functional.interpolate(target, size=(32,32,32), mode='trilinear', align_corners=False)
#         source_64 = torch.nn.functional.interpolate(source, size=(64,64,64), mode='trilinear', align_corners=False)
#         target_64 = torch.nn.functional.interpolate(target, size=(64,64,64), mode='trilinear', align_corners=False)
#         source_128 = source
#         target_128 = target

#         x_16 = torch.cat([source_16, target_16], dim=1)
#         x_16 = self.unet_model_16(x_16)
#         flow_field_16 = self.flow_16(x_16)

#         flow_field_16_32_resize = torch.nn.functional.interpolate(flow_field_16, size=(32,32,32), mode='trilinear', align_corners=False)
#         flow_field_16_32_resize = flow_field_16_32_resize*2
#         moved_32 = self.transformer_32(source_32, flow_field_16_32_resize)

#         x_32 = torch.cat([moved_32, target_32], dim=1)        
#         x_32 = self.unet_model_32(x_32)
#         flow_field_32 = self.flow_32(x_32)

#         flow_field_16_64_resize = torch.nn.functional.interpolate(flow_field_16, size=(64,64,64), mode='trilinear', align_corners=False)
#         flow_field_32_64_resize = torch.nn.functional.interpolate(flow_field_32, size=(64,64,64), mode='trilinear', align_corners=False)
#         flow_field_16_64_resize = flow_field_16_64_resize*4
#         flow_field_32_64_resize = flow_field_32_64_resize*2

#         moved_64 = self.transformer_64(source_64, flow_field_16_64_resize)
#         moved_64 = self.transformer_64(moved_64, flow_field_32_64_resize)

#         x_64 = torch.cat([moved_64, target_64], dim=1)        
#         x_64 = self.unet_model_64(x_64)
#         flow_field_64 = self.flow_64(x_64)

#         flow_field_16_128_resize = torch.nn.functional.interpolate(flow_field_16, size=(128,128,128), mode='trilinear', align_corners=False)
#         flow_field_32_128_resize = torch.nn.functional.interpolate(flow_field_32, size=(128,128,128), mode='trilinear', align_corners=False)
#         flow_field_64_128_resize = torch.nn.functional.interpolate(flow_field_64, size=(128,128,128), mode='trilinear', align_corners=False) 
#         flow_field_16_128_resize = flow_field_16_128_resize*8
#         flow_field_32_128_resize = flow_field_32_128_resize*4
#         flow_field_64_128_resize = flow_field_64_128_resize*2

#         moved_128_11 = self.transformer_128(source_128, flow_field_16_128_resize)
#         moved_128_22 = self.transformer_128(moved_128_11, flow_field_32_128_resize)
#         moved_128_33 = self.transformer_128(moved_128_22, flow_field_64_128_resize)


#         x_128 = torch.cat([moved_128_33, target_128], dim=1)        
#         x_128 = self.unet_model_128(x_128)
#         flow_field_128 = self.flow_128(x_128)

#         flow_field = flow_field_128

# #------------------------------------------------------------------------------------------------------------------------
# #----------------------------------------------------------------Wavelet変換するぜ---------------------------------------------------
#         # import pywt
#         # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#         # level = 3

#         # source_np = source.cpu().numpy()
#         # target_np = target.cpu().numpy()

#         # coeffs_Moving = pywt.wavedecn(source_np, 'db1', mode='symmetric', level=level)
#         # coeffs_Fixed = pywt.wavedecn(target_np, 'db1', mode='symmetric', level=level)

#         # coeffs_Moving_16 = coeffs_Moving[0]
#         # coeffs_Fixed_16 = coeffs_Fixed[0]

#         # coeffs_Moving_16 = torch.tensor(coeffs_Moving_16)
#         # coeffs_Moving_16 = coeffs_Moving_16.unsqueeze(0).unsqueeze(0).expand(2, 1, 16, 16, 16)
#         # coeffs_Fixed_16 = torch.tensor(coeffs_Fixed_16)
#         # coeffs_Fixed_16 = coeffs_Fixed_16.unsqueeze(0).unsqueeze(0).expand(2, 1, 16, 16, 16)

#         # coeffs_Moving_16 = coeffs_Moving_16.to(device)
#         # coeffs_Fixed_16 = coeffs_Fixed_16.to(device)    

#         # x_16 = torch.cat([coeffs_Moving_16, coeffs_Fixed_16], dim=1)
#         # # print(x_16.shape)
#         # x_16 = self.unet_model_16(x_16)
#         # flow_field_16 = self.flow_16(x_16)

#         # coeffs_Moving[1]['aad'] = torch.tensor(coeffs_Moving[1]['aad'])
#         # coeffs_Moving[1]['aad'] = coeffs_Moving[1]['aad'].unsqueeze(0).unsqueeze(0).expand(2, 1, 16, 16, 16)
#         # coeffs_Moving[1]['aad'] = coeffs_Moving[1]['aad'].to(device)       
#         # coeffs_Moving[1]['ada'] = torch.tensor(coeffs_Moving[1]['ada'])
#         # coeffs_Moving[1]['ada'] = coeffs_Moving[1]['ada'].unsqueeze(0).unsqueeze(0).expand(2, 1, 16, 16, 16)
#         # coeffs_Moving[1]['ada'] = coeffs_Moving[1]['ada'].to(device)     
#         # coeffs_Moving[1]['add'] = torch.tensor(coeffs_Moving[1]['add'])
#         # coeffs_Moving[1]['add'] = coeffs_Moving[1]['add'].unsqueeze(0).unsqueeze(0).expand(2, 1, 16, 16, 16)
#         # coeffs_Moving[1]['add'] = coeffs_Moving[1]['add'].to(device)     
#         # coeffs_Moving[1]['daa'] = torch.tensor(coeffs_Moving[1]['daa'])
#         # coeffs_Moving[1]['daa'] = coeffs_Moving[1]['daa'].unsqueeze(0).unsqueeze(0).expand(2, 1, 16, 16, 16)
#         # coeffs_Moving[1]['daa'] = coeffs_Moving[1]['daa'].to(device)     
#         # coeffs_Moving[1]['dad'] = torch.tensor(coeffs_Moving[1]['dad'])
#         # coeffs_Moving[1]['dad'] = coeffs_Moving[1]['dad'].unsqueeze(0).unsqueeze(0).expand(2, 1, 16, 16, 16)
#         # coeffs_Moving[1]['dad'] = coeffs_Moving[1]['dad'].to(device)     
#         # coeffs_Moving[1]['dda'] = torch.tensor(coeffs_Moving[1]['dda'])
#         # coeffs_Moving[1]['dda'] = coeffs_Moving[1]['dda'].unsqueeze(0).unsqueeze(0).expand(2, 1, 16, 16, 16)
#         # coeffs_Moving[1]['dda'] = coeffs_Moving[1]['daa'].to(device)     
#         # coeffs_Moving[1]['ddd'] = torch.tensor(coeffs_Moving[1]['ddd'])
#         # coeffs_Moving[1]['ddd'] = coeffs_Moving[1]['ddd'].unsqueeze(0).unsqueeze(0).expand(2, 1, 16, 16, 16)
#         # coeffs_Moving[1]['ddd'] = coeffs_Moving[1]['ddd'].to(device)     

#         # moved_level_1 = self.transformer_16(coeffs_Moving_16, flow_field_16)
#         # moved_level_1_aad = self.transformer_16(coeffs_Moving[1]['aad'], flow_field_16)
#         # moved_level_1_ada = self.transformer_16(coeffs_Moving[1]['ada'], flow_field_16)
#         # moved_level_1_add = self.transformer_16(coeffs_Moving[1]['add'], flow_field_16)
#         # moved_level_1_daa = self.transformer_16(coeffs_Moving[1]['daa'], flow_field_16)
#         # moved_level_1_dad = self.transformer_16(coeffs_Moving[1]['dad'], flow_field_16)
#         # moved_level_1_dda = self.transformer_16(coeffs_Moving[1]['dda'], flow_field_16)
#         # moved_level_1_ddd = self.transformer_16(coeffs_Moving[1]['ddd'], flow_field_16)


#         # moved_level_1 = moved_level_1.reshape(2 * 1, 16, 16, 16)
#         # moved_level_1 = moved_level_1[0]  
#         # moved_level_1 = moved_level_1.detach().cpu().numpy()

#         # moved_level_1_aad = moved_level_1_aad.reshape(2 * 1, 16, 16, 16)
#         # moved_level_1_aad = moved_level_1_aad[0] 
#         # moved_level_1_aad = moved_level_1_aad.detach().cpu().numpy()

#         # moved_level_1_ada = moved_level_1_ada.reshape(2 * 1, 16, 16, 16)
#         # moved_level_1_ada = moved_level_1_ada[0]
#         # moved_level_1_ada = moved_level_1_ada.detach().cpu().numpy()

#         # moved_level_1_add = moved_level_1_add.reshape(2 * 1, 16, 16, 16)
#         # moved_level_1_add = moved_level_1_add[0]
#         # moved_level_1_add = moved_level_1_add.detach().cpu().numpy()

#         # moved_level_1_daa = moved_level_1_daa.reshape(2 * 1, 16, 16, 16)
#         # moved_level_1_daa = moved_level_1_daa[0]
#         # moved_level_1_daa = moved_level_1_daa.detach().cpu().numpy()

#         # moved_level_1_dad = moved_level_1_dad.reshape(2 * 1, 16, 16, 16)
#         # moved_level_1_dad = moved_level_1_dad[0]
#         # moved_level_1_dad = moved_level_1_dad.detach().cpu().numpy()

#         # moved_level_1_dda = moved_level_1_dda.reshape(2 * 1, 16, 16, 16)
#         # moved_level_1_dda = moved_level_1_dda[0]
#         # moved_level_1_dda = moved_level_1_dda.detach().cpu().numpy()

#         # moved_level_1_ddd = moved_level_1_ddd.reshape(2 * 1, 16, 16, 16)
#         # moved_level_1_ddd = moved_level_1_ddd[0]
#         # moved_level_1_ddd = moved_level_1_ddd.detach().cpu().numpy()

#         # coeffs_Moving_32 = pywt.waverecn((moved_level_1, {
#         #     'aad': moved_level_1_aad,
#         #     'ada': moved_level_1_ada,
#         #     'add': moved_level_1_add,
#         #     'daa': moved_level_1_daa,
#         #     'dad': moved_level_1_dad,
#         #     'dda': moved_level_1_dda,
#         #     'ddd': moved_level_1_ddd
#         # }), 'db1', mode='symmetric')

#         # Fixed_level_1 = coeffs_Fixed_16
#         # Fixed_level_1 = Fixed_level_1.reshape(2 * 1, 16, 16, 16)
#         # Fixed_level_1 = Fixed_level_1[0]  
#         # Fixed_level_1 = Fixed_level_1.detach().cpu().numpy()           
#         # Fixed_level_1_aad = coeffs_Fixed[1]['aad']
#         # Fixed_level_1_ada = coeffs_Fixed[1]['ada']
#         # Fixed_level_1_add = coeffs_Fixed[1]['add']
#         # Fixed_level_1_daa = coeffs_Fixed[1]['daa']
#         # Fixed_level_1_dad = coeffs_Fixed[1]['dad']
#         # Fixed_level_1_dda = coeffs_Fixed[1]['dda']
#         # Fixed_level_1_ddd = coeffs_Fixed[1]['ddd']

#         # coeffs_Fixed_32 = pywt.waverecn((Fixed_level_1, {
#         #     'aad': Fixed_level_1_aad,
#         #     'ada': Fixed_level_1_ada,
#         #     'add': Fixed_level_1_add,
#         #     'daa': Fixed_level_1_daa,
#         #     'dad': Fixed_level_1_dad,
#         #     'dda': Fixed_level_1_dda,
#         #     'ddd': Fixed_level_1_ddd
#         # }), 'db1', mode='symmetric')

#         # coeffs_Moving_32 = torch.tensor(coeffs_Moving_32)
#         # coeffs_Moving_32 = coeffs_Moving_32.unsqueeze(0).unsqueeze(0).expand(2, 1, 32, 32, 32)
#         # coeffs_Fixed_32 = torch.tensor(coeffs_Fixed_32)
#         # coeffs_Fixed_32 = coeffs_Fixed_32.unsqueeze(0).unsqueeze(0).expand(2, 1, 32, 32, 32)

#         # coeffs_Moving_32 = coeffs_Moving_32.to(device)
#         # coeffs_Fixed_32 = coeffs_Fixed_32.to(device)

#         # x_32 = torch.cat([coeffs_Moving_32, coeffs_Fixed_32], dim=1)
#         # x_32 = self.unet_model_32(x_32)
#         # flow_field_32 = self.flow_32(x_32)

#         # coeffs_Moving[2]['aad'] = torch.tensor(coeffs_Moving[2]['aad'])
#         # coeffs_Moving[2]['aad'] = coeffs_Moving[2]['aad'].unsqueeze(0).unsqueeze(0).expand(2, 1, 32, 32, 32)
#         # coeffs_Moving[2]['aad'] = coeffs_Moving[2]['aad'].to(device)

#         # coeffs_Moving[2]['ada'] = torch.tensor(coeffs_Moving[2]['ada'])
#         # coeffs_Moving[2]['ada'] = coeffs_Moving[2]['ada'].unsqueeze(0).unsqueeze(0).expand(2, 1, 32, 32, 32)
#         # coeffs_Moving[2]['ada'] = coeffs_Moving[2]['ada'].to(device)

#         # coeffs_Moving[2]['add'] = torch.tensor(coeffs_Moving[2]['add'])
#         # coeffs_Moving[2]['add'] = coeffs_Moving[2]['add'].unsqueeze(0).unsqueeze(0).expand(2, 1, 32, 32, 32)
#         # coeffs_Moving[2]['add'] = coeffs_Moving[2]['add'].to(device)

#         # coeffs_Moving[2]['daa'] = torch.tensor(coeffs_Moving[2]['daa'])
#         # coeffs_Moving[2]['daa'] = coeffs_Moving[2]['daa'].unsqueeze(0).unsqueeze(0).expand(2, 1, 32, 32, 32)
#         # coeffs_Moving[2]['daa'] = coeffs_Moving[2]['daa'].to(device)

#         # coeffs_Moving[2]['dad'] = torch.tensor(coeffs_Moving[2]['dad'])
#         # coeffs_Moving[2]['dad'] = coeffs_Moving[2]['dad'].unsqueeze(0).unsqueeze(0).expand(2, 1, 32, 32, 32)
#         # coeffs_Moving[2]['dad'] = coeffs_Moving[2]['dad'].to(device)

#         # coeffs_Moving[2]['dda'] = torch.tensor(coeffs_Moving[2]['dda'])
#         # coeffs_Moving[2]['dda'] = coeffs_Moving[2]['dda'].unsqueeze(0).unsqueeze(0).expand(2, 1, 32, 32, 32) 
#         # coeffs_Moving[2]['dda'] = coeffs_Moving[2]['dda'].to(device)

#         # coeffs_Moving[2]['ddd'] = torch.tensor(coeffs_Moving[2]['ddd'])
#         # coeffs_Moving[2]['ddd'] = coeffs_Moving[2]['ddd'].unsqueeze(0).unsqueeze(0).expand(2, 1, 32, 32, 32)
#         # coeffs_Moving[2]['ddd'] = coeffs_Moving[2]['ddd'].to(device)   

#         # moved_level_2 = self.transformer_32(coeffs_Moving_32, flow_field_32)
#         # moved_level_2_aad = self.transformer_32(coeffs_Moving[2]['aad'], flow_field_32)
#         # moved_level_2_ada = self.transformer_32(coeffs_Moving[2]['ada'], flow_field_32)
#         # moved_level_2_add = self.transformer_32(coeffs_Moving[2]['add'], flow_field_32)
#         # moved_level_2_daa = self.transformer_32(coeffs_Moving[2]['daa'], flow_field_32)
#         # moved_level_2_dad = self.transformer_32(coeffs_Moving[2]['dad'], flow_field_32)
#         # moved_level_2_dda = self.transformer_32(coeffs_Moving[2]['dda'], flow_field_32)
#         # moved_level_2_ddd = self.transformer_32(coeffs_Moving[2]['ddd'], flow_field_32)

#         # moved_level_2 = moved_level_2.reshape(2 * 1, 32, 32, 32)
#         # moved_level_2 = moved_level_2[0]  
#         # moved_level_2 = moved_level_2.detach().cpu().numpy()

#         # moved_level_2_aad = moved_level_2_aad.reshape(2 * 1, 32, 32, 32)
#         # moved_level_2_aad = moved_level_2_aad[0] 
#         # moved_level_2_aad = moved_level_2_aad.detach().cpu().numpy()

#         # moved_level_2_ada = moved_level_2_ada.reshape(2 * 1, 32, 32, 32)
#         # moved_level_2_ada = moved_level_2_ada[0]
#         # moved_level_2_ada = moved_level_2_ada.detach().cpu().numpy()

#         # moved_level_2_add = moved_level_2_add.reshape(2 * 1, 32, 32, 32)
#         # moved_level_2_add = moved_level_2_add[0]
#         # moved_level_2_add = moved_level_2_add.detach().cpu().numpy()

#         # moved_level_2_daa = moved_level_2_daa.reshape(2 * 1, 32, 32, 32)
#         # moved_level_2_daa = moved_level_2_daa[0]
#         # moved_level_2_daa = moved_level_2_daa.detach().cpu().numpy()

#         # moved_level_2_dad = moved_level_2_dad.reshape(2 * 1, 32, 32, 32)
#         # moved_level_2_dad = moved_level_2_dad[0]
#         # moved_level_2_dad = moved_level_2_dad.detach().cpu().numpy()

#         # moved_level_2_dda = moved_level_2_dda.reshape(2 * 1, 32, 32, 32)
#         # moved_level_2_dda = moved_level_2_dda[0]
#         # moved_level_2_dda = moved_level_2_dda.detach().cpu().numpy()

#         # moved_level_2_ddd = moved_level_2_ddd.reshape(2 * 1, 32, 32, 32)
#         # moved_level_2_ddd = moved_level_2_ddd[0]
#         # moved_level_2_ddd = moved_level_2_ddd.detach().cpu().numpy()

#         # coeffs_Moving_64 = pywt.waverecn((moved_level_2, {
#         #     'aad': moved_level_2_aad,
#         #     'ada': moved_level_2_ada,
#         #     'add': moved_level_2_add,
#         #     'daa': moved_level_2_daa,
#         #     'dad': moved_level_2_dad,
#         #     'dda': moved_level_2_dda,
#         #     'ddd': moved_level_2_ddd
#         # }), 'db1', mode='symmetric')

#         # Fixed_level_2 = coeffs_Fixed_32
#         # Fixed_level_2 = Fixed_level_2.reshape(2 * 1, 32, 32, 32)
#         # Fixed_level_2 = Fixed_level_2[0]  
#         # Fixed_level_2 = Fixed_level_2.detach().cpu().numpy()
#         # Fixed_level_2_aad = coeffs_Fixed[2]['aad']
#         # Fixed_level_2_ada = coeffs_Fixed[2]['ada']
#         # Fixed_level_2_add = coeffs_Fixed[2]['add']
#         # Fixed_level_2_daa = coeffs_Fixed[2]['daa']
#         # Fixed_level_2_dad = coeffs_Fixed[2]['dad']
#         # Fixed_level_2_dda = coeffs_Fixed[2]['dda']
#         # Fixed_level_2_ddd = coeffs_Fixed[2]['ddd']

#         # coeffs_Fixed_64 = pywt.waverecn((Fixed_level_2, {
#         #     'aad': Fixed_level_2_aad,
#         #     'ada': Fixed_level_2_ada,
#         #     'add': Fixed_level_2_add,
#         #     'daa': Fixed_level_2_daa,
#         #     'dad': Fixed_level_2_dad,
#         #     'dda': Fixed_level_2_dda,
#         #     'ddd': Fixed_level_2_ddd
#         # }), 'db1', mode='symmetric')

#         # coeffs_Moving_64 = torch.tensor(coeffs_Moving_64)
#         # coeffs_Moving_64 = coeffs_Moving_64.unsqueeze(0).unsqueeze(0).expand(2, 1, 64, 64, 64)
#         # coeffs_Fixed_64 = torch.tensor(coeffs_Fixed_64)
#         # coeffs_Fixed_64 = coeffs_Fixed_64.unsqueeze(0).unsqueeze(0).expand(2, 1, 64, 64, 64)

#         # coeffs_Moving_64 = coeffs_Moving_64.to(device)
#         # coeffs_Fixed_64 = coeffs_Fixed_64.to(device)

#         # x_64 = torch.cat([coeffs_Moving_64, coeffs_Fixed_64], dim=1)
#         # x_64 = self.unet_model_64(x_64)
#         # flow_field_64 = self.flow_64(x_64)

#         # coeffs_Moving[3]['aad'] = torch.tensor(coeffs_Moving[3]['aad'])
#         # coeffs_Moving[3]['aad'] = coeffs_Moving[3]['aad'].unsqueeze(0).unsqueeze(0).expand(2, 1, 64, 64, 64) 
#         # coeffs_Moving[3]['aad'] = coeffs_Moving[3]['aad'].to(device)

#         # coeffs_Moving[3]['ada'] = torch.tensor(coeffs_Moving[3]['ada'])
#         # coeffs_Moving[3]['ada'] = coeffs_Moving[3]['ada'].unsqueeze(0).unsqueeze(0).expand(2, 1, 64, 64, 64) 
#         # coeffs_Moving[3]['ada'] = coeffs_Moving[3]['ada'].to(device)

#         # coeffs_Moving[3]['add'] = torch.tensor(coeffs_Moving[3]['add'])
#         # coeffs_Moving[3]['add'] = coeffs_Moving[3]['add'].unsqueeze(0).unsqueeze(0).expand(2, 1, 64, 64, 64) 
#         # coeffs_Moving[3]['add'] = coeffs_Moving[3]['add'].to(device)

#         # coeffs_Moving[3]['daa'] = torch.tensor(coeffs_Moving[3]['daa'])
#         # coeffs_Moving[3]['daa'] = coeffs_Moving[3]['daa'].unsqueeze(0).unsqueeze(0).expand(2, 1, 64, 64, 64)
#         # coeffs_Moving[3]['daa'] = coeffs_Moving[3]['daa'].to(device)

#         # coeffs_Moving[3]['dad'] = torch.tensor(coeffs_Moving[3]['dad'])
#         # coeffs_Moving[3]['dad'] = coeffs_Moving[3]['dad'].unsqueeze(0).unsqueeze(0).expand(2, 1, 64, 64, 64) 
#         # coeffs_Moving[3]['dad'] = coeffs_Moving[3]['dad'].to(device)

#         # coeffs_Moving[3]['dda'] = torch.tensor(coeffs_Moving[3]['dda'])
#         # coeffs_Moving[3]['dda'] = coeffs_Moving[3]['dda'].unsqueeze(0).unsqueeze(0).expand(2, 1, 64, 64, 64)
#         # coeffs_Moving[3]['dda'] = coeffs_Moving[3]['dda'].to(device)

#         # coeffs_Moving[3]['ddd'] = torch.tensor(coeffs_Moving[3]['ddd'])
#         # coeffs_Moving[3]['ddd'] = coeffs_Moving[3]['ddd'].unsqueeze(0).unsqueeze(0).expand(2, 1, 64, 64, 64) 
#         # coeffs_Moving[3]['ddd'] = coeffs_Moving[3]['ddd'].to(device)   

#         # moved_level_3 = self.transformer_64(coeffs_Moving_64, flow_field_64)
#         # moved_level_3_aad = self.transformer_64(coeffs_Moving[3]['aad'], flow_field_64)
#         # moved_level_3_ada = self.transformer_64(coeffs_Moving[3]['ada'], flow_field_64)
#         # moved_level_3_add = self.transformer_64(coeffs_Moving[3]['add'], flow_field_64)
#         # moved_level_3_daa = self.transformer_64(coeffs_Moving[3]['daa'], flow_field_64)
#         # moved_level_3_dad = self.transformer_64(coeffs_Moving[3]['dad'], flow_field_64)
#         # moved_level_3_dda = self.transformer_64(coeffs_Moving[3]['dda'], flow_field_64)
#         # moved_level_3_ddd = self.transformer_64(coeffs_Moving[3]['ddd'], flow_field_64)

#         # moved_level_3 = moved_level_3.reshape(2 * 1, 64, 64, 64)
#         # moved_level_3 = moved_level_3[0]  
#         # moved_level_3 = moved_level_3.detach().cpu().numpy()

#         # moved_level_3_aad = moved_level_3_aad.reshape(2 * 1, 64, 64, 64)
#         # moved_level_3_aad = moved_level_3_aad[0] 
#         # moved_level_3_aad = moved_level_3_aad.detach().cpu().numpy()

#         # moved_level_3_ada = moved_level_3_ada.reshape(2 * 1, 64, 64, 64)
#         # moved_level_3_ada = moved_level_3_ada[0]
#         # moved_level_3_ada = moved_level_3_ada.detach().cpu().numpy()

#         # moved_level_3_add = moved_level_3_add.reshape(2 * 1, 64, 64, 64)
#         # moved_level_3_add = moved_level_3_add[0]
#         # moved_level_3_add = moved_level_3_add.detach().cpu().numpy()

#         # moved_level_3_daa = moved_level_3_daa.reshape(2 * 1, 64, 64, 64)
#         # moved_level_3_daa = moved_level_3_daa[0]
#         # moved_level_3_daa = moved_level_3_daa.detach().cpu().numpy()

#         # moved_level_3_dad = moved_level_3_dad.reshape(2 * 1, 64, 64, 64)
#         # moved_level_3_dad = moved_level_3_dad[0]
#         # moved_level_3_dad = moved_level_3_dad.detach().cpu().numpy()

#         # moved_level_3_dda = moved_level_3_dda.reshape(2 * 1, 64, 64, 64)
#         # moved_level_3_dda = moved_level_3_dda[0]
#         # moved_level_3_dda = moved_level_3_dda.detach().cpu().numpy()

#         # moved_level_3_ddd = moved_level_3_ddd.reshape(2 * 1, 64, 64, 64)
#         # moved_level_3_ddd = moved_level_3_ddd[0]
#         # moved_level_3_ddd = moved_level_3_ddd.detach().cpu().numpy()


#         # coeffs_Moving_128 = pywt.waverecn((moved_level_3, {
#         #     'aad': moved_level_3_aad,
#         #     'ada': moved_level_3_ada,
#         #     'add': moved_level_3_add,
#         #     'daa': moved_level_3_daa,
#         #     'dad': moved_level_3_dad,
#         #     'dda': moved_level_3_dda,
#         #     'ddd': moved_level_3_ddd
#         # }), 'db1', mode='symmetric')

#         # Fixed_level_3 = coeffs_Fixed_64
#         # Fixed_level_3 = Fixed_level_3.reshape(2 * 1, 64, 64, 64)
#         # Fixed_level_3 = Fixed_level_3[0]  
#         # Fixed_level_3 = Fixed_level_3.detach().cpu().numpy()
#         # Fixed_level_3_aad = coeffs_Fixed[3]['aad']
#         # Fixed_level_3_ada = coeffs_Fixed[3]['ada']
#         # Fixed_level_3_add = coeffs_Fixed[3]['add']
#         # Fixed_level_3_daa = coeffs_Fixed[3]['daa']
#         # Fixed_level_3_dad = coeffs_Fixed[3]['dad']
#         # Fixed_level_3_dda = coeffs_Fixed[3]['dda']
#         # Fixed_level_3_ddd = coeffs_Fixed[3]['ddd']

#         # coeffs_Fixed_128 = pywt.waverecn((Fixed_level_3, {
#         #     'aad': Fixed_level_3_aad,
#         #     'ada': Fixed_level_3_ada,
#         #     'add': Fixed_level_3_add,
#         #     'daa': Fixed_level_3_daa,
#         #     'dad': Fixed_level_3_dad,
#         #     'dda': Fixed_level_3_dda,
#         #     'ddd': Fixed_level_3_ddd
#         # }), 'db1', mode='symmetric')

#         # coeffs_Moving_128 = torch.tensor(coeffs_Moving_128)
#         # coeffs_Moving_128 = coeffs_Moving_128.unsqueeze(0).unsqueeze(0).expand(2, 1, 128, 128, 128)
#         # coeffs_Fixed_128 = torch.tensor(coeffs_Fixed_128)
#         # coeffs_Fixed_128 = coeffs_Fixed_128.unsqueeze(0).unsqueeze(0).expand(2, 1, 128, 128, 128)

#         # coeffs_Moving_128 = coeffs_Moving_128.to(device)
#         # coeffs_Fixed_128 = coeffs_Fixed_128.to(device)

#         # x_128 = torch.cat([coeffs_Moving_128, coeffs_Fixed_128], dim=1)
#         # x_128 = self.unet_model_128(x_128)
#         # flow_field_128 = self.flow_128(x_128)

#         # moved_128 = self.transformer_128(coeffs_Moving_128, flow_field_128)







# #------------------------------------------------------------------------------------------------------------------------
# #-------------------------------------------------Kihon----------------------------------------------------------------------

#         # resize flow for integration
#         pos_flow = flow_field
#         if self.resize:
#             print("A1")
#             pos_flow = self.resize(pos_flow)
#         preint_flow = pos_flow

#         # pos_flowは変形ベクトル

#         # negate flow for bidirectional model
#         neg_flow = -pos_flow if self.bidir else None

#         # integrate to produce diffeomorphic warp
#         if self.integrate:
#             print("A2")
#             pos_flow = self.integrate(pos_flow)
#             neg_flow = self.integrate(neg_flow) if self.bidir else None

#             # resize to final resolution
#             if self.fullsize:
#                 print("A3")
#                 pos_flow = self.fullsize(pos_flow)
#                 neg_flow = self.fullsize(neg_flow) if self.bidir else None

#         # warp image with flow field
#         # preint_flow = torch.ones_like(pos_flow)*(50)
#         # print(pos_flow.mean())




#         y_source = self.transformer(source, pos_flow)




# #--------------------------------------------kihonn--------------------------------------------------------------------------

#         # moved_16_2 = self.transformer_16(source_16, flow_field_16)
#         # moved_32_2 = self.transformer_32(moved_32, flow_field_32)
#         # moved_64_2 = self.transformer_64(moved_64, flow_field_64)
#         # moved_128_2 = self.transformer_128(moved_128, pos_flow)

#         # flow_field_all = flow_field_16_128_resize + flow_field_32_128_resize + flow_field_64_128_resize + flow_field_128
#         # moved_128_final = self.transformer_128(source_128, flow_field_all)


# #------------------------------------------------------------------------------------------------------------------------

# #--------------------------------------------kihonn2--------------------------------------------------------------------------

#         moved_16_2 = self.transformer_16(source_16, flow_field_16)
#         moved_32_2 = self.transformer_32(moved_32, flow_field_32)
#         moved_64_2 = self.transformer_64(moved_64, flow_field_64)
#         moved_128_2 = self.transformer_128(moved_128_33, pos_flow)
#         flow_field_all = flow_field_16_128_resize + flow_field_32_128_resize + flow_field_64_128_resize + flow_field_128
#         moved_128_final = self.transformer_128(moved_128_33, flow_field)


# #------------------------------------------------------------------------------------------------------------------------




#         # y_source = self.transformer(source, preint_flow)
#         y_target = self.transformer(target, neg_flow) if self.bidir else None

#         # return non-integrated flow field if training
#         if not registration:
#             # print("aaaa", preint_flow.max())
#             # return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
#             #return (moved_128, y_target, preint_flow) if self.bidir else (moved_16, target_16, moved_32, target_32, moved_64, target_64,  moved_128, target, pos_flow) kihonn
#             # return (moved_128, y_target, preint_flow) if self.bidir else (moved_16_2, target_16, moved_32_2, target_32, moved_64_2, target_64,  moved_128_2, target_128, pos_flow)
#             # return (moved_128, y_target, preint_flow) if self.bidir else ( source_16, target_16, moved_16_2, flow_field_16, source_32, moved_32, target_32, moved_32_2, flow_field_32, source_64,  moved_64, target_64, moved_64_2, flow_field_64, source_128, moved_128, target_128, moved_128_2, pos_flow)     
#             # return (moved_128, y_target, preint_flow) if self.bidir else ( source_64, target_64, moved_64_2, flow_field_64, source_128, moved_128, target_128, moved_128_2, pos_flow)     
#             return (moved_128_11, y_target, preint_flow) if self.bidir else ( source_16, target_16, moved_16_2, flow_field_16, source_32, moved_32, \
#                                                                           target_32, moved_32_2, flow_field_32, source_64,  moved_64, target_64, moved_64_2, \
#                                                                             flow_field_64, source_128, moved_128_11, target_128, moved_128_2, pos_flow, flow_field_all, moved_128_final,moved_128_11, moved_128_22, moved_128_33)  
#             # return (moved_128, y_target, preint_flow) \
#             #     if self.bidir else ( moved_level_1, moved_level_1_aad, moved_level_1_ada, moved_level_1_add, moved_level_1_daa, moved_level_1_dad, moved_level_1_dda, moved_level_1_ddd, \
#             #                             Fixed_level_1, Fixed_level_1_aad, Fixed_level_1_ada, Fixed_level_1_add, Fixed_level_1_daa, Fixed_level_1_dad, Fixed_level_1_dda, Fixed_level_1_ddd, \
#             #                             coeffs_Moving_32, coeffs_Fixed_32, \
#             #                             moved_level_2, moved_level_2_aad, moved_level_2_ada, moved_level_2_add, moved_level_2_daa, moved_level_2_dad, moved_level_2_dda, moved_level_2_ddd, \
#             #                             Fixed_level_2, Fixed_level_2_aad, Fixed_level_2_ada, Fixed_level_2_add, Fixed_level_2_daa, Fixed_level_2_dad, Fixed_level_2_dda, Fixed_level_2_ddd, \
#             #                             coeffs_Moving_64, coeffs_Fixed_64, \
#             #                             moved_level_3, moved_level_3_aad, moved_level_3_ada, moved_level_3_add, moved_level_3_daa, moved_level_3_dad, moved_level_3_dda, moved_level_3_ddd, \
#             #                             Fixed_level_3, Fixed_level_3_aad, Fixed_level_3_ada, Fixed_level_3_add, Fixed_level_3_daa, Fixed_level_3_dad, Fixed_level_3_dda, Fixed_level_3_ddd, \
#             #                             coeffs_Moving_128, coeffs_Fixed_128, \
#             #                             moved_128, flow_field_128, flow_field_64, flow_field_32, flow_field_16) 
                                                
        
#         else:
#             pos_flow = torch.ones_like(pos_flow)*10
#             # print("bbbb", pos_flow.mean())
#             # return y_source, pos_flow
#             # return moved_16, target_16, moved_32, target_32, moved_64, target_64,  moved_128, target, pos_flow
#             # return moved_16_2, target_16, moved_32_2, target_32, moved_64_2, target_64,  moved_128_2, target_128, pos_flow
#             # return  source_16, target_16, moved_16_2, flow_field_16, source_32, moved_32, target_32, moved_32_2, flow_field_32, source_64, moved_64, target_64, moved_64_2, flow_field_64, source_128, moved_128, target_128, moved_128_2, pos_flow     
#             return  source_64, target_64, moved_64_2, flow_field_64, source_128, moved_128, target_128, moved_128_2, pos_flow     
#             # return ( moved_level_1, moved_level_1_aad, moved_level_1_ada, moved_level_1_add, moved_level_1_daa, moved_level_1_dad, moved_level_1_dda, moved_level_1_ddd, \
#             #                             Fixed_level_1, Fixed_level_1_aad, Fixed_level_1_ada, Fixed_level_1_add, Fixed_level_1_daa, Fixed_level_1_dad, Fixed_level_1_dda, Fixed_level_1_ddd, \
#             #                             coeffs_Moving_32, coeffs_Fixed_32, \
#             #                             moved_level_2, moved_level_2_aad, moved_level_2_ada, moved_level_2_add, moved_level_2_daa, moved_level_2_dad, moved_level_2_dda, moved_level_2_ddd, \
#             #                             Fixed_level_2, Fixed_level_2_aad, Fixed_level_2_ada, Fixed_level_2_add, Fixed_level_2_daa, Fixed_level_2_dad, Fixed_level_2_dda, Fixed_level_2_ddd, \
#             #                             coeffs_Moving_64, coeffs_Fixed_64, \
#             #                             moved_level_3, moved_level_3_aad, moved_level_3_ada, moved_level_3_add, moved_level_3_daa, moved_level_3_dad, moved_level_3_dda, moved_level_3_ddd, \
#             #                             Fixed_level_3, Fixed_level_3_aad, Fixed_level_3_ada, Fixed_level_3_add, Fixed_level_3_daa, Fixed_level_3_dad, Fixed_level_3_dda, Fixed_level_3_ddd, \
#             #                             coeffs_Moving_128, coeffs_Fixed_128, \
#             #                             moved_128, flow_field_128, flow_field_64, flow_field_32, flow_field_16) 
        

# # class ConvBlock(nn.Module):
# #     """
# #     Specific convolutional block followed by leakyrelu for unet.
# #     """

# #     def __init__(self, ndims, in_channels, out_channels, stride=1):
# #         super().__init__()

# #         Conv = getattr(nn, 'Conv%dd' % ndims)
# #         self.main = Conv(in_channels, out_channels, 3, stride, 1)
# #         self.activation = nn.LeakyReLU(0.2)

# #     def forward(self, x):
# #         out = self.main(x)
# #         out = self.activation(out)
# #         return out

# class ConvBlock(nn.Module):
#     """
#     Specific convolutional block followed by leakyrelu for unet.
#     """

#     def __init__(self, ndims, in_channels, out_channels, stride=1):
#         super().__init__()

#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.main = Conv(in_channels, out_channels, 3, stride, 1)
#         self.activation = nn.LeakyReLU(0.2)

#     def forward(self, x):
#         # Conv3dが呼ばれる前にxの型を確認するデバッグ用コード
#         # print("xの型（ConvBlockの中）:", type(x))
#         # print("（ConvBlockの中）:", x.shape)
        
#         out = self.main(x)
#         out = self.activation(out)
#         # print("aaaa1111", out.max())
#         # print("（ConvBlockの中）:", out.shape)
#         return out






# # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo







# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions.normal import Normal

# from .. import default_unet_features
# from . import layers
# from .modelio import LoadableModel, store_config_args


# class Unet(nn.Module):
#     """
#     A unet architecture. Layer features can be specified directly as a list of encoder and decoder
#     features or as a single integer along with a number of unet levels. The default network features
#     per layer (when no options are specified) are:

#         encoder: [16, 32, 32, 32]
#         decoder: [32, 32, 32, 32, 32, 16, 16]
#     """
        
#     def __init__(self,
#                  inshape=None,
#                  infeats=None,
#                  nb_features=None,
#                  nb_levels=None,
#                  max_pool=2,
#                  feat_mult=1,
#                  nb_conv_per_level=1,
#                  half_res=False):
        
#         """
#         Parameters:
#             inshape: Input shape. e.g. (192, 192, 192)
#             infeats: Number of input features.
#             nb_features: Unet convolutional features. Can be specified via a list of lists with
#                 the form [[encoder feats], [decoder feats]], or as a single integer. 
#                 If None (default), the unet features are defined by the default config described in 
#                 the class documentation.
#             nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
#                 Default is None.
#             feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
#                 Default is 1.
#             nb_conv_per_level: Number of convolutions per unet level. Default is 1.
#             half_res: Skip the last decoder upsampling. Default is False.
#         """
#         super().__init__()

#         # ensure correct dimensionality
#         ndims = len(inshape)
#         assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

#         # cache some parameters
#         self.half_res = half_res

#         # default encoder and decoder layer features if nothing provided
#         if nb_features is None:
#             nb_features = default_unet_features()
#             print("1")

#         # build feature list automatically
#         if isinstance(nb_features, int):
#             if nb_levels is None:
#                 print("2")
#                 raise ValueError('must provide unet nb_levels if nb_features is an integer')
#             feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
#             print("3")
#             nb_features = [
#                 np.repeat(feats[:-1], nb_conv_per_level),
#                 np.repeat(np.flip(feats), nb_conv_per_level)
#             ]
#         elif nb_levels is not None:
#             print("4")
#             raise ValueError('cannot use nb_levels if nb_features is not an integer')

#         # extract any surplus (full resolution) decoder convolutions
#         enc_nf, dec_nf = nb_features
#         nb_dec_convs = len(enc_nf)
#         final_convs = dec_nf[nb_dec_convs:]
#         dec_nf = dec_nf[:nb_dec_convs]
#         self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

#         if isinstance(max_pool, int):
#             print("5")
#             max_pool = [max_pool] * self.nb_levels

#         # cache downsampling / upsampling operations
#         MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
#         self.pooling = [MaxPooling(s) for s in max_pool]
#         self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

#         # configure encoder (down-sampling path)
#         prev_nf = infeats
#         encoder_nfs = [prev_nf]
#         self.encoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             print("6")
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 print("7")
#                 nf = enc_nf[level * nb_conv_per_level + conv]
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#             self.encoder.append(convs)
#             encoder_nfs.append(prev_nf)

#         # configure decoder (up-sampling path)
#         encoder_nfs = np.flip(encoder_nfs)
#         self.decoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             print("8")
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 print("9")
#                 nf = dec_nf[level * nb_conv_per_level + conv]
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#             self.decoder.append(convs)
#             if not half_res or level < (self.nb_levels - 2):
#                 print("10")
#                 prev_nf += encoder_nfs[level]

#         # now we take care of any remaining convolutions
#         self.remaining = nn.ModuleList()
#         for num, nf in enumerate(final_convs):
#             print("11")
#             self.remaining.append(ConvBlock(ndims, prev_nf, nf))
#             prev_nf = nf

#         # cache final number of features
#         self.final_nf = prev_nf

#     def forward(self, x):

#         # encoder forward pass
#         x_history = [x]
#         for level, convs in enumerate(self.encoder):
#             for conv in convs:
#                 x = conv(x)
#             x_history.append(x)
#             # print("After encoder level（エンコーダの中）", level, ":", x.shape)  # 追加
#             x = self.pooling[level](x)

#         # decoder forward pass with upsampling and concatenation
#         for level, convs in enumerate(self.decoder):
#             for conv in convs:
#                 x = conv(x)
#             if not self.half_res or level < (self.nb_levels - 2):
#                 x = self.upsampling[level](x)
#                 # print("Before concatenation（デコーダーの中）:", x.shape, x_history[-1].shape)  # 追加
#                 x = torch.cat([x, x_history.pop()], dim=1)
#                 # print("After concatenation（デコーダーの中）:", x.shape)  # 追加

#         # remaining convs at full resolution
#         for conv in self.remaining:
#             x = conv(x)

#         return x


# class VxmDense(LoadableModel):
#     """
#     VoxelMorph network for (unsupervised) nonlinear registration between two images.
#     """

#     @store_config_args
#     def __init__(self,
#                  inshape,
#                  nb_unet_features=None,
#                  nb_unet_levels=None,
#                  unet_feat_mult=1,
#                  nb_unet_conv_per_level=1,
#                  int_steps=7,
#                  int_downsize=2,
#                  bidir=False,
#                  use_probs=False,
#                  src_feats=1,
#                  trg_feats=1,
#                  unet_half_res=False):
#         """ 
#         Parameters:
#             inshape: Input shape. e.g. (192, 192, 192)
#             nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
#                 the form [[encoder feats], [decoder feats]], or as a single integer. 
#                 If None (default), the unet features are defined by the default config described in 
#                 the unet class documentation.
#             nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
#                 Default is None.
#             unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
#                 Default is 1.
#             nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
#             int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
#                 value is 0.
#             int_downsize: Integer specifying the flow downsample factor for vector integration. 
#                 The flow field is not downsampled when this value is 1.
#             bidir: Enable bidirectional cost function. Default is False.
#             use_probs: Use probabilities in flow field. Default is False.
#             src_feats: Number of source image features. Default is 1.
#             trg_feats: Number of target image features. Default is 1.
#             unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
#                 Default is False.
#         """
#         super().__init__()

#         # internal flag indicating whether to return flow or integrated warp during inference
#         self.training = True

#         # ensure correct dimensionality
#         ndims = len(inshape)
#         assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

#         # # configure core unet model
#         # self.unet_model = Unet(
#         #     inshape,
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )

# #------------------------------------------------------------------------------------------------------------------------
    
#         # self.unet_model_128_1 = Unet(
#         #     (128,128,128),
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )     

#         # self.unet_model_128_2 = Unet(
#         #     (128,128,128),
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )  

#         # self.unet_model_128_3 = Unet(
#         #     (128,128,128),
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )  

#         # self.unet_model_128_4 = Unet(
#         #     (128,128,128),
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )                             
 

#         # Conv_1 = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow_128_1 = Conv_1(self.unet_model_128_1.final_nf, ndims, kernel_size=3, padding=1)
#         # self.flow_128_1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128_1.weight.shape))
#         # self.flow_128_1.bias = nn.Parameter(torch.zeros(self.flow_128_1.bias.shape))

#         # Conv_2 = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow_128_2 = Conv_2(self.unet_model_128_2.final_nf, ndims, kernel_size=3, padding=1)
#         # self.flow_128_2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128_2.weight.shape))
#         # self.flow_128_2.bias = nn.Parameter(torch.zeros(self.flow_128_2.bias.shape))

#         # Conv_3 = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow_128_3 = Conv_3(self.unet_model_128_3.final_nf, ndims, kernel_size=3, padding=1)
#         # self.flow_128_3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128_3.weight.shape))
#         # self.flow_128_3.bias = nn.Parameter(torch.zeros(self.flow_128_3.bias.shape))

#         # Conv_4 = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow_128_4 = Conv_4(self.unet_model_128_4.final_nf, ndims, kernel_size=3, padding=1)
#         # self.flow_128_4.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128_4.weight.shape))
#         # self.flow_128_4.bias = nn.Parameter(torch.zeros(self.flow_128_4.bias.shape))  
        
#         self.unet_model_16 = Unet(
#             (16,16,16),
#             infeats=(src_feats + trg_feats),
#             nb_features=nb_unet_features,
#             nb_levels=nb_unet_levels,
#             feat_mult=unet_feat_mult,
#             nb_conv_per_level=nb_unet_conv_per_level,
#             half_res=unet_half_res,
#         )

#         self.unet_model_32 = Unet(
#             (32,32,32),
#             infeats=(src_feats + trg_feats),
#             nb_features=nb_unet_features,
#             nb_levels=nb_unet_levels,
#             feat_mult=unet_feat_mult,
#             nb_conv_per_level=nb_unet_conv_per_level,
#             half_res=unet_half_res,
#         )

#         self.unet_model_64 = Unet(
#             (64,64,64),
#             infeats=(src_feats + trg_feats),
#             nb_features=nb_unet_features,
#             nb_levels=nb_unet_levels,
#             feat_mult=unet_feat_mult,
#             nb_conv_per_level=nb_unet_conv_per_level,
#             half_res=unet_half_res,
#         )

#         self.unet_model_128 = Unet(
#             (128,128,128),
#             infeats=(src_feats + trg_feats),
#             nb_features=nb_unet_features,
#             nb_levels=nb_unet_levels,
#             feat_mult=unet_feat_mult,
#             nb_conv_per_level=nb_unet_conv_per_level,
#             half_res=unet_half_res,
#         )        
 
#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.flow_16 = Conv(self.unet_model_16.final_nf, ndims, kernel_size=3, padding=1)
#         self.flow_16.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_16.weight.shape))
#         self.flow_16.bias = nn.Parameter(torch.zeros(self.flow_16.bias.shape))

#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.flow_32 = Conv(self.unet_model_32.final_nf, ndims, kernel_size=3, padding=1)
#         self.flow_32.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_32.weight.shape))
#         self.flow_32.bias = nn.Parameter(torch.zeros(self.flow_32.bias.shape))

#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.flow_64 = Conv(self.unet_model_64.final_nf, ndims, kernel_size=3, padding=1)
#         self.flow_64.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_64.weight.shape))
#         self.flow_64.bias = nn.Parameter(torch.zeros(self.flow_64.bias.shape))

#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.flow_128 = Conv(self.unet_model_128.final_nf, ndims, kernel_size=3, padding=1)
#         self.flow_128.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128.weight.shape))
#         self.flow_128.bias = nn.Parameter(torch.zeros(self.flow_128.bias.shape))

# #------------------------------------------------------------------------------------------------------------------------

#         # # configure unet to flow field layer
#         # Conv = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

#         # # init flow layer with small weights and bias
#         # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
#         # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))





                
#         # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
#         # self.flow.bias = nn.Parameter(torch.ones(self.flow.bias.shape))

#         # probabilities are not supported in pytorch
#         if use_probs:
#             raise NotImplementedError(
#                 'Flow variance has not been implemented in pytorch - set use_probs to False')

#         # configure optional resize layers (downsize)
#         if not unet_half_res and int_steps > 0 and int_downsize > 1:
#             self.resize = layers.ResizeTransform(int_downsize, ndims)
#         else:
#             self.resize = None

#         # resize to full res
#         if int_steps > 0 and int_downsize > 1:
#             self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
#         else:
#             self.fullsize = None

#         # configure bidirectional training
#         self.bidir = bidir

#         # configure optional integration layer for diffeomorphic warp
#         down_shape = [int(dim / int_downsize) for dim in inshape]
#         print(down_shape)
#         self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
#         # print("insahpeのサイズ",inshape)

#         # configure transformer

#         # new_inshape = inshape.squeeze(dim=2)
#         self.transformer = layers.SpatialTransformer(inshape)

# #------------------------------------------------------------------------------------------------------------------------

#         self.transformer_16 = layers.SpatialTransformer((16,16,16))
#         self.transformer_32 = layers.SpatialTransformer((32,32,32))
#         self.transformer_64 = layers.SpatialTransformer((64,64,64))
#         self.transformer_128 = layers.SpatialTransformer((128,128,128))
       
# #------------------------------------------------------------------------------------------------------------------------


#     def forward(self, source, target, registration=False):
#         '''
#         Parameters:
#             source: Source image tensor.
#             target: Target image tensor.
#             registration: Return transformed image and flow. Default is False.
#         '''

#         # # concatenate inputs and propagate unet
#         # # print("（読み込んだ画像サイズ）:", source.shape, target.shape)
#         # x = torch.cat([source, target], dim=1)
#         # # x = x.unsqueeze(-1)
#         # print("（Catした画像サイズ）:", x.shape)
#         # x = self.unet_model(x)
#         # print("aaaa2222", x.max())
#         # print("（Xサイズ）:", x.shape)

#         # # transform into flow field
#         # flow_field = self.flow(x)
#         # print("aaaa3333", flow_field.max())
#         # print("（flow_fieldサイズ）:", flow_field.shape)

 
# #----------------------------------------------------------おうよう------------------------------------------------------------

#         x = torch.cat([source, target], dim=1)

#         source_16 = torch.nn.functional.interpolate(source, size=(16,16,16), mode='trilinear', align_corners=False)
#         target_16 = torch.nn.functional.interpolate(target, size=(16,16,16), mode='trilinear', align_corners=False)
#         source_32 = torch.nn.functional.interpolate(source, size=(32,32,32), mode='trilinear', align_corners=False)
#         target_32 = torch.nn.functional.interpolate(target, size=(32,32,32), mode='trilinear', align_corners=False)
#         source_64 = torch.nn.functional.interpolate(source, size=(64,64,64), mode='trilinear', align_corners=False)
#         target_64 = torch.nn.functional.interpolate(target, size=(64,64,64), mode='trilinear', align_corners=False)
#         source_128 = source
#         target_128 = target


#         source_16_128 = torch.nn.functional.interpolate(source_16, size=(128,128,128), mode='trilinear', align_corners=False)
#         target_16_128 = torch.nn.functional.interpolate(target_16, size=(128,128,128), mode='trilinear', align_corners=False)
#         x_16 = torch.cat([source_16_128, target_16_128], dim=1)
#         x_16 = self.unet_model_128_1(x_16)
#         flow_field_16 = self.flow_128_1(x_16)
#         unet_model_128_1_state_dict = self.unet_model_128_1.state_dict()
#         flow_128_1_state_dict = self.flow_128_1.state_dict()


#         self.unet_model_128_2.load_state_dict(unet_model_128_1_state_dict)
#         self.flow_128_2.load_state_dict(flow_128_1_state_dict)
#         source_32_128 = torch.nn.functional.interpolate(source_32, size=(128,128,128), mode='trilinear', align_corners=False)
#         target_32_128 = torch.nn.functional.interpolate(target_32, size=(128,128,128), mode='trilinear', align_corners=False)
#         x_32 = torch.cat([source_32_128, target_32_128], dim=1)        
#         x_32 = self.unet_model_128_2(x_32)
#         flow_field_32 = self.flow_128_2(x_32)
#         unet_model_128_2_state_dict = self.unet_model_128_2.state_dict()
#         flow_128_2_state_dict = self.flow_128_2.state_dict()

#         self.unet_model_128_3.load_state_dict(unet_model_128_2_state_dict)
#         self.flow_128_3.load_state_dict(flow_128_2_state_dict)
#         source_64_128 = torch.nn.functional.interpolate(source_64, size=(128,128,128), mode='trilinear', align_corners=False)
#         target_64_128 = torch.nn.functional.interpolate(target_64, size=(128,128,128), mode='trilinear', align_corners=False)
#         x_64 = torch.cat([source_64_128, target_64_128], dim=1)        
#         x_64 = self.unet_model_128_3(x_64)
#         flow_field_64 = self.flow_128_3(x_64)
#         unet_model_128_3_state_dict = self.unet_model_128_3.state_dict()
#         flow_128_3_state_dict = self.flow_128_3.state_dict()

#         self.unet_model_128_4.load_state_dict(unet_model_128_3_state_dict)
#         self.flow_128_4.load_state_dict(flow_128_3_state_dict)
#         x_128 = torch.cat([source_128, target_128], dim=1)        
#         x_128 = self.unet_model_128_4(x_128)
#         flow_field_128 = self.flow_128_4(x_128)

#         flow_field = flow_field_128

      
# #------------------------------------------------------------------------------------------------------------------------


# #-------------------------------------------------Kihon----------------------------------------------------------------------

#         # resize flow for integration
#         pos_flow = flow_field
#         if self.resize:
#             print("A1")
#             pos_flow = self.resize(pos_flow)
#         preint_flow = pos_flow

#         # pos_flowは変形ベクトル

#         # negate flow for bidirectional model
#         neg_flow = -pos_flow if self.bidir else None

#         # integrate to produce diffeomorphic warp
#         if self.integrate:
#             print("A2")
#             pos_flow = self.integrate(pos_flow)
#             neg_flow = self.integrate(neg_flow) if self.bidir else None

#             # resize to final resolution
#             if self.fullsize:
#                 print("A3")
#                 pos_flow = self.fullsize(pos_flow)
#                 neg_flow = self.fullsize(neg_flow) if self.bidir else None

#         # warp image with flow field
#         # preint_flow = torch.ones_like(pos_flow)*(50)
#         # print(pos_flow.mean())




#         y_source = self.transformer(source, pos_flow)




# #--------------------------------------------kihonn--------------------------------------------------------------------------

#         moved_16_2 = self.transformer_128(source_16_128, flow_field_16)
#         moved_32_2 = self.transformer_128(source_32_128, flow_field_32)
#         moved_64_2 = self.transformer_128(source_64_128, flow_field_64)
#         moved_128_2 = self.transformer_128(source_128, pos_flow)

# #------------------------------------------------------------------------------------------------------------------------



#         # y_source = self.transformer(source, preint_flow)
#         y_target = self.transformer(target, neg_flow) if self.bidir else None

#         # return non-integrated flow field if training
#         if not registration:
#             print("aaaa", preint_flow.max())
#             return (y_source, y_target, preint_flow) if self.bidir else (source_16, target_16, source_16_128, target_16_128, source_32, target_32, source_32_128, target_32_128, \
#                                                                          source_64, target_64, source_64_128, target_64_128, source_128, target_128, moved_16_2, moved_32_2, moved_64_2, moved_128_2
#                                                                          )

        
#         else:
#             pos_flow = torch.ones_like(pos_flow)*10
#             print("bbbb", pos_flow.mean())
#             return y_source, pos_flow

        

# # class ConvBlock(nn.Module):
# #     """
# #     Specific convolutional block followed by leakyrelu for unet.
# #     """

# #     def __init__(self, ndims, in_channels, out_channels, stride=1):
# #         super().__init__()

# #         Conv = getattr(nn, 'Conv%dd' % ndims)
# #         self.main = Conv(in_channels, out_channels, 3, stride, 1)
# #         self.activation = nn.LeakyReLU(0.2)

# #     def forward(self, x):
# #         out = self.main(x)
# #         out = self.activation(out)
# #         return out

# class ConvBlock(nn.Module):
#     """
#     Specific convolutional block followed by leakyrelu for unet.
#     """

#     def __init__(self, ndims, in_channels, out_channels, stride=1):
#         super().__init__()

#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.main = Conv(in_channels, out_channels, 3, stride, 1)
#         self.activation = nn.LeakyReLU(0.2)

#     def forward(self, x):
#         # Conv3dが呼ばれる前にxの型を確認するデバッグ用コード
#         # print("xの型（ConvBlockの中）:", type(x))
#         # print("（ConvBlockの中）:", x.shape)
        
#         out = self.main(x)
#         out = self.activation(out)
#         # print("aaaa1111", out.max())
#         # print("（ConvBlockの中）:", out.shape)
#         return out



# oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

    

# # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo







# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions.normal import Normal

# from .. import default_unet_features
# from . import layers
# from .modelio import LoadableModel, store_config_args


# class Unet(nn.Module):
#     """
#     A unet architecture. Layer features can be specified directly as a list of encoder and decoder
#     features or as a single integer along with a number of unet levels. The default network features
#     per layer (when no options are specified) are:

#         encoder: [16, 32, 32, 32]
#         decoder: [32, 32, 32, 32, 32, 16, 16]
#     """
        
#     def __init__(self,
#                  inshape=None,
#                  infeats=None,
#                  nb_features=None,
#                  nb_levels=None,
#                  max_pool=2,
#                  feat_mult=1,
#                  nb_conv_per_level=1,
#                  half_res=False):
        
#         """
#         Parameters:
#             inshape: Input shape. e.g. (192, 192, 192)
#             infeats: Number of input features.
#             nb_features: Unet convolutional features. Can be specified via a list of lists with
#                 the form [[encoder feats], [decoder feats]], or as a single integer. 
#                 If None (default), the unet features are defined by the default config described in 
#                 the class documentation.
#             nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
#                 Default is None.
#             feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
#                 Default is 1.
#             nb_conv_per_level: Number of convolutions per unet level. Default is 1.
#             half_res: Skip the last decoder upsampling. Default is False.
#         """
#         super().__init__()

#         # ensure correct dimensionality
#         ndims = len(inshape)
#         assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

#         # cache some parameters
#         self.half_res = half_res

#         # default encoder and decoder layer features if nothing provided
#         if nb_features is None:
#             nb_features = default_unet_features()
#             print("1")

#         # build feature list automatically
#         if isinstance(nb_features, int):
#             if nb_levels is None:
#                 print("2")
#                 raise ValueError('must provide unet nb_levels if nb_features is an integer')
#             feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
#             print("3")
#             nb_features = [
#                 np.repeat(feats[:-1], nb_conv_per_level),
#                 np.repeat(np.flip(feats), nb_conv_per_level)
#             ]
#         elif nb_levels is not None:
#             print("4")
#             raise ValueError('cannot use nb_levels if nb_features is not an integer')

#         # extract any surplus (full resolution) decoder convolutions
#         enc_nf, dec_nf = nb_features
#         nb_dec_convs = len(enc_nf)
#         final_convs = dec_nf[nb_dec_convs:]
#         dec_nf = dec_nf[:nb_dec_convs]
#         self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

#         if isinstance(max_pool, int):
#             print("5")
#             max_pool = [max_pool] * self.nb_levels

#         # cache downsampling / upsampling operations
#         MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
#         self.pooling = [MaxPooling(s) for s in max_pool]
#         self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

#         # configure encoder (down-sampling path)
#         prev_nf = infeats
#         encoder_nfs = [prev_nf]
#         self.encoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             print("6")
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 print("7")
#                 nf = enc_nf[level * nb_conv_per_level + conv]
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#             self.encoder.append(convs)
#             encoder_nfs.append(prev_nf)

#         # configure decoder (up-sampling path)
#         encoder_nfs = np.flip(encoder_nfs)
#         self.decoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             print("8")
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 print("9")
#                 nf = dec_nf[level * nb_conv_per_level + conv]
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#             self.decoder.append(convs)
#             if not half_res or level < (self.nb_levels - 2):
#                 print("10")
#                 prev_nf += encoder_nfs[level]

#         # now we take care of any remaining convolutions
#         self.remaining = nn.ModuleList()
#         for num, nf in enumerate(final_convs):
#             print("11")
#             self.remaining.append(ConvBlock(ndims, prev_nf, nf))
#             prev_nf = nf

#         # cache final number of features
#         self.final_nf = prev_nf

#     def forward(self, x):

#         # encoder forward pass
#         x_history = [x]
#         for level, convs in enumerate(self.encoder):
#             for conv in convs:
#                 x = conv(x)
#             x_history.append(x)
#             # print("After encoder level（エンコーダの中）", level, ":", x.shape)  # 追加
#             x = self.pooling[level](x)

#         # decoder forward pass with upsampling and concatenation
#         for level, convs in enumerate(self.decoder):
#             for conv in convs:
#                 x = conv(x)
#             if not self.half_res or level < (self.nb_levels - 2):
#                 x = self.upsampling[level](x)
#                 # print("Before concatenation（デコーダーの中）:", x.shape, x_history[-1].shape)  # 追加
#                 x = torch.cat([x, x_history.pop()], dim=1)
#                 # print("After concatenation（デコーダーの中）:", x.shape)  # 追加

#         # remaining convs at full resolution
#         for conv in self.remaining:
#             x = conv(x)

#         return x


# class VxmDense(LoadableModel):
#     """
#     VoxelMorph network for (unsupervised) nonlinear registration between two images.
#     """

#     @store_config_args
#     def __init__(self,
#                  inshape,
#                  nb_unet_features=None,
#                  nb_unet_levels=None,
#                  unet_feat_mult=1,
#                  nb_unet_conv_per_level=1,
#                  int_steps=7,
#                  int_downsize=2,
#                  bidir=False,
#                  use_probs=False,
#                  src_feats=1,
#                  trg_feats=1,
#                  unet_half_res=False):
#         """ 
#         Parameters:
#             inshape: Input shape. e.g. (192, 192, 192)
#             nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
#                 the form [[encoder feats], [decoder feats]], or as a single integer. 
#                 If None (default), the unet features are defined by the default config described in 
#                 the unet class documentation.
#             nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
#                 Default is None.
#             unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
#                 Default is 1.
#             nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
#             int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
#                 value is 0.
#             int_downsize: Integer specifying the flow downsample factor for vector integration. 
#                 The flow field is not downsampled when this value is 1.
#             bidir: Enable bidirectional cost function. Default is False.
#             use_probs: Use probabilities in flow field. Default is False.
#             src_feats: Number of source image features. Default is 1.
#             trg_feats: Number of target image features. Default is 1.
#             unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
#                 Default is False.
#         """
#         super().__init__()

#         # internal flag indicating whether to return flow or integrated warp during inference
#         self.training = True

#         # ensure correct dimensionality
#         ndims = len(inshape)
#         assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

#         # # configure core unet model
#         # self.unet_model = Unet(
#         #     inshape,
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )

# #------------------------------------------------------------------------------------------------------------------------
    
#         # self.unet_model_128_1 = Unet(
#         #     (128,128,128),
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )     

#         # self.unet_model_128_2 = Unet(
#         #     (128,128,128),
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )  

#         # self.unet_model_128_3 = Unet(
#         #     (128,128,128),
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )  

#         # self.unet_model_128_4 = Unet(
#         #     (128,128,128),
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )                             
 

#         # Conv_1 = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow_128_1 = Conv_1(self.unet_model_128_1.final_nf, ndims, kernel_size=3, padding=1)
#         # self.flow_128_1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128_1.weight.shape))
#         # self.flow_128_1.bias = nn.Parameter(torch.zeros(self.flow_128_1.bias.shape))

#         # Conv_2 = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow_128_2 = Conv_2(self.unet_model_128_2.final_nf, ndims, kernel_size=3, padding=1)
#         # self.flow_128_2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128_2.weight.shape))
#         # self.flow_128_2.bias = nn.Parameter(torch.zeros(self.flow_128_2.bias.shape))

#         # Conv_3 = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow_128_3 = Conv_3(self.unet_model_128_3.final_nf, ndims, kernel_size=3, padding=1)
#         # self.flow_128_3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128_3.weight.shape))
#         # self.flow_128_3.bias = nn.Parameter(torch.zeros(self.flow_128_3.bias.shape))

#         # Conv_4 = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow_128_4 = Conv_4(self.unet_model_128_4.final_nf, ndims, kernel_size=3, padding=1)
#         # self.flow_128_4.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128_4.weight.shape))
#         # self.flow_128_4.bias = nn.Parameter(torch.zeros(self.flow_128_4.bias.shape))  
        
#         self.unet_model_16 = Unet(
#             (16,16,16),
#             infeats=(src_feats + trg_feats),
#             nb_features=nb_unet_features,
#             nb_levels=nb_unet_levels,
#             feat_mult=unet_feat_mult,
#             nb_conv_per_level=nb_unet_conv_per_level,
#             half_res=unet_half_res,
#         )

#         self.unet_model_32 = Unet(
#             (32,32,32),
#             infeats=(src_feats + trg_feats),
#             nb_features=nb_unet_features,
#             nb_levels=nb_unet_levels,
#             feat_mult=unet_feat_mult,
#             nb_conv_per_level=nb_unet_conv_per_level,
#             half_res=unet_half_res,
#         )

#         self.unet_model_64 = Unet(
#             (64,64,64),
#             infeats=(src_feats + trg_feats),
#             nb_features=nb_unet_features,
#             nb_levels=nb_unet_levels,
#             feat_mult=unet_feat_mult,
#             nb_conv_per_level=nb_unet_conv_per_level,
#             half_res=unet_half_res,
#         )

#         self.unet_model_128 = Unet(
#             (128,128,128),
#             infeats=(src_feats + trg_feats),
#             nb_features=nb_unet_features,
#             nb_levels=nb_unet_levels,
#             feat_mult=unet_feat_mult,
#             nb_conv_per_level=nb_unet_conv_per_level,
#             half_res=unet_half_res,
#         )        
 
#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.flow_16 = Conv(self.unet_model_16.final_nf, ndims, kernel_size=3, padding=1)
#         self.flow_16.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_16.weight.shape))
#         self.flow_16.bias = nn.Parameter(torch.zeros(self.flow_16.bias.shape))

#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.flow_32 = Conv(self.unet_model_32.final_nf, ndims, kernel_size=3, padding=1)
#         self.flow_32.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_32.weight.shape))
#         self.flow_32.bias = nn.Parameter(torch.zeros(self.flow_32.bias.shape))

#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.flow_64 = Conv(self.unet_model_64.final_nf, ndims, kernel_size=3, padding=1)
#         self.flow_64.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_64.weight.shape))
#         self.flow_64.bias = nn.Parameter(torch.zeros(self.flow_64.bias.shape))

#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.flow_128 = Conv(self.unet_model_128.final_nf, ndims, kernel_size=3, padding=1)
#         self.flow_128.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128.weight.shape))
#         self.flow_128.bias = nn.Parameter(torch.zeros(self.flow_128.bias.shape))

# #------------------------------------------------------------------------------------------------------------------------

#         # # configure unet to flow field layer
#         # Conv = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

#         # # init flow layer with small weights and bias
#         # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
#         # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))





                
#         # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
#         # self.flow.bias = nn.Parameter(torch.ones(self.flow.bias.shape))

#         # probabilities are not supported in pytorch
#         if use_probs:
#             raise NotImplementedError(
#                 'Flow variance has not been implemented in pytorch - set use_probs to False')

#         # configure optional resize layers (downsize)
#         if not unet_half_res and int_steps > 0 and int_downsize > 1:
#             self.resize = layers.ResizeTransform(int_downsize, ndims)
#         else:
#             self.resize = None

#         # resize to full res
#         if int_steps > 0 and int_downsize > 1:
#             self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
#         else:
#             self.fullsize = None

#         # configure bidirectional training
#         self.bidir = bidir

#         # configure optional integration layer for diffeomorphic warp
#         down_shape = [int(dim / int_downsize) for dim in inshape]
#         print(down_shape)
#         self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
#         # print("insahpeのサイズ",inshape)

#         # configure transformer

#         # new_inshape = inshape.squeeze(dim=2)
#         self.transformer = layers.SpatialTransformer(inshape)

# #------------------------------------------------------------------------------------------------------------------------

#         self.transformer_16 = layers.SpatialTransformer((16,16,16))
#         self.transformer_32 = layers.SpatialTransformer((32,32,32))
#         self.transformer_64 = layers.SpatialTransformer((64,64,64))
#         self.transformer_128 = layers.SpatialTransformer((128,128,128))
       
# #------------------------------------------------------------------------------------------------------------------------


#     def forward(self, source, target, registration=False):
#         '''
#         Parameters:
#             source: Source image tensor.
#             target: Target image tensor.
#             registration: Return transformed image and flow. Default is False.
#         '''

#         # # concatenate inputs and propagate unet
#         # # print("（読み込んだ画像サイズ）:", source.shape, target.shape)
#         # x = torch.cat([source, target], dim=1)
#         # # x = x.unsqueeze(-1)
#         # print("（Catした画像サイズ）:", x.shape)
#         # x = self.unet_model(x)
#         # print("aaaa2222", x.max())
#         # print("（Xサイズ）:", x.shape)

#         # # transform into flow field
#         # flow_field = self.flow(x)
#         # print("aaaa3333", flow_field.max())
#         # print("（flow_fieldサイズ）:", flow_field.shape)

 
# #----------------------------------------------------------おうよう------------------------------------------------------------

#         x = torch.cat([source, target], dim=1)

#         source_16 = torch.nn.functional.interpolate(source, size=(16,16,16), mode='trilinear', align_corners=False)
#         target_16 = torch.nn.functional.interpolate(target, size=(16,16,16), mode='trilinear', align_corners=False)
#         source_32 = torch.nn.functional.interpolate(source, size=(32,32,32), mode='trilinear', align_corners=False)
#         target_32 = torch.nn.functional.interpolate(target, size=(32,32,32), mode='trilinear', align_corners=False)
#         source_64 = torch.nn.functional.interpolate(source, size=(64,64,64), mode='trilinear', align_corners=False)
#         target_64 = torch.nn.functional.interpolate(target, size=(64,64,64), mode='trilinear', align_corners=False)
#         source_128 = source
#         target_128 = target


#         x_16 = torch.cat([source_16, target_16], dim=1)
#         x_16 = self.unet_model_16(x_16)
#         flow_field_16 = self.flow_16(x_16)

#         copy_weights(self.unet_model_16, self.unet_model_32)
 
#         x_32 = torch.cat([source_32, target_32], dim=1)        
#         x_32 = self.unet_model_32(x_32)
#         flow_field_32 = self.flow_32(x_32)

#         copy_weights(self.unet_model_32, self.unet_model_64)

#         x_64 = torch.cat([source_64, target_64], dim=1)        
#         x_64 = self.unet_model_64(x_64)
#         flow_field_64 = self.flow_64(x_64)

#         copy_weights(self.unet_model_64, self.unet_model_128)

#         x_128 = torch.cat([source_128, target_128], dim=1)        
#         x_128 = self.unet_model_128(x_128)
#         flow_field_128 = self.flow_128(x_128)

#         flow_field_16_32_resize = torch.nn.functional.interpolate(flow_field_16, size=(32,32,32), mode='trilinear', align_corners=False)
#         flow_field_16_32_resize = flow_field_16_32_resize*2
#         flow_field_32_64_resize = torch.nn.functional.interpolate(flow_field_32, size=(64,64,64), mode='trilinear', align_corners=False)
#         flow_field_32_64_resize = flow_field_32_64_resize*2       
#         flow_field_64_128_resize = torch.nn.functional.interpolate(flow_field_64, size=(128,128,128), mode='trilinear', align_corners=False)
#         flow_field_64_128_resize = flow_field_64_128_resize*2
#         flow_field = flow_field_128

      
# #------------------------------------------------------------------------------------------------------------------------


# #-------------------------------------------------Kihon----------------------------------------------------------------------

#         # resize flow for integration
#         pos_flow = flow_field
#         if self.resize:
#             print("A1")
#             pos_flow = self.resize(pos_flow)
#         preint_flow = pos_flow

#         # pos_flowは変形ベクトル

#         # negate flow for bidirectional model
#         neg_flow = -pos_flow if self.bidir else None

#         # integrate to produce diffeomorphic warp
#         if self.integrate:
#             print("A2")
#             pos_flow = self.integrate(pos_flow)
#             neg_flow = self.integrate(neg_flow) if self.bidir else None

#             # resize to final resolution
#             if self.fullsize:
#                 print("A3")
#                 pos_flow = self.fullsize(pos_flow)
#                 neg_flow = self.fullsize(neg_flow) if self.bidir else None

#         # warp image with flow field
#         # preint_flow = torch.ones_like(pos_flow)*(50)
#         # print(pos_flow.mean())




#         y_source = self.transformer(source, pos_flow)




# #--------------------------------------------kihonn--------------------------------------------------------------------------

#         moved_16_2 = self.transformer_16(source_16, flow_field_16)
#         moved_32_2 = self.transformer_32(source_32, flow_field_32)
#         moved_64_2 = self.transformer_64(source_64, flow_field_64)
#         moved_128_2 = self.transformer_128(source_128, pos_flow)

# #------------------------------------------------------------------------------------------------------------------------



#         # y_source = self.transformer(source, preint_flow)
#         y_target = self.transformer(target, neg_flow) if self.bidir else None

#         # return non-integrated flow field if training
#         if not registration:
#             # print("aaaa", preint_flow.max())
#             return (y_source, y_target, preint_flow) if self.bidir else (source_16, target_16, source_32, target_32, source_64, target_64, source_128, target_128, \
#                                                                          moved_16_2, moved_32_2, moved_64_2, moved_128_2, flow_field_16_32_resize, flow_field_32_64_resize,\
#                                                                         flow_field_64_128_resize, flow_field, flow_field_32, flow_field_64
#                                                                          )

        
#         else:
#             pos_flow = torch.ones_like(pos_flow)*10
#             print("bbbb", pos_flow.mean())
#             return y_source, pos_flow

        

# # class ConvBlock(nn.Module):
# #     """
# #     Specific convolutional block followed by leakyrelu for unet.
# #     """

# #     def __init__(self, ndims, in_channels, out_channels, stride=1):
# #         super().__init__()

# #         Conv = getattr(nn, 'Conv%dd' % ndims)
# #         self.main = Conv(in_channels, out_channels, 3, stride, 1)
# #         self.activation = nn.LeakyReLU(0.2)

# #     def forward(self, x):
# #         out = self.main(x)
# #         out = self.activation(out)
# #         return out

# class ConvBlock(nn.Module):
#     """
#     Specific convolutional block followed by leakyrelu for unet.
#     """

#     def __init__(self, ndims, in_channels, out_channels, stride=1):
#         super().__init__()

#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.main = Conv(in_channels, out_channels, 3, stride, 1)
#         self.activation = nn.LeakyReLU(0.2)

#     def forward(self, x):
#         # Conv3dが呼ばれる前にxの型を確認するデバッグ用コード
#         # print("xの型（ConvBlockの中）:", type(x))
#         # print("（ConvBlockの中）:", x.shape)
        
#         out = self.main(x)
#         out = self.activation(out)
#         # print("aaaa1111", out.max())
#         # print("（ConvBlockの中）:", out.shape)
#         return out
    
    
# def copy_weights(src_model, dst_model):
#     src_state_dict = src_model.state_dict()
#     dst_state_dict = dst_model.state_dict()

#     # src_modelの重みがdst_modelに含まれていることを確認する
#     for key in src_state_dict:
#         if key in dst_state_dict:
#             dst_state_dict[key] = src_state_dict[key]

#     dst_model.load_state_dict(dst_state_dict)  


    # # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo







# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions.normal import Normal

# from .. import default_unet_features
# from . import layers
# from .modelio import LoadableModel, store_config_args


# class Unet(nn.Module):
#     """
#     A unet architecture. Layer features can be specified directly as a list of encoder and decoder
#     features or as a single integer along with a number of unet levels. The default network features
#     per layer (when no options are specified) are:

#         encoder: [16, 32, 32, 32]
#         decoder: [32, 32, 32, 32, 32, 16, 16]
#     """
        
#     def __init__(self,
#                  inshape=None,
#                  infeats=None,
#                  nb_features=None,
#                  nb_levels=None,
#                  max_pool=2,
#                  feat_mult=1,
#                  nb_conv_per_level=1,
#                  half_res=False):
        
#         """
#         Parameters:
#             inshape: Input shape. e.g. (192, 192, 192)
#             infeats: Number of input features.
#             nb_features: Unet convolutional features. Can be specified via a list of lists with
#                 the form [[encoder feats], [decoder feats]], or as a single integer. 
#                 If None (default), the unet features are defined by the default config described in 
#                 the class documentation.
#             nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
#                 Default is None.
#             feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
#                 Default is 1.
#             nb_conv_per_level: Number of convolutions per unet level. Default is 1.
#             half_res: Skip the last decoder upsampling. Default is False.
#         """
#         super().__init__()

#         # ensure correct dimensionality
#         ndims = len(inshape)
#         assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

#         # cache some parameters
#         self.half_res = half_res

#         # default encoder and decoder layer features if nothing provided
#         if nb_features is None:
#             nb_features = default_unet_features()
#             print("1")

#         # build feature list automatically
#         if isinstance(nb_features, int):
#             if nb_levels is None:
#                 print("2")
#                 raise ValueError('must provide unet nb_levels if nb_features is an integer')
#             feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
#             print("3")
#             nb_features = [
#                 np.repeat(feats[:-1], nb_conv_per_level),
#                 np.repeat(np.flip(feats), nb_conv_per_level)
#             ]
#         elif nb_levels is not None:
#             print("4")
#             raise ValueError('cannot use nb_levels if nb_features is not an integer')

#         # extract any surplus (full resolution) decoder convolutions
#         enc_nf, dec_nf = nb_features
#         nb_dec_convs = len(enc_nf)
#         final_convs = dec_nf[nb_dec_convs:]
#         dec_nf = dec_nf[:nb_dec_convs]
#         self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

#         if isinstance(max_pool, int):
#             print("5")
#             max_pool = [max_pool] * self.nb_levels

#         # cache downsampling / upsampling operations
#         MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
#         self.pooling = [MaxPooling(s) for s in max_pool]
#         self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

#         # configure encoder (down-sampling path)
#         prev_nf = infeats
#         encoder_nfs = [prev_nf]
#         self.encoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             print("6")
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 print("7")
#                 nf = enc_nf[level * nb_conv_per_level + conv]
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#             self.encoder.append(convs)
#             encoder_nfs.append(prev_nf)

#         # configure decoder (up-sampling path)
#         encoder_nfs = np.flip(encoder_nfs)
#         self.decoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             print("8")
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 print("9")
#                 nf = dec_nf[level * nb_conv_per_level + conv]
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#             self.decoder.append(convs)
#             if not half_res or level < (self.nb_levels - 2):
#                 print("10")
#                 prev_nf += encoder_nfs[level]

#         # now we take care of any remaining convolutions
#         self.remaining = nn.ModuleList()
#         for num, nf in enumerate(final_convs):
#             print("11")
#             self.remaining.append(ConvBlock(ndims, prev_nf, nf))
#             prev_nf = nf

#         # cache final number of features
#         self.final_nf = prev_nf

#     def forward(self, x):

#         # encoder forward pass
#         x_history = [x]
#         for level, convs in enumerate(self.encoder):
#             for conv in convs:
#                 x = conv(x)
#             x_history.append(x)
#             # print("After encoder level（エンコーダの中）", level, ":", x.shape)  # 追加
#             x = self.pooling[level](x)

#         # decoder forward pass with upsampling and concatenation
#         for level, convs in enumerate(self.decoder):
#             for conv in convs:
#                 x = conv(x)
#             if not self.half_res or level < (self.nb_levels - 2):
#                 x = self.upsampling[level](x)
#                 # print("Before concatenation（デコーダーの中）:", x.shape, x_history[-1].shape)  # 追加
#                 x = torch.cat([x, x_history.pop()], dim=1)
#                 # print("After concatenation（デコーダーの中）:", x.shape)  # 追加

#         # remaining convs at full resolution
#         for conv in self.remaining:
#             x = conv(x)

#         return x


# class VxmDense(LoadableModel):
#     """
#     VoxelMorph network for (unsupervised) nonlinear registration between two images.
#     """

#     @store_config_args
#     def __init__(self,
#                  inshape,
#                  nb_unet_features=None,
#                  nb_unet_levels=None,
#                  unet_feat_mult=1,
#                  nb_unet_conv_per_level=1,
#                  int_steps=7,
#                  int_downsize=2,
#                  bidir=False,
#                  use_probs=False,
#                  src_feats=1,
#                  trg_feats=1,
#                  unet_half_res=False):
#         """ 
#         Parameters:
#             inshape: Input shape. e.g. (192, 192, 192)
#             nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
#                 the form [[encoder feats], [decoder feats]], or as a single integer. 
#                 If None (default), the unet features are defined by the default config described in 
#                 the unet class documentation.
#             nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
#                 Default is None.
#             unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
#                 Default is 1.
#             nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
#             int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
#                 value is 0.
#             int_downsize: Integer specifying the flow downsample factor for vector integration. 
#                 The flow field is not downsampled when this value is 1.
#             bidir: Enable bidirectional cost function. Default is False.
#             use_probs: Use probabilities in flow field. Default is False.
#             src_feats: Number of source image features. Default is 1.
#             trg_feats: Number of target image features. Default is 1.
#             unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
#                 Default is False.
#         """
#         super().__init__()

#         # internal flag indicating whether to return flow or integrated warp during inference
#         self.training = True

#         # ensure correct dimensionality
#         ndims = len(inshape)
#         assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

#         # # configure core unet model
#         # self.unet_model = Unet(
#         #     inshape,
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )

# #------------------------------------------------------------------------------------------------------------------------
    
#         # self.unet_model_128_1 = Unet(
#         #     (128,128,128),
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )     

#         # self.unet_model_128_2 = Unet(
#         #     (128,128,128),
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )  

#         # self.unet_model_128_3 = Unet(
#         #     (128,128,128),
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )  

#         # self.unet_model_128_4 = Unet(
#         #     (128,128,128),
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )                             
 

#         # Conv_1 = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow_128_1 = Conv_1(self.unet_model_128_1.final_nf, ndims, kernel_size=3, padding=1)
#         # self.flow_128_1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128_1.weight.shape))
#         # self.flow_128_1.bias = nn.Parameter(torch.zeros(self.flow_128_1.bias.shape))

#         # Conv_2 = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow_128_2 = Conv_2(self.unet_model_128_2.final_nf, ndims, kernel_size=3, padding=1)
#         # self.flow_128_2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128_2.weight.shape))
#         # self.flow_128_2.bias = nn.Parameter(torch.zeros(self.flow_128_2.bias.shape))

#         # Conv_3 = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow_128_3 = Conv_3(self.unet_model_128_3.final_nf, ndims, kernel_size=3, padding=1)
#         # self.flow_128_3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128_3.weight.shape))
#         # self.flow_128_3.bias = nn.Parameter(torch.zeros(self.flow_128_3.bias.shape))

#         # Conv_4 = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow_128_4 = Conv_4(self.unet_model_128_4.final_nf, ndims, kernel_size=3, padding=1)
#         # self.flow_128_4.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128_4.weight.shape))
#         # self.flow_128_4.bias = nn.Parameter(torch.zeros(self.flow_128_4.bias.shape))  
        

#         self.unet_model_128_1 = Unet(
#             (128,128,128),
#             infeats=(src_feats + trg_feats),
#             nb_features=nb_unet_features,
#             nb_levels=nb_unet_levels,
#             feat_mult=unet_feat_mult,
#             nb_conv_per_level=nb_unet_conv_per_level,
#             half_res=unet_half_res,
#         )        
 

#         self.unet_model_128_2 = Unet(
#             (128,128,128),
#             infeats=(src_feats + trg_feats),
#             nb_features=nb_unet_features,
#             nb_levels=nb_unet_levels,
#             feat_mult=unet_feat_mult,
#             nb_conv_per_level=nb_unet_conv_per_level,
#             half_res=unet_half_res,
#         )     



#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.flow_128_1 = Conv(self.unet_model_128_1.final_nf, ndims, kernel_size=3, padding=1)
#         self.flow_128_1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128_1.weight.shape))
#         self.flow_128_1.bias = nn.Parameter(torch.zeros(self.flow_128_1.bias.shape))

#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.flow_128_2 = Conv(self.unet_model_128_2.final_nf, ndims, kernel_size=3, padding=1)
#         self.flow_128_2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_128_2.weight.shape))
#         self.flow_128_2.bias = nn.Parameter(torch.zeros(self.flow_128_2.bias.shape))

# #------------------------------------------------------------------------------------------------------------------------

#         # # configure unet to flow field layer
#         # Conv = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

#         # # init flow layer with small weights and bias
#         # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
#         # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))





                
#         # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
#         # self.flow.bias = nn.Parameter(torch.ones(self.flow.bias.shape))

#         # probabilities are not supported in pytorch
#         if use_probs:
#             raise NotImplementedError(
#                 'Flow variance has not been implemented in pytorch - set use_probs to False')

#         # configure optional resize layers (downsize)
#         if not unet_half_res and int_steps > 0 and int_downsize > 1:
#             self.resize = layers.ResizeTransform(int_downsize, ndims)
#         else:
#             self.resize = None

#         # resize to full res
#         if int_steps > 0 and int_downsize > 1:
#             self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
#         else:
#             self.fullsize = None

#         # configure bidirectional training
#         self.bidir = bidir

#         # configure optional integration layer for diffeomorphic warp
#         down_shape = [int(dim / int_downsize) for dim in inshape]
#         print(down_shape)
#         self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
#         # print("insahpeのサイズ",inshape)

#         # configure transformer

#         # new_inshape = inshape.squeeze(dim=2)
#         self.transformer = layers.SpatialTransformer(inshape)

# #------------------------------------------------------------------------------------------------------------------------

#         self.transformer_128 = layers.SpatialTransformer((128,128,128))
       
# #------------------------------------------------------------------------------------------------------------------------


#     def forward(self, source, target, registration=False):
#         '''
#         Parameters:
#             source: Source image tensor.
#             target: Target image tensor.
#             registration: Return transformed image and flow. Default is False.
#         '''

#         # # concatenate inputs and propagate unet
#         # # print("（読み込んだ画像サイズ）:", source.shape, target.shape)
#         # x = torch.cat([source, target], dim=1)
#         # # x = x.unsqueeze(-1)
#         # print("（Catした画像サイズ）:", x.shape)
#         # x = self.unet_model(x)
#         # print("aaaa2222", x.max())
#         # print("（Xサイズ）:", x.shape)

#         # # transform into flow field
#         # flow_field = self.flow(x)
#         # print("aaaa3333", flow_field.max())
#         # print("（flow_fieldサイズ）:", flow_field.shape)

 
# #----------------------------------------------------------おうよう------------------------------------------------------------

#         x = torch.cat([source, target], dim=1)


#         # source_64 = torch.nn.functional.interpolate(source, size=(64,64,64), mode='trilinear', align_corners=False)
#         # target_64 = torch.nn.functional.interpolate(target, size=(64,64,64), mode='trilinear', align_corners=False)
#         source_128 = source
#         target_128 = target


#         # source_64_128 = torch.nn.functional.interpolate(source_64, size=(128,128,128), mode='trilinear', align_corners=False)
#         # target_64_128 = torch.nn.functional.interpolate(target_64, size=(128,128,128), mode='trilinear', align_corners=False)
#         # x_64 = torch.cat([source_64_128, target_64_128], dim=1)        
#         # x_64 = self.unet_model_128_1(x_64)
#         # flow_field_64 = self.flow_128_1(x_64)
#         # unet_model_128_1_state_dict = self.unet_model_128_1.state_dict()
#         # flow_128_1_state_dict = self.flow_128_1.state_dict()


#         # self.unet_model_128_2.load_state_dict(unet_model_128_1_state_dict)
#         # self.flow_128_2.load_state_dict(flow_128_1_state_dict)
#         x_128 = torch.cat([source_128, target_128], dim=1)        
#         x_128 = self.unet_model_128_2(x_128)
#         flow_field_128 = self.flow_128_2(x_128)

     
#         flow_field_64_256_resize = torch.nn.functional.interpolate(flow_field_128, size=(256,256,256), mode='trilinear', align_corners=False)
#         flow_field_64_256_resize = flow_field_64_256_resize*2
#         flow_field_64_256_128_resize = torch.nn.functional.interpolate(flow_field_64_256_resize, size=(128, 128, 128), mode='nearest')
#         flow_field_64_256_128_resize = flow_field_64_256_128_resize/2

#         flow_field = flow_field_128

      
# #------------------------------------------------------------------------------------------------------------------------


# #-------------------------------------------------Kihon----------------------------------------------------------------------

#         # resize flow for integration
#         pos_flow = flow_field
#         if self.resize:
#             print("A1")
#             pos_flow = self.resize(pos_flow)
#         preint_flow = pos_flow

#         # pos_flowは変形ベクトル

#         # negate flow for bidirectional model
#         neg_flow = -pos_flow if self.bidir else None

#         # integrate to produce diffeomorphic warp
#         if self.integrate:
#             print("A2")
#             pos_flow = self.integrate(pos_flow)
#             neg_flow = self.integrate(neg_flow) if self.bidir else None

#             # resize to final resolution
#             if self.fullsize:
#                 print("A3")
#                 pos_flow = self.fullsize(pos_flow)
#                 neg_flow = self.fullsize(neg_flow) if self.bidir else None

#         # warp image with flow field
#         # preint_flow = torch.ones_like(pos_flow)*(50)
#         # print(pos_flow.mean())




#         y_source = self.transformer(source, pos_flow)




# #--------------------------------------------kihonn--------------------------------------------------------------------------


#         moved_64_2 = self.transformer_128(source_128, flow_field_64_256_128_resize)
#         # moved_128_2 = self.transformer_128(source_128, pos_flow)

# #------------------------------------------------------------------------------------------------------------------------



#         # y_source = self.transformer(source, preint_flow)
#         y_target = self.transformer(target, neg_flow) if self.bidir else None

#         # return non-integrated flow field if training
#         if not registration:
#             # print("aaaa", preint_flow.max())
#             # return (y_source, y_target, preint_flow) if self.bidir else ( source_64, target_64, source_64_128, target_64_128, source_128, target_128, \
#             #                                                               moved_64_2, moved_128_2, flow_field_64, flow_field_64_256_resize,\
#             #                                                             flow_field_64_256_128_resize, flow_field
#             #                                                              )

#              return (y_source, y_target, preint_flow) if self.bidir else ( source_128, target_128, \
#                                                                           moved_64_2,  flow_field_64_256_resize,\
#                                                                         flow_field_64_256_128_resize, flow_field
#                                                                          )       
#         else:
#             pos_flow = torch.ones_like(pos_flow)*10
#             print("bbbb", pos_flow.mean())
#             return y_source, pos_flow

        

# # class ConvBlock(nn.Module):
# #     """
# #     Specific convolutional block followed by leakyrelu for unet.
# #     """

# #     def __init__(self, ndims, in_channels, out_channels, stride=1):
# #         super().__init__()

# #         Conv = getattr(nn, 'Conv%dd' % ndims)
# #         self.main = Conv(in_channels, out_channels, 3, stride, 1)
# #         self.activation = nn.LeakyReLU(0.2)

# #     def forward(self, x):
# #         out = self.main(x)
# #         out = self.activation(out)
# #         return out

# class ConvBlock(nn.Module):
#     """
#     Specific convolutional block followed by leakyrelu for unet.
#     """

#     def __init__(self, ndims, in_channels, out_channels, stride=1):
#         super().__init__()

#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         self.main = Conv(in_channels, out_channels, 3, stride, 1)
#         self.activation = nn.LeakyReLU(0.2)

#     def forward(self, x):
#         # Conv3dが呼ばれる前にxの型を確認するデバッグ用コード
#         # print("xの型（ConvBlockの中）:", type(x))
#         # print("（ConvBlockの中）:", x.shape)
        
#         out = self.main(x)
#         out = self.activation(out)
#         # print("aaaa1111", out.max())
#         # print("（ConvBlockの中）:", out.shape)
#         return out
    
    
# def copy_weights(src_model, dst_model):
#     src_state_dict = src_model.state_dict()
#     dst_state_dict = dst_model.state_dict()

#     # src_modelの重みがdst_modelに含まれていることを確認する
#     for key in src_state_dict:
#         if key in dst_state_dict:
#             dst_state_dict[key] = src_state_dict[key]

#     dst_model.load_state_dict(dst_state_dict)   














# # # SWINNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions.normal import Normal
# import pywt
# import torchvision.transforms.functional as TF

# import torch
# from torch import nn, einsum
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
# from typing import Union, List
# import numpy as np
# from timm.models.layers import trunc_normal_


# class CyclicShift3D(nn.Module):
#     def __init__(self, displacement):
#         super().__init__()

#         assert type(displacement) is int or len(displacement) == 3, f'displacement must be 1 or 3 dimension'
#         if type(displacement) is int:
#             displacement = np.array([displacement, displacement, displacement])
#         self.displacement = displacement

#     def forward(self, x):
#         return torch.roll(x, shifts=(self.displacement[0], self.displacement[1], self.displacement[2]), dims=(1, 2, 3))


# class Residual3D(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         return self.fn(x, **kwargs) + x


# class PreNorm3D(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)


# class FeedForward3D(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout: float = 0.0):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, dim),
#         )
#         self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

#     def forward(self, x):
#         x = self.net(x)
#         x = self.drop(x)
#         return x


# def create_mask3D(window_size: Union[int, List[int]], displacement: Union[int, List[int]],
#                   x_shift: bool, y_shift: bool, z_shift: bool):
#     assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
#     if type(window_size) is int:
#         window_size = np.array([window_size, window_size, window_size])

#     assert type(displacement) is int or len(displacement) == 3, f'displacement must be 1 or 3 dimension'
#     if type(displacement) is int:
#         displacement = np.array([displacement, displacement, displacement])

#     assert len(window_size) == len(displacement)
#     for i in range(len(window_size)):
#         assert 0 < displacement[i] < window_size[i], \
#             f'在第{i}轴的偏移量不正确，维度包括X(i=0)，Y(i=1)和Z(i=2)'

#     mask = torch.zeros(window_size[0] * window_size[1] * window_size[2],
#                        window_size[0] * window_size[1] * window_size[2])  # (wx*wy*wz, wx*wy*wz)
#     mask = rearrange(mask, '(x1 y1 z1) (x2 y2 z2) -> x1 y1 z1 x2 y2 z2',
#                      x1=window_size[0], y1=window_size[1], x2=window_size[0], y2=window_size[1])

#     x_dist, y_dist, z_dist = displacement[0], displacement[1], displacement[2]

#     if x_shift:
#         #      x1     y1 z1     x2     y2 z2
#         mask[-x_dist:, :, :, :-x_dist, :, :] = float('-inf')
#         mask[:-x_dist, :, :, -x_dist:, :, :] = float('-inf')

#     if y_shift:
#         #   x1   y1       z1 x2  y2       z2
#         mask[:, -y_dist:, :, :, :-y_dist, :] = float('-inf')
#         mask[:, :-y_dist, :, :, -y_dist:, :] = float('-inf')

#     if z_shift:
#         #   x1  y1  z1       x2 y2  z2
#         mask[:, :, -z_dist:, :, :, :-z_dist] = float('-inf')
#         mask[:, :, :-z_dist, :, :, -z_dist:] = float('-inf')

#     mask = rearrange(mask, 'x1 y1 z1 x2 y2 z2 -> (x1 y1 z1) (x2 y2 z2)')
#     return mask


# # 参考自video_swin_transformer:
# # #https://github.com/MohammadRezaQaderi/Video-Swin-Transformer/blob/c3cd8639decf19a25303615db0b6c1195495f5bb/mmaction/models/backbones/swin_transformer.py#L119
# def get_relative_distances(window_size):
#     assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
#     if type(window_size) is int:
#         window_size = np.array([window_size, window_size, window_size])
#     indices = torch.tensor(
#         np.array(
#             [[x, y, z] for x in range(window_size[0]) for y in range(window_size[1]) for z in range(window_size[2])]))

#     distances = indices[None, :, :] - indices[:, None, :]
#     # distance:(n,n,3) n =window_size[0]*window_size[1]*window_size[2]
#     return distances


# class WindowAttention3D(nn.Module):
#     def __init__(self, dim: int, heads: int, head_dim: int, shifted: bool, window_size: Union[int, List[int]],
#                  relative_pos_embedding: bool = True):
#         super().__init__()

#         assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
#         if type(window_size) is int:
#             window_size = np.array([window_size, window_size, window_size])
#         else:
#             window_size = np.array(window_size)

#         inner_dim = head_dim * heads
#         self.heads = heads
#         self.scale = head_dim ** -0.5
#         self.window_size = window_size
#         # self.relative_pos_embedding = relative_pos_embedding
#         self.shifted = shifted

#         if self.shifted:
#             displacement = window_size // 2
#             self.cyclic_shift = CyclicShift3D(-displacement)
#             self.cyclic_back_shift = CyclicShift3D(displacement)
#             self.x_mask = nn.Parameter(create_mask3D(window_size=window_size, displacement=displacement,
#                                                      x_shift=True, y_shift=False, z_shift=False), requires_grad=False)
#             self.y_mask = nn.Parameter(create_mask3D(window_size=window_size, displacement=displacement,
#                                                      x_shift=False, y_shift=True, z_shift=False), requires_grad=False)
#             self.z_mask = nn.Parameter(create_mask3D(window_size=window_size, displacement=displacement,
#                                                      x_shift=False, y_shift=False, z_shift=True), requires_grad=False)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # QKV三个

#         # if self.relative_pos_embedding:
#         #     self.relative_indices = get_relative_distances(window_size)
#         #     # relative_indices的形状为 (n,n,3) n=window_size[0]*window_size[1]*window_size[2],
#         #
#         #     for i in range(len(window_size)):  # 在每个维度上进行偏移
#         #         self.relative_indices[:, :, i] += window_size[i] - 1
#         #
#         #     self.pos_embedding = nn.Parameter(
#         #         torch.randn(2 * window_size[0] - 1, 2 * window_size[1] - 1, 2 * window_size[2] - 1)
#         #     )
#         # else:
#         # self.pos_embedding = nn.Parameter(torch.randn(window_size[0] * window_size[1] * window_size[2],
#         #                                               window_size[0] * window_size[1] * window_size[2]))

#         self.softmax = nn.Softmax(dim=-1)
#         self.to_out = nn.Linear(inner_dim, dim)

#     def forward(self, x):
#         if self.shifted:
#             x = self.cyclic_shift(x)

#         b, n_x, n_y, n_z, _, h = *x.shape, self.heads

#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         nw_x = n_x // self.window_size[0]
#         nw_y = n_y // self.window_size[1]
#         nw_z = n_z // self.window_size[2]

#         q, k, v = map(
#             lambda t: rearrange(t, 'b (nw_x w_x) (nw_y w_y) (nw_z w_z) (h d) -> b h (nw_x nw_y nw_z) (w_x w_y w_z) d',
#                                 h=h, w_x=self.window_size[0], w_y=self.window_size[1], w_z=self.window_size[2]), qkv)

#         dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale  # q和k的矩阵乘法

#         # if self.relative_pos_embedding:
#         #     dots += self.pos_embedding[self.relative_indices[:, :, 0].long(), self.relative_indices[:, :, 1].long(),
#         #                                self.relative_indices[:, :, 2].long()]
#         # else:
#         #   dots += self.pos_embedding  # 触发了广播机制

#         if self.shifted:
#             # 将x轴的窗口数量移至尾部，便于和x轴上对应的mask叠加，下同
#             dots = rearrange(dots, 'b h (n_x n_y n_z) i j -> b h n_y n_z n_x i j',
#                              n_x=nw_x, n_y=nw_y)
#             #   b   h n_y n_z n_x i j
#             dots[:, :, :, :, -1] += self.x_mask

#             dots = rearrange(dots, 'b h n_y n_z n_x i j -> b h n_x n_z n_y i j')
#             dots[:, :, :, :, -1] += self.y_mask

#             dots = rearrange(dots, 'b h n_x n_z n_y i j -> b h n_x n_y n_z i j')
#             dots[:, :, :, :, -1] += self.z_mask

#             dots = rearrange(dots, 'b h n_y n_z n_x i j -> b h (n_x n_y n_z) i j')

#         # attn = dots.softmax(dim=-1)
#         attn = self.softmax(dots)
#         out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)  # 进行attn和v的矩阵乘法

#         # nw_x 表示x轴上窗口的数量 , nw_y 表示 y轴上窗口的数量，nw_Z表示z轴上窗口的数量
#         # w_x 表示 x_window_size, w_y 表示 y_window_size， w_z表示z_window_size
#         #                     b 3  (8,8,8)         （7,  7,  7） 96 -> b  56          56          56        288
#         out = rearrange(out, 'b h (nw_x nw_y nw_z) (w_x w_y w_z) d -> b (nw_x w_x) (nw_y w_y) (nw_z w_z) (h d)',
#                         h=h, w_x=self.window_size[0], w_y=self.window_size[1], w_z=self.window_size[2],
#                         nw_x=nw_x, nw_y=nw_y, nw_z=nw_z)
#         out = self.to_out(out)

#         if self.shifted:
#             out = self.cyclic_back_shift(out)
#         return out


# class SwinBlock3D(nn.Module):  # 不会改变输入空间分辨率
#     def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size: Union[int, List[int]],
#                  relative_pos_embedding: bool = True, dropout: float = 0.0):
#         super().__init__()
#         self.attention_block = Residual3D(PreNorm3D(dim, WindowAttention3D(dim=dim,
#                                                                            heads=heads,
#                                                                            head_dim=head_dim,
#                                                                            shifted=shifted,
#                                                                            window_size=window_size,
#                                                                            relative_pos_embedding=relative_pos_embedding)))
#         self.mlp_block = Residual3D(PreNorm3D(dim, FeedForward3D(dim=dim, hidden_dim=mlp_dim, dropout=dropout)))

#     def forward(self, x):
#         x = self.attention_block(x)
#         x = self.mlp_block(x)
#         return x


# class Norm(nn.Module):
#     def __init__(self, dim, channel_first: bool = True):
#         super(Norm, self).__init__()
#         if channel_first:
#             self.net = nn.Sequential(
#                 Rearrange('b c h w d -> b h w d c'),
#                 nn.LayerNorm(dim),
#                 Rearrange('b h w d c -> b c h w d')
#             )

#             # self.net = nn.InstanceNorm3d(dim, eps=1e-5, momentum=0.1, affine=False)
#         else:
#             self.net = nn.LayerNorm(dim)

#     def forward(self, x):
#         x = self.net(x)
#         return x


# class PatchMerging3D(nn.Module):
#     def __init__(self, in_dim, out_dim, downscaling_factor):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv3d(in_dim, out_dim, kernel_size=downscaling_factor, stride=downscaling_factor),
#             Norm(dim=out_dim),
#         )

#     def forward(self, x):
#         # x: B, C, H, W, D
#         x = self.net(x)
#         return x  # B,  H //down_scaling, W//down_scaling, D//down_scaling, out_dim


# class PatchExpand3D(nn.Module):
#     def __init__(self, in_dim, out_dim, up_scaling_factor):
#         super(PatchExpand3D, self).__init__()

#         stride = up_scaling_factor
#         kernel_size = up_scaling_factor
#         padding = (kernel_size - stride) // 2
#         self.net = nn.Sequential(
#             nn.ConvTranspose3d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
#             Norm(out_dim),
#         )

#     def forward(self, x):
#         '''X: B,C,X,Y,Z'''
#         x = self.net(x)
#         return x


# class FinalExpand3D(nn.Module):  # 体素最终分类时使用
#     def __init__(self, in_dim, out_dim, up_scaling_factor):  # stl为second_to_last的缩写
#         super(FinalExpand3D, self).__init__()

#         stride = up_scaling_factor
#         kernel_size = up_scaling_factor
#         padding = (kernel_size - stride) // 2
#         self.net = nn.Sequential(
#             nn.ConvTranspose3d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
#             Norm(out_dim),
#             nn.PReLU()
#         )

#     def forward(self, x):
#         '''X: B,C,H,W,D'''
#         x = self.net(x)
#         return x


# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(ConvBlock, self).__init__()
#         groups = min(in_ch, out_ch)
#         self.net = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=groups),
#             Norm(dim=out_ch),
#             nn.PReLU(),

#             nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=groups),
#             Norm(dim=out_ch),
#             nn.PReLU(),
#         )

#     def forward(self, x):
#         x2 = x.clone()
#         x = self.net(x) * x2
#         return x


# class StageModuleDownScaling3D(nn.Module):
#     def __init__(self, in_dims, hidden_dimension, layers, downscaling_factor, num_heads, head_dim,
#                  window_size: Union[int, List[int]], relative_pos_embedding: bool = True, dropout: float = 0.0):
#         super().__init__()
#         assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

#         self.patch_partition = PatchMerging3D(in_dim=in_dims, out_dim=hidden_dimension,
#                                               downscaling_factor=downscaling_factor)
#         self.conv_block = ConvBlock(in_ch=hidden_dimension, out_ch=hidden_dimension)

#         self.re1 = Rearrange('b c h w d -> b h w d c')
#         self.swin_layers = nn.ModuleList([])
#         for _ in range(layers // 2):
#             self.swin_layers.append(nn.ModuleList([
#                 SwinBlock3D(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
#                             shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
#                             dropout=dropout),
#                 SwinBlock3D(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
#                             shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
#                             dropout=dropout),
#             ]))
#         self.re2 = Rearrange('b  h w d c -> b c h w d')

#     def forward(self, x):
#         x = self.patch_partition(x)
#         x2 = self.conv_block(x)  # 卷积块学习短距离依赖

#         x = self.re1(x)
#         for regular_block, shifted_block in self.swin_layers:  # swin_layers块学习长距离依赖
#             x = regular_block(x)
#             x = shifted_block(x)
#         x = self.re2(x)

#         x = x + x2  # 对长短距离依赖信息进行融合
#         return x


# class StageModuleUpScaling3D(nn.Module):
#     def __init__(self, in_dims, out_dims, layers, up_scaling_factor, num_heads, head_dim,
#                  window_size: Union[int, List[int]], relative_pos_embedding, dropout: float = 0.0):
#         super().__init__()
#         assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

#         self.patch_expand = PatchExpand3D(in_dim=in_dims, out_dim=out_dims,
#                                           up_scaling_factor=up_scaling_factor)

#         self.conv_block = ConvBlock(in_ch=out_dims, out_ch=out_dims)
#         self.re1 = Rearrange('b c h w d -> b h w d c')
#         self.swin_layers = nn.ModuleList([])
#         for _ in range(layers // 2):
#             self.swin_layers.append(nn.ModuleList([
#                 SwinBlock3D(dim=out_dims, heads=num_heads, head_dim=head_dim, mlp_dim=out_dims * 4,
#                             shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
#                             dropout=dropout),
#                 SwinBlock3D(dim=out_dims, heads=num_heads, head_dim=head_dim, mlp_dim=out_dims * 4,
#                             shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
#                             dropout=dropout),
#             ]))
#         self.re2 = Rearrange('b h w d c -> b c h w d')

#     def forward(self, x):
#         x = self.patch_expand(x)

#         x2 = self.conv_block(x)

#         x = self.re1(x)
#         for regular_block, shifted_block in self.swin_layers:
#             x = regular_block(x)
#             x = shifted_block(x)
#         x = self.re2(x)

#         x = x + x2
#         return x


# class Converge(nn.Module):
#     def __init__(self, dim: int):
#         '''
#         stack:融合方式以堆叠+线性变换实现
#         add 跳跃连接通过直接相加的方式实现
#         '''
#         super(Converge, self).__init__()
#         self.norm = Norm(dim=dim)

#     def forward(self, x, enc_x):
#         '''
#          x: B,C,X,Y,Z
#         enc_x:B,C,X,Y,Z
#         '''
#         assert x.shape == enc_x.shape
#         x = x + enc_x
#         x = self.norm(x)
#         return x


# class SwinUnet3D(nn.Module):
#     def __init__(self, *, hidden_dim, layers, heads, in_channel=1, num_classes=2, head_dim=32,
#                  window_size: Union[int, List[int]] = 7, downscaling_factors=(4, 2, 2, 2),
#                  relative_pos_embedding=True, dropout: float = 0.0, skip_style='stack',
#                  stl_channels: int = 32):  # second_to_last_channels
#         super().__init__()

#         self.dsf = downscaling_factors
#         self.window_size = window_size

#         self.down_stage12 = StageModuleDownScaling3D(in_dims=in_channel, hidden_dimension=hidden_dim, layers=layers[0],
#                                                      downscaling_factor=downscaling_factors[0], num_heads=heads[0],
#                                                      head_dim=head_dim, window_size=window_size, dropout=dropout,
#                                                      relative_pos_embedding=relative_pos_embedding)
#         self.down_stage3 = StageModuleDownScaling3D(in_dims=hidden_dim, hidden_dimension=hidden_dim * 2,
#                                                     layers=layers[1],
#                                                     downscaling_factor=downscaling_factors[1], num_heads=heads[1],
#                                                     head_dim=head_dim, window_size=window_size, dropout=dropout,
#                                                     relative_pos_embedding=relative_pos_embedding)
#         self.down_stage4 = StageModuleDownScaling3D(in_dims=hidden_dim * 2, hidden_dimension=hidden_dim * 4,
#                                                     layers=layers[2],
#                                                     downscaling_factor=downscaling_factors[2], num_heads=heads[2],
#                                                     head_dim=head_dim, window_size=window_size, dropout=dropout,
#                                                     relative_pos_embedding=relative_pos_embedding)
#         self.features = StageModuleDownScaling3D(in_dims=hidden_dim * 4, hidden_dimension=hidden_dim * 8,
#                                                  layers=layers[3],
#                                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3],
#                                                  head_dim=head_dim, window_size=window_size, dropout=dropout,
#                                                  relative_pos_embedding=relative_pos_embedding)

#         self.up_stage4 = StageModuleUpScaling3D(in_dims=hidden_dim * 8, out_dims=hidden_dim * 4,
#                                                 layers=layers[2],
#                                                 up_scaling_factor=downscaling_factors[3], num_heads=heads[2],
#                                                 head_dim=head_dim, window_size=window_size, dropout=dropout,
#                                                 relative_pos_embedding=relative_pos_embedding)

#         self.up_stage3 = StageModuleUpScaling3D(in_dims=hidden_dim * 4, out_dims=hidden_dim * 2,
#                                                 layers=layers[1],
#                                                 up_scaling_factor=downscaling_factors[2], num_heads=heads[1],
#                                                 head_dim=head_dim, window_size=window_size, dropout=dropout,
#                                                 relative_pos_embedding=relative_pos_embedding)

#         self.up_stage12 = StageModuleUpScaling3D(in_dims=hidden_dim * 2, out_dims=hidden_dim,
#                                                  layers=layers[0],
#                                                  up_scaling_factor=downscaling_factors[1], num_heads=heads[0],
#                                                  head_dim=head_dim, window_size=window_size, dropout=dropout,
#                                                  relative_pos_embedding=relative_pos_embedding)

#         self.converge4 = Converge(hidden_dim * 4)
#         self.converge3 = Converge(hidden_dim * 2)
#         self.converge12 = Converge(hidden_dim)

#         self.final = FinalExpand3D(in_dim=hidden_dim, out_dim=stl_channels,
#                                    up_scaling_factor=downscaling_factors[0])
#         self.out = nn.Sequential(
#             # nn.Linear(stl_channels, num_classes),
#             # Rearrange('b h w d c -> b c h w d'),
#             nn.Conv3d(stl_channels, num_classes, kernel_size=1)
#         )
#         # 参数初始化
#         self.init_weight()

#     def forward(self, img):
#         window_size = self.window_size
#         assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
#         if type(window_size) is int:
#             window_size = np.array([window_size, window_size, window_size])
#         _, _, x_s, y_s, z_s = img.shape
#         x_ws, y_ws, z_ws = window_size

#         assert x_s % (x_ws * 32) == 0, f'x轴上的尺寸必须能被x_window_size*32 整除'
#         assert y_s % (y_ws * 32) == 0, f'y轴上的尺寸必须能被y_window_size*32 整除'
#         assert z_s % (z_ws * 32) == 0, f'y轴上的尺寸必须能被z_window_size*32 整除'

#         down12_1 = self.down_stage12(img)  # (B,C, X//4, Y//4, Z//4)
#         down3 = self.down_stage3(down12_1)  # (B, 2C,X//8, Y//8, Z//8)
#         down4 = self.down_stage4(down3)  # (B, 4C,X//16, Y//16, Z//16)
#         features = self.features(down4)  # (B, 8C,X//32, Y//32, Z//32)

#         up4 = self.up_stage4(features)  # (B, 8C, X//16, Y//16, Z//16 )
#         # up1和 down3融合
#         up4 = self.converge4(up4, down4)  # (B, 4C, X//16, Y//16, Z//16)

#         up3 = self.up_stage3(up4)  # ((B, 2C,X//8, Y//8, Z//8)
#         # up2和 down2融合
#         up3 = self.converge3(up3, down3)  # (B,2C, X//8, Y//8)

#         up12 = self.up_stage12(up3)  # (B,C, X//4, Y//4, Z// 4)
#         # up3和 down1融合
#         up12 = self.converge12(up12, down12_1)  # (B,C, X//4, Y//4, Z//4)

#         out = self.final(up12)  # (B,num_classes, X, Y, Z)
#         out = self.out(out)
#         return out

#     def init_weight(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
#                 trunc_normal_(m.weight, std=0.02)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             if isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)


# # 原始论文中 layers=[2,2,6,2]
# def swinUnet_t_3D(hidden_dim=96, layers=(2, 2, 4, 2), heads=(3, 6, 9, 12), num_classes: int = 2, **kwargs):
#     return SwinUnet3D(hidden_dim=hidden_dim, layers=layers, heads=heads, num_classes=num_classes, **kwargs)


# class Unet(nn.Module):
#     """
#     A unet architecture. Layer features can be specified directly as a list of encoder and decoder
#     features or as a single integer along with a number of unet levels. The default network features
#     per layer (when no options are specified) are:

#         encoder: [16, 32, 32, 32]
#         decoder: [32, 32, 32, 32, 32, 16, 16]
#     """
        
#     def __init__(self,
#                  inshape=None,
#                  infeats=None,
#                  nb_features=None,
#                  nb_levels=None,
#                  max_pool=2,
#                  feat_mult=1,
#                  nb_conv_per_level=1,
#                  half_res=False):
        
#         """
#         Parameters:
#             inshape: Input shape. e.g. (192, 192, 192)
#             infeats: Number of input features.
#             nb_features: Unet convolutional features. Can be specified via a list of lists with
#                 the form [[encoder feats], [decoder feats]], or as a single integer. 
#                 If None (default), the unet features are defined by the default config described in 
#                 the class documentation.
#             nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
#                 Default is None.
#             feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
#                 Default is 1.
#             nb_conv_per_level: Number of convolutions per unet level. Default is 1.
#             half_res: Skip the last decoder upsampling. Default is False.
#         """
#         super().__init__()

#         # ensure correct dimensionality
#         ndims = len(inshape)
#         assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

#         # cache some parameters
#         self.half_res = half_res

#         # default encoder and decoder layer features if nothing provided
#         if nb_features is None:
#             nb_features = default_unet_features()
#             print("1")

#         # build feature list automatically
#         if isinstance(nb_features, int):
#             if nb_levels is None:
#                 print("2")
#                 raise ValueError('must provide unet nb_levels if nb_features is an integer')
#             feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
#             print("3")
#             nb_features = [
#                 np.repeat(feats[:-1], nb_conv_per_level),
#                 np.repeat(np.flip(feats), nb_conv_per_level)
#             ]
#         elif nb_levels is not None:
#             print("4")
#             raise ValueError('cannot use nb_levels if nb_features is not an integer')

#         # extract any surplus (full resolution) decoder convolutions
#         enc_nf, dec_nf = nb_features
#         nb_dec_convs = len(enc_nf)
#         final_convs = dec_nf[nb_dec_convs:]
#         dec_nf = dec_nf[:nb_dec_convs]
#         self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

#         if isinstance(max_pool, int):
#             print("5")
#             max_pool = [max_pool] * self.nb_levels

#         # cache downsampling / upsampling operations
#         MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
#         self.pooling = [MaxPooling(s) for s in max_pool]
#         self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

#         # configure encoder (down-sampling path)
#         prev_nf = infeats
#         encoder_nfs = [prev_nf]
#         self.encoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             print("6")
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 print("7")
#                 nf = enc_nf[level * nb_conv_per_level + conv]
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#             self.encoder.append(convs)
#             encoder_nfs.append(prev_nf)

#         # configure decoder (up-sampling path)
#         encoder_nfs = np.flip(encoder_nfs)
#         self.decoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             print("8")
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 print("9")
#                 nf = dec_nf[level * nb_conv_per_level + conv]
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#             self.decoder.append(convs)
#             if not half_res or level < (self.nb_levels - 2):
#                 print("10")
#                 prev_nf += encoder_nfs[level]

#         # now we take care of any remaining convolutions
#         self.remaining = nn.ModuleList()
#         for num, nf in enumerate(final_convs):
#             print("11")
#             self.remaining.append(ConvBlock(ndims, prev_nf, nf))
#             prev_nf = nf

#         # cache final number of features
#         self.final_nf = prev_nf

#     # def forward(self, x):
#     def forward(self, source, target):
#         x = torch.cat([source, target], dim=1)


#         # encoder forward pass
#         x_history = [x]
#         for level, convs in enumerate(self.encoder):
#             for conv in convs:
#                 x = conv(x)
#             x_history.append(x)
#             # print("After encoder level（エンコーダの中）", level, ":", x.shape)  # 追加
#             x = self.pooling[level](x)

#         # decoder forward pass with upsampling and concatenation
#         for level, convs in enumerate(self.decoder):
#             for conv in convs:
#                 x = conv(x)
#             if not self.half_res or level < (self.nb_levels - 2):
#                 x = self.upsampling[level](x)
#                 # print("Before concatenation（デコーダーの中）:", x.shape, x_history[-1].shape)  # 追加
#                 x = torch.cat([x, x_history.pop()], dim=1)
#                 # print("After concatenation（デコーダーの中）:", x.shape)  # 追加

#         # remaining convs at full resolution
#         for conv in self.remaining:
#             x = conv(x)

#         return x

# class VxmDense_Swin(LoadableModel):
#     """
#     VoxelMorph network for (unsupervised) nonlinear registration between two images.
#     """

#     @store_config_args
#     def __init__(self,
#                  inshape,
#                  nb_unet_features=None,
#                  nb_unet_levels=None,
#                  unet_feat_mult=1,
#                  nb_unet_conv_per_level=1,
#                  int_steps=7,
#                  int_downsize=2,
#                  bidir=False,
#                  use_probs=False,
#                  src_feats=1,
#                  trg_feats=1,
#                  unet_half_res=False):
#         """ 
#         Parameters:
#             inshape: Input shape. e.g. (192, 192, 192)
#             nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
#                 the form [[encoder feats], [decoder feats]], or as a single integer. 
#                 If None (default), the unet features are defined by the default config described in 
#                 the unet class documentation.
#             nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
#                 Default is None.
#             unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
#                 Default is 1.
#             nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
#             int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
#                 value is 0.
#             int_downsize: Integer specifying the flow downsample factor for vector integration. 
#                 The flow field is not downsampled when this value is 1.
#             bidir: Enable bidirectional cost function. Default is False.
#             use_probs: Use probabilities in flow field. Default is False.
#             src_feats: Number of source image features. Default is 1.
#             trg_feats: Number of target image features. Default is 1.
#             unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
#                 Default is False.
#         """
#         super().__init__()

#         # internal flag indicating whether to return flow or integrated warp during inference
#         self.training = True

#         # ensure correct dimensionality
#         ndims = len(inshape)
#         assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

#         # # configure core unet model
#         # self.unet_model = Unet(
#         #     inshape,
#         #     infeats=(src_feats + trg_feats),
#         #     nb_features=nb_unet_features,
#         #     nb_levels=nb_unet_levels,
#         #     feat_mult=unet_feat_mult,
#         #     nb_conv_per_level=nb_unet_conv_per_level,
#         #     half_res=unet_half_res,
#         # )
#         self.unet_model = SwinUnet3D(
#             hidden_dim=48,          # 各ステージの基本チャネル数
#             layers=[2, 2, 2, 2],    # 各ステージの Transformer ブロック数
#             heads=[3, 6, 12, 24],   # 各ステージの Attention ヘッド数
#             in_channel=2,            # moving/fixed の 2チャネル入力
#             num_classes=3,           # 出力は 3D displacement field
#             window_size=2            # Swin Transformer の window size
#         )
  
#         # # configure unet to flow field layer
#         # Conv = getattr(nn, 'Conv%dd' % ndims)
#         # self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

#         # # init flow layer with small weights and bias
#         # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
#         # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

#         # # probabilities are not supported in pytorch
#         # if use_probs:
#         #     raise NotImplementedError(
#         #         'Flow variance has not been implemented in pytorch - set use_probs to False')

#         # # configure optional resize layers (downsize)
#         # if not unet_half_res and int_steps > 0 and int_downsize > 1:
#         #     self.resize = layers.ResizeTransform(int_downsize, ndims)
#         # else:
#         #     self.resize = None

#         # # resize to full res
#         # if int_steps > 0 and int_downsize > 1:
#         #     self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
#         # else:
#         #     self.fullsize = None

#         # # configure bidirectional training
#         # self.bidir = bidir

#         # # configure optional integration layer for diffeomorphic warp
#         # down_shape = [int(dim / int_downsize) for dim in inshape]
#         # print(down_shape)
#         # self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
#         # # print("insahpeのサイズ",inshape)

#         # configure transformer

#         # new_inshape = inshape.squeeze(dim=2)
#         self.transformer = layers.SpatialTransformer(inshape)


#     def forward(self, source, target, catimage, registration=False):
#         '''
#         Parameters:
#             source: Source image tensor.
#             target: Target image tensor.
#             registration: Return transformed image and flow. Default is False.
#         '''

#         # concatenate inputs and propagate unet
#         # print("（読み込んだ画像サイズ）:", source.shape, target.shape)
#         # x = torch.cat([source, target], dim=1)
#         # x = x.unsqueeze(-1)
#         # print("（Catした画像サイズ）:", x.shape)
#         x = self.unet_model(catimage)
#         # print("aaaa2222", x.max())
#         print("（Xサイズ）:", x.shape)

#         # transform into flow field

#         # flow_field = self.flow(x)
#         flow_field = x

#         # print("aaaa3333", flow_field.max())
#         print("（flow_fieldサイズ）:", flow_field.shape)

#         y_source = self.transformer(source, flow_field)
#         return y_source, flow_field


# # #DAvoxelmorphwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
# # coding=utf-8
# # from __future__ import absolute_import
# # from __future__ import division
# # from __future__ import print_function

# import copy
# import logging
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as nnf
# from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
# from torch.nn.modules.utils import _pair, _triple
# import configs as configs
# from torch.distributions.normal import Normal

# logger = logging.getLogger(__name__)


# ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
# ATTENTION_K = "MultiHeadDotProductAttention_1/key"
# ATTENTION_V = "MultiHeadDotProductAttention_1/value"
# ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
# FC_0 = "MlpBlock_3/Dense_0"
# FC_1 = "MlpBlock_3/Dense_1"
# ATTENTION_NORM = "LayerNorm_0"
# MLP_NORM = "LayerNorm_2"


# def np2th(weights, conv=False):
#     """Possibly convert HWIO to OIHW."""
#     if conv:
#         weights = weights.transpose([3, 2, 0, 1])
#     return torch.from_numpy(weights)


# def swish(x):
#     return x * torch.sigmoid(x)


# ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


# class Attention(nn.Module):
#     def __init__(self, config, vis):
#         super(Attention, self).__init__()
#         self.vis = vis
#         self.num_attention_heads = config.transformer["num_heads"]
#         self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = Linear(config.hidden_size, self.all_head_size)
#         self.key = Linear(config.hidden_size, self.all_head_size)
#         self.value = Linear(config.hidden_size, self.all_head_size)

#         self.out = Linear(config.hidden_size, config.hidden_size)
#         self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
#         self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

#         self.softmax = Softmax(dim=-1)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(self, hidden_states):
#         mixed_query_layer = self.query(hidden_states)
#         mixed_key_layer = self.key(hidden_states)
#         mixed_value_layer = self.value(hidden_states)

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         attention_probs = self.softmax(attention_scores)
#         weights = attention_probs if self.vis else None
#         attention_probs = self.attn_dropout(attention_probs)

#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         attention_output = self.out(context_layer)
#         attention_output = self.proj_dropout(attention_output)
#         return attention_output, weights


# class Mlp(nn.Module):
#     def __init__(self, config):
#         super(Mlp, self).__init__()
#         self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
#         self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
#         self.act_fn = ACT2FN["gelu"]
#         self.dropout = Dropout(config.transformer["dropout_rate"])

#         self._init_weights()

#     def _init_weights(self):
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.normal_(self.fc1.bias, std=1e-6)
#         nn.init.normal_(self.fc2.bias, std=1e-6)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act_fn(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x


# class Embeddings(nn.Module):
#     """Construct the embeddings from patch, position embeddings.
#     """
#     def __init__(self, config, img_size):
#         super(Embeddings, self).__init__()
#         self.config = config
#         down_factor = config.down_factor
#         patch_size = _triple(config.patches["size"])
#         n_patches = int((img_size[0]/2**down_factor// patch_size[0]) * (img_size[1]/2**down_factor// patch_size[1]) * (img_size[2]/2**down_factor// patch_size[2]))
#         self.hybrid_model = CNNEncoder(config, n_channels=2)
#         in_channels = config['encoder_channels'][-1]
#         self.patch_embeddings = Conv3d(in_channels=in_channels,
#                                        out_channels=config.hidden_size,
#                                        kernel_size=patch_size,
#                                        stride=patch_size)
#         self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

#         self.dropout = Dropout(config.transformer["dropout_rate"])

#     def forward(self, x):
#         x, features = self.hybrid_model(x)
#         x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
#         x = x.flatten(2)
#         x = x.transpose(-1, -2)  # (B, n_patches, hidden)
#         embeddings = x + self.position_embeddings
#         embeddings = self.dropout(embeddings)
#         return embeddings, features


# class Block(nn.Module):
#     def __init__(self, config, vis):
#         super(Block, self).__init__()
#         self.hidden_size = config.hidden_size
#         self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn = Mlp(config)
#         self.attn = Attention(config, vis)

#     def forward(self, x):
#         h = x

#         x = self.attention_norm(x)
#         x, weights = self.attn(x)
#         x = x + h

#         h = x
#         x = self.ffn_norm(x)
#         x = self.ffn(x)
#         x = x + h
#         return x, weights

# class Encoder(nn.Module):
#     def __init__(self, config, vis):
#         super(Encoder, self).__init__()
#         self.vis = vis
#         self.layer = nn.ModuleList()
#         self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         for _ in range(config.transformer["num_layers"]):
#             layer = Block(config, vis)
#             self.layer.append(copy.deepcopy(layer))

#     def forward(self, hidden_states):
#         attn_weights = []
#         for layer_block in self.layer:
#             hidden_states, weights = layer_block(hidden_states)
#             if self.vis:
#                 attn_weights.append(weights)
#         encoded = self.encoder_norm(hidden_states)
#         return encoded, attn_weights


# class Transformer(nn.Module):
#     def __init__(self, config, img_size, vis):
#         super(Transformer, self).__init__()
#         self.embeddings = Embeddings(config, img_size=img_size)
#         self.encoder = Encoder(config, vis)

#     def forward(self, input_ids):
#         embedding_output, features = self.embeddings(input_ids)
#         encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
#         return encoded, attn_weights, features


# class Conv3dReLU(nn.Sequential):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             kernel_size,
#             padding=0,
#             stride=1,
#             use_batchnorm=True,
#     ):
#         conv = nn.Conv3d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding=padding,
#             bias=not (use_batchnorm),
#         )
#         relu = nn.ReLU(inplace=True)

#         bn = nn.BatchNorm3d(out_channels)

#         super(Conv3dReLU, self).__init__(conv, bn, relu)


# class DecoderBlock(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             skip_channels=0,
#             use_batchnorm=True,
#     ):
#         super().__init__()
#         self.conv1 = Conv3dReLU(
#             in_channels + skip_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.conv2 = Conv3dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

#     def forward(self, x, skip=None):
#         x = self.up(x)
#         if skip is not None:
#             x = torch.cat([x, skip], dim=1)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x

# class DecoderCup(nn.Module):
#     def __init__(self, config, img_size):
#         super().__init__()
#         self.config = config
#         self.down_factor = config.down_factor
#         head_channels = config.conv_first_channel
#         self.img_size = img_size
#         self.conv_more = Conv3dReLU(
#             config.hidden_size,
#             head_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=True,
#         )
#         decoder_channels = config.decoder_channels
#         in_channels = [head_channels] + list(decoder_channels[:-1])
#         out_channels = decoder_channels
#         self.patch_size = _triple(config.patches["size"])
#         skip_channels = self.config.skip_channels
#         blocks = [
#             DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
#         ]
#         self.blocks = nn.ModuleList(blocks)

#     def forward(self, hidden_states, features=None):
#         B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
#         l, h, w = (self.img_size[0]//2**self.down_factor//self.patch_size[0]), (self.img_size[1]//2**self.down_factor//self.patch_size[1]), (self.img_size[2]//2**self.down_factor//self.patch_size[2])
#         x = hidden_states.permute(0, 2, 1)
#         x = x.contiguous().view(B, hidden, l, h, w)
#         x = self.conv_more(x)
#         for i, decoder_block in enumerate(self.blocks):
#             if features is not None:
#                 skip = features[i] if (i < self.config.n_skip) else None
#                 #print(skip.shape)
#             else:
#                 skip = None
#             x = decoder_block(x, skip=skip)
#         return x

# class SpatialTransformer(nn.Module):
#     """
#     N-D Spatial Transformer

#     Obtained from https://github.com/voxelmorph/voxelmorph
#     """

#     def __init__(self, size, mode='bilinear'):
#         super().__init__()

#         self.mode = mode

#         # create sampling grid
#         vectors = [torch.arange(0, s) for s in size]
#         grids = torch.meshgrid(vectors)
#         grid = torch.stack(grids)
#         grid = torch.unsqueeze(grid, 0)
#         grid = grid.type(torch.FloatTensor)

#         # registering the grid as a buffer cleanly moves it to the GPU, but it also
#         # adds it to the state dict. this is annoying since everything in the state dict
#         # is included when saving weights to disk, so the model files are way bigger
#         # than they need to be. so far, there does not appear to be an elegant solution.
#         # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
#         self.register_buffer('grid', grid)

#     def forward(self, src, flow):
#         # new locations
#         new_locs = self.grid + flow
#         shape = flow.shape[2:]

#         # need to normalize grid values to [-1, 1] for resampler
#         for i in range(len(shape)):
#             new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

#         # move channels dim to last position
#         # also not sure why, but the channels need to be reversed
#         if len(shape) == 2:
#             new_locs = new_locs.permute(0, 2, 3, 1)
#             new_locs = new_locs[..., [1, 0]]
#         elif len(shape) == 3:
#             new_locs = new_locs.permute(0, 2, 3, 4, 1)
#             new_locs = new_locs[..., [2, 1, 0]]

#         return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool3d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)

# class CNNEncoder(nn.Module):
#     def __init__(self, config, n_channels=2):
#         super(CNNEncoder, self).__init__()
#         self.n_channels = n_channels
#         decoder_channels = config.decoder_channels
#         encoder_channels = config.encoder_channels
#         self.down_num = config.down_num
#         self.inc = DoubleConv(n_channels, encoder_channels[0])
#         self.down1 = Down(encoder_channels[0], encoder_channels[1])
#         self.down2 = Down(encoder_channels[1], encoder_channels[2])
#         self.width = encoder_channels[-1]
#     def forward(self, x):
#         features = []
#         x1 = self.inc(x)
#         features.append(x1)
#         x2 = self.down1(x1)
#         features.append(x2)
#         feats = self.down2(x2)
#         features.append(feats)
#         feats_down = feats
#         for i in range(self.down_num):
#             feats_down = nn.MaxPool3d(2)(feats_down)
#             features.append(feats_down)
#         return feats, features[::-1]

# class RegistrationHead(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
#         conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
#         conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
#         conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
#         super().__init__(conv3d)

# class ViTVNet(nn.Module):
#     def __init__(self, config, img_size=(64, 256, 256), int_steps=7, vis=False):
#         super(ViTVNet, self).__init__()
#         self.transformer = Transformer(config, img_size, vis)
#         self.decoder = DecoderCup(config, img_size)
#         self.reg_head = RegistrationHead(
#             in_channels=config.decoder_channels[-1],
#             out_channels=config['n_dims'],
#             kernel_size=3,
#         )
#         self.spatial_trans = SpatialTransformer(img_size)
#         self.config = config
#         #self.integrate = VecInt(img_size, int_steps)
#     def forward(self, x):

#         source = x[:,0:1,:,:]

#         x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
#         x = self.decoder(x, features)
#         flow = self.reg_head(x)
#         #flow = self.integrate(flow)
#         out = self.spatial_trans(source, flow)
#         return out, flow

# class VecInt(nn.Module):
#     """
#     Integrates a vector field via scaling and squaring.

#     Obtained from https://github.com/voxelmorph/voxelmorph
#     """

#     def __init__(self, inshape, nsteps):
#         super().__init__()

#         assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
#         self.nsteps = nsteps
#         self.scale = 1.0 / (2 ** self.nsteps)
#         self.transformer = SpatialTransformer(inshape)

#     def forward(self, vec):
#         vec = vec * self.scale
#         for _ in range(self.nsteps):
#             vec = vec + self.transformer(vec, vec)
#         return vec

# CONFIGS = {
#     'ViT-V-Net': configs.get_3DReg_config(),
# }