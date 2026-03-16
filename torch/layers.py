# ============================================================
# layers.py
# ------------------------------------------------------------
# このファイルには，VoxelMorph 系モデルで使う
# 「画像を変形するための基本レイヤ」が定義されています。
# 
# 主に入っているのは次の3つです。
# 1. SpatialTransformer
#    - 変形ベクトル場（flow, DVF）を使って画像を実際にワープする
# 2. VecInt
#    - ベクトル場を scaling and squaring によって積分する
# 3. ResizeTransform
#    - 変形場のサイズ変更と，ベクトル量のスケーリングを行う
# 
# 位置合わせでは，ネットワークが予測した flow をそのまま使うだけでなく，
# 画像に適用したり，必要に応じて積分したり，解像度を合わせたりする必要がある。
# このファイルは，そのための中核処理をまとめたもの。
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as nnf


# =====================
# Spatial Transformer
# =====================
# flow（変形場）を使って，入力画像 src を実際に変形するクラス
class SpatialTransformer(nn.Module):
    """
    N次元の Spatial Transformer。
    2次元画像にも3次元画像にも使えるように書かれている。
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        # grid_sample で使う補間方法を保存する
        self.mode = mode

        # 画像全体の各座標を表す基準グリッドを作る
        # 各軸について 0, 1, 2, ... の座標列を作る
        vectors = [torch.arange(0, s) for s in size]
        # 各軸の座標列から，N次元の格子座標を作る
        grids = torch.meshgrid(vectors)
        # 軸ごとの座標をまとめて 1つのテンソルにする
        grid = torch.stack(grids)
        # 先頭にバッチ次元を追加して扱いやすくする
        grid = torch.unsqueeze(grid, 0)
        # 補間計算で使えるように float 型へ変換する
        grid = grid.type(torch.FloatTensor)

        # grid は学習パラメータではないが，モデルと一緒に保持したい値なので
        # register_buffer で登録している。
        # こうしておくと，model.to(device) したときに grid も一緒に GPU / CPU へ移動する。
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # 各座標に flow を足して，変形後に参照したい座標を作る
        new_locs = self.grid + flow
        # 空間サイズ（2Dなら H, W / 3Dなら D, H, W）を取り出す
        shape = flow.shape[2:]

        # grid_sample に渡すため，座標値を [-1, 1] の範囲へ正規化する
        # 各軸ごとに，画素番号ベースの座標を正規化座標へ変換する
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # grid_sample が要求する形に合わせて，座標チャネルを最後の次元へ移す
        # さらに，内部の軸順と grid_sample 側の軸順を合わせるために順序を入れ替える
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        # 正規化した座標に基づいて src を補間し，変形後画像を返す
        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


# =====================
# Vector Integration
# =====================
# ベクトル場を scaling and squaring で積分し，より滑らかな変形場を作るクラス
class VecInt(nn.Module):
    """
    scaling and squaring によってベクトル場を積分する。
    大きな変形を，小さな変形の繰り返しとして扱うための処理。
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        # 積分ステップ数は 0 以上である必要がある
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        # 何回 scaling and squaring を行うかを保存する
        self.nsteps = nsteps
        # 最初にベクトル場を小さくしてから繰り返し合成するための倍率
        self.scale = 1.0 / (2 ** self.nsteps)
        # ベクトル場同士の合成にも SpatialTransformer を使う
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        # まず小さな変形にスケールダウンする
        vec = vec * self.scale
        # 小さな変形を繰り返し自分自身に合成して積分する
        for _ in range(self.nsteps):
            # vec を vec に従ってワープし，自分自身に足して変形を合成する
            vec = vec + self.transformer(vec, vec)
        return vec


# =====================
# Resize Transform
# =====================
# 変形場のサイズ変更と，ベクトル量のスケール調整を行うクラス
class ResizeTransform(nn.Module):
    """
    変形場のサイズを変更する。
    ただ大きさを変えるだけでなく，ベクトル値そのものも倍率に合わせて補正する。
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        # リサイズ倍率を計算する
        self.factor = 1.0 / vel_resize
        # 基本の補間モードを決める
        self.mode = 'linear'
        # 次元数に応じて bilinear / trilinear を使い分ける
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        # factor < 1 のときは縮小する処理
        if self.factor < 1:
            # 先にサイズを小さくしてから，ベクトル値も縮小倍率に合わせる
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        # factor > 1 のときは拡大する処理
        elif self.factor > 1:
            # 先にベクトル値を拡大倍率に合わせてから，サイズを大きくする
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # factor == 1 のときは何もせずそのまま返す
        return x
