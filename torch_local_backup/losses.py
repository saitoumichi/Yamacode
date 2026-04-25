# ============================================================
# losses.py
# ------------------------------------------------------------
# このファイルには，VoxelMorph 系モデルで使う代表的な損失関数が
# まとめられています。
# 
# 主に入っているのは次の4つです。
# 1. NCC
#    - 局所的な正規化相互相関を用いる損失
# 2. MSE
#    - 平均二乗誤差
# 3. Dice
#    - セグメンテーションの重なり具合を見る損失
# 4. Grad
#    - 変形場が急に変わりすぎないようにする正則化損失
# 
# 位置合わせでは，画像同士がどれだけ似ているかを見る損失だけでなく，
# 変形が不自然になりすぎないようにする損失も重要になる。
# このファイルは，そのための基本的な損失をまとめたもの。
# ============================================================
import torch_local_backup
import torch.nn.functional as F
import numpy as np
import math


# =====================
# NCC Loss
# =====================
# 局所領域ごとの正規化相互相関を用いた損失
class NCC:
    """
    局所ウィンドウ内で計算する正規化相互相関損失。
    画像の濃淡パターンの似方を，局所領域ごとに評価する。
    """

    def __init__(self, win=None):
        # 相関を計算する局所ウィンドウサイズを保存する
        self.win = win

    def loss(self, y_true, y_pred):

        # y_true と y_pred を，式で使う I, J として扱う
        Ii = y_true
        Ji = y_pred

        # 画像の次元数を取得する
        # ここでは Ii, Ji が [batch_size, channel, ...] 形式で入っていると考える
        ndims = len(list(Ii.size())) - 2
        # 1次元・2次元・3次元のいずれかであることを確認する
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # 局所相関を計算するウィンドウサイズを決める
        win = [9] * ndims if self.win is None else self.win

        # 局所和を求めるための，全部1のフィルタを作る
        # 現状の実装ではフィルタを CUDA 上に置いている
        sum_filt = torch_local_backup.ones([1, 1, *win]).to("cuda")

        # 畳み込み後にサイズを保ちやすいよう，パディング量を計算する
        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # 次元数に応じて conv1d / conv2d / conv3d を選ぶ
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # 相関計算に必要な二乗項と積の項を作る
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # 局所ウィンドウごとの総和を畳み込みで計算する
        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        # ウィンドウ内の要素数から局所平均を計算する
        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        # 共分散項と分散項を計算する
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        # 正規化相互相関を計算する
        cc = cross * cross / (I_var * J_var + 1e-5)

        # 相関は大きいほど良いので，損失としてはマイナスを付けて最小化する
        return -torch_local_backup.mean(cc)


# =====================
# MSE Loss
# =====================
# 平均二乗誤差による基本的な損失
class MSE:
    """
    平均二乗誤差損失。
    画素値やボクセル値の差を直接評価する基本的な損失。
    """

    def loss(self, y_true, y_pred):
        # 予測値と正解値の差を二乗して平均する
        return torch_local_backup.mean((y_true - y_pred) ** 2)


# =====================
# Dice Loss
# =====================
# セグメンテーションの重なり具合を見る損失
class Dice:
    """
    N次元の Dice 損失。
    セグメンテーション結果どうしの重なり具合を評価する。
    """

    def loss(self, y_true, y_pred):
        # 空間次元数を求める
        ndims = len(list(y_pred.size())) - 2
        # 空間方向に和を取るための軸番号を作る
        vol_axes = list(range(2, ndims + 2))
        # 分子は重なった領域の 2倍
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        # 分母は両者の体積和で，0除算を避けるために下限を設ける
        bottom = torch_local_backup.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        # バッチ全体で平均 Dice を求める
        dice = torch_local_backup.mean(top / bottom)
        # Dice は大きいほど良いので，損失としてはマイナスを付ける
        return -dice


# =====================
# Gradient Loss
# =====================
# 変形場が急激に変わりすぎないようにする正則化損失
class Grad:
    """
    N次元の勾配損失。
    隣り合う位置で変形量が急に変わらないように抑えるための正則化。
    """

    def __init__(self, penalty='l1', loss_mult=None):
        # l1 か l2 のどちらで勾配を評価するかを保存する
        self.penalty = penalty
        # 必要なら最後に掛ける倍率を保存する
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        # 各軸方向で隣接ボクセルとの差分を取り，変形の変化量を調べる
        dy = torch_local_backup.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch_local_backup.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch_local_backup.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        # l2 の場合は差分を二乗して，より大きな変化を強く罰する
        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        # 各方向の平均勾配を足し合わせる
        d = torch_local_backup.mean(dx) + torch_local_backup.mean(dy) + torch_local_backup.mean(dz)
        # 3方向の平均を取って最終的な勾配損失にする
        grad = d / 3.0

        # 倍率指定があれば最後に反映する
        if self.loss_mult is not None:
            grad *= self.loss_mult
        # 値が小さいほど，より滑らかな変形場であることを表す
        return grad
