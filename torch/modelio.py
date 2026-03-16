# ============================================================
# modelio.py
# ------------------------------------------------------------
# このファイルには，モデルの設定情報を保存したり，
# 学習済みモデルを読み込みやすくしたりするための補助機能が
# まとめられています。
# 
# 主に入っているのは次の2つです。
# 1. store_config_args
#    - __init__ に渡した引数を self.config に保存するデコレータ
# 2. LoadableModel
#    - モデル構造の設定と重みをまとめて保存・読込できる基底クラス
# 
# VoxelMorph 系モデルでは，学習済み重みを読み込むときに
# 「どんな引数でそのモデルを作ったか」も分かると便利なので，
# この仕組みを使って同じ構成のモデルを再生成できるようにしている。
# ============================================================
import torch
import torch.nn as nn
import inspect
import functools


# =====================
# 設定引数を保存するデコレータ
# =====================
# __init__ に渡された引数を self.config に記録するために使う
def store_config_args(func):
    """
    クラスメソッド用のデコレータ。
    関数に渡された引数を辞書として self.config に保存する。
    これにより，あとで同じ設定のモデルを再構築しやすくなる。
    """
    # 関数の引数名やデフォルト値の情報を取り出す
    attrs, varargs, varkw, defaults = inspect.getargspec(func)

    # 元の関数情報を保ったままラッパー関数を作る
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # モデル設定を保存する辞書を初期化する
        self.config = {}

        # まずデフォルト引数を self.config に入れる
        if defaults:
            # 引数名とデフォルト値を対応づけて保存する
            for attr, val in zip(reversed(attrs), reversed(defaults)):
                self.config[attr] = val

        # 次に，位置引数として渡された値で上書きする
        # attrs[0] は通常 self なので飛ばしている
        for attr, val in zip(attrs[1:], args):
            self.config[attr] = val

        # 最後に，キーワード引数として渡された値で上書きする
        if kwargs:
            for attr, val in kwargs.items():
                self.config[attr] = val

        # 元の関数本体を実行する
        return func(self, *args, **kwargs)
    return wrapper


# =====================
# 保存・読込しやすいモデルの基底クラス
# =====================
# モデル構造の設定と重みをまとめて扱いやすくするためのクラス
class LoadableModel(nn.Module):
    """
    PyTorch モデルを読み込みやすくするための基底クラス。

    モデル生成時の引数を self.config に保存しておくことで，
    ファイルから読み込むときに同じ構成のモデルを自動で再生成できる。
    __init__ に @store_config_args を付けておくと，引数が自動保存される。
    """

    # このコンストラクタは，self.config が用意されているかを確認するためのもの
    # @store_config_args を付け忘れると，ロード時にモデル構成を再現できなくなる
    def __init__(self, *args, **kwargs):
        # self.config が無ければ，設定保存の仕組みが使われていないと判断する
        if not hasattr(self, 'config'):
            raise RuntimeError('LoadableModel を継承するモデルでは，コンストラクタに @store_config_args を付ける必要があります')
        super().__init__(*args, **kwargs)

    def save(self, path):
        """
        モデル設定と重みをまとめて PyTorch ファイルとして保存する。
        """
        # 現在のモデル重みを辞書として取得する
        sd = self.state_dict().copy()
        # .grid で終わる buffer 名を探す
        grid_buffers = [key for key in sd.keys() if key.endswith('.grid')]
        # 不要な grid buffer を state_dict から削除する
        # transformer の grid バッファは再生成できるので，保存対象から外す
        for key in grid_buffers:
            sd.pop(key)
        # モデル構成と重みを 1つのファイルにまとめて保存する
        torch.save({'config': self.config, 'model_state': sd}, path)

    @classmethod
    def load(cls, path, device):
        """
        保存済みファイルから，モデル構成と重みを読み込む。
        """
        # 保存済みチェックポイントを指定 device 上に読み込む
        checkpoint = torch.load(path, map_location=torch.device(device))
        # 保存しておいた config を使って，同じ構成のモデルを再生成する
        model = cls(**checkpoint['config'])
        # 保存済み重みをモデルに読み込む
        model.load_state_dict(checkpoint['model_state'], strict=False)
        # 復元したモデルを返す
        return model
