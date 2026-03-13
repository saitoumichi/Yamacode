# Yamacode (Medical Image Registration)

基礎は VoxelMorph による 3D 医療画像登録 (非線形変形) の PyTorch 実装です。論文: https://ieeexplore.ieee.org/document/11406103

## 目的

- CT / MRI などの 3D ボリューム画像 (D,H,W) に対して、学習ベースの画像登録を実施.
- VxmDense (UNet ベース) による移動画像 (moving) -> 固定画像 (fixed) の変形場推定.
- 変形場のスケーリング・積分 (VecInt) により準可逆 (diffeomorphic) 変形を実現.
- 128/256 解像度データを使う実験スクリプトを含む.

## リポジトリ構成

- `torch/` : PyTorch 実装
  - `network.py` : UNet + VxmDense (Voxelmorph) + ConvBlock
  - `layers.py` : SpatialTransformer, VecInt, ResizeTransform など変形演算
  - `losses.py` : NCC, MSE, Dice, Grad (画像類似度 + スムージング)
  - `modelio.py` : LoadableModel, state_dict 付き保存/ロード
  - `utils.py` : （今は torch だけインポートで拡張余地あり）

- `transformer/` : 同等の UNet/registration 実装 (兼用)
  - `networks.py`, `layers.py`, `losses.py`, `utils.py`

- `voxelmorph/` : Voxelmorph API サブパッケージ
  - `__init__.py`, `generators.py` (既存 Voxelmorph 互換)

- `essay/` : 学習・検証実験スクリプト
  - `128model_Train.py` : 学習ループ、データジェネレータ、loss 定義、保存
  - `128model_Test.py` : 推論・評価用 SSIM/FSIM/NCC/DSC 等メトリクス
  - `256_128model_Train.py`, `256_128model_Test.py` : 解像度拡張版

## 主要コンポーネント

### UNet (torch/network.py)

- `Unet`:
  - `source` + `target` を ch 方向結合して入力
  - 4 層程度のエンコーダ/デコーダ構造
  - `nb_features` を指定可能: list 形式または scalar+levels
  - `half_res` で decoder の最後のアップサンプリングをスキップ

### VxmDense (torch/network.py)

- `VxmDense`:
  - `unet_model` により SVF 予測
  - 3D Conv で `flow_field` 生成
  - `ResizeTransform` + `VecInt` でスケールと積分 (nonlinear) を実現
  - `SpatialTransformer` で warp を適用
  - bidirectional `bidir` オプション対応
  - `registration=False` なら中間 flow (preint) を返す

### 変形演算 (transformer/layers.py)

- `SpatialTransformer`: サンプリンググリッド + `grid_sample`
- `VecInt`: Scaling-and-squaring ベースの velocity integration
- `ResizeTransform`: flow のリサイズとスケーリング

### 損失 (torch/losses.py)

- `NCC`, `MSE`, `Dice`, `Grad`
- `Grad` で空間勾配ペナルティ (smooth系) あり

### モデル保存/読み込み (torch/modelio.py)

- `LoadableModel.save(path)` : config + weights を保持
- `LoadableModel.load(path, device)` : config からモデル復元

## 使い方

### 1) 依存環境

```bash
pip install torch torchvision numpy scipy scikit-image pywt tabulate piq neurite
pip install matplotlib tqdm
```

> 注意: 本リポジトリは `voxelmorph` の内部にある `torch` / `transformer` 実装を直接参照します。
> そのため、pip インストール済みの `voxelmorph` とバージョン競合が出ないように環境を整えるか、仮想環境を推奨。

### 2) 学習

`essay/128model_Train.py` を実行すると、 `a.pth`、`a2.pth` などにチェックポイント保存されます。

- 入力データは NumPy `.npz` 形式 (例: `TrainData_NoBed.npz`) で読み込み
- `vxm_data_generator` で moving/fixed ペアをランダム生成
- `loss = MSE(moving
eval, fixed) + 0.01 * Grad(flow)` のような形式

### 3) 評価

`essay/128model_Test.py` を実行して `a2.pth` をロードし、
- RMSE, NCC, MS-SSIM, FSIM 等を計算
- セントラルスライス、ボリューム全体比較

## 論文と理論背景

- リンク: https://ieeexplore.ieee.org/document/11406103
- 3D 医療画像登録における深層学習アプローチ (VoxelMorph 系)
- 論文のメインアイデア: 速度場予測+スケーリングアンドスコアリングで可逆性保証

## 実装メモ

- `torch/network.py` の `print` でデバグ出力が多数あり、実運用では削除推奨
- `Transformer` で `grid` が `register_buffer` されるため、モデル保存時に grid が state_dict に含まれないよう `modelio` が除去
- `VxmDense` の `forward` 呼び出し中の `pos_flow.shape`/`source.shape` 等を出力しているため、デバッグ用途向け

## 拡張案

- `voxelmorph/generators.py` にデータ拡張追加 (回転、スケール、歪曲)
- infer サンプルとして `runner.py` を作成し、コマンドラインパラメタを整理
- `torch/losses.py` に LNCC、MI など追加

## クレジット

- このコードは scipy/tensorflow 版 Voxelmorph をベースに独自改変 (torch 移植)
- `torch` と `transformer` の融合で 1 ステップ登録計算を実現

