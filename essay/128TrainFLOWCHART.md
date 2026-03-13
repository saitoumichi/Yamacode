# Yamacode フローチャート

以下は `essay/128model_Train.py` を中心とした主要な処理フローです。

```mermaid
flowchart TD
    A[Start: main() 実行] --> B[load_training_data(DATA_PATH)]
    B --> B1[np.load で TrainData_NoBed.npz を読み込む]
    B1 --> B2[Train キーを取得<br/>shape 例: (H,W,D,N)]
    B2 --> B3[np.transpose で軸順変更<br/>shape: (H,W,D,N) → (N,H,W,D)]
    B3 --> B4[各症例を scipy.ndimage.zoom で 0.5 倍に縮小<br/>shape: (N,H,W,D) → (N,H/2,W/2,D/2)]
    B4 --> B5[前処理後データ x_train を返す<br/>最終 shape 例: (N,64,128,128)]

    B5 --> C[vxm_data_generator(x_train, BATCH_SIZE) を作成]
    C --> C1[vol_shape = x_data.shape[1:]<br/>ndims = len(vol_shape)]
    C1 --> C2[zero_phi を作成<br/>shape: (B,D,H,W,ndims)]
    C2 --> D[サンプル1バッチを next() で取得して shape 確認]
    D --> D1[in_sample[0] = moving_images<br/>shape: (B,1,D,H,W)]
    D1 --> D2[in_sample[1] = fixed_images<br/>shape: (B,1,D,H,W)]
    D2 --> D3[out_sample[0] = fixed_images<br/>shape: (B,1,D,H,W)]
    D3 --> D4[out_sample[1] = zero_phi<br/>shape: (B,D,H,W,3)]

    D4 --> E[build_model(INPUT_SHAPE)]
    E --> E1[VxmDense1 を構築<br/>入力 shape: (B,1,64,128,128) × 2]
    E1 --> E2[model.to(device)]
    E2 --> E3[Adam optimizer を作成]
    E3 --> F[SpatialTransformer(INPUT_SHAPE) を作成]

    F --> G[pretrain_with_synthetic_dvf 開始]

    subgraph PretrainLoop[1段階目: 人工DVFによる事前学習]
      G1[for epoch in range(PRETRAIN_EPOCHS)] --> G2[一定間隔ごとに shift_range を増やす<br/>easy-to-hard の curriculum learning]
      G2 --> G3[generator から moving_images を取得<br/>shape: (B,1,64,128,128)]
      G3 --> G4[sample_synthetic_dvf 呼び出し]
      G4 --> G41[粗い解像度でランダムDVF生成<br/>shape: (B,3,8,16,16)]
      G41 --> G42[gaussian_smooth_3d で空間方向のみ平滑化<br/>shape 不変: (B,3,8,16,16)]
      G42 --> G43[trilinear interpolate で INPUT_SHAPE へ拡大<br/>shape: (B,3,64,128,128)]
      G43 --> G5[人工DVF = displacement_field を得る]
      G5 --> G6[transformer で moving_images を warping]
      G6 --> G7[擬似fixed画像 moving_images2 を作成<br/>shape: (B,1,64,128,128)]
      G7 --> G8[model(moving_images, moving_images2)]
      G8 --> G81[出力1: transformed_image<br/>shape: (B,1,64,128,128)]
      G8 --> G82[出力2: vec 推定DVF<br/>shape: (B,3,64,128,128)]
      G81 --> G9[loss_image = MSE(moving_images2, transformed_image)]
      G82 --> G10[loss_vec = MSE(displacement_field, vec)]
      G9 --> G11[loss = loss_image × IMAGE_LOSS_WEIGHT + loss_vec × VEC_LOSS_WEIGHT]
      G10 --> G11
      G11 --> G12[loss.backward()]
      G12 --> G13[optimizer.step()]
      G13 --> G14[loss 履歴を保存]
      G14 --> G15[一定間隔で plot_losses]
      G15 --> G16[一定間隔で a.pth を保存]
      G16 --> G1
    end

    G --> H[build_model(INPUT_SHAPE) でモデル再作成]
    H --> H1[a.pth を load_state_dict で読み込む]
    H1 --> I[finetune_registration 開始]

    subgraph FinetuneLoop[2段階目: 通常の位置合わせ学習]
      I1[for epoch in range(FINETUNE_EPOCHS)] --> I2[generator から moving_images, fixed_images を取得<br/>shape: どちらも (B,1,64,128,128)]
      I2 --> I3[model(moving_images, fixed_images)]
      I3 --> I31[出力1: transformed_image<br/>shape: (B,1,64,128,128)]
      I3 --> I32[出力2: 推定DVF<br/>shape: (B,3,64,128,128)]
      I31 --> I4[loss = MSE(fixed_images, transformed_image)]
      I4 --> I5[loss.backward()]
      I5 --> I6[optimizer.step()]
      I6 --> I7[一定間隔で a2.pth を保存]
      I7 --> I1
    end

    I --> J[End]

    %% generator の中身
    subgraph GeneratorDetail[vxm_data_generator の詳細]
      C3[idx1, idx2 をランダム生成<br/>shape: (B,)] --> C4[moving 用症例と fixed 用症例を選ぶ<br/>x_data[idx, ...] の shape: (B,D,H,W)]
      C4 --> C5[各画像にチャネル次元を追加<br/>shape: (B,D,H,W) → (B,D,H,W,1)]
      C5 --> C6[torch.from_numpy(...).permute(...).float()<br/>shape: (B,D,H,W,1) → (B,1,D,H,W)]
      C6 --> C7[inputs = [moving_images, fixed_images]]
      C7 --> C8[outputs = [fixed_images, zero_phi]]
    end

    C --> C3

    %% モデル内部
    subgraph ModelInternal[VxmDense1 の内部イメージ]
      M1[source と target をチャネル方向で結合<br/>shape: (B,1,D,H,W)+(B,1,D,H,W) → (B,2,D,H,W)] --> M2[U-Net 系 encoder-decoder で特徴抽出]
      M2 --> M3[flow field 変位ベクトル場を推定<br/>shape: (B,3,D,H,W)]
      M3 --> M4[int_steps=0 のため積分なし]
      M4 --> M5[SpatialTransformer で source を変形]
      M5 --> M6[出力: transformed_image shape (B,1,D,H,W)<br/>flow shape (B,3,D,H,W)]
    end

    E1 --> M1
    G8 --> M6
    I3 --> M6
```

## フローチャートの詳しい説明

このフローチャートは、`essay/128model_Train.py` が **どの順番で処理を進めているか** を、日本語で細かく整理したものです。大きく分けると、次の4段階で進みます。

1. **学習データを読み込んで前処理する**  
2. **generator で moving / fixed 画像のバッチを作る**  
3. **人工DVFを使った事前学習を行う**  
4. **事前学習済みモデルを使って通常の位置合わせ学習を行う**

以下、それぞれを順番に説明します。

---

### 1. `main()` が最初に呼ばれる

コードは `main()` から始まります。ここが全体の入口です。

`main()` の中では、主に次のことを順番にしています。

- 学習データを読み込む
- generator を作る
- サンプル1バッチを取り出して shape を確認する
- 1段階目の事前学習用モデルを作る
- 人工DVFを使った事前学習を行う
- その後、モデルを作り直して重みを読み込み、2段階目の学習を行う

つまり `main()` は、**このコード全体の進行管理をしている部分**です。

---

### 2. `load_training_data(DATA_PATH)` でデータを読み込む

最初に `load_training_data(DATA_PATH)` が呼ばれます。ここでは、学習に使う胸部CTデータを `.npz` ファイルから読み出しています。

#### ここでやっていること

まず、

- `np.load(...)` でファイルを開く
- その中の `Train` キーのデータを取り出す

という処理をしています。

この時点では、データの軸順はまだ学習しやすい形ではない可能性があります。そこで次に、

- `np.transpose(x_train, (3, 0, 1, 2))`

を使って軸の順番を変えています。

#### 軸順を変える理由

元データは例えば `(H, W, D, N)` のように、症例番号 `N` が最後にある形かもしれません。これだと、

- 1番目の症例
- 2番目の症例
- 3番目の症例

というふうにデータを取り出しづらいです。

そこで `(N, H, W, D)` に並べ替えることで、**最初の次元が「何番目の症例か」** という扱いやすい形になります。

---

### 3. 各3D画像を半分に縮小する

軸順を変えたあと、各症例の3D画像を `scipy.ndimage.zoom` で 0.5 倍に縮小しています。

これは、

- 元の3D CT画像はサイズが大きい
- そのままだとGPUメモリを多く使う
- 学習時間も長くなる

という問題があるためです。

つまり、この縮小処理は **計算しやすくするための前処理** です。

#### ここでの shape の変化

例えば元が `(N, H, W, D)` だったとすると、縮小後は

- `(N, H/2, W/2, D/2)`

のようになります。

このコードでは、最終的に `INPUT_SHAPE = (64, 128, 128)` に合わせて学習する想定になっているので、前処理後のデータ shape もその前提に揃っていきます。

---

### 4. `vxm_data_generator(...)` で学習用バッチを作る

次に `vxm_data_generator(x_train, BATCH_SIZE)` を作っています。これは **学習のたびに moving 画像と fixed 画像をランダムに作る関数** です。

#### generator の役割

毎回すべてのデータを一気にモデルへ入れるのではなく、少しずつミニバッチとして取り出して学習するために generator を使っています。

この generator の中では、

- `idx1` と `idx2` をランダムに作る
- `idx1` で moving 用の症例を選ぶ
- `idx2` で fixed 用の症例を選ぶ

という流れになっています。

つまり、moving と fixed は **それぞれ独立にランダム選択** されています。

#### チャネル次元を追加する理由

CT画像そのものは1チャネルのグレースケール画像です。しかし PyTorch の3D CNNでは、通常

- `(B, C, D, H, W)`

の順で tensor を持ちます。

そのため generator の中で、

- `np.newaxis`

を使ってチャネル次元を追加し、

- `(B, D, H, W)` → `(B, D, H, W, 1)`

にしてから、さらに `permute(...)` で

- `(B, 1, D, H, W)`

の形へ変換しています。

これで PyTorch の3D CNNにそのまま入れられる形になります。

---

### 5. `zero_phi` を作っている理由

generator の中では `zero_phi` も作っています。これは **ゼロの変形場** です。

shape が `(B, D, H, W, 3)` になっているのは、各ボクセルに対して

- x方向の変位
- y方向の変位
- z方向の変位

の3つを持つからです。

つまり、`3` は **3次元空間でどちらにどれだけ動くか** を表す成分数です。

このコードでは、自前の学習ループの中で `outputs` はほぼ直接使っていませんが、VoxelMorph の一般的な形式に合わせて残してあります。

---

### 6. サンプル1バッチを取り出して shape を確認する

`main()` の中では、学習を始める前に一度 `next(train_generator)` を呼んで、サンプル1バッチを取得しています。

ここで確認しているのは、

- moving_images の shape
- fixed_images の shape
- zero_phi の shape

です。

これは **「データの形が想定通りか」を確かめるための確認作業** です。

深層学習では、shape が少しでもずれるとその後すぐエラーになるので、この確認はかなり大事です。

---

### 7. `build_model(INPUT_SHAPE)` で VoxelMorph モデルを作る

次に `build_model(INPUT_SHAPE)` でモデルを作成します。

ここで使っているのは、

- `vxm.networks.VxmDense1(...)`

です。これは VoxelMorph のネットワーク本体です。

#### このモデルがしていること

入力として

- moving 画像
- fixed 画像

の2枚を受け取り、

- moving を fixed に近づけるための DVF（変位ベクトル場）
- そのDVFで変形した後の画像

を出力します。

つまりこのモデルは、**「どのように画像を変形すれば合うか」を学ぶネットワーク** です。

#### `model.to(device)` の意味

モデルを CPU ではなく GPU で計算したい場合、モデル自体を GPU に乗せる必要があります。そのために `model.to(device)` をしています。

---

### 8. `SpatialTransformer(INPUT_SHAPE)` を作る

次に `SpatialTransformer` を作っています。これは **DVF を使って実際に画像を変形する層** です。

たとえば、

- moving 画像
- displacement field（DVF）

があれば、SpatialTransformer は

- 「この位置の画素をどこへ動かすか」

を計算して、変形後画像を作れます。

このコードでは特に、**人工DVFで moving 画像を変形し、擬似的な教師データを作るため** に使っています。

---

### 9. 1段階目：人工DVFを使った事前学習を行う

ここから `pretrain_with_synthetic_dvf(...)` に入ります。これが **1段階目の学習** です。

この段階の目的は、モデルにいきなり本物の難しい位置合わせをさせるのではなく、まずは

- 人工的に作った変形
- 人工的に作った正解DVF

を使って、**「変形の当て方」を先に覚えさせること** です。

---

### 10. `shift_range` を徐々に増やす

事前学習ループの中では、一定間隔ごとに `shift_range` を増やしています。

これは curriculum learning の考え方です。

#### なぜ増やすのか

最初から大きく複雑な変形だけを学習させると、モデルは不安定になりやすいです。そこでまずは

- 小さい変形

から始めて、慣れてきたら

- 大きい変形

も扱うようにしています。

つまりこれは、**簡単な課題から難しい課題へ進める学習法** です。

---

### 11. `sample_synthetic_dvf(...)` で人工DVFを作る

事前学習では、正解として使う人工DVFを `sample_synthetic_dvf(...)` で作っています。

#### 作り方の流れ

まず最初に、粗い解像度 `(B, 3, 8, 16, 16)` でランダムなDVFを作ります。

ここで `3` になっているのは、

- x方向
- y方向
- z方向

の3成分を持つからです。

そのあと、`gaussian_smooth_3d(...)` で空間方向だけ平滑化します。これをしないと、DVF がガタガタで不自然な形になりやすいです。

さらに `interpolate(...)` を使って `(B, 3, 64, 128, 128)` に拡大します。こうして、最終的に入力画像と同じサイズのDVFになります。

---

### 12. 人工DVFで moving 画像を変形し、擬似 fixed を作る

作った人工DVFを `transformer` に渡すと、moving 画像を変形できます。

その結果できるのが `moving_images2` です。

これは、本来の fixed 画像ではなく、**人工DVFによって moving から作った「擬似 fixed」** です。

この段階では、

- 元画像 `moving_images`
- 人工的に変形した画像 `moving_images2`

の対応関係が分かっています。また、

- どのDVFで変形したか

も分かっています。

つまり、**画像とDVFの両方に正解がある状態** で学習できます。

---

### 13. モデルに `(moving_images, moving_images2)` を入れる

次にモデルへ

- moving_images
- moving_images2

を入力します。

するとモデルは、

- `transformed_image`
- `vec`

を出します。

#### それぞれの意味

- `transformed_image`  
  → モデルが「moving をこう変形すれば target に近い」と考えて作った変形後画像

- `vec`  
  → モデルが推定したDVF

ここでの target は `moving_images2` です。つまりモデルは、**元の moving を人工的に作った target に近づける変形** を学んでいます。

---

### 14. 事前学習で計算している loss

事前学習では、2種類の損失を計算しています。

#### `loss_image`

これは

- `moving_images2`
- `transformed_image`

の MSE です。

意味としては、**モデルが作った変形後画像が、人工的に作った正解画像にどれだけ近いか** を見ています。

#### `loss_vec`

これは

- 人工正解DVF `displacement_field`
- モデルの推定DVF `vec`

の MSE です。

意味としては、**モデルが予測した変形場そのものが、正解DVFにどれだけ近いか** を見ています。

#### 総損失

最後に、

- `loss_image × IMAGE_LOSS_WEIGHT`
- `loss_vec × VEC_LOSS_WEIGHT`

を足して総損失にしています。

つまりこの段階では、**画像の一致** と **DVFそのものの一致** の両方を見ながら学習しています。

---

### 15. `loss.backward()` と `optimizer.step()`

総損失を計算したあとは、

- `loss.backward()`
- `optimizer.step()`

を行います。

#### それぞれの意味

- `loss.backward()`  
  → 誤差を各パラメータに逆向きに伝えて、どの重みをどれだけ直すべきかを計算する

- `optimizer.step()`  
  → 実際に重みを更新する

これが1回の学習更新です。

---

### 16. 途中で loss を保存・表示・モデル保存する

事前学習中は、

- loss の履歴をリストに保存する
- 一定間隔で `plot_losses(...)` でグラフ表示する
- 一定間隔で `a.pth` に重みを保存する

という処理もしています。

これは、**ちゃんと学習が進んでいるかを確認するため** です。

---

### 17. 2段階目の前にモデルを作り直して重みを読み込む

1段階目が終わったら、もう一度 `build_model(INPUT_SHAPE)` でモデルを作り直しています。

そのあと、

- `model.load_state_dict(torch.load('a.pth', ...))`

で、1段階目で保存した重みを読み込んでいます。

つまり2段階目は、**まっさらなモデルからではなく、事前学習済みのモデルから始める** ということです。

---

### 18. 2段階目：通常の位置合わせ学習を行う

次に `finetune_registration(...)` に入ります。これが **2段階目の学習** です。

この段階では、1段階目と違って人工DVFは作りません。

代わりに generator からそのまま

- moving_images
- fixed_images

を取り出して、モデルに入力します。

ここでモデルは、

- moving を fixed に近づける変形後画像
- そのための推定DVF

を出力します。

---

### 19. 2段階目では画像の一致だけで学習する

2段階目の loss は、

- `MSE(fixed_images, transformed_image)`

だけです。

つまりここでは、**DVFそのものの正解は使っていません**。見ているのは、あくまで

- 変形後画像が fixed にどれだけ近いか

だけです。

これは、実際の画像位置合わせでよくある **非教師あり registration に近い学習** です。

---

### 20. 2段階目でも逆伝播と保存を繰り返す

2段階目でも、

- `loss.backward()`
- `optimizer.step()`

で学習し、一定間隔で

- `a2.pth` に保存
- loss を表示

しています。

こうして、最終的に **事前学習で変形の基礎を覚えたモデルを、実際の registration に適応させる** 流れになっています。

---

### 21. このコード全体を一言でまとめると

このコードは、

- まず人工的に正解DVFを作って「変形の仕方」を学ばせる
- その後、実際の moving / fixed 画像どうしで位置合わせ学習をする

という **2段階構成の VoxelMorph 学習コード** です。

1段階目では教師ありに近い形でDVFも学び、2段階目では画像一致を中心に微調整するので、

- 最初から難しい registration を直接学習するより安定しやすい
- 大きな変形にもある程度強くなりやすい

という狙いがあります。

---

### 22. このフローチャートを見るときのポイント

この図を見るときは、次の3つを意識すると読みやすいです。

#### ① 画像とDVFを区別する

- 画像はだいたい `(B, 1, D, H, W)`
- DVF はだいたい `(B, 3, D, H, W)`

です。

#### ② 1段階目と2段階目を分けて考える

- 1段階目 → 人工DVFあり
- 2段階目 → 人工DVFなし

です。

#### ③ `transformed_image` と `vec` の役割を意識する

- `transformed_image` は「変形後画像」
- `vec` は「変形ベクトル場」

です。

この2つをセットで見ると、モデルが何を出しているかが分かりやすくなります。

### shape の見方
- **B**: バッチサイズ
- **D, H, W**: 3次元画像の奥行き・高さ・幅
- **1**: CT画像のチャネル数（グレースケール画像1枚分）
- **3**: DVF の x, y, z 方向の3成分
- **zero_phi の shape が (B,D,H,W,3)** なのは、各ボクセルごとに3方向の変位を持つため


## shape 変化の一覧表

| 処理段階 | 変数 | shape | 意味 |
|---|---|---|---|
| 元データ読込直後 | `x_train` | `(H, W, D, N)` | npz 内に保存されていた元の並び |
| 軸順変更後 | `x_train` | `(N, H, W, D)` | 症例番号を先頭にした並び |
| 縮小後 | `x_train_resized` | `(N, H/2, W/2, D/2)` | 各症例を 0.5 倍にリサイズした後 |
| generator 内で症例抽出後 | `x_data[idx1, ...]` | `(B, D, H, W)` | moving 用に選ばれたミニバッチ |
| generator 内でチャネル追加後 | `x_data[idx1, ..., np.newaxis]` | `(B, D, H, W, 1)` | CT画像にチャネル次元を追加 |
| generator 出力後 | `moving_images` | `(B, 1, D, H, W)` | PyTorch 用の moving 画像 |
| generator 出力後 | `fixed_images` | `(B, 1, D, H, W)` | PyTorch 用の fixed 画像 |
| generator 出力後 | `zero_phi` | `(B, D, H, W, 3)` | ゼロ変形場 |
| 事前学習: 人工DVF生成直後 | `displacement_field` | `(B, 3, 8, 16, 16)` | 粗い解像度で作ったランダムDVF |
| 事前学習: 平滑化後 | `displacement_field` | `(B, 3, 8, 16, 16)` | shape はそのままで中身だけ滑らかになる |
| 事前学習: 拡大後 | `displacement_field` | `(B, 3, 64, 128, 128)` | 入力画像サイズに合わせたDVF |
| 事前学習: warping 後 | `moving_images2` | `(B, 1, 64, 128, 128)` | 人工DVFで変形した擬似 fixed |
| モデル入力時 | `moving_images`, `moving_images2` | どちらも `(B, 1, 64, 128, 128)` | VoxelMorph に入る2枚の画像 |
| モデル内部結合後 | `source + target` | `(B, 2, 64, 128, 128)` | チャネル方向に2枚を連結した入力 |
| モデル出力 | `transformed_image` | `(B, 1, 64, 128, 128)` | 変形後画像 |
| モデル出力 | `vec` / `flow` | `(B, 3, 64, 128, 128)` | 推定されたDVF |
| 本学習入力 | `moving_images`, `fixed_images` | どちらも `(B, 1, 64, 128, 128)` | 通常の位置合わせに使う入力 |
| 本学習出力 | `transformed_image` | `(B, 1, 64, 128, 128)` | fixed に近づけたい出力画像 |
| 本学習出力 | `推定DVF` | `(B, 3, 64, 128, 128)` | 変形ベクトル場 |


### 見るときのコツ
- **画像**はだいたい `(B, 1, D, H, W)` で流れる。
- **DVF**はだいたい `(B, 3, D, H, W)` で流れる。
- **画像の 1** はグレースケール1チャネル、**DVF の 3** は x・y・z 方向の3成分を表す。