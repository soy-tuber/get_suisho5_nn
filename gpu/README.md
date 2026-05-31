# gpu/ — 水匠5評価関数を Python / GPU(RTX 5090) で動かす

水匠5の評価関数（NNUE, HalfKP 256x2-32-32）を **Python で完全再現**し、
RTX 5090 などの GPU で **大量局面をバッチ評価**するための実装です。

> **棋力度外視の前提について**
> NNUE+αβ探索の「対局」は1局面ずつ評価するため CPU 向きで、GPU はカーネル起動
> レイテンシでむしろ不利です（対局エンジンの GPU 化は無意味）。
> GPU が真価を発揮するのは **何百万局面もまとめて評価するバッチ用途**——
> 局面データセットのラベリング、評価値ヒートマップ、解析の前処理など。
> RTX 5090 の出番はこちらで、ここではその形に最適化しています。

## ファイル

| ファイル | 役割 |
|---|---|
| `suisho5_eval.py` | nn.bin パーサ＋SFEN→HalfKP特徴量＋前進計算（numpy参照実装 / torch GPU実装） |
| `verify_against_engine.py` | 本物の水匠5エンジンの評価値と総当たり照合（正当性検証） |
| `batch_eval_gpu.py` | GPUバッチ評価のデモ（自己検証 / スループット計測 / SFEN一括評価） |

## 正当性（検証済み）

本実装(numpy)は **オリジナル水匠5 6.50（FV_SCALE=24）と評価値が完全一致**することを確認済み：

```
ランダム合法局面 250/250 で厳密一致（手番b/w・中終盤・持ち駒・成駒すべて含む）
```

実装は水匠5のC++ソース（旧nodchipフォーマット, kVersion=0x7AF32F16）に厳密準拠：

- **FeatureTransformer**: bias int16[256] → weight int16[125388][256]（行=特徴量）
  `acc[色] = bias + Σ weight[活性特徴]`、出力 = `clamp(acc, 0, 127)`
- **入力512次元** = `[手番側256, 相手側256]`
- **AffineTransform**（512→32→32→1）: bias int32 → weight int8[out][padded_in] 行優先
- **量子化**: 層間ClippedReLU = `clamp(z >> 6, 0, 127)`、最終 `score = trunc(z / 24)`
- 評価値は **手番側から見た値**

> 注: 盤上＋持ち駒で歩が19枚等の「不正局面」では、エンジンの駒リスト(eval_list)が
> 溢れるため一致しません。これは実装の誤りではなく、合法局面では完全一致します。

## 使い方

### 0. 準備
```bash
pip install numpy            # 必須（numpy版の評価・検証）
pip install python-shogi     # 検証/局面生成に使用（任意）
pip install torch            # GPUバッチ評価に使用（5090なら CUDA 版）
```

### 1. 単一局面の評価（numpy, CPU）
```bash
python3 gpu/suisho5_eval.py suisho5_nn/nn.bin
# => architecture や startpos の評価値を表示
```
```python
from gpu.suisho5_eval import Suisho5Net
net = Suisho5Net("suisho5_nn/nn.bin")
print(net.eval_numpy("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"))
```

### 2. 実エンジンと照合（正当性検証）
```bash
# 水匠5(halfKP256)対応エンジンを用意して:
python3 gpu/verify_against_engine.py <engine> suisho5_nn/nn.bin 250
```

### 3. GPU でバッチ評価（RTX 5090）
```bash
# まず torch版が numpy版と一致するか自己検証（GPUを信頼する前に）
python3 gpu/batch_eval_gpu.py --selfcheck --device cuda suisho5_nn/nn.bin

# スループット計測（100万局面）
python3 gpu/batch_eval_gpu.py --bench --device cuda --n 1000000 suisho5_nn/nn.bin

# SFEN一括評価（1行1局面）
python3 gpu/batch_eval_gpu.py --device cuda --infile positions.sfen suisho5_nn/nn.bin
```

## 実装メモ（GPU側の数値の正しさ）

- FeatureTransformer は `torch.nn.functional.embedding(...).sum()`（疎な重み集約）。
- 後段は float32 行列積。入力0..127・重みint8で、層内総和の最大は ~8.3M < 2²⁴ なので
  **float32 で厳密**（丸め誤差なし）。`>>6` は `floor(x/64)`、最終は `trunc(x/24)` で整数一致。
- そのため torch(GPU) 版は numpy 参照版と同一結果になる（`--selfcheck` で確認可能）。
- このコンテナにはGPUが無いため、GPU実機での実行は 5090 側で `--device cuda` を指定して
  行ってください（torch CPU でも `--device cpu` で動作・自己検証可能）。

## ライセンス
水匠5は GPL v3。本実装も同ライセンスの範囲で学習・解析目的に扱います。
