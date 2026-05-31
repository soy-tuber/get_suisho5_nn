# engine/ — ビルド済みやねうら王（水匠5評価関数用）

## YaneuraOu-NNUE-AVX512VNNI-linux

最新やねうら王を **標準NNUE(halfKP256)** 構成でビルドしたもので、
本リポジトリの `suisho5_nn/nn.bin`（水匠5評価関数）を読み込める実行ファイルです。

| 項目 | 内容 |
|---|---|
| 版数 | `YaneuraOu NNUE 9.30git 64AVX512VNNI` |
| エディション | `YANEURAOU_ENGINE_NNUE`（= halfKP256, 水匠5と構造一致） |
| ターゲットCPU | `AVX512VNNI`（`-march=cascadelake`） |
| OS / ABI | **Linux x86-64 のみ** |
| ビルダ | g++ 13.3 (LTO, -O3 -ffast-math) |

### ⚠️ 注意
- **Linux 専用**です。Windows では動きません（Windowsはやねうら王公式配布のexe、
  または `scripts/build_yaneuraou_for_suisho5.sh` をWSL/MSYS等でビルドして使用）。
- **AVX512-VNNI 必須**。非対応CPUでは起動しません。その場合は
  `scripts/build_yaneuraou_for_suisho5.sh` を `TARGET_CPU=AVX2` 等で再ビルドしてください。

### 使い方
```bash
# 同じディレクトリ構成で eval/nn.bin を用意して起動
mkdir -p eval && cp ../suisho5_nn/nn.bin eval/nn.bin
printf 'usi\nisready\nquit\n' | ./YaneuraOu-NNUE-AVX512VNNI-linux

# ベンチ（水匠5評価関数で実探索）
printf 'bench\n' | ./YaneuraOu-NNUE-AVX512VNNI-linux
```

### 再ビルド
`scripts/build_yaneuraou_for_suisho5.sh` を実行すると、最新やねうら王を取得して
同構成でビルドし、`nn.bin` のロード確認まで自動で行います。
