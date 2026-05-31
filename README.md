# get\_suisho5\_nn

[水匠5](https://drive.google.com/file/d/1T-Go2KImMfKD_4m_j4fQFXrEfaGgAcS_/)のソースコードに埋め込まれた評価関数テーブルをファイルに書き出します。

## Usage

```bash
(cd 水匠5; unrar e source.rar) && \
g++ -O2 get_suisho5_nn.cpp -o get_suisho5_nn && \
./get_suisho5_nn
```

suisho5_nn/nn.bin に書き出されます。

## 学習用ドキュメント / 使い方

水匠5評価関数(NNUE)の仕組み・最新やねうら王や8.60との組み合わせ方・推奨設定をまとめています。

- [`docs/水匠5評価関数_解説と使い方.md`](docs/水匠5評価関数_解説と使い方.md)
  … NNUEの仕組み、公式Suisho5とのSHA256一致検証、互換性、推奨USI設定、ベンチ実証
- [`docs/NNUEネット構造_比較.md`](docs/NNUEネット構造_比較.md)
  … 水匠5(256x2-32-32)と新型ネット(512x2/1024x2等)の構造・サイズ比較
- [`scripts/build_yaneuraou_for_suisho5.sh`](scripts/build_yaneuraou_for_suisho5.sh)
  … 最新やねうら王を標準NNUE(halfKP256)でビルドし nn.bin のロード確認まで自動化
- [`engine/`](engine/)
  … ビルド済みやねうら王（Linux/AVX512VNNI, 水匠5評価関数で動作確認済み）
- [`gpu/`](gpu/)
  … 水匠5評価関数を Python で完全再現し、GPU(RTX 5090)で大量局面をバッチ評価する実装
    （オリジナル水匠5 6.50 と評価値が合法局面 250/250 で厳密一致することを検証済み）

> 補足: 水匠5評価関数は **標準NNUE(halfKP256)ビルドのやねうら王**であれば、
> バージョン（8.60〜最新）に関係なく読み込めます（構造ハッシュは版数に依存しないため）。

## License

GPL v3.
