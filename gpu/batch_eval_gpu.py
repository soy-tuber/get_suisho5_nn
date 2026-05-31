#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_eval_gpu.py — 水匠5評価関数で「大量の局面」をGPU(RTX 5090)で一括評価するデモ。

  # 自己検証（torch版がnumpy版と一致するか。GPUでもCPUでも可）
  python3 gpu/batch_eval_gpu.py --selfcheck --device cpu  suisho5_nn/nn.bin
  python3 gpu/batch_eval_gpu.py --selfcheck --device cuda suisho5_nn/nn.bin

  # スループット計測（RTX 5090 で大量バッチを評価）
  python3 gpu/batch_eval_gpu.py --bench --device cuda --n 1000000 suisho5_nn/nn.bin

  # SFENファイル(1行1局面)を一括評価して評価値を出力
  python3 gpu/batch_eval_gpu.py --device cuda --infile positions.sfen suisho5_nn/nn.bin

設計思想:
  NNUE+αβ探索の「対局」は1局面ずつの評価でCPU向き（GPUはレイテンシで不利）。
  GPUが効くのは「何百万局面もまとめて評価する」バッチ用途。RTX 5090 の出番はこちら。
  FeatureTransformer を疎集約(embedding sum)、後段を行列積でバッチ処理する。
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suisho5_eval import Suisho5Net, build_torch_evaluator


def gen_legal_sfens(n, seed=0):
    import random, shogi
    random.seed(seed)
    out = []
    while len(out) < n:
        b = shogi.Board()
        for _ in range(random.randint(0, 80)):
            mv = list(b.legal_moves)
            if not mv or b.is_game_over():
                break
            b.push(random.choice(mv))
        out.append(b.sfen())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nnbin")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--infile")
    args = ap.parse_args()

    net = Suisho5Net(args.nnbin)
    evaluate = build_torch_evaluator(net, device=args.device)

    if args.selfcheck:
        sfens = gen_legal_sfens(2000, seed=7)
        gpu = evaluate(sfens)
        ng = sum(1 for s, g in zip(sfens, gpu) if int(g) != net.eval_numpy(s))
        print(f"[selfcheck/{args.device}] torch vs numpy: 一致 {len(sfens)-ng}/{len(sfens)}  不一致 {ng}")
        sys.exit(1 if ng else 0)

    if args.infile:
        sfens = [l.strip() for l in open(args.infile) if l.strip()]
        scores = []
        for i in range(0, len(sfens), args.batch):
            scores.extend(evaluate(sfens[i:i + args.batch]).tolist())
        for s, v in zip(sfens, scores):
            print(v, s)
        return

    if args.bench:
        sfens = gen_legal_sfens(min(args.n, 50000), seed=1)
        # 必要数までタイル
        sfens = (sfens * (args.n // len(sfens) + 1))[:args.n]
        t0 = time.time()
        done = 0
        for i in range(0, len(sfens), args.batch):
            evaluate(sfens[i:i + args.batch]); done += min(args.batch, len(sfens) - i)
        dt = time.time() - t0
        print(f"[bench/{args.device}] {done} 局面を {dt:.2f}s  =>  {done/dt:,.0f} pos/s")


if __name__ == "__main__":
    main()
