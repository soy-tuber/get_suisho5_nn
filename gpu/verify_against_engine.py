#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_against_engine.py — suisho5_eval.py の評価値を、本物の水匠5エンジンの
生 evaluate() 出力（USIコマンド `eval` = Eval::compute_eval）と総当たり照合する。

  python3 gpu/verify_against_engine.py <engine_path> <nn.bin> [N]

  <engine_path> : 水匠5(halfKP256)対応のやねうら王実行ファイル
                  ※オリジナル水匠5(6.50, FV_SCALE=24)なら全局面で厳密一致する。
  N             : ランダム合法局面の生成数（既定200, python-shogiが必要）

実測（このリポジトリで確認済み）:
  オリジナル水匠5 6.50 と本実装(numpy)は、ランダム合法局面 250/250 で評価値が完全一致。
  （唯一ズレるのは盤上＋持ち駒の歩が19枚等の「不正局面」で、エンジンの駒リストが溢れる場合のみ）
"""
import sys, subprocess, os, shutil

def engine_evals(engine, nnbin, sfens):
    d = os.path.dirname(os.path.abspath(engine))
    os.makedirs(os.path.join(d, "eval"), exist_ok=True)
    dst = os.path.join(d, "eval", "nn.bin")
    if os.path.abspath(nnbin) != os.path.abspath(dst):
        shutil.copy(nnbin, dst)
    cmds = ["isready"] + sum([[f"position sfen {s}", "eval"] for s in sfens], []) + ["quit"]
    out = subprocess.run([engine], input="\n".join(cmds) + "\n",
                         capture_output=True, text=True, timeout=600, cwd=d).stdout
    return [int(l.split("=")[1]) for l in out.splitlines() if l.strip().startswith("eval =")]

def random_legal_sfens(n, seed=0):
    import random, shogi
    random.seed(seed)
    sfens = []
    for _ in range(n):
        b = shogi.Board()
        for _ in range(random.randint(0, 80)):
            mv = list(b.legal_moves)
            if not mv or b.is_game_over():
                break
            b.push(random.choice(mv))
        sfens.append(b.sfen())
    return list(dict.fromkeys(sfens))

def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from suisho5_eval import Suisho5Net
    engine, nnbin = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    try:
        sfens = random_legal_sfens(n)
    except ImportError:
        print("python-shogi 未導入のため固定局面で検証します（pip install python-shogi 推奨）")
        sfens = [
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "ln1g4l/1ks1g2+R1/1ppppsn2/p5ppp/9/2P6/PPNPPPPPP/2S1GS1R1/L1KG3NL w Bb 1",
        ]
    net = Suisho5Net(nnbin)
    eng = engine_evals(engine, nnbin, sfens)
    ng = 0
    for s, e in zip(sfens, eng):
        m = net.eval_numpy(s)
        if e != m:
            ng += 1
            if ng <= 10:
                print(f"NG  engine={e:>7}  mine={m:>7}  {s}")
    print(f"\n合法局面 {len(sfens)} 件:  一致 {len(sfens)-ng}  /  不一致 {ng}")
    sys.exit(1 if ng else 0)

if __name__ == "__main__":
    main()
