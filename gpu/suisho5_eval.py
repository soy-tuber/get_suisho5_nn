#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
suisho5_eval.py — 水匠5(NNUE, HalfKP 256x2-32-32)評価関数を Python で再現する。

目的（棋力度外視）:
  - 水匠5の nn.bin をそのまま読み込み、評価値を計算する
  - 大量局面の「バッチ評価」を GPU(RTX 5090) で高速に回す
    （NNUE+αβ探索の対局用途は CPU が適切。GPU はバッチ一括評価でこそ効く）

実装の正解仕様は、水匠5のC++ソース（旧nodchipフォーマット, kVersion=0x7AF32F16）に厳密準拠:
  FeatureTransformer:  bias int16[256] -> weight int16[125388][256]   (行=特徴量, 列=次元)
  各 AffineTransform:  bias int32[out]  -> weight int8[out][padded_in] (行優先)
  量子化:  FT出力 = clamp(acc, 0, 127)
           層間ClippedReLU = clamp(z >> 6, 0, 127)            (>>6 = kWeightScaleBits)
           最終 score = trunc(z_out / 24)                      (FV_SCALE=24)
  視点:    入力512次元 = [手番側256, 相手側256]、評価値は手番側から見た値

CPU(numpy)実装は実エンジン `eval` と一致することを verify_against_engine.py で検証する。
GPU(torch)実装は同じ重み・同じ整数演算をバッチで行い、 .to('cuda') で 5090 に載せる。
"""
import struct
import numpy as np

# ---------------------------------------------------------------------------
# 定数（水匠5 evaluate.h / nnue_common.h より）
# ---------------------------------------------------------------------------
SQ_NB   = 81
FE_END  = 1548                 # = HalfKP の P 次元
KDIM    = SQ_NB * FE_END       # = 125388  (FeatureTransformer 入力次元)
HALF    = 256                  # kHalfDimensions
FV_SCALE = 24
WEIGHT_SHIFT = 6               # kWeightScaleBits
KVERSION = 0x7AF32F16

def INV(sq: int) -> int:
    """盤面を点対称に反転（後手視点）。SQ_NB-1-sq"""
    return SQ_NB - 1 - sq

# BonaPiece 基準値（fe_hand_end=90 の標準Apery配置）
f_hand = {  # [type] = (fb_base, fw_base) for BLACK の手駒。WHITEはfb/fw入替。
    'P': (1, 20), 'L': (39, 44), 'N': (49, 54), 'S': (59, 64),
    'B': (79, 82), 'R': (85, 88), 'G': (69, 74),
}
# 盤駒 BonaPiece 基準値（BLACKのfb, fw）。WHITEはfb/fw入替・升はINV。
f_board = {
    'P': (90, 171), 'L': (252, 333), 'N': (414, 495), 'S': (576, 657),
    'G': (738, 819), 'B': (900, 981), 'R': (1224, 1305),
    '+P': (738, 819), '+L': (738, 819), '+N': (738, 819), '+S': (738, 819),
    '+B': (1062, 1143), '+R': (1386, 1467),  # 馬, 龍
    'K': (1548, 1629),  # f_king, e_king（HalfKPの活性特徴には使わない。玉の升取得用）
}

# ---------------------------------------------------------------------------
# nn.bin パーサ
# ---------------------------------------------------------------------------
class Suisho5Net:
    def __init__(self, path: str):
        with open(path, 'rb') as f:
            data = f.read()
        off = 0
        version, hashval, arch_len = struct.unpack_from('<III', data, off); off += 12
        if version != KVERSION:
            raise ValueError(f'unexpected version 0x{version:08X}')
        self.architecture = data[off:off+arch_len].decode('ascii', 'replace'); off += arch_len

        # --- FeatureTransformer ---
        self.ft_hash, = struct.unpack_from('<I', data, off); off += 4
        ft_bias = np.frombuffer(data, '<i2', HALF, off).astype(np.int32); off += HALF * 2
        ft_w = np.frombuffer(data, '<i2', KDIM * HALF, off).astype(np.int16); off += KDIM * HALF * 2
        self.ft_bias = ft_bias                       # int32[256]
        self.ft_weight = ft_w.reshape(KDIM, HALF)    # int16[125388, 256]  (行=特徴量)

        # --- Network (recursive: Affine1 -> Affine2 -> Output) ---
        self.net_hash, = struct.unpack_from('<I', data, off); off += 4
        def read_affine(in_dim, out_dim):
            nonlocal off
            pad = ((in_dim + 31) // 32) * 32
            b = np.frombuffer(data, '<i4', out_dim, off).astype(np.int32); off += out_dim * 4
            w = np.frombuffer(data, '<i1', out_dim * pad, off).astype(np.int32); off += out_dim * pad
            w = w.reshape(out_dim, pad)[:, :in_dim].copy()   # padding列を捨てる
            return b, w
        self.b1, self.w1 = read_affine(HALF * 2, 32)   # 512 -> 32
        self.b2, self.w2 = read_affine(32, 32)         # 32  -> 32
        self.b3, self.w3 = read_affine(32, 1)          # 32  -> 1
        assert off == len(data), f'parse mismatch: off={off}, size={len(data)}'

    # ----- SFEN -> HalfKP 活性特徴量（両視点）------------------------------
    @staticmethod
    def sfen_to_features(sfen: str):
        """returns (active_black:list[int], active_white:list[int], side_to_move:'b'|'w')"""
        parts = sfen.split()
        if parts[0] == 'sfen':
            parts = parts[1:]
        board_s, stm, hands_s = parts[0], parts[1], parts[2]

        bona_fb = []   # 非玉駒の BonaPiece(先手視点)
        bona_fw = []   # 非玉駒の BonaPiece(後手視点)
        bk_sq = wk_sq = None

        # 盤面: 行=rank1..9(上から), 行内左から file9..1
        rows = board_s.split('/')
        for r, row in enumerate(rows):           # r = Rank (0..8)
            k = 0
            i = 0
            while i < len(row):
                ch = row[i]
                if ch.isdigit():
                    k += int(ch); i += 1; continue
                promoted = ''
                if ch == '+':
                    promoted = '+'; i += 1; ch = row[i]
                i += 1
                file = 8 - k                      # File (0..8): 左端が file9=8
                sq = file * 9 + r                 # YaneuraOu Square = file*9 + rank
                k += 1
                is_black = ch.isupper()
                t = ch.upper()
                key = promoted + t
                if t == 'K':
                    if is_black: bk_sq = sq
                    else:        wk_sq = sq
                    continue
                fb_base, fw_base = f_board[key]
                if is_black:
                    bona_fb.append(fb_base + sq)
                    bona_fw.append(fw_base + INV(sq))
                else:
                    # 後手駒: fb/fw 入替（kpp_board_index の後手側）
                    bona_fb.append(fw_base + sq)
                    bona_fw.append(fb_base + INV(sq))

        # 手駒
        if hands_s != '-':
            i = 0
            while i < len(hands_s):
                num = 0
                while hands_s[i].isdigit():
                    num = num * 10 + int(hands_s[i]); i += 1
                if num == 0: num = 1
                ch = hands_s[i]; i += 1
                is_black = ch.isupper()
                t = ch.upper()
                fb_base, fw_base = f_hand[t]
                for n in range(num):
                    if is_black:
                        bona_fb.append(fb_base + n)
                        bona_fw.append(fw_base + n)
                    else:
                        bona_fb.append(fw_base + n)
                        bona_fw.append(fb_base + n)

        # MakeIndex(sq_k, p) = FE_END * sq_k + p
        ab = [FE_END * bk_sq + p for p in bona_fb]              # BLACK視点: 先手玉の升
        aw = [FE_END * INV(wk_sq) + p for p in bona_fw]         # WHITE視点: INV(後手玉の升)
        return ab, aw, stm

    # ----- numpy 前進計算（1局面、整数演算で実エンジンと一致）-------------
    def eval_numpy(self, sfen: str) -> int:
        ab, aw, stm = self.sfen_to_features(sfen)
        acc_b = self.ft_bias + self.ft_weight[ab].astype(np.int32).sum(0)   # int32[256]
        acc_w = self.ft_bias + self.ft_weight[aw].astype(np.int32).sum(0)
        cb = np.clip(acc_b, 0, 127)
        cw = np.clip(acc_w, 0, 127)
        x = np.concatenate([cb, cw] if stm == 'b' else [cw, cb]).astype(np.int32)  # [512]
        z1 = self.b1 + self.w1 @ x                       # int32[32]
        a1 = np.clip(z1 >> WEIGHT_SHIFT, 0, 127)         # >>6 は算術右シフト(floor)
        z2 = self.b2 + self.w2 @ a1
        a2 = np.clip(z2 >> WEIGHT_SHIFT, 0, 127)
        z3 = int(self.b3[0] + int(self.w3[0] @ a2))
        return int(z3 / FV_SCALE)                        # C++ の / は0方向切捨て


# ---------------------------------------------------------------------------
# GPU(torch) バッチ評価 — RTX 5090 でまとめて評価する用
# ---------------------------------------------------------------------------
def build_torch_evaluator(net: 'Suisho5Net', device: str = 'cuda'):
    """
    torch ベースのバッチ評価器を返す。
      evaluator(list_of_sfen) -> np.ndarray[int]  （手番側から見た評価値）
    FeatureTransformer は EmbeddingBag(sum) による疎集約、後段は float32 行列積。
    （入力0..127・重みint8、層内総和は最大 ~8.3M < 2^24 なので float32 で厳密）
    """
    import torch
    dev = torch.device(device)

    # FT を EmbeddingBag に。padding_idx 用に末尾へ0行を足す。
    ft_w = torch.from_numpy(net.ft_weight.astype(np.float32))
    pad_row = torch.zeros((1, HALF), dtype=torch.float32)
    emb = torch.cat([ft_w, pad_row], 0).to(dev)             # [KDIM+1, 256]
    PAD = KDIM
    ft_bias = torch.from_numpy(net.ft_bias.astype(np.float32)).to(dev)
    w1 = torch.from_numpy(net.w1.astype(np.float32)).to(dev)   # [32,512]
    b1 = torch.from_numpy(net.b1.astype(np.float32)).to(dev)
    w2 = torch.from_numpy(net.w2.astype(np.float32)).to(dev)
    b2 = torch.from_numpy(net.b2.astype(np.float32)).to(dev)
    w3 = torch.from_numpy(net.w3.astype(np.float32)).to(dev)   # [1,32]
    b3 = torch.from_numpy(net.b3.astype(np.float32)).to(dev)

    def floor_shift(x):   # clamp(x >> 6, 0, 127)  （floor 相当）
        return torch.clamp(torch.floor(x / (1 << WEIGHT_SHIFT)), 0, 127)

    @torch.no_grad()
    def evaluate(sfens):
        B = len(sfens)
        feats_b, feats_w, stm = [], [], []
        for s in sfens:
            ab, aw, t = net.sfen_to_features(s)
            feats_b.append(ab); feats_w.append(aw); stm.append(t)
        maxlen = max(max(len(x) for x in feats_b), max(len(x) for x in feats_w))
        def pad(rows):
            m = torch.full((B, maxlen), PAD, dtype=torch.long)
            for i, r in enumerate(rows):
                if r: m[i, :len(r)] = torch.tensor(r, dtype=torch.long)
            return m.to(dev)
        idx_b, idx_w = pad(feats_b), pad(feats_w)
        # EmbeddingBag(sum) = 各局面の活性特徴の重み和
        acc_b = torch.nn.functional.embedding(idx_b, emb).sum(1) + ft_bias   # [B,256]
        acc_w = torch.nn.functional.embedding(idx_w, emb).sum(1) + ft_bias
        cb = torch.clamp(acc_b, 0, 127)
        cw = torch.clamp(acc_w, 0, 127)
        is_b = torch.tensor([t == 'b' for t in stm], device=dev).unsqueeze(1)
        own  = torch.where(is_b, cb, cw)
        opp  = torch.where(is_b, cw, cb)
        x = torch.cat([own, opp], 1)                       # [B,512]
        z1 = x @ w1.T + b1
        a1 = floor_shift(z1)
        z2 = a1 @ w2.T + b2
        a2 = floor_shift(z2)
        z3 = a2 @ w3.T + b3                                 # [B,1]
        score = torch.trunc(z3.squeeze(1) / FV_SCALE)
        return score.to('cpu').numpy().astype(np.int64)

    return evaluate


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'suisho5_nn/nn.bin'
    net = Suisho5Net(path)
    print('architecture :', net.architecture)
    print(f'FT hash=0x{net.ft_hash:08X}  Network hash=0x{net.net_hash:08X}')
    start = 'lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1'
    print('startpos eval (numpy) :', net.eval_numpy(start))
