#!/usr/bin/env bash
#
# build_yaneuraou_for_suisho5.sh
#
# 最新やねうら王を「標準NNUE(halfKP256)」構成でビルドし、
# 本リポジトリで取り出した水匠5の評価関数 suisho5_nn/nn.bin を読み込ませて
# 起動・ベンチ確認まで自動で行うスクリプト（Linux / g++ 想定）。
#
# 水匠5(halfKP256)と構造が一致するのは「標準NNUE = YANEURAOU_ENGINE_NNUE」ビルドのみ。
# これ以外(halfkpe9 / kp256 / 1024x2 など)では構造ハッシュ不一致で起動に失敗する。
#
# 使い方:
#   bash scripts/build_yaneuraou_for_suisho5.sh
#
# 主な環境変数（省略可）:
#   TARGET_CPU   ... AVX512VNNI(既定) / AVX512 / AVXVNNI / AVX2 / ...
#   JOBS         ... 並列ビルド数（既定: nproc）
#   WORKDIR      ... 作業ディレクトリ（既定: ./.build_yaneuraou）
#   YANEURAOU_REF... チェックアウトするブランチ/タグ（既定: 既定ブランチ）
#
set -euo pipefail

# --- 設定 -------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NN_BIN="${REPO_ROOT}/suisho5_nn/nn.bin"
TARGET_CPU="${TARGET_CPU:-AVX512VNNI}"
JOBS="${JOBS:-$(nproc)}"
WORKDIR="${WORKDIR:-${REPO_ROOT}/.build_yaneuraou}"
YANEURAOU_REF="${YANEURAOU_REF:-}"
EDITION="YANEURAOU_ENGINE_NNUE"   # = 標準NNUE(halfKP256)。水匠5と一致させるため固定。

echo "==> 設定"
echo "    REPO_ROOT  = ${REPO_ROOT}"
echo "    nn.bin     = ${NN_BIN}"
echo "    TARGET_CPU = ${TARGET_CPU}"
echo "    EDITION    = ${EDITION} (標準NNUE=halfKP256)"
echo "    JOBS       = ${JOBS}"
echo "    WORKDIR    = ${WORKDIR}"

# --- 事前チェック -----------------------------------------------------------
command -v git >/dev/null || { echo "git が必要です"; exit 1; }
command -v make >/dev/null || { echo "make が必要です"; exit 1; }
command -v g++  >/dev/null || { echo "g++ が必要です"; exit 1; }
[ -f "${NN_BIN}" ] || { echo "評価関数が見つかりません: ${NN_BIN}"; exit 1; }

# --- ソース取得 -------------------------------------------------------------
mkdir -p "${WORKDIR}"
if [ ! -d "${WORKDIR}/YaneuraOu/.git" ]; then
  echo "==> やねうら王を clone"
  git clone --depth 1 ${YANEURAOU_REF:+--branch "${YANEURAOU_REF}"} \
      https://github.com/yaneurao/YaneuraOu.git "${WORKDIR}/YaneuraOu"
else
  echo "==> 既存の clone を再利用"
fi

SRC="${WORKDIR}/YaneuraOu/source"

# --- ビルド -----------------------------------------------------------------
echo "==> ビルド (make normal, ${EDITION}, ${TARGET_CPU})"
make -C "${SRC}" -j"${JOBS}" normal \
     YANEURAOU_EDITION="${EDITION}" \
     TARGET_CPU="${TARGET_CPU}" \
     COMPILER=g++

BIN="${SRC}/YaneuraOu-by-gcc"
[ -x "${BIN}" ] || { echo "ビルド失敗: ${BIN} が生成されていません"; exit 1; }
echo "==> ビルド成功: ${BIN}"

# --- 評価関数を配置 ---------------------------------------------------------
mkdir -p "${SRC}/eval"
cp -f "${NN_BIN}" "${SRC}/eval/nn.bin"
echo "==> 水匠5 nn.bin を ${SRC}/eval/nn.bin に配置"

# --- ロード確認 -------------------------------------------------------------
echo "==> 起動して水匠5評価関数のロードを確認"
printf 'usi\nisready\nquit\n' | "${BIN}" 2>&1 \
  | grep -iE "id name|loading eval|readyok|Error" \
  | grep -viE "book" || true

echo
echo "==> 完了。エンジン: ${BIN}"
echo "    対局GUIには このエンジンを登録し、同じ eval フォルダに nn.bin を置いてください。"
echo "    ベンチを取るには:  printf 'bench\\n' | \"${BIN}\""
