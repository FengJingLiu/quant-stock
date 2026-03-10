#!/usr/bin/env bash
set -euo pipefail

PCS="/opt/BaiduPCS/BaiduPCS-Go-v4.0.0-linux-amd64/BaiduPCS-Go"

# Cron 环境里避免找不到配置
export BAIDUPCS_GO_CONFIG_DIR="/home/autumn/.config/BaiduPCS-Go"
export LANG="C.UTF-8"

REMOTE_ROOT="/openclaw_sync"
REMOTE_L1="$REMOTE_ROOT/l1"
REMOTE_L2="$REMOTE_ROOT/l2"

LOCAL_L1="/home/autumn/quant/stock/A股数据_zip"
LOCAL_L2="/home/autumn/quant/stock/A股数据_每日指标"

# After syncing source files, we sync them into the local Parquet Lake + DuckDB.
PROJECT_ROOT="/home/autumn/quant/stock"
UV_BIN="/home/autumn/.local/bin/uv"
SYNC_REPROCESS_DAYS="${SYNC_REPROCESS_DAYS:-5}"
SYNC_UPDATE_FACTORS="${SYNC_UPDATE_FACTORS:-1}"

# Links
L1_URL="https://pan.baidu.com/s/1oFeJDuspFDonVrsYFdYDgg"
L1_PWD="3kfj"
L1_DIRNAME="A股数据_zip"
L1_FILES=(
  "daily_qfq.zip"
  "daily.zip"
  "daily_hfq.zip"
  "股票列表.csv"
)

L2_URL="https://pan.baidu.com/s/1XBmNm1KLayjUidr3rP-PHw"
L2_PWD="jib9"
L2_DIRNAME="A股数据_每日指标"
L2_FULL_FILES=(
  "股票列表.csv"
  "退市股票列表.csv"
)
L2_INC_PARENT="增量数据"
L2_INC_DIR="每日指标"

LOCK_FILE="/tmp/baidupan_sync.lock"

log() {
  # 用北京时间打点，方便对齐“每天20点”
  TZ=Asia/Shanghai date "+[%F %T]" | tr -d '\n'
  echo " $*"
}

cleanup_recycle_prefix() {
  # 百度网盘偶发风控会拦截 recycle delete（提示“安全验证”）。
  # 这里做 best-effort：能删就删，删不了也不影响主流程。
  local prefix="$1"
  local ids

  ids=$(
    "$PCS" recycle list 2>/dev/null \
      | awk -v p="$prefix" 'index($0,p)>0 {print $2}' \
      | tr "\n" " " \
      | xargs echo -n
  )

  if [[ -n "${ids:-}" ]]; then
    log "清理回收站(尽力而为): $prefix"
    "$PCS" recycle delete $ids >/dev/null 2>&1 || true
  fi
}

remote_refresh_share_into() {
  # $1 = remote parent dir (must exist)
  # $2 = share dirname (created by transfer)
  # $3 = url
  # $4 = pwd
  local parent="$1"; shift
  local dirname="$1"; shift
  local url="$1"; shift
  local pwd="$1"; shift

  "$PCS" mkdir "$REMOTE_ROOT" >/dev/null 2>&1 || true
  "$PCS" mkdir "$parent"      >/dev/null 2>&1 || true

  # 避免转存产生 (1)(2) 重复目录：先删旧的，再转存
  "$PCS" rm "$parent/$dirname" >/dev/null 2>&1 || true
  cleanup_recycle_prefix "$parent/$dirname"

  "$PCS" cd "$parent" >/dev/null
  "$PCS" transfer "$url" "$pwd"
}

sync_l1() {
  log "L1: 开始 (更新 daily_*.zip + 股票列表.csv; 本地: $LOCAL_L1)"

  remote_refresh_share_into "$REMOTE_L1" "$L1_DIRNAME" "$L1_URL" "$L1_PWD"

  for f in "${L1_FILES[@]}"; do
    local remote_path="$REMOTE_L1/$L1_DIRNAME/$f"
    if "$PCS" meta "$remote_path" >/dev/null 2>&1; then
      log "L1: 覆盖更新 $f"
      # 这些文件名固定但内容会变：必须覆盖更新，否则不会变新
      "$PCS" download --ow --mtime --saveto "$LOCAL_L1" "$remote_path" || log "WARN: L1 下载失败: $f"
    else
      log "WARN: L1 远端不存在: $f"
    fi
  done

  # 清理远端临时目录，避免占用网盘空间
  "$PCS" rm "$REMOTE_L1/$L1_DIRNAME" >/dev/null 2>&1 || true
  cleanup_recycle_prefix "$REMOTE_L1/$L1_DIRNAME"

  log "L1: 完成"
}

sync_l2() {
  log "L2: 开始 (覆盖更新CSV + 增量更新 /$L2_INC_PARENT/$L2_INC_DIR; 本地: $LOCAL_L2)"

  remote_refresh_share_into "$REMOTE_L2" "$L2_DIRNAME" "$L2_URL" "$L2_PWD"

  # 1) CSV 全量更新（覆盖）
  for f in "${L2_FULL_FILES[@]}"; do
    local remote_path="$REMOTE_L2/$L2_DIRNAME/$f"
    if "$PCS" meta "$remote_path" >/dev/null 2>&1; then
      log "L2: 覆盖更新 $f"
      "$PCS" download --ow --mtime --saveto "$LOCAL_L2" "$remote_path" || log "WARN: L2 下载失败: $f"
    else
      log "WARN: L2 远端不存在: $f"
    fi
  done

  # 2) 指定路径增量更新
  local remote_dir="$REMOTE_L2/$L2_DIRNAME/$L2_INC_PARENT/$L2_INC_DIR"
  local local_base="$LOCAL_L2/$L2_INC_PARENT"
  mkdir -p "$local_base"

  if "$PCS" meta "$remote_dir" >/dev/null 2>&1; then
    log "L2: 增量更新目录 $L2_INC_PARENT/$L2_INC_DIR (不覆盖已存在文件)"
    "$PCS" download --mtime --saveto "$local_base" "$remote_dir" || log "WARN: L2 下载失败: $L2_INC_PARENT/$L2_INC_DIR"
  else
    log "WARN: L2 远端不存在目录: $remote_dir"
  fi

  # 清理远端临时目录
  "$PCS" rm "$REMOTE_L2/$L2_DIRNAME" >/dev/null 2>&1 || true
  cleanup_recycle_prefix "$REMOTE_L2/$L2_DIRNAME"

  log "L2: 完成"
}

sync_lake() {
  log "L3: 同步到本地 Lake (bars/indicators/factors + dim_symbol + duckdb views)"

  if [[ ! -x "$UV_BIN" ]]; then
    log "ERROR: uv 不存在或不可执行: $UV_BIN"
    return 1
  fi

  local args=("--reprocess-days" "$SYNC_REPROCESS_DAYS")
  if [[ "$SYNC_UPDATE_FACTORS" == "1" ]]; then
    args+=("--update-factors")
  fi

  (cd "$PROJECT_ROOT" && "$UV_BIN" run python scripts/sync_lake_daily.py "${args[@]}")

  log "L3: 完成"
}

main() {
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    log "已有同步任务在跑，跳过本次。"
    exit 0
  fi

  log "开始百度网盘同步"

  # 确认登录态
  "$PCS" who >/dev/null

  mkdir -p "$LOCAL_L1" "$LOCAL_L2"

  sync_l1
  sync_l2
  sync_lake

  log "全部完成"
}

main "$@"
