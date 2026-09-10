#!/usr/bin/env bash
set -euo pipefail
cd /home/exx/Projects/ECOT-Alignment
suffix="$1"
shift
case "$suffix" in
  20) weight=2.0 ;;
  50) weight=5.0 ;;
  *) exit 2 ;;
esac
exec bash runs/round3/train.sh "$suffix" "$weight" 32 1 "$@"
