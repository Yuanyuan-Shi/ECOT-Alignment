#!/usr/bin/env bash
set -euo pipefail
cd /home/exx/Projects/ECOT-Alignment
exec bash runs/round3/evaluate.sh "$1"
