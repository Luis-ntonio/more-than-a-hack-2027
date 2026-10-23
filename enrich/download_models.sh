#!/usr/bin/env bash
# Roboflow sports weights (MIT, https://github.com/roboflow/sports), same Google Drive ids as their
# examples/soccer/setup.sh. They run locally: no frame ever leaves the machine.
#   docker compose run --rm gpu bash enrich/download_models.sh
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$DIR"
fetch() {
  if [ -s "$DIR/$1" ]; then echo "ok    $1"; return; fi
  gdown -q -O "$DIR/$1" "https://drive.google.com/uc?id=$2"
  echo "got   $1 ($(du -h "$DIR/$1" | cut -f1))"
}
fetch football-ball-detection.pt   1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V
fetch football-player-detection.pt 17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q
fetch football-pitch-detection.pt  1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf
sha256sum "$DIR"/*.pt | sed "s#$DIR/##"
