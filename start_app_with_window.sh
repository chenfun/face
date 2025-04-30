#!/bin/bash

# 進入專案目錄
cd "$(dirname "$0")"

# 啟動程式，並顯示視窗
python3 app/headless_main.py --show-window
