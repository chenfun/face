#!/bin/bash

# 設置顏色輸出
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 設置日誌函數
log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${BLUE}${BOLD}=== $1 ===${NC}\n"
}

log_header "臉部辨識系統啟動"

# 檢查虛擬環境是否存在
if [ ! -d "venv" ]; then
    log_error "未找到虛擬環境，請先執行 setup_venv.sh"
    exit 1
fi

# 啟用虛擬環境
log_info "啟用虛擬環境..."
source venv/bin/activate || { log_error "啟用虛擬環境失敗"; exit 1; }

# 提供選擇執行哪個版本的應用程式
log_header "選擇執行模式"
echo -e "請選擇要執行的應用程式版本:"
echo -e "1. ${GREEN}無視窗版本${NC} (headless_main.py) - 不需要圖形界面，適合在服務器或後台運行"
echo -e "2. ${GREEN}視窗版本${NC} (main.py) - 帶有圖形界面，可視化處理結果"
echo -e "3. ${GREEN}備用視窗版本${NC} (open_rtsp.py) - 備用界面"
echo -e "4. ${RED}退出${NC}"

read -p "請選擇 [1-4]: " choice

case $choice in
    1)
        log_info "啟動無視窗版本..."
        python app/headless_main.py
        ;;
    2)
        log_info "啟動視窗版本..."
        python app/main.py
        ;;
    3)
        log_info "啟動備用視窗版本..."
        python app/open_rtsp.py
        ;;
    4)
        log_info "退出程序..."
        deactivate
        exit 0
        ;;
    *)
        log_error "無效選項，退出程序..."
        deactivate
        exit 1
        ;;
esac

# 程序結束後保持終端開啟
log_success "程序已結束"
deactivate
