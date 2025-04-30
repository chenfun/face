#!/bin/bash

# ====================
# 臉部辨識系統環境設置腳本
# ====================

# 設置顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
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

# 檢查依賴函數
check_dependency() {
    if ! command -v $1 &> /dev/null; then
        log_error "找不到 $1 命令。請確保已安裝 $2"
        exit 1
    fi
}

# 顯示標題
log_header "臉部辨識系統環境設置"

# 檢查必要依賴
log_info "檢查系統依賴..."
check_dependency python3 "Python 3"
check_dependency pip3 "Python pip"

# 檢查是否已經存在虛擬環境
if [ -d "venv" ]; then
    log_info "檢測到已存在的虛擬環境"
    
    # 詢問是否重新創建
    echo -e "${YELLOW}是否要重新創建虛擬環境？(y/n)${NC}"
    read -r recreate
    
    if [[ "$recreate" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        log_info "移除舊的虛擬環境"
        rm -rf venv
        log_info "創建新的虛擬環境"
        python3 -m venv venv || { log_error "創建虛擬環境失敗"; exit 1; }
    fi
else
    # 創建虛擬環境
    log_info "創建虛擬環境"
    python3 -m venv venv || { log_error "創建虛擬環境失敗"; exit 1; }
fi

# 啟用虛擬環境
log_info "啟用虛擬環境"
source venv/bin/activate || { log_error "啟用虛擬環境失敗"; exit 1; }

# 檢查是否成功啟用虛擬環境
if [[ "$VIRTUAL_ENV" == "" ]]; then
    log_error "虛擬環境啟用失敗"
    exit 1
fi

# 升級 pip
log_info "升級 pip..."
pip install --upgrade pip || { log_error "升級 pip 失敗"; exit 1; }

# 檢查 requirements.txt 文件
if [ ! -f "requirements.txt" ]; then
    log_error "找不到 requirements.txt 文件"
    exit 1
fi

# 安裝依賴
log_info "安裝所需套件 (可能需要一些時間)..."
pip install -r requirements.txt || { log_error "安裝套件失敗"; exit 1; }

# 檢查 dlib 安裝是否成功 (由於 dlib 安裝常出問題)
log_info "驗證關鍵套件安裝..."
if ! python -c "import dlib" &> /dev/null; then
    log_error "dlib 安裝失敗，可能需要安裝 CMake 和 C++ 編譯器"
    echo -e "${YELLOW}在 Mac 上，您可以使用: ${NC}brew install cmake"
    exit 1
fi

if ! python -c "import tensorflow" &> /dev/null; then
    log_error "tensorflow 安裝失敗"
    exit 1
fi

if ! python -c "import cv2, face_recognition, deepface" &> /dev/null; then
    log_error "某些關鍵套件安裝失敗"
    exit 1
fi

# 成功安裝
log_success "虛擬環境設置完成！所有套件已成功安裝"
log_header "使用方法"
echo -e "1. 啟用虛擬環境: ${GREEN}source venv/bin/activate${NC}"
echo -e "2. 執行無視窗程式: ${GREEN}python app/headless_main.py${NC}"
echo -e "3. 執行視窗程式: ${GREEN}python app/main.py${NC}"
echo -e "4. 停用虛擬環境: ${GREEN}deactivate${NC}"

# 保持虛擬環境處於啟用狀態
log_success "虛擬環境已啟用，可以直接執行應用程式。"

# 提供選擇執行哪個版本的應用程式
echo -e "${YELLOW}請選擇要執行的應用程式:${NC}"
echo -e "1. 執行無視窗版本 (headless_main.py)"
echo -e "2. 執行視窗版本 (main.py)"
echo -e "3. 不執行任何程式"

read -p "請選擇 [1-3]: " choice

case $choice in
    1)
        log_success "正在啟動無視窗版本..."
        python app/headless_main.py
        ;;
    2)
        log_success "正在啟動視窗版本..."
        python app/main.py
        ;;
    *)
        log_info "您可以稍後手動執行應用程式。"
        ;;
esac

# 顯示結束資訊
log_header "環境設置完成"
echo -e "${GREEN}臉部辨識系統環境已設置完成。${NC}"