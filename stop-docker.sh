#!/bin/bash

# 設定顏色輸出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 顯示標題
echo -e "${RED}====================================${NC}"
echo -e "${RED}     臉部識別應用程式停止腳本       ${NC}"
echo -e "${RED}====================================${NC}"

# 停止 Docker 容器
echo -e "${YELLOW}正在停止 Docker 容器...${NC}"
docker-compose down

# 確認容器已停止
if [ $? -eq 0 ]; then
    echo -e "${GREEN}====================================${NC}"
    echo -e "${GREEN}所有服務已成功停止${NC}"
    echo -e "${GREEN}您可以隨時使用 ./start.sh 重新啟動服務${NC}"
    echo -e "${GREEN}====================================${NC}"
else
    echo -e "${RED}停止服務時發生錯誤，請檢查 Docker 狀態${NC}"
fi