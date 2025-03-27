#!/bin/bash

# 設定顏色輸出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 顯示標題
echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}     臉部識別應用程式啟動腳本       ${NC}"
echo -e "${GREEN}====================================${NC}"

# 檢查 Docker 是否正在運行
if ! docker info > /dev/null 2>&1; then
    echo -e "${YELLOW}Docker 未運行，請先啟動 Docker...${NC}"
    # 視作業系統而定，可能需要調整此行
    open -a Docker
    
    # 等待 Docker 啟動
    echo -e "等待 Docker 引擎啟動中..."
    sleep 10
fi

# 啟動 Docker 容器
echo -e "${GREEN}正在啟動 Docker 容器...${NC}"
xhost +local:docker
docker-compose up -d

# 等待容器完全啟動
echo -e "${GREEN}等待服務準備就緒...${NC}"
sleep 5

# 顯示容器狀態
echo -e "${GREEN}容器狀態:${NC}"
docker-compose ps

# 提取運行端口
PORT=$(docker-compose port face-recognition 8000 | cut -d: -f2)
if [ -z "$PORT" ]; then
    PORT="8000" # 默認端口
fi

# 輸出應用程式訪問資訊
echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}臉部識別應用程式已啟動!${NC}"
echo -e "${GREEN}訪問地址: http://localhost:${PORT}${NC}"
echo -e "${GREEN}====================================${NC}"

# 提供其他有用的命令
echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}其他有用的命令:${NC}"
echo -e "${YELLOW}停止服務:${NC} ./stop.sh"
echo -e "${YELLOW}重啟服務:${NC} docker-compose restart"
echo -e "${YELLOW}進入容器:${NC} docker-compose exec face-recognition bash"
echo -e "${GREEN}====================================${NC}"

# 執行應用程式
docker-compose exec face-recognition python /app/main.py