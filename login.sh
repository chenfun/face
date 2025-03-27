#!/bin/bash

# 獲取容器狀態
container_running=$(docker ps --format "{{.Names}}" | grep -E 'face.*recognition|face-recognition|face_recognition')

# 檢查容器是否正在運行
if [ -z "$container_running" ]; then
    echo "錯誤：找不到正在運行的人臉識別容器。"
    echo "請先使用 'start.sh' 腳本啟動容器。"
    exit 1
fi

echo "正在登入到容器：$container_running"
# 使用 docker exec 命令登入容器，使用 bash shell
docker exec -it $container_running /bin/bash

# 如果容器內沒有 bash，則嘗試使用 sh
if [ $? -ne 0 ]; then
    echo "嘗試使用 sh 登入..."
    docker exec -it $container_running /bin/sh
fi