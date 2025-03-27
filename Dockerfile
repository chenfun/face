# 從 Python 3.9 基礎映像開始
FROM python:3.9
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 只複製 requirements.txt 並安裝 Python 依賴
# 這樣當 requirements.txt 沒有變更時可以利用快取
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案文件 - 這一步放在依賴安裝後，確保代碼變更不會觸發重新安裝依賴
# 注意：由於使用 volume 映射，這步可能不是必要的
# 複製 app/ 目錄到 WORKDIR
# COPY app/ .

# 執行應用
# CMD ["python", "main.py"]