# 展覽分析系統 (Exhibition Analytics System)

這是一個基於電腦視覺和人臉辨識技術的展覽會場人流分析系統，能夠追蹤訪客在展區內的移動軌跡、停留時間，以及分析訪客的基本人口統計資料（如性別、年齡）。

## 功能特色

- **即時人臉辨識**：使用 face_recognition 和 DeepFace 進行高精度人臉偵測與辨識
- **訪客追蹤**：追蹤訪客在不同展區/攤位之間的移動
- **人口統計分析**：分析訪客的性別和年齡分佈
- **停留時間分析**：計算訪客在各個展區的停留時間
- **數據保存**：將訪客資料儲存至 MongoDB 資料庫
- **Docker 支援**：提供 Docker 容器化部署方案，確保環境一致性

## 技術架構

- **Python 3.x**：主要程式語言
- **OpenCV**：影像處理與電腦視覺
- **face_recognition**：人臉偵測與辨識
- **DeepFace**：臉部屬性分析（性別、年齡）
- **Pandas & Matplotlib**：數據處理與視覺化
- **MongoDB**：數據儲存
- **Docker**：容器化部署

## 系統需求

- Python 3.7+
- 相機設備（網路攝影機或 IP 攝影機）
- Docker 和 Docker Compose（選用，如需容器化部署）
- MongoDB（如需使用資料庫功能）

## 安裝指南

### 使用 Docker（建議）

1. 確保您的系統已安裝 Docker 和 Docker Compose
2. 複製專案到本地
3. 在專案根目錄執行：

```bash
./start-docker.sh
```

### 手動安裝

1. 確保您的系統已安裝 Python 3.7+
2. 複製專案到本地
3. 安裝相依套件：

```bash
pip install -r requirements.txt
```

4. 執行程式：

```bash
python app/main.py
```

## 系統配置

可通過修改環境變數來調整系統配置：

- `MONGO_URI`：MongoDB 連接 URI（預設值為 `mongodb://localhost:27017/`）
- `CAMERA_SOURCE`：攝影機來源，可以是數字索引或 RTSP 串流 URL

## 使用說明

1. 系統啟動後會自動開始偵測與追蹤展區內的訪客
2. 系統會記錄訪客的移動軌跡和停留時間
3. 透過 MongoDB 資料庫可以查詢和分析收集到的數據

## 專案結構

```
.
├── app
│   ├── main.py          # 主程式入口
│   ├── main2.py         # 替代主程式（可選）
│   └── visitor_data.csv # 訪客數據樣本
├── docker-compose.yml   # Docker Compose 配置
├── Dockerfile           # Docker 映像檔配置
├── login.sh             # Docker 登入腳本
├── requirements.txt     # Python 相依套件
├── start-docker.sh      # Docker 啟動腳本
└── stop-docker.sh       # Docker 停止腳本
```

## 聯絡與支援

如有任何問題或需要支援，請聯絡：chenfun@example.com

## 授權條款

[授權條款待定]