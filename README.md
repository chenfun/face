# 人臉辨識與追蹤系統

這是一個使用人臉辨識技術來追踪展會訪客的系統。它能夠透過攝像頭識別人臉，追踪訪客在不同攤位的移動，並提供性別、年齡及停留時間等統計數據。

## 功能特點

- 人臉偵測與辨識
- 訪客追蹤與統計
- 性別和年齡分析
- 攤位流量統計
- 攤位平均停留時間統計
- 支援多台攝像頭
- 數據本地CSV存儲

## 系統需求

- Python 3.8+
- 攝像頭 (USB或RTSP串流)
- 足夠的CPU資源進行實時人臉辨識

## 虛擬環境安裝

1. 克隆或下載此專案
2. 執行設置腳本:

```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

這將會:
- 創建Python虛擬環境
- 安裝所有必要的依賴項
- 配置開發環境

## 啟動應用程式

```bash
./start_app.sh
```

## 環境變數

可以通過設置以下環境變數來自定義行為:

- `MONGO_URI`: MongoDB連接字串 (預設: `mongodb://localhost:27017/`)
- `MONGO_ENABLED`: 是否啟用MongoDB儲存 (預設: `false`)

若要啟用MongoDB儲存:

```bash
export MONGO_ENABLED=true
./start_app.sh
```

## 數據分析

該系統會產生以下統計數據:

1. 訪客總數
2. 性別分布
3. 年齡段分布
4. 各攤位流量
5. 各攤位平均停留時間

數據將保存到 `app/visitor_data.csv` 文件中，可使用Excel或其他分析工具進行進一步分析。

## 故障排除

### 攝像頭相關問題

- 確保攝像頭已連接且工作正常
- 檢查RTSP串流URL是否正確
- 嘗試使用不同的攝像頭索引 (0, 1, 2等)

### 安裝相關問題

如果在安裝過程中出現錯誤，特別是與dlib相關的錯誤，您可能需要先安裝以下系統依賴:

```bash
# macOS
brew install cmake
brew install boost
brew install boost-python3

# Linux (Ubuntu/Debian)
sudo apt-get install -y build-essential cmake
sudo apt-get install -y libboost-all-dev
```

### 數據存儲問題

如果需要使用MongoDB存儲功能:

1. 安裝並運行MongoDB服務
2. 啟用MongoDB存儲: `export MONGO_ENABLED=true`
