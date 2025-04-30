import cv2
import time
import threading
import numpy as np
import queue

# RTSP 串流來源列表
camera_sources = [
    "rtsp://192.168.100.94:554/ch01.264",  # 第一台相機
    "rtsp://192.168.100.5:554/ch01.264"    # 第二台相機
]

# 創建全局隊列和停止事件
frame_queues = []
stop_event = threading.Event()

def capture_stream(camera_url, queue_idx):
    """從RTSP串流讀取影像並放入隊列，在獨立執行緒中運行"""
    # 連接到 RTSP 串流
    cap = cv2.VideoCapture(camera_url)
    
    # 設置解碼器和緩衝區大小以改善串流效能
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # 設置緩衝區大小
    
    if not cap.isOpened():
        print(f"無法連接到RTSP串流: {camera_url}")
        # 放入一個空幀的標記，表示連接失敗
        frame_queues[queue_idx].put(None)
        return
    
    print(f"已成功連接到RTSP串流: {camera_url}")
    
    # 設定讀取間隔時間（秒）
    interval = 0.03  # 可以調整這個值來平衡流暢度和資源使用
    
    try:
        while not stop_event.is_set():
            start_time = time.time()
            
            ret, frame = cap.read()
            if ret:
                # 在影像上顯示相機ID
                cv2.putText(frame, f"Camera {queue_idx + 1}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 將幀放入隊列，如果隊列已滿則丟棄舊幀
                try:
                    frame_queues[queue_idx].put(frame, block=False)
                except queue.Full:
                    # 清除隊列中最舊的幀
                    try:
                        frame_queues[queue_idx].get_nowait()
                        frame_queues[queue_idx].put(frame, block=False)
                    except Exception as e:
                        print(f"隊列處理異常: {e}")
            else:
                print(f"無法讀取相機 {queue_idx + 1} 的影像幀，嘗試重新連接...")
                cap.release()
                time.sleep(1)  # 等待一秒再嘗試
                cap = cv2.VideoCapture(camera_url)
                continue
                
            # 控制循環的速率
            elapsed = time.time() - start_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
    except Exception as e:
        print(f"相機 {queue_idx + 1} 執行緒異常: {e}")
    finally:
        if cap is not None:
            cap.release()
        print(f"相機 {queue_idx + 1} 執行緒已結束")

def show_all_cameras():
    """在主線程中顯示所有相機的視窗"""
    # 為每個相機創建一個視窗
    window_names = []
    
    try:
        for i in range(len(camera_sources)):
            window_name = f"Camera {i + 1}"
            window_names.append(window_name)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # 安排視窗位置
            x_pos = (i % 2) * 650 + 50  # 水平位置
            y_pos = (i // 2) * 520 + 50  # 垂直位置
            
            cv2.moveWindow(window_name, x_pos, y_pos)
            cv2.resizeWindow(window_name, 640, 480)
        
        # 創建一個背景幀，在無影像時顯示
        background = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 循環顯示所有相機的影像
        while not stop_event.is_set():
            for i, queue in enumerate(frame_queues):
                try:
                    # 嘗試從隊列獲取最新影像，不阻塞
                    if not queue.empty():
                        frame = queue.get_nowait()
                        if frame is not None:
                            cv2.imshow(window_names[i], frame)
                        else:
                            # 顯示背景幀和錯誤訊息
                            error_frame = background.copy()
                            cv2.putText(error_frame, f"Camera {i+1} Disconnected", (50, 240), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.imshow(window_names[i], error_frame)
                except Exception as e:
                    print(f"顯示相機 {i+1} 影像時發生異常: {e}")
                    
            # 每個迴圈只檢查一次鍵盤輸入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
                break
            
            # 限制主循環速率
            time.sleep(0.01)
    except Exception as e:
        print(f"顯示視窗異常: {e}")
    finally:
        # 確保所有視窗都被關閉
        for name in window_names:
            cv2.destroyWindow(name)

# 主程式
if __name__ == "__main__":
    try:
        # 初始化影像隊列
        for _ in range(len(camera_sources)):
            # 設定隊列大小為1，只保留最新的影像
            frame_queues.append(queue.Queue(maxsize=2))
        
        # 創建並啟動每個相機的擷取執行緒
        capture_threads = []
        for i, camera_source in enumerate(camera_sources):
            t = threading.Thread(target=capture_stream, args=(camera_source, i))
            t.daemon = True  # 設置為守護執行緒，主程序結束時自動終止
            capture_threads.append(t)
            t.start()
            time.sleep(0.5)  # 間隔啟動，避免資源衝突
        
        # 在主執行緒中顯示所有相機的視窗
        show_all_cameras()
        
    except Exception as e:
        print(f"主程序異常: {e}")
    finally:
        # 設置停止事件，通知所有執行緒退出
        stop_event.set()
        
        # 等待所有執行緒結束（設置超時以避免程序卡住）
        for t in capture_threads:
            t.join(timeout=2)
        
        # 確保所有視窗都被關閉
        cv2.destroyAllWindows()
        print("程式已結束")