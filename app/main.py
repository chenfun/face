import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
import time
import datetime
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import threading
import logging
import subprocess
from pymongo import MongoClient

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 設定MongoDB連接
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = 'exhibition_analytics'

class PersonTracker:
    def __init__(self):
        self.next_id = 0
        self.persons = {}  # 格式: {person_id: {'face_encoding': encoding, 'last_seen': timestamp, 'first_seen': timestamp}}
        self.location_history = defaultdict(list)  # 格式: {person_id: [(booth_id, enter_time, exit_time)]}
        self.current_location = {}  # 格式: {person_id: (booth_id, enter_time)}
        self.gender_age_data = {}  # 格式: {person_id: {'gender': gender, 'age': age}}
        
    def add_person(self, face_encoding, booth_id, timestamp, face_image=None):
        person_id = self.next_id
        self.next_id += 1
        self.persons[person_id] = {
            'face_encoding': face_encoding,
            'last_seen': timestamp,
            'first_seen': timestamp
        }
        self.current_location[person_id] = (booth_id, timestamp)
        
        # 在背景執行性別和年齡分析
        if face_image is not None:
            threading.Thread(target=self.analyze_demographics, args=(person_id, face_image)).start()
            
        return person_id
    
    def analyze_demographics(self, person_id, face_image):
        try:
            # 確保圖像尺寸適當
            if face_image.shape[0] < 20 or face_image.shape[1] < 20:
                logger.warning(f"人臉圖像太小，無法進行分析: {face_image.shape}")
                # 設定默認值
                self.gender_age_data[person_id] = {
                    'gender': 'N/A',
                    'age': 'N/A'  # 默認年齡
                }
                return
                
            # 檢查圖像是否為彩色
            if len(face_image.shape) < 3:
                logger.warning("圖像不是彩色的，轉換為彩色")
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
            
            # 放大圖像以提高分析準確度
            face_image = cv2.resize(face_image, (0, 0), fx=2, fy=2)
            
            # 使用簡化的參數調用DeepFace
            result = DeepFace.analyze(
                img_path=face_image, 
                actions=['age', 'gender'],
                enforce_detection=False,  # 避免再次進行人臉檢測
                silent=True  # 減少輸出日誌
            )
            
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            gender = result.get('gender', '未知')
            age = result.get('age', 30)
            
            self.gender_age_data[person_id] = {
                'gender': gender,
                'age': age
            }
            logger.info(f"人物ID {person_id} 的人口統計分析：性別={gender}, 年齡={age}")
        except Exception as e:
            logger.error(f"人口統計分析錯誤：{str(e)}")
            # 發生錯誤時設定默認值
            self.gender_age_data[person_id] = {
                'gender': '未知',
                'age': 30
            }
    
    def update_person(self, person_id, booth_id, timestamp):
        if person_id in self.persons:
            self.persons[person_id]['last_seen'] = timestamp
            
            # 如果攤位變化，記錄離開前一個攤位的時間並記錄進入新攤位的時間
            if person_id in self.current_location:
                prev_booth, enter_time = self.current_location[person_id]
                if prev_booth != booth_id:
                    self.location_history[person_id].append((prev_booth, enter_time, timestamp))
                    self.current_location[person_id] = (booth_id, timestamp)
            else:
                self.current_location[person_id] = (booth_id, timestamp)
    
    def recognize_person(self, face_encoding, tolerance=0.6):
        """嘗試識別一個人，如果無法識別則返回None"""
        if len(self.persons) == 0:
            return None
            
        known_encodings = [person['face_encoding'] for person in self.persons.values()]
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        min_distance_idx = np.argmin(distances)
        
        if distances[min_distance_idx] <= tolerance:
            return list(self.persons.keys())[min_distance_idx]
        return None
    
    def get_statistics(self):
        """獲取各種統計數據"""
        total_visitors = len(self.persons)
        
        # 性別統計

class BoothCamera:
    def __init__(self, booth_id, camera_source, person_tracker):
        self.booth_id = booth_id
        self.camera_source = camera_source
        self.person_tracker = person_tracker
        self.cap = None
        self.running = False
        self.thread = None
        self.current_display_frame = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run)
            self.thread.daemon = True
            self.thread.start()

    def run(self):
        max_retries = 3
        retry_count = 0
        retry_delay = 2  # 秒

        while retry_count < max_retries and self.running:
            try:
                logger.info(f"嘗試開啟攝像頭 {self.camera_source} (重試次數: {retry_count})")
                self.cap = cv2.VideoCapture(self.camera_source)
                
                if not self.cap.isOpened():
                    logger.error(f"無法開啟攝像頭來源: {self.camera_source}")
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        time.sleep(retry_delay)
                        continue
                    return

                logger.info(f"成功開啟攝像頭 {self.camera_source}")
                
                # 避免在執行緒中創建視窗，先檢查攝像頭是否可正常讀取
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    logger.error("無法讀取攝像頭測試畫面")
                    raise Exception("無法讀取攝像頭測試畫面")
                    
                logger.info("相機讀取測試成功")
                
                # 正常相機處理循環
                while self.running:
                    try:
                        ret, frame = self.cap.read()
                        if not ret:
                            logger.error("無法讀取攝像頭畫面")
                            break
                        
                        # 處理影像（進階功能）
                        try:
                            # 創建副本以避免修改原始幀
                            process_frame = frame.copy()
                            self.process_frame(process_frame)
                            # 釋放記憶體
                            del process_frame
                        except Exception as process_error:
                            logger.error(f"處理畫面時發生錯誤: {str(process_error)}")
                            # 繼續嘗試讀取畫面，但不進行處理
                            time.sleep(0.1)
                    except Exception as frame_error:
                        logger.error(f"讀取畫面時發生錯誤: {str(frame_error)}")
                        time.sleep(0.5)  # 短暫休眠後重試
                    
            except Exception as e:
                logger.error(f"攝像頭處理發生錯誤: {str(e)}")
                # 列出更詳細的錯誤資訊
                import traceback
                logger.error(f"詳細錯誤追蹤:\n{traceback.format_exc()}")
                
                if retry_count < max_retries - 1:
                    retry_count += 1
                    time.sleep(retry_delay)
                    continue
                break
            finally:
                if self.cap is not None:
                    self.cap.release()

    def process_frame(self, frame):
        try:
            # 找到所有的臉
            face_locations = face_recognition.face_locations(frame)
            
            # 顯示攝像頭畫面
            display_frame = frame.copy()
            
            # 列印偵測到的人臉資料
            if len(face_locations) > 0:
                logger.info(f"攝像頭 {self.booth_id} 偵測到 {len(face_locations)} 個人臉")
                for i, face_location in enumerate(face_locations):
                    top, right, bottom, left = face_location
                    logger.info(f"人臉 #{i+1} - 位置: 上={top}, 右={right}, 下={bottom}, 左={left}, 尺寸: {bottom-top}x{right-left}")
            
            if len(face_locations) > 0:
                try:
                    face_encodings = face_recognition.face_encodings(frame, face_locations)
                    
                    for face_encoding, face_location in zip(face_encodings, face_locations):
                        # 嘗試識別人臉
                        person_id = self.person_tracker.recognize_person(face_encoding)
                        current_time = time.time()
                        
                        # 在畫面上標記臉部位置
                        top, right, bottom, left = face_location
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        
                        if person_id is None:
                            # 新的訪客
                            face_img = frame[top:bottom, left:right].copy()  # 創建副本
                            person_id = self.person_tracker.add_person(
                                face_encoding, 
                                self.booth_id, 
                                current_time,
                                face_img
                            )
                            # 設定初始默認值，等待背景線程更新
                            self.person_tracker.gender_age_data[person_id] = {
                                'gender': '分析中',
                                'age': '--'
                            }
                        else:
                            # 更新已知訪客的位置
                            self.person_tracker.update_person(person_id, self.booth_id, current_time)
                        
                        # 顯示人物ID
                        cv2.putText(display_frame, f"ID: {person_id}", (left, top - 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 始終顯示性別和年齡資訊
                        gender = "未知"
                        age = "未知"
                        
                        # 如果有性別和年齡數據，則使用；否則使用默認值
                        if person_id in self.person_tracker.gender_age_data:
                            gender_data = self.person_tracker.gender_age_data[person_id].get('gender')
                            age_data = self.person_tracker.gender_age_data[person_id].get('age')
                            
                            # 處理性別顯示
                            if gender_data == 'Woman' or gender_data == 'Female':
                                gender = '女'
                            elif gender_data == 'Man' or gender_data == 'Male':
                                gender = '男'
                            else:
                                gender = gender_data if gender_data else "未知"
                            
                            # 處理年齡顯示
                            if age_data:
                                if isinstance(age_data, (int, float)):
                                    age = str(int(age_data))
                                else:
                                    age = str(age_data)
                        
                        # 顯示性別
                        cv2.putText(display_frame, f"性別: {gender}", (left, top - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        # 顯示年齡
                        cv2.putText(display_frame, f"年齡: {age}", (left, top + 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    
                except Exception as encode_error:
                    logger.error(f"處理人臉編碼時發生錯誤: {str(encode_error)}")
            
            # 避免在多執行緒環境中直接顯示窗口
            # 使用類屬性來存儲當前幀，讓主執行緒進行顯示
            self.current_display_frame = display_frame
            
        except Exception as e:
            logger.error(f"處理畫面詳細錯誤: {str(e)}")
            import traceback
            logger.error(f"人臉處理錯誤追蹤:\n{traceback.format_exc()}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()

def main():
    try:
        # 初始化人員跟踪器
        person_tracker = PersonTracker()
        
        # 在 macOS 上使用系統默認相機
        camera_source = 0  # macOS 使用數字索引來訪問相機
        
        # 如果是 macOS，嘗試其他相機索引
        if not cv2.VideoCapture(camera_source).isOpened():
            for i in range(1, 10):  # 嘗試不同的相機索引
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_source = i
                    cap.release()
                    break
        
        cameras = [
            BoothCamera(1, camera_source, person_tracker)
        ]
        
        # 啟動所有攝像頭
        for camera in cameras:
            camera.start()
        
        try:
            # 在主執行緒中顯示窗口，避免多執行緒的 OpenCV GUI 問題
            logger.info("開始主畫面顯示循環")
            while True:
                # 主執行緒顯示窗口
                for camera in cameras:
                    if camera.current_display_frame is not None:
                        try:
                            cv2.imshow(f"Booth {camera.booth_id}", camera.current_display_frame)
                            key = cv2.waitKey(1)  # 必須調用waitKey以更新窗口
                            
                            # 按 ESC 鍵退出
                            if key == 27:
                                raise KeyboardInterrupt
                        except Exception as display_error:
                            logger.error(f"顯示窗口錯誤: {str(display_error)}")
                
                time.sleep(0.033)  # 約 30 FPS，更平滑的刷新率
        except KeyboardInterrupt:
            logger.info("正在關閉程式...")
        finally:
            # 停止所有攝像頭
            for camera in cameras:
                camera.stop()
            # 關閉所有視窗
            cv2.destroyAllWindows()
    
    except Exception as e:
        logger.error(f"程式執行錯誤: {str(e)}")
        import traceback
        logger.error(f"主程式錯誤追蹤:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
