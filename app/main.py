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
            result = DeepFace.analyze(face_image, actions=['age', 'gender'])
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            self.gender_age_data[person_id] = {
                'gender': result.get('gender', None),
                'age': result.get('age', None)
            }
            logger.info(f"人物ID {person_id} 的人口統計分析：性別={result.get('gender')}, 年齡={result.get('age')}")
        except Exception as e:
            logger.error(f"人口統計分析錯誤：{str(e)}")
    
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
                
                while self.running:
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.error("無法讀取攝像頭畫面")
                        break
                    
                    # 處理影像
                    self.process_frame(frame)
                    
            except Exception as e:
                logger.error(f"攝像頭處理發生錯誤: {str(e)}")
                if retry_count < max_retries - 1:
                    retry_count += 1
                    time.sleep(retry_delay)
                    continue
                break
            finally:
                if self.cap is not None:
                    self.cap.release()

    def process_frame(self, frame):
        # 找到所有的臉
        face_locations = face_recognition.face_locations(frame)
        if len(face_locations) > 0:
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # 嘗試識別人臉
                person_id = self.person_tracker.recognize_person(face_encoding)
                current_time = time.time()
                
                if person_id is None:
                    # 新的訪客
                    top, right, bottom, left = face_location
                    face_image = frame[top:bottom, left:right]
                    person_id = self.person_tracker.add_person(
                        face_encoding, 
                        self.booth_id, 
                        current_time,
                        face_image
                    )
                else:
                    # 更新已知訪客的位置
                    self.person_tracker.update_person(person_id, self.booth_id, current_time)

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
            while True:
                time.sleep(1)  # 主迴圈休眠
        except KeyboardInterrupt:
            logger.info("正在關閉程式...")
        finally:
            # 停止所有攝像頭
            for camera in cameras:
                camera.stop()
    
    except Exception as e:
        logger.error(f"程式執行錯誤: {str(e)}")

if __name__ == "__main__":
    main()
