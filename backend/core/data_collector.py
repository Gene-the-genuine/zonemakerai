import json
import time
import threading
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import win32gui
import win32process
import win32api
import win32con
import psutil
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WindowInfo:
    """윈도우 정보를 담는 데이터 클래스"""
    handle: int
    title: str
    class_name: str
    process_id: int
    process_name: str
    rect: Tuple[int, int, int, int]  # (left, top, right, bottom)
    is_visible: bool
    is_minimized: bool
    is_maximized: bool
    z_order: int
    timestamp: float

@dataclass
class UserActivity:
    """사용자 활동 정보를 담는 데이터 클래스"""
    timestamp: float
    active_window: Optional[WindowInfo]
    all_windows: List[WindowInfo]
    mouse_position: Tuple[int, int]
    keyboard_active: bool
    activity_type: str  # 'click', 'keyboard', 'window_change', 'idle'

class DataCollector:
    """실시간 윈도우 데이터 수집기"""
    
    def __init__(self):
        self.last_activity_time = time.time()
        self.last_window_states = {}
        self.collection_active = False
        self.collection_thread = None
        
    def get_window_info(self, hwnd: int) -> Optional[WindowInfo]:
        """단일 윈도우 정보 수집"""
        try:
            if not win32gui.IsWindow(hwnd):
                return None
                
            # 윈도우가 보이는지 확인
            if not win32gui.IsWindowVisible(hwnd):
                return None
                
            # 윈도우 제목 가져오기
            title = win32gui.GetWindowText(hwnd)
            if not title:  # 제목이 없으면 건너뛰기
                return None
                
            # 윈도우 클래스명 가져오기
            class_name = win32gui.GetClassName(hwnd)
            
            # 프로세스 ID 가져오기
            try:
                _, process_id = win32process.GetWindowThreadProcessId(hwnd)
                process = psutil.Process(process_id)
                process_name = process.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                process_id = 0
                process_name = "Unknown"
            
            # 윈도우 위치와 크기 가져오기
            rect = win32gui.GetWindowRect(hwnd)
            
            # 윈도우 상태 확인 (실제 존재하는 메서드만 사용)
            is_visible = win32gui.IsWindowVisible(hwnd)
            
            # 최소화/최대화 상태 확인
            placement = win32gui.GetWindowPlacement(hwnd)
            is_minimized = placement[1] == win32con.SW_SHOWMINIMIZED
            is_maximized = placement[1] == win32con.SW_SHOWMAXIMIZED
            
            # Z-order는 간단한 방법으로 추정 (EnumWindows 순서)
            z_order = 0  # 실제 Z-order는 복잡하므로 0으로 설정
            
            return WindowInfo(
                handle=hwnd,
                title=title,
                class_name=class_name,
                process_id=process_id,
                process_name=process_name,
                rect=rect,
                is_visible=is_visible,
                is_minimized=is_minimized,
                is_maximized=is_maximized,
                z_order=z_order,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.warning(f"윈도우 정보 수집 실패 (handle: {hwnd}): {e}")
            return None
    
    def get_active_window_info(self) -> Optional[WindowInfo]:
        """현재 활성 윈도우 정보 수집"""
        try:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                return self.get_window_info(hwnd)
        except Exception as e:
            logger.error(f"활성 윈도우 정보 수집 실패: {e}")
        return None
    
    def get_all_windows(self) -> List[WindowInfo]:
        """모든 보이는 윈도우 정보 수집"""
        windows = []
        
        def enum_windows_callback(hwnd, windows_list):
            try:
                window_info = self.get_window_info(hwnd)
                if window_info:
                    windows_list.append(window_info)
            except Exception as e:
                logger.warning(f"윈도우 열거 중 오류: {e}")
            return True
        
        try:
            win32gui.EnumWindows(enum_windows_callback, windows)
        except Exception as e:
            logger.error(f"윈도우 열거 실패: {e}")
        
        return windows
    
    def get_mouse_position(self) -> Tuple[int, int]:
        """현재 마우스 위치 가져오기"""
        try:
            cursor_pos = win32gui.GetCursorPos()
            return cursor_pos
        except Exception as e:
            logger.error(f"마우스 위치 가져오기 실패: {e}")
            return (0, 0)
    
    def detect_activity_changes(self, current_windows: List[WindowInfo]) -> bool:
        """윈도우 상태 변화 감지"""
        try:
            current_states = {}
            for window in current_windows:
                key = f"{window.handle}_{window.rect}_{window.is_minimized}_{window.is_maximized}"
                current_states[key] = window.timestamp
            
            # 이전 상태와 비교
            if self.last_window_states != current_states:
                self.last_window_states = current_states.copy()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"활동 변화 감지 실패: {e}")
            return False
    
    def collect_activity_sample(self) -> UserActivity:
        """현재 시점의 활동 샘플 수집"""
        try:
            timestamp = time.time()
            active_window = self.get_active_window_info()
            all_windows = self.get_all_windows()
            mouse_position = self.get_mouse_position()
            
            # 키보드 활동 여부 (간단한 방법)
            keyboard_active = False
            try:
                # GetAsyncKeyState로 주요 키들의 상태 확인
                keys_to_check = [win32con.VK_SHIFT, win32con.VK_CONTROL, win32con.VK_MENU]
                for key in keys_to_check:
                    if win32api.GetAsyncKeyState(key) & 0x8000:
                        keyboard_active = True
                        break
            except:
                pass
            
            # 활동 타입 결정
            activity_type = 'idle'
            if self.detect_activity_changes(all_windows):
                activity_type = 'window_change'
            elif keyboard_active:
                activity_type = 'keyboard'
            elif mouse_position != (0, 0):
                activity_type = 'click'
            
            self.last_activity_time = timestamp
            
            return UserActivity(
                timestamp=timestamp,
                active_window=active_window,
                all_windows=all_windows,
                mouse_position=mouse_position,
                keyboard_active=keyboard_active,
                activity_type=activity_type
            )
            
        except Exception as e:
            logger.error(f"활동 샘플 수집 실패: {e}")
            return None
    
    def start_continuous_collection(self, duration_seconds: int = 30, callback=None):
        """연속 데이터 수집 시작 (30초 관찰)"""
        if self.collection_active:
            logger.warning("이미 데이터 수집이 진행 중입니다.")
            return
        
        self.collection_active = True
        self.collection_thread = threading.Thread(
            target=self._continuous_collection_worker,
            args=(duration_seconds, callback),
            daemon=True
        )
        self.collection_thread.start()
        logger.info(f"연속 데이터 수집 시작 (지속 시간: {duration_seconds}초)")
    
    def _continuous_collection_worker(self, duration_seconds: int, callback=None):
        """연속 데이터 수집 워커 스레드"""
        try:
            start_time = time.time()
            samples = []
            
            while self.collection_active and (time.time() - start_time) < duration_seconds:
                sample = self.collect_activity_sample()
                if sample:
                    samples.append(sample)
                
                # 0.1초마다 샘플링
                time.sleep(0.1)
            
            if callback and samples:
                callback(samples)
                
        except Exception as e:
            logger.error(f"연속 데이터 수집 중 오류: {e}")
        finally:
            self.collection_active = False
    
    def stop_collection(self):
        """데이터 수집 중지"""
        self.collection_active = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=1.0)
        logger.info("데이터 수집 중지됨")
    
    def collect_data_sample(self, duration_seconds: int = 30) -> List[UserActivity]:
        """지정된 시간 동안 데이터 샘플 수집 (동기 방식)"""
        samples = []
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration_seconds:
                sample = self.collect_activity_sample()
                if sample:
                    samples.append(sample)
                
                # 0.1초마다 샘플링
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"데이터 샘플 수집 실패: {e}")
        
        logger.info(f"데이터 샘플 수집 완료: {len(samples)}개 샘플")
        return samples
    
    def save_data(self, activities: List[UserActivity], filename: str = None):
        """수집된 데이터를 JSON 파일로 저장"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"window_activity_data_{timestamp}.json"
        
        try:
            # 데이터를 JSON 직렬화 가능한 형태로 변환
            data_to_save = []
            for activity in activities:
                activity_dict = asdict(activity)
                # WindowInfo 객체들도 딕셔너리로 변환
                if activity.active_window:
                    activity_dict['active_window'] = asdict(activity.active_window)
                activity_dict['all_windows'] = [asdict(w) for w in activity.all_windows]
                data_to_save.append(activity_dict)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
            logger.info(f"데이터 저장 완료: {filename}")
            
        except Exception as e:
            logger.error(f"데이터 저장 실패: {e}")
    
    def load_data(self, filename: str) -> List[UserActivity]:
        """JSON 파일에서 데이터 로드"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSON 데이터를 UserActivity 객체로 변환
            activities = []
            for item in data:
                # WindowInfo 객체들 복원
                active_window = None
                if item.get('active_window'):
                    active_window = WindowInfo(**item['active_window'])
                
                all_windows = [WindowInfo(**w) for w in item.get('all_windows', [])]
                
                activity = UserActivity(
                    timestamp=item['timestamp'],
                    active_window=active_window,
                    all_windows=all_windows,
                    mouse_position=tuple(item['mouse_position']),
                    keyboard_active=item['keyboard_active'],
                    activity_type=item['activity_type']
                )
                activities.append(activity)
            
            logger.info(f"데이터 로드 완료: {filename} ({len(activities)}개 샘플)")
            return activities
            
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return []

if __name__ == "__main__":
    # 테스트 실행
    collector = DataCollector()
    
    print("30초간 데이터 수집 테스트...")
    activities = collector.collect_data_sample(30)
    
    print(f"수집된 샘플 수: {len(activities)}")
    if activities:
        print("첫 번째 샘플:")
        print(f"  타임스탬프: {activities[0].timestamp}")
        if activities[0].active_window:
            print(f"  활성 윈도우: {activities[0].active_window.title}")
        print(f"  마우스 위치: ({activities[0].mouse_position[0]}, {activities[0].mouse_position[1]})")
        print(f"  키보드 활성: {activities[0].keyboard_active}")
        print(f"  활동 타입: {activities[0].activity_type}")
    
    # 데이터 저장
    collector.save_data(activities, "test_collection.json")
