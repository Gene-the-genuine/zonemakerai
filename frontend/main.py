#!/usr/bin/env python3
"""
Zonemaker AI - 프론트엔드
PySide6 기반 사용자 인터페이스
"""

import sys
import os
import json
import requests
from typing import Dict, List, Optional
from datetime import datetime

# PySide6 임포트
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QTextEdit, QListWidget, QListWidgetItem,
        QProgressBar, QMessageBox, QInputDialog, QLineEdit, QTabWidget,
        QGroupBox, QGridLayout, QSpinBox, QComboBox, QCheckBox
    )
    from PySide6.QtCore import QThread, QTimer, pyqtSignal, Qt
    from PySide6.QtGui import QFont, QPixmap, QIcon
except ImportError:
    print("PySide6가 설치되어 있지 않습니다.")
    print("pip install PySide6[all]로 설치하세요.")
    sys.exit(1)

# API 클라이언트
class APIClient:
    """백엔드 API 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get(self, endpoint: str) -> Optional[Dict]:
        """GET 요청"""
        try:
            response = self.session.get(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API GET 오류: {e}")
            return None
    
    def post(self, endpoint: str, data: Dict = None) -> Optional[Dict]:
        """POST 요청"""
        try:
            response = self.session.post(f"{self.base_url}{endpoint}", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API POST 오류: {e}")
            return None
    
    def delete(self, endpoint: str) -> Optional[Dict]:
        """DELETE 요청"""
        try:
            response = self.session.delete(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API DELETE 오류: {e}")
            return None

# 백그라운드 작업 스레드
class BackgroundWorker(QThread):
    """백그라운드 작업 스레드"""
    
    # 시그널 정의
    data_collection_progress = pyqtSignal(int, int)  # current, total
    training_progress = pyqtSignal(int, int)  # current, total
    inference_status = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    work_completed = pyqtSignal(dict)
    
    def __init__(self, api_client: APIClient):
        super().__init__()
        self.api_client = api_client
        self.work_type = None
        self.work_params = {}
    
    def set_work(self, work_type: str, **params):
        """작업 설정"""
        self.work_type = work_type
        self.work_params = params
    
    def run(self):
        """작업 실행"""
        try:
            if self.work_type == "data_collection":
                self._run_data_collection()
            elif self.work_type == "training":
                self._run_training()
            elif self.work_type == "inference":
                self._run_inference()
            else:
                self.error_occurred.emit("알 수 없는 작업 유형")
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _run_data_collection(self):
        """데이터 수집 실행"""
        duration = self.work_params.get('duration', 60)
        
        # 데이터 수집 시작
        result = self.api_client.post("/data/collect", {"duration_seconds": duration})
        if not result:
            self.error_occurred.emit("데이터 수집 시작 실패")
            return
        
        # 진행률 모니터링
        for i in range(duration):
            if self.isInterruptionRequested():
                break
            
            progress = int((i + 1) / duration * 100)
            self.data_collection_progress.emit(i + 1, duration)
            self.msleep(1000)  # 1초 대기
        
        self.work_completed.emit({"type": "data_collection", "result": result})
    
    def _run_training(self):
        """모델 훈련 실행"""
        workstation_name = self.work_params.get('workstation_name', 'default')
        
        # 훈련 시작
        result = self.api_client.post("/training/start", {
            "workstation_name": workstation_name,
            "duration_minutes": 10
        })
        
        if not result:
            self.error_occurred.emit("훈련 시작 실패")
            return
        
        # 훈련 상태 모니터링
        max_epochs = 50
        for epoch in range(max_epochs):
            if self.isInterruptionRequested():
                break
            
            # 훈련 상태 확인
            status = self.api_client.get("/training/status")
            if status and status.get('status') == 'completed':
                break
            
            progress = int((epoch + 1) / max_epochs * 100)
            self.training_progress.emit(epoch + 1, max_epochs)
            self.msleep(2000)  # 2초 대기
        
        self.work_completed.emit({"type": "training", "result": result})
    
    def _run_inference(self):
        """추론 실행"""
        model_path = self.work_params.get('model_path', '')
        
        # 추론 시작
        result = self.api_client.post("/inference/start", {
            "model_path": model_path,
            "inference_interval": 1.0,
            "max_inference_time": 0.5
        })
        
        if not result:
            self.error_occurred.emit("추론 시작 실패")
            return
        
        # 추론 상태 모니터링
        for i in range(10):  # 10초간 모니터링
            if self.isInterruptionRequested():
                break
            
            status = self.api_client.get("/inference/status")
            if status:
                self.inference_status.emit(status)
            
            self.msleep(1000)  # 1초 대기
        
        self.work_completed.emit({"type": "inference", "result": result})

# 메인 윈도우
class MainWindow(QMainWindow):
    """메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.api_client = APIClient()
        self.background_worker = BackgroundWorker(self.api_client)
        self.setup_ui()
        self.setup_connections()
        self.refresh_data()
    
    def setup_ui(self):
        """UI 설정"""
        self.setWindowTitle("Zonemaker AI - 윈도우 배열 최적화 시스템")
        self.setGeometry(100, 100, 1200, 800)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)
        
        # 제목
        title_label = QLabel("Zonemaker AI")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 서브타이틀
        subtitle_label = QLabel("AI 기반 윈도우 배열 최적화 시스템")
        subtitle_label.setFont(QFont("Arial", 14))
        subtitle_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle_label)
        
        # 탭 위젯
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 탭 생성
        self.create_home_tab()
        self.create_workstation_tab()
        self.create_training_tab()
        self.create_inference_tab()
        self.create_system_tab()
    
    def create_home_tab(self):
        """홈 탭 생성"""
        home_widget = QWidget()
        layout = QVBoxLayout(home_widget)
        
        # 시스템 상태
        status_group = QGroupBox("시스템 상태")
        status_layout = QGridLayout(status_group)
        
        self.backend_status_label = QLabel("백엔드: 연결 확인 중...")
        self.training_status_label = QLabel("훈련: 대기 중")
        self.inference_status_label = QLabel("추론: 중지됨")
        
        status_layout.addWidget(QLabel("백엔드 상태:"), 0, 0)
        status_layout.addWidget(self.backend_status_label, 0, 1)
        status_layout.addWidget(QLabel("훈련 상태:"), 1, 0)
        status_layout.addWidget(self.training_status_label, 1, 1)
        status_layout.addWidget(QLabel("추론 상태:"), 2, 0)
        status_layout.addWidget(self.inference_status_label, 2, 1)
        
        layout.addWidget(status_group)
        
        # 빠른 액션
        action_group = QGroupBox("빠른 액션")
        action_layout = QHBoxLayout(action_group)
        
        self.refresh_btn = QPushButton("새로고침")
        self.health_check_btn = QPushButton("헬스 체크")
        self.system_info_btn = QPushButton("시스템 정보")
        
        action_layout.addWidget(self.refresh_btn)
        action_layout.addWidget(self.health_check_btn)
        action_layout.addWidget(self.system_info_btn)
        
        layout.addWidget(action_group)
        
        # 로그 영역
        log_group = QGroupBox("시스템 로그")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        self.tab_widget.addTab(home_widget, "홈")
    
    def create_workstation_tab(self):
        """워크스테이션 탭 생성"""
        workstation_widget = QWidget()
        layout = QVBoxLayout(workstation_widget)
        
        # 워크스테이션 생성
        create_group = QGroupBox("워크스테이션 생성")
        create_layout = QGridLayout(create_group)
        
        create_layout.addWidget(QLabel("이름:"), 0, 0)
        self.ws_name_input = QLineEdit()
        create_layout.addWidget(self.ws_name_input, 0, 1)
        
        create_layout.addWidget(QLabel("설명:"), 1, 0)
        self.ws_desc_input = QLineEdit()
        create_layout.addWidget(self.ws_desc_input, 1, 1)
        
        create_layout.addWidget(QLabel("프로그램:"), 2, 0)
        self.ws_programs_input = QLineEdit()
        self.ws_programs_input.setPlaceholderText("프로그램1,프로그램2,프로그램3")
        create_layout.addWidget(self.ws_programs_input, 2, 1)
        
        self.create_ws_btn = QPushButton("워크스테이션 생성")
        create_layout.addWidget(self.create_ws_btn, 3, 0, 1, 2)
        
        layout.addWidget(create_group)
        
        # 워크스테이션 목록
        list_group = QGroupBox("워크스테이션 목록")
        list_layout = QVBoxLayout(list_group)
        
        self.ws_list = QListWidget()
        list_layout.addWidget(self.ws_list)
        
        ws_buttons_layout = QHBoxLayout()
        self.refresh_ws_btn = QPushButton("새로고침")
        self.delete_ws_btn = QPushButton("삭제")
        
        ws_buttons_layout.addWidget(self.refresh_ws_btn)
        ws_buttons_layout.addWidget(self.delete_ws_btn)
        
        list_layout.addLayout(ws_buttons_layout)
        
        layout.addWidget(list_group)
        
        self.tab_widget.addTab(workstation_widget, "워크스테이션")
    
    def create_training_tab(self):
        """훈련 탭 생성"""
        training_widget = QWidget()
        layout = QVBoxLayout(training_widget)
        
        # 훈련 설정
        config_group = QGroupBox("훈련 설정")
        config_layout = QGridLayout(config_group)
        
        config_layout.addWidget(QLabel("워크스테이션:"), 0, 0)
        self.training_ws_combo = QComboBox()
        config_layout.addWidget(self.training_ws_combo, 0, 1)
        
        config_layout.addWidget(QLabel("지속 시간(분):"), 1, 0)
        self.training_duration_spin = QSpinBox()
        self.training_duration_spin.setRange(1, 60)
        self.training_duration_spin.setValue(10)
        config_layout.addWidget(self.training_duration_spin, 1, 1)
        
        self.start_training_btn = QPushButton("훈련 시작")
        self.stop_training_btn = QPushButton("훈련 중지")
        self.stop_training_btn.setEnabled(False)
        
        config_layout.addWidget(self.start_training_btn, 2, 0)
        config_layout.addWidget(self.stop_training_btn, 2, 1)
        
        layout.addWidget(config_group)
        
        # 훈련 진행률
        progress_group = QGroupBox("훈련 진행률")
        progress_layout = QVBoxLayout(progress_group)
        
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        progress_layout.addWidget(self.training_progress)
        
        self.training_status_text = QLabel("훈련 대기 중...")
        progress_layout.addWidget(self.training_status_text)
        
        layout.addWidget(progress_group)
        
        self.tab_widget.addTab(training_widget, "모델 훈련")
    
    def create_inference_tab(self):
        """추론 탭 생성"""
        inference_widget = QWidget()
        layout = QVBoxLayout(inference_widget)
        
        # 추론 설정
        config_group = QGroupBox("추론 설정")
        config_layout = QGridLayout(config_group)
        
        config_layout.addWidget(QLabel("모델 경로:"), 0, 0)
        self.inference_model_input = QLineEdit()
        self.inference_model_input.setPlaceholderText("data/models/workstation_xxx_best.pth")
        config_layout.addWidget(self.inference_model_input, 0, 1)
        
        config_layout.addWidget(QLabel("추론 간격(초):"), 1, 0)
        self.inference_interval_spin = QSpinBox()
        self.inference_interval_spin.setRange(1, 10)
        self.inference_interval_spin.setValue(1)
        self.inference_interval_spin.setSuffix("초")
        config_layout.addWidget(self.inference_interval_spin, 1, 1)
        
        self.start_inference_btn = QPushButton("추론 시작")
        self.stop_inference_btn = QPushButton("추론 중지")
        self.stop_inference_btn.setEnabled(False)
        
        config_layout.addWidget(self.start_inference_btn, 2, 0)
        config_layout.addWidget(self.stop_inference_btn, 2, 1)
        
        layout.addWidget(config_group)
        
        # 추론 상태
        status_group = QGroupBox("추론 상태")
        status_layout = QVBoxLayout(status_group)
        
        self.inference_status_text = QLabel("추론 중지됨")
        status_layout.addWidget(self.inference_status_text)
        
        self.inference_stats_text = QTextEdit()
        self.inference_stats_text.setMaximumHeight(150)
        self.inference_stats_text.setReadOnly(True)
        status_layout.addWidget(self.inference_stats_text)
        
        layout.addWidget(status_group)
        
        self.tab_widget.addTab(inference_widget, "실시간 추론")
    
    def create_system_tab(self):
        """시스템 탭 생성"""
        system_widget = QWidget()
        layout = QVBoxLayout(system_widget)
        
        # 시스템 정보
        info_group = QGroupBox("시스템 정보")
        info_layout = QGridLayout(info_group)
        
        self.system_info_text = QTextEdit()
        self.system_info_text.setMaximumHeight(200)
        self.system_info_text.setReadOnly(True)
        
        info_layout.addWidget(self.system_info_text, 0, 0, 1, 2)
        
        self.refresh_system_info_btn = QPushButton("시스템 정보 새로고침")
        info_layout.addWidget(self.refresh_system_info_btn, 1, 0, 1, 2)
        
        layout.addWidget(info_group)
        
        # 데이터 수집
        data_group = QGroupBox("데이터 수집")
        data_layout = QGridLayout(data_group)
        
        data_layout.addWidget(QLabel("지속 시간(초):"), 0, 0)
        self.data_duration_spin = QSpinBox()
        self.data_duration_spin.setRange(60, 3600)
        self.data_duration_spin.setValue(600)
        self.data_duration_spin.setSuffix("초")
        data_layout.addWidget(self.data_duration_spin, 0, 1)
        
        self.start_data_collection_btn = QPushButton("데이터 수집 시작")
        data_layout.addWidget(self.start_data_collection_btn, 1, 0, 1, 2)
        
        layout.addWidget(data_group)
        
        # NPU 변환
        npu_group = QGroupBox("NPU 변환")
        npu_layout = QGridLayout(npu_group)
        
        npu_layout.addWidget(QLabel("모델 경로:"), 0, 0)
        self.npu_model_input = QLineEdit()
        self.npu_model_input.setPlaceholderText("data/models/workstation_xxx_best.pth")
        npu_layout.addWidget(self.npu_model_input, 0, 1)
        
        self.convert_npu_btn = QPushButton("NPU 변환")
        npu_layout.addWidget(self.convert_npu_btn, 1, 0, 1, 2)
        
        layout.addWidget(npu_group)
        
        self.tab_widget.addTab(system_widget, "시스템")
    
    def setup_connections(self):
        """시그널-슬롯 연결"""
        # 홈 탭
        self.refresh_btn.clicked.connect(self.refresh_data)
        self.health_check_btn.clicked.connect(self.health_check)
        self.system_info_btn.clicked.connect(self.show_system_info)
        
        # 워크스테이션 탭
        self.create_ws_btn.clicked.connect(self.create_workstation)
        self.refresh_ws_btn.clicked.connect(self.refresh_workstations)
        self.delete_ws_btn.clicked.connect(self.delete_workstation)
        
        # 훈련 탭
        self.start_training_btn.clicked.connect(self.start_training)
        self.stop_training_btn.clicked.connect(self.stop_training)
        
        # 추론 탭
        self.start_inference_btn.clicked.connect(self.start_inference)
        self.stop_inference_btn.clicked.connect(self.stop_inference)
        
        # 시스템 탭
        self.refresh_system_info_btn.clicked.connect(self.show_system_info)
        self.start_data_collection_btn.clicked.connect(self.start_data_collection)
        self.convert_npu_btn.clicked.connect(self.convert_to_npu)
        
        # 백그라운드 워커
        self.background_worker.data_collection_progress.connect(self.update_data_collection_progress)
        self.background_worker.training_progress.connect(self.update_training_progress)
        self.background_worker.inference_status.connect(self.update_inference_status)
        self.background_worker.error_occurred.connect(self.show_error)
        self.background_worker.work_completed.connect(self.work_completed)
        
        # 타이머
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(5000)  # 5초마다 상태 업데이트
    
    def refresh_data(self):
        """데이터 새로고침"""
        self.log_message("데이터 새로고침 시작...")
        self.refresh_workstations()
        self.update_status()
        self.log_message("데이터 새로고침 완료")
    
    def health_check(self):
        """헬스 체크"""
        self.log_message("헬스 체크 시작...")
        
        try:
            result = self.api_client.get("/health")
            if result:
                self.log_message(f"헬스 체크 성공: {result}")
                QMessageBox.information(self, "헬스 체크", f"시스템 상태: {result.get('status', 'unknown')}")
            else:
                self.log_message("헬스 체크 실패")
                QMessageBox.warning(self, "헬스 체크", "백엔드에 연결할 수 없습니다.")
        except Exception as e:
            self.log_message(f"헬스 체크 오류: {e}")
            QMessageBox.critical(self, "헬스 체크", f"오류 발생: {e}")
    
    def show_system_info(self):
        """시스템 정보 표시"""
        self.log_message("시스템 정보 조회 중...")
        
        try:
            result = self.api_client.get("/system/info")
            if result:
                info_text = f"CPU 개수: {result.get('cpu_count', 'N/A')}\n"
                info_text += f"메모리 총량: {result.get('memory_total_gb', 'N/A')} GB\n"
                info_text += f"사용 가능 메모리: {result.get('memory_available_gb', 'N/A')} GB\n"
                info_text += f"디스크 사용률: {result.get('disk_usage_percent', 'N/A')}%\n"
                info_text += f"Python 버전: {result.get('python_version', 'N/A')}\n"
                info_text += f"타임스탬프: {result.get('timestamp', 'N/A')}"
                
                self.system_info_text.setText(info_text)
                self.log_message("시스템 정보 조회 완료")
            else:
                self.log_message("시스템 정보 조회 실패")
        except Exception as e:
            self.log_message(f"시스템 정보 조회 오류: {e}")
    
    def create_workstation(self):
        """워크스테이션 생성"""
        name = self.ws_name_input.text().strip()
        description = self.ws_desc_input.text().strip()
        programs_text = self.ws_programs_input.text().strip()
        
        if not name:
            QMessageBox.warning(self, "입력 오류", "워크스테이션 이름을 입력하세요.")
            return
        
        programs = [p.strip() for p in programs_text.split(',') if p.strip()]
        
        self.log_message(f"워크스테이션 생성 중: {name}")
        
        try:
            result = self.api_client.post("/workstations", {
                "name": name,
                "description": description,
                "programs": programs
            })
            
            if result:
                self.log_message(f"워크스테이션 생성 성공: {result.get('workstation_id', 'N/A')}")
                QMessageBox.information(self, "성공", "워크스테이션이 생성되었습니다.")
                
                # 입력 필드 초기화
                self.ws_name_input.clear()
                self.ws_desc_input.clear()
                self.ws_programs_input.clear()
                
                # 워크스테이션 목록 새로고침
                self.refresh_workstations()
            else:
                self.log_message("워크스테이션 생성 실패")
                QMessageBox.warning(self, "실패", "워크스테이션 생성에 실패했습니다.")
        except Exception as e:
            self.log_message(f"워크스테이션 생성 오류: {e}")
            QMessageBox.critical(self, "오류", f"워크스테이션 생성 중 오류 발생: {e}")
    
    def refresh_workstations(self):
        """워크스테이션 목록 새로고침"""
        try:
            result = self.api_client.get("/workstations")
            if result:
                self.ws_list.clear()
                workstations = result.get('workstations', [])
                
                for ws in workstations:
                    item_text = f"{ws['name']} ({ws['id']})"
                    if ws.get('description'):
                        item_text += f" - {ws['description']}"
                    
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.UserRole, ws)
                    self.ws_list.addItem(item)
                
                # 훈련 탭의 워크스테이션 콤보박스 업데이트
                self.training_ws_combo.clear()
                for ws in workstations:
                    self.training_ws_combo.addItem(ws['name'], ws['id'])
                
                self.log_message(f"워크스테이션 목록 새로고침 완료: {len(workstations)}개")
            else:
                self.log_message("워크스테이션 목록 조회 실패")
        except Exception as e:
            self.log_message(f"워크스테이션 목록 조회 오류: {e}")
    
    def delete_workstation(self):
        """워크스테이션 삭제"""
        current_item = self.ws_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "선택 오류", "삭제할 워크스테이션을 선택하세요.")
            return
        
        ws_data = current_item.data(Qt.UserRole)
        ws_id = ws_data['id']
        ws_name = ws_data['name']
        
        reply = QMessageBox.question(
            self, 
            "확인", 
            f"워크스테이션 '{ws_name}'을(를) 삭제하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.log_message(f"워크스테이션 삭제 중: {ws_name}")
            
            try:
                result = self.api_client.delete(f"/workstations/{ws_id}")
                if result:
                    self.log_message(f"워크스테이션 삭제 성공: {ws_name}")
                    QMessageBox.information(self, "성공", "워크스테이션이 삭제되었습니다.")
                    self.refresh_workstations()
                else:
                    self.log_message("워크스테이션 삭제 실패")
                    QMessageBox.warning(self, "실패", "워크스테이션 삭제에 실패했습니다.")
            except Exception as e:
                self.log_message(f"워크스테이션 삭제 오류: {e}")
                QMessageBox.critical(self, "오류", f"워크스테이션 삭제 중 오류 발생: {e}")
    
    def start_training(self):
        """훈련 시작"""
        if self.training_ws_combo.count() == 0:
            QMessageBox.warning(self, "오류", "훈련할 워크스테이션이 없습니다.")
            return
        
        workstation_name = self.training_ws_combo.currentText()
        duration = self.training_duration_spin.value()
        
        self.log_message(f"훈련 시작: {workstation_name}, 지속 시간: {duration}분")
        
        # UI 상태 변경
        self.start_training_btn.setEnabled(False)
        self.stop_training_btn.setEnabled(True)
        self.training_progress.setValue(0)
        self.training_status_text.setText("훈련 시작 중...")
        
        # 백그라운드에서 훈련 실행
        self.background_worker.set_work("training", workstation_name=workstation_name, duration=duration)
        self.background_worker.start()
    
    def stop_training(self):
        """훈련 중지"""
        self.log_message("훈련 중지 요청...")
        
        try:
            result = self.api_client.post("/training/stop")
            if result:
                self.log_message("훈련 중지 성공")
            else:
                self.log_message("훈련 중지 실패")
        except Exception as e:
            self.log_message(f"훈련 중지 오류: {e}")
        
        # UI 상태 복원
        self.start_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
        self.training_status_text.setText("훈련 중지됨")
    
    def start_inference(self):
        """추론 시작"""
        model_path = self.inference_model_input.text().strip()
        if not model_path:
            QMessageBox.warning(self, "입력 오류", "모델 경로를 입력하세요.")
            return
        
        inference_interval = self.inference_interval_spin.value()
        
        self.log_message(f"추론 시작: {model_path}, 간격: {inference_interval}초")
        
        # UI 상태 변경
        self.start_inference_btn.setEnabled(False)
        self.stop_inference_btn.setEnabled(True)
        self.inference_status_text.setText("추론 시작 중...")
        
        # 백그라운드에서 추론 실행
        self.background_worker.set_work("inference", model_path=model_path, interval=inference_interval)
        self.background_worker.start()
    
    def stop_inference(self):
        """추론 중지"""
        self.log_message("추론 중지 요청...")
        
        try:
            result = self.api_client.post("/inference/stop")
            if result:
                self.log_message("추론 중지 성공")
            else:
                self.log_message("추론 중지 실패")
        except Exception as e:
            self.log_message(f"추론 중지 오류: {e}")
        
        # UI 상태 복원
        self.start_inference_btn.setEnabled(True)
        self.stop_inference_btn.setEnabled(False)
        self.inference_status_text.setText("추론 중지됨")
    
    def start_data_collection(self):
        """데이터 수집 시작"""
        duration = self.data_duration_spin.value()
        
        self.log_message(f"데이터 수집 시작: {duration}초")
        
        # 백그라운드에서 데이터 수집 실행
        self.background_worker.set_work("data_collection", duration=duration)
        self.background_worker.start()
    
    def convert_to_npu(self):
        """NPU 변환"""
        model_path = self.npu_model_input.text().strip()
        if not model_path:
            QMessageBox.warning(self, "입력 오류", "모델 경로를 입력하세요.")
            return
        
        self.log_message(f"NPU 변환 시작: {model_path}")
        
        try:
            result = self.api_client.post("/npu/convert", {
                "model_path": model_path,
                "target_platform": "snapdragon"
            })
            
            if result:
                self.log_message("NPU 변환 성공")
                QMessageBox.information(self, "성공", "NPU 변환이 완료되었습니다.")
            else:
                self.log_message("NPU 변환 실패")
                QMessageBox.warning(self, "실패", "NPU 변환에 실패했습니다.")
        except Exception as e:
            self.log_message(f"NPU 변환 오류: {e}")
            QMessageBox.critical(self, "오류", f"NPU 변환 중 오류 발생: {e}")
    
    def update_status(self):
        """상태 업데이트"""
        try:
            # 백엔드 상태 확인
            health = self.api_client.get("/health")
            if health:
                self.backend_status_label.setText(f"백엔드: {health.get('status', 'unknown')}")
            else:
                self.backend_status_label.setText("백엔드: 연결 실패")
            
            # 훈련 상태 확인
            training_status = self.api_client.get("/training/status")
            if training_status:
                status = training_status.get('status', 'unknown')
                self.training_status_label.setText(f"훈련: {status}")
            else:
                self.training_status_label.setText("훈련: 상태 확인 실패")
            
            # 추론 상태 확인
            inference_status = self.api_client.get("/inference/status")
            if inference_status:
                status = inference_status.get('status', 'unknown')
                self.inference_status_label.setText(f"추론: {status}")
            else:
                self.inference_status_label.setText("추론: 상태 확인 실패")
                
        except Exception as e:
            self.log_message(f"상태 업데이트 오류: {e}")
    
    def update_data_collection_progress(self, current: int, total: int):
        """데이터 수집 진행률 업데이트"""
        progress = int(current / total * 100)
        self.log_message(f"데이터 수집 진행률: {progress}% ({current}/{total})")
    
    def update_training_progress(self, current: int, total: int):
        """훈련 진행률 업데이트"""
        progress = int(current / total * 100)
        self.training_progress.setValue(progress)
        self.training_status_text.setText(f"훈련 진행 중... {progress}% ({current}/{total})")
    
    def update_inference_status(self, status: dict):
        """추론 상태 업데이트"""
        self.inference_status_text.setText(f"추론 실행 중... {status.get('status', 'unknown')}")
        
        # 통계 정보 업데이트
        stats = status.get('statistics', {})
        if stats:
            stats_text = f"총 추론 수: {stats.get('total_inferences', 0)}\n"
            stats_text += f"성공: {stats.get('successful_inferences', 0)}\n"
            stats_text += f"실패: {stats.get('failed_inferences', 0)}\n"
            stats_text += f"평균 추론 시간: {stats.get('average_inference_time', 0):.4f}초"
            
            self.inference_stats_text.setText(stats_text)
    
    def work_completed(self, result: dict):
        """작업 완료 처리"""
        work_type = result.get('type', 'unknown')
        self.log_message(f"{work_type} 작업 완료")
        
        if work_type == "training":
            self.start_training_btn.setEnabled(True)
            self.stop_training_btn.setEnabled(False)
            self.training_status_text.setText("훈련 완료")
            QMessageBox.information(self, "훈련 완료", "모델 훈련이 완료되었습니다.")
        
        elif work_type == "inference":
            self.start_inference_btn.setEnabled(True)
            self.stop_inference_btn.setEnabled(False)
            self.inference_status_text.setText("추론 완료")
            QMessageBox.information(self, "추론 완료", "추론 작업이 완료되었습니다.")
    
    def show_error(self, error_message: str):
        """오류 메시지 표시"""
        self.log_message(f"오류 발생: {error_message}")
        QMessageBox.critical(self, "오류", error_message)
    
    def log_message(self, message: str):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)
        
        # 스크롤을 맨 아래로
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 애플리케이션 정보 설정
    app.setApplicationName("Zonemaker AI")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Zonemaker AI Team")
    
    # 메인 윈도우 생성 및 표시
    window = MainWindow()
    window.show()
    
    # 이벤트 루프 실행
    sys.exit(app.exec())

if __name__ == "__main__":
    main()