from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import json
import os
import time
from datetime import datetime
import threading
import asyncio

# ML 모듈 임포트
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model import create_model, WindowArrangementPredictor
from ml.trainer import ModelTrainer, prepare_training_data
from ml.inference import RealTimeInferenceEngine, BatchInferenceEngine
from ml.npu_converter import create_npu_converter
from core.data_collector import DataCollector, UserActivity

# FastAPI 앱 생성
app = FastAPI(
    title="Zonemaker AI API",
    description="AI 기반 윈도우 배열 최적화 시스템",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 모델
class WorkstationCreate(BaseModel):
    name: str
    description: Optional[str] = None
    programs: List[str]

class TrainingRequest(BaseModel):
    workstation_name: str
    duration_minutes: int = 10
    model_config: Optional[Dict[str, Any]] = None

class InferenceRequest(BaseModel):
    model_path: str
    inference_interval: float = 1.0
    max_inference_time: float = 0.5

class WindowArrangementResponse(BaseModel):
    program_id: str
    x: float
    y: float
    width: float
    height: float
    confidence: float

# 전역 상태
class AppState:
    def __init__(self):
        self.data_collector = DataCollector()
        self.inference_engine = None
        self.training_status = "idle"
        self.inference_status = "stopped"
        self.workstations = {}
        self.current_workstation = None
        self.training_thread = None
        self.inference_thread = None

app_state = AppState()

# API 엔드포인트
@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Zonemaker AI API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "training_status": app_state.training_status,
        "inference_status": app_state.inference_status
    }

# 워크스테이션 관리
@app.post("/workstations")
async def create_workstation(workstation: WorkstationCreate):
    """워크스테이션 생성"""
    try:
        workstation_id = f"ws_{int(time.time())}"
        
        app_state.workstations[workstation_id] = {
            "id": workstation_id,
            "name": workstation.name,
            "description": workstation.description,
            "programs": workstation.programs,
            "created_at": datetime.now().isoformat(),
            "status": "created"
        }
        
        return {
            "message": "워크스테이션 생성 완료",
            "workstation_id": workstation_id,
            "workstation": app_state.workstations[workstation_id]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"워크스테이션 생성 실패: {str(e)}")

@app.get("/workstations")
async def list_workstations():
    """워크스테이션 목록 조회"""
    return {
        "workstations": list(app_state.workstations.values()),
        "total_count": len(app_state.workstations)
    }

@app.get("/workstations/{workstation_id}")
async def get_workstation(workstation_id: str):
    """워크스테이션 상세 조회"""
    if workstation_id not in app_state.workstations:
        raise HTTPException(status_code=404, detail="워크스테이션을 찾을 수 없습니다")
    
    return app_state.workstations[workstation_id]

@app.delete("/workstations/{workstation_id}")
async def delete_workstation(workstation_id: str):
    """워크스테이션 삭제"""
    if workstation_id not in app_state.workstations:
        raise HTTPException(status_code=404, detail="워크스테이션을 찾을 수 없습니다")
    
    deleted_workstation = app_state.workstations.pop(workstation_id)
    
    return {
        "message": "워크스테이션 삭제 완료",
        "deleted_workstation": deleted_workstation
    }

# 데이터 수집
@app.post("/data/collect")
async def start_data_collection(duration_seconds: int = 600):
    """데이터 수집 시작"""
    try:
        if duration_seconds < 60:
            raise HTTPException(status_code=400, detail="최소 60초 이상 수집해야 합니다")
        
        # 백그라운드에서 데이터 수집 실행
        def collect_data():
            try:
                print(f"데이터 수집 시작: {duration_seconds}초")
                activities = app_state.data_collector.collect_data_sample(duration_seconds)
                
                # 데이터 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"workstation_data_{timestamp}.json"
                filepath = app_state.data_collector.save_data(activities, filename)
                
                print(f"데이터 수집 완료: {len(activities)}개 샘플, 저장: {filepath}")
                
            except Exception as e:
                print(f"데이터 수집 오류: {e}")
        
        # 별도 스레드에서 실행
        collection_thread = threading.Thread(target=collect_data, daemon=True)
        collection_thread.start()
        
        return {
            "message": "데이터 수집 시작",
            "duration_seconds": duration_seconds,
            "estimated_completion": datetime.now().timestamp() + duration_seconds
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 수집 시작 실패: {str(e)}")

@app.get("/data/status")
async def get_data_collection_status():
    """데이터 수집 상태 조회"""
    data_dir = "data/raw"
    if not os.path.exists(data_dir):
        return {"status": "no_data_directory", "files": []}
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
    
    file_info = []
    for file in files[:10]:  # 최근 10개 파일만
        filepath = os.path.join(data_dir, file)
        file_info.append({
            "filename": file,
            "size_mb": round(os.path.getsize(filepath) / 1024 / 1024, 2),
            "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
        })
    
    return {
        "status": "available",
        "total_files": len(files),
        "recent_files": file_info
    }

# 모델 훈련
@app.post("/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """모델 훈련 시작"""
    try:
        if app_state.training_status == "training":
            raise HTTPException(status_code=400, detail="이미 훈련이 진행 중입니다")
        
        # 워크스테이션 확인
        workstation = None
        for ws in app_state.workstations.values():
            if ws["name"] == request.workstation_name:
                workstation = ws
                break
        
        if not workstation:
            raise HTTPException(status_code=404, detail="워크스테이션을 찾을 수 없습니다")
        
        app_state.current_workstation = workstation
        app_state.training_status = "training"
        
        # 백그라운드에서 훈련 실행
        def run_training():
            try:
                print(f"훈련 시작: {workstation['name']}")
                
                # 데이터 수집 (10분)
                print("훈련 데이터 수집 중...")
                activities = app_state.data_collector.collect_data_sample(600)  # 10분
                
                if len(activities) < 100:
                    raise Exception("충분한 훈련 데이터를 수집할 수 없습니다")
                
                # 데이터로더 준비
                train_loader, val_loader = prepare_training_data(
                    activities, 
                    batch_size=16,
                    sequence_length=600
                )
                
                # 모델 생성
                model_config = request.model_config or {}
                model = create_model(
                    input_dim=model_config.get('input_dim', 64),
                    hidden_dim=model_config.get('hidden_dim', 128),
                    num_heads=model_config.get('num_heads', 4),
                    num_layers=model_config.get('num_layers', 3)
                )
                
                # 훈련기 생성 및 훈련 실행
                trainer = ModelTrainer(model, learning_rate=1e-4)
                
                results = trainer.train(
                    train_loader,
                    val_loader,
                    num_epochs=50,  # 50 에포크
                    save_dir="data/models",
                    model_name=f"workstation_{workstation['id']}"
                )
                
                print(f"훈련 완료: {results}")
                app_state.training_status = "completed"
                
            except Exception as e:
                print(f"훈련 오류: {e}")
                app_state.training_status = "failed"
        
        # 별도 스레드에서 실행
        app_state.training_thread = threading.Thread(target=run_training, daemon=True)
        app_state.training_thread.start()
        
        return {
            "message": "훈련 시작",
            "workstation": workstation,
            "duration_minutes": request.duration_minutes,
            "status": "training"
        }
        
    except Exception as e:
        app_state.training_status = "failed"
        raise HTTPException(status_code=500, detail=f"훈련 시작 실패: {str(e)}")

@app.get("/training/status")
async def get_training_status():
    """훈련 상태 조회"""
    return {
        "status": app_state.training_status,
        "current_workstation": app_state.current_workstation,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/training/stop")
async def stop_training():
    """훈련 중지"""
    if app_state.training_status != "training":
        raise HTTPException(status_code=400, detail="현재 훈련 중이 아닙니다")
    
    app_state.training_status = "stopped"
    
    return {
        "message": "훈련 중지",
        "status": "stopped"
    }

# 모델 추론
@app.post("/inference/start")
async def start_inference(request: InferenceRequest):
    """실시간 추론 시작"""
    try:
        if app_state.inference_status == "running":
            raise HTTPException(status_code=400, detail="이미 추론이 실행 중입니다")
        
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail="모델 파일을 찾을 수 없습니다")
        
        # 추론 엔진 생성
        app_state.inference_engine = RealTimeInferenceEngine(
            model_path=request.model_path,
            inference_interval=request.inference_interval,
            max_inference_time=request.max_inference_time
        )
        
        # 추론 시작
        app_state.inference_engine.start_inference()
        app_state.inference_status = "running"
        
        return {
            "message": "실시간 추론 시작",
            "model_path": request.model_path,
            "inference_interval": request.inference_interval,
            "status": "running"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 시작 실패: {str(e)}")

@app.post("/inference/stop")
async def stop_inference():
    """실시간 추론 중지"""
    if app_state.inference_status != "running":
        raise HTTPException(status_code=400, detail="현재 추론 중이 아닙니다")
    
    if app_state.inference_engine:
        app_state.inference_engine.stop_inference()
        app_state.inference_status = "stopped"
    
    return {
        "message": "실시간 추론 중지",
        "status": "stopped"
    }

@app.get("/inference/status")
async def get_inference_status():
    """추론 상태 조회"""
    if app_state.inference_engine:
        stats = app_state.inference_engine.get_inference_stats()
    else:
        stats = {}
    
    return {
        "status": app_state.inference_status,
        "inference_engine": app_state.inference_engine is not None,
        "statistics": stats,
        "timestamp": datetime.now().isoformat()
    }

# NPU 변환
@app.post("/npu/convert")
async def convert_to_npu(model_path: str, target_platform: str = "snapdragon"):
    """NPU 변환"""
    try:
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="모델 파일을 찾을 수 없습니다")
        
        # NPU 변환기 생성
        converter = create_npu_converter(
            model_path=model_path,
            target_platform=target_platform
        )
        
        # 변환 실행
        results = converter.convert_to_npu()
        
        if 'error' in results:
            raise HTTPException(status_code=500, detail=f"NPU 변환 실패: {results['error']}")
        
        return {
            "message": "NPU 변환 완료",
            "results": results,
            "target_platform": target_platform
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NPU 변환 실패: {str(e)}")

@app.post("/npu/benchmark")
async def benchmark_npu_model(model_path: str, num_runs: int = 100):
    """NPU 모델 벤치마크"""
    try:
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="모델 파일을 찾을 수 없습니다")
        
        # NPU 변환기 생성
        converter = create_npu_converter(model_path=model_path)
        
        # 벤치마크 실행
        results = converter.benchmark_model(num_runs=num_runs)
        
        return {
            "message": "벤치마크 완료",
            "results": results,
            "num_runs": num_runs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벤치마크 실패: {str(e)}")

# 윈도우 배열 예측
@app.post("/predict/arrangement")
async def predict_window_arrangement(program_id: str, activities: List[Dict]):
    """윈도우 배열 예측"""
    try:
        # 활동 데이터를 UserActivity 객체로 변환
        user_activities = []
        for activity_data in activities:
            activity = UserActivity(
                timestamp=activity_data.get('timestamp', time.time()),
                program_id=activity_data.get('program_id', program_id),
                is_active=activity_data.get('is_active', False),
                is_minimized=activity_data.get('is_minimized', False),
                is_maximized=activity_data.get('is_maximized', False),
                click_count=activity_data.get('click_count', 0),
                keystroke_count=activity_data.get('keystroke_count', 0),
                width_resize_count=activity_data.get('width_resize_count', 0),
                height_resize_count=activity_data.get('height_resize_count', 0),
                scroll_down_count=activity_data.get('scroll_down_count', 0),
                scroll_up_count=activity_data.get('scroll_up_count', 0),
                x=activity_data.get('x', 0),
                y=activity_data.get('y', 0),
                width=activity_data.get('width', 800),
                height=activity_data.get('height', 600),
                z_order=activity_data.get('z_order', 0)
            )
            user_activities.append(activity)
        
        # 기본 예측 (실제로는 훈련된 모델 사용)
        # 여기서는 간단한 규칙 기반 예측
        if len(user_activities) > 0:
            last_activity = user_activities[-1]
            
            # 간단한 예측 로직
            pred_x = max(0, last_activity.x + 50)
            pred_y = max(0, last_activity.y + 30)
            pred_w = max(100, last_activity.width)
            pred_h = max(100, last_activity.height)
            
            # 화면 경계 내로 제한
            pred_x = min(pred_x, 1920 - pred_w)
            pred_y = min(pred_y, 1080 - pred_h)
            
            prediction = WindowArrangementResponse(
                program_id=program_id,
                x=pred_x,
                y=pred_y,
                width=pred_w,
                height=pred_h,
                confidence=0.8
            )
        else:
            # 기본값
            prediction = WindowArrangementResponse(
                program_id=program_id,
                x=100,
                y=100,
                width=800,
                height=600,
                confidence=0.5
            )
        
        return {
            "message": "윈도우 배열 예측 완료",
            "prediction": prediction,
            "input_activities_count": len(user_activities)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")

# 시스템 정보
@app.get("/system/info")
async def get_system_info():
    """시스템 정보 조회"""
    import psutil
    
    return {
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2),
        "memory_available_gb": round(psutil.virtual_memory().available / 1024 / 1024 / 1024, 2),
        "disk_usage_percent": psutil.disk_usage('/').percent,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # 개발 서버 실행
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
