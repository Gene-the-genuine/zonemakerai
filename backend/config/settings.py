import os
from typing import Dict, Any
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 데이터 디렉토리
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
NPU_MODELS_DIR = MODELS_DIR / "npu_converted"

# 백엔드 설정
BACKEND_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True,
    "reload": True,
    "workers": 1
}

# ML 모델 설정
ML_CONFIG = {
    "model": {
        "input_dim": 64,
        "hidden_dim": 128,
        "num_heads": 4,
        "num_layers": 3,
        "max_sequence_length": 600,  # 10분 * 60초
        "num_classes": 4  # x, y, w, h
    },
    "training": {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "num_epochs": 100,
        "validation_split": 0.2,
        "early_stopping_patience": 15,
        "checkpoint_interval": 10
    },
    "inference": {
        "interval_seconds": 1.0,
        "max_inference_time": 0.5,
        "confidence_threshold": 0.7
    }
}

# NPU 설정
NPU_CONFIG = {
    "target_platform": "snapdragon",
    "optimization": {
        "quantization": True,
        "pruning": True,
        "fusion": True,
        "memory_optimization": True,
        "precision": "int8"
    },
    "benchmark": {
        "num_runs": 100,
        "warmup_runs": 10
    }
}

# 데이터 수집 설정
DATA_COLLECTION_CONFIG = {
    "sampling_interval": 1.0,  # 1초마다 샘플링
    "min_collection_duration": 60,  # 최소 1분
    "max_collection_duration": 3600,  # 최대 1시간
    "default_collection_duration": 600,  # 기본 10분
    "window_detection": {
        "min_window_size": (100, 100),  # 최소 윈도우 크기
        "max_window_size": (1920, 1080),  # 최대 윈도우 크기
        "ignore_invisible": True
    }
}

# 윈도우 관리 설정
WINDOW_MANAGEMENT_CONFIG = {
    "arrangement": {
        "min_spacing": 20,  # 윈도우 간 최소 간격
        "snap_threshold": 50,  # 스냅 감지 임계값
        "animation_duration": 0.3,  # 애니메이션 지속 시간
        "max_z_order": 1000
    },
    "constraints": {
        "screen_bounds": (0, 0, 1920, 1080),  # 화면 경계
        "taskbar_height": 40,  # 작업 표시줄 높이
        "titlebar_height": 30   # 제목 표시줄 높이
    }
}

# 워크스테이션 설정
WORKSTATION_CONFIG = {
    "max_programs": 10,  # 워크스테이션당 최대 프로그램 수
    "auto_save_interval": 300,  # 자동 저장 간격 (5분)
    "backup_count": 5  # 백업 파일 개수
}

# 로깅 설정
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": {
        "enabled": True,
        "path": PROJECT_ROOT / "logs" / "zonemaker.log",
        "max_size": "10MB",
        "backup_count": 5
    },
    "console": {
        "enabled": True,
        "level": "INFO"
    }
}

# 성능 모니터링 설정
PERFORMANCE_CONFIG = {
    "monitoring": {
        "enabled": True,
        "interval": 5.0,  # 5초마다 모니터링
        "metrics": ["cpu", "memory", "gpu", "inference_time"]
    },
    "profiling": {
        "enabled": False,
        "output_dir": PROJECT_ROOT / "benchmark_results"
    }
}

# 보안 설정
SECURITY_CONFIG = {
    "cors": {
        "allowed_origins": ["*"],
        "allowed_methods": ["*"],
        "allowed_headers": ["*"]
    },
    "rate_limiting": {
        "enabled": False,
        "max_requests": 100,
        "window_seconds": 60
    }
}

def get_config() -> Dict[str, Any]:
    """전체 설정 반환"""
    return {
        "project_root": str(PROJECT_ROOT),
        "data_directories": {
            "raw": str(RAW_DATA_DIR),
            "processed": str(PROCESSED_DATA_DIR),
            "models": str(MODELS_DIR),
            "npu_models": str(NPU_MODELS_DIR)
        },
        "backend": BACKEND_CONFIG,
        "ml": ML_CONFIG,
        "npu": NPU_CONFIG,
        "data_collection": DATA_COLLECTION_CONFIG,
        "window_management": WINDOW_MANAGEMENT_CONFIG,
        "workstation": WORKSTATION_CONFIG,
        "logging": LOGGING_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "security": SECURITY_CONFIG
    }

def ensure_directories():
    """필요한 디렉토리 생성"""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        NPU_MODELS_DIR,
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "benchmark_results"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"디렉토리 확인/생성: {directory}")

def get_model_path(workstation_id: str, model_type: str = "best") -> str:
    """모델 파일 경로 반환"""
    if model_type == "npu":
        return str(NPU_MODELS_DIR / f"workstation_{workstation_id}_npu")
    else:
        return str(MODELS_DIR / f"workstation_{workstation_id}_{model_type}.pth")

def get_data_file_path(filename: str) -> str:
    """데이터 파일 경로 반환"""
    return str(RAW_DATA_DIR / filename)

def validate_config() -> bool:
    """설정 유효성 검사"""
    try:
        # 필수 디렉토리 확인
        ensure_directories()
        
        # 설정 값 검증
        if ML_CONFIG["training"]["batch_size"] <= 0:
            raise ValueError("배치 크기는 0보다 커야 합니다")
        
        if ML_CONFIG["inference"]["interval_seconds"] <= 0:
            raise ValueError("추론 간격은 0보다 커야 합니다")
        
        if DATA_COLLECTION_CONFIG["sampling_interval"] <= 0:
            raise ValueError("샘플링 간격은 0보다 커야 합니다")
        
        print("설정 유효성 검사 통과")
        return True
        
    except Exception as e:
        print(f"설정 유효성 검사 실패: {e}")
        return False

def update_config(updates: Dict[str, Any]):
    """설정 업데이트"""
    global ML_CONFIG, NPU_CONFIG, DATA_COLLECTION_CONFIG
    
    for key, value in updates.items():
        if key == "ml":
            ML_CONFIG.update(value)
        elif key == "npu":
            NPU_CONFIG.update(value)
        elif key == "data_collection":
            DATA_COLLECTION_CONFIG.update(value)
        else:
            print(f"알 수 없는 설정 키: {key}")

if __name__ == "__main__":
    # 설정 테스트
    print("Zonemaker AI 설정 테스트")
    print("=" * 50)
    
    # 디렉토리 생성
    ensure_directories()
    
    # 설정 유효성 검사
    if validate_config():
        print("설정이 올바르게 구성되었습니다.")
        
        # 전체 설정 출력
        config = get_config()
        print("\n현재 설정:")
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    else:
        print("설정에 문제가 있습니다.")