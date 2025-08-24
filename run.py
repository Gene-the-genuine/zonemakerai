
#!/usr/bin/env python3
"""
Zonemaker AI - AI 기반 윈도우 배열 최적화 시스템
실행 스크립트
"""

import argparse
import sys
import os
import time
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_backend():
    """백엔드 API 서버 실행"""
    try:
        from backend.api.main import app
        import uvicorn
        
        logger.info("백엔드 API 서버 시작...")
        uvicorn.run(
            app, 
            host="127.0.0.1", 
            port=8000, 
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"백엔드 실행 실패: {e}")
        return False
    return True

def run_data_collection(duration: int = 30):
    """데이터 수집 실행 (30초 관찰)"""
    try:
        from backend.core.data_collector import DataCollector
        
        logger.info(f"데이터 수집 테스트 시작 (지속 시간: {duration}초)")
        
        collector = DataCollector()
        
        # 30초간 데이터 수집
        activities = collector.collect_data_sample(duration)
        
        if activities:
            # 데이터 저장
            filename = f"data/window_activity_data_{int(time.time())}.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            collector.save_data(activities, filename)
            
            logger.info(f"데이터 수집 완료: {len(activities)}개 샘플, 저장: {filename}")
            return True
        else:
            logger.error("데이터 수집 실패: 샘플이 수집되지 않았습니다.")
            return False
            
    except Exception as e:
        logger.error(f"데이터 수집 실패: {e}")
        return False

def run_training(data_file: str = None, epochs: int = 50):
    """모델 훈련 실행"""
    try:
        from backend.ml.model import create_model
        from backend.ml.trainer import ModelTrainer
        
        logger.info("모델 훈련 시작...")
        
        # 모델 생성
        model = create_model()
        logger.info("모델 생성 완료")
        
        # 훈련기 생성
        trainer = ModelTrainer(model)
        
        # 훈련 데이터 준비
        if not data_file:
            # 최신 데이터 파일 찾기
            data_dir = Path("data")
            if data_dir.exists():
                data_files = list(data_dir.glob("window_activity_data_*.json"))
                if data_files:
                    data_file = str(max(data_files, key=lambda x: x.stat().st_mtime))
                    logger.info(f"자동으로 데이터 파일 선택: {data_file}")
                else:
                    logger.error("훈련할 데이터 파일을 찾을 수 없습니다.")
                    return False
            else:
                logger.error("데이터 디렉토리가 존재하지 않습니다.")
                return False
        
        # 훈련 데이터 준비
        train_loader, val_loader = trainer.prepare_training_data(
            data_file, 
            train_split=0.8, 
            batch_size=4,  # 메모리 절약을 위해 작은 배치 크기
            sequence_length=30
        )
        
        # 훈련 실행
        save_dir = "data/models"
        os.makedirs(save_dir, exist_ok=True)
        
        training_history = trainer.train(
            train_loader, 
            val_loader, 
            num_epochs=epochs,
            patience=15,
            save_dir=save_dir
        )
        
        logger.info(f"훈련 완료! 최종 손실: {training_history['val_loss'][-1]:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"모델 훈련 실패: {e}")
        return False

def run_inference(model_path: str = None, duration: int = 60):
    """실시간 추론 실행"""
    try:
        from backend.ml.inference import RealTimeInferenceEngine
        
        logger.info("실시간 추론 엔진 시작...")
        
        # 모델 경로 설정
        if not model_path:
            # 최신 모델 파일 찾기
            models_dir = Path("data/models")
            if models_dir.exists():
                model_files = list(models_dir.glob("*.pth"))
                if model_files:
                    model_path = str(max(model_files, key=lambda x: x.stat().st_mtime))
                    logger.info(f"자동으로 모델 파일 선택: {model_path}")
                else:
                    logger.error("추론할 모델 파일을 찾을 수 없습니다.")
                    return False
            else:
                logger.error("모델 디렉토리가 존재하지 않습니다.")
                return False
        
        # 추론 엔진 생성
        engine = RealTimeInferenceEngine(model_path, prediction_interval=1.0)
        
        # 추론 시작
        if engine.start_inference():
            logger.info(f"추론 엔진이 {duration}초간 실행됩니다...")
            time.sleep(duration)
            
            # 추론 중지
            try:
                engine.stop_inference()
                
                # 상태 및 통계 출력 (안전하게)
                try:
                    status = engine.get_inference_status()
                    logger.info(f"추론 완료 - 상태: {status}")
                except Exception as status_error:
                    logger.warning(f"상태 확인 실패: {status_error}")
                
                # 로그 저장 (안전하게)
                try:
                    engine.save_inference_log()
                except Exception as log_error:
                    logger.warning(f"로그 저장 실패: {log_error}")
                
                return True
                
            except Exception as stop_error:
                logger.error(f"추론 엔진 중지 중 오류: {stop_error}")
                return False
        else:
            logger.error("추론 엔진 시작 실패")
            return False
            
    except Exception as e:
        logger.error(f"실시간 추론 실패: {e}")
        return False

def run_npu_conversion(model_path: str = None):
    """NPU 변환 실행"""
    try:
        from backend.ml.npu_converter import NPUConverter
        
        logger.info("NPU 변환 시작...")
        
        # 모델 경로 설정
        if not model_path:
            # 최신 모델 파일 찾기
            models_dir = Path("data/models")
            if models_dir.exists():
                model_files = list(models_dir.glob("*.pth"))
                if model_files:
                    model_path = str(max(model_files, key=lambda x: x.stat().st_mtime))
                    logger.info(f"자동으로 모델 파일 선택: {model_path}")
                else:
                    logger.error("변환할 모델 파일을 찾을 수 없습니다.")
                    return False
            else:
                logger.error("모델 디렉토리가 존재하지 않습니다.")
                return False
        
        # NPU 변환기 생성
        converter = NPUConverter(model_path)
        
        # NPU 변환 실행
        success = converter.convert_to_npu()
        
        if success:
            # 벤치마크 실행
            benchmark_results = converter.benchmark_model()
            logger.info(f"NPU 변환 완료! 벤치마크 결과: {benchmark_results}")
            return True
        else:
            logger.error("NPU 변환 실패")
            return False
            
    except Exception as e:
        logger.error(f"NPU 변환 실패: {e}")
        return False

def run_frontend():
    """프론트엔드 GUI 실행"""
    try:
        from frontend.main import MainWindow
        from PySide6.QtWidgets import QApplication
        
        logger.info("프론트엔드 GUI 시작...")
        
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"프론트엔드 실행 실패: {e}")
        return False

def run_all():
    """전체 시스템 실행 (백엔드 + 프론트엔드)"""
    try:
        import threading
        import time
        
        logger.info("전체 시스템 실행 시작...")
        
        # 백엔드를 별도 스레드에서 실행
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        # 백엔드 시작 대기
        time.sleep(3)
        
        # 프론트엔드 실행
        run_frontend()
        
    except Exception as e:
        logger.error(f"전체 시스템 실행 실패: {e}")
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Zonemaker AI - AI 기반 윈도우 배열 최적화 시스템"
    )
    
    parser.add_argument(
        '--mode', 
        choices=['backend', 'data-collect', 'train', 'inference', 'npu-convert', 'frontend', 'all'],
        default='all',
        help='실행 모드 선택'
    )
    
    parser.add_argument(
        '--duration', 
        type=int, 
        default=30,
        help='데이터 수집 또는 추론 지속 시간 (초)'
    )
    
    parser.add_argument(
        '--data-file', 
        type=str,
        help='훈련 또는 추론에 사용할 데이터 파일 경로'
    )
    
    parser.add_argument(
        '--model-path', 
        type=str,
        help='훈련, 추론 또는 NPU 변환에 사용할 모델 파일 경로'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help='훈련 에포크 수'
    )
    
    args = parser.parse_args()
    
    # 프로젝트 정보 출력
    print("=" * 60)
    print("Zonemaker AI - AI 기반 윈도우 배열 최적화 시스템")
    print("=" * 60)
    print(f"실행 모드: {args.mode}")
    print(f"프로젝트 루트: {project_root}")
    print("=" * 60)
    
    # 모드별 실행
    success = False
    
    try:
        if args.mode == 'backend':
            success = run_backend()
        elif args.mode == 'data-collect':
            success = run_data_collection(args.duration)
        elif args.mode == 'train':
            success = run_training(args.data_file, args.epochs)
        elif args.mode == 'inference':
            success = run_inference(args.model_path, args.duration)
        elif args.mode == 'npu-convert':
            success = run_npu_conversion(args.model_path)
        elif args.mode == 'frontend':
            success = run_frontend()
        elif args.mode == 'all':
            success = run_all()
        else:
            logger.error(f"알 수 없는 실행 모드: {args.mode}")
            return False
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        return True
    except Exception as e:
        logger.error(f"실행 중 예상치 못한 오류: {e}")
        return False
    
    if success:
        logger.info(f"{args.mode} 모드 실행 완료")
    else:
        logger.error(f"{args.mode} 모드 실행 실패")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)