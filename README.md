# Zonemaker AI - AI 기반 윈도우 배열 최적화 시스템

## 📖 개요

Zonemaker AI는 Microsoft Copilot PC의 Snapdragon NPU를 활용하여 사용자의 윈도우 사용 패턴을 학습하고, 실시간으로 최적의 윈도우 배열을 제안하는 AI 시스템입니다. Vision Transformer 기반의 경량화된 모델을 사용하여 로컬에서 빠른 추론을 수행합니다.

## ✨ 주요 기능

- **실시간 윈도우 모니터링**: Windows API를 통한 지속적인 윈도우 상태 추적
- **AI 기반 배열 예측**: 30초 관찰 후 다음 순간의 윈도우 위치 예측
- **연속 최적화**: 1초마다 새로운 예측으로 지속적인 최적화
- **NPU 최적화**: Snapdragon NPU 전용 모델 변환 및 최적화
- **직관적 UI**: PySide6 기반의 사용자 친화적 인터페이스
- **RESTful API**: FastAPI 기반의 확장 가능한 백엔드

## 🏗️ 시스템 요구사항

### 하드웨어 요구사항
- **CPU**: Intel/AMD 64비트 프로세서
- **메모리**: 최소 8GB RAM (16GB 권장)
- **저장공간**: 최소 2GB 여유 공간
- **NPU**: Snapdragon NPU (선택사항, 성능 향상)

### 소프트웨어 요구사항
- **OS**: Windows 10/11 (64비트)
- **Python**: 3.8 이상
- **가상환경**: conda 또는 venv 권장

## 🚀 설치 및 설정

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/zonemakerai.git
cd zonemakerai
```

### 2. 가상환경 생성 및 활성화
```bash
# conda 사용
conda create -n zonemakeraiconda python=3.9
conda activate zonemakeraiconda

# 또는 venv 사용
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 프로젝트 구조 확인
```
zonemakerai/
├── backend/
│   ├── api/           # FastAPI 백엔드
│   ├── core/          # 핵심 기능 (데이터 수집)
│   ├── ml/            # ML 파이프라인
│   └── config/        # 설정 파일
├── frontend/          # PySide6 GUI
├── data/              # 데이터 및 모델 저장
├── logs/              # 로그 파일
└── run.py             # 메인 실행 스크립트
```

## 🧠 ML 파이프라인 상세 가이드

### 📊 데이터 수집 파이프라인

#### 구조 및 원리
```
사용자 활동 → 윈도우 상태 모니터링 → 30초 시퀀스 생성 → 실시간 버퍼링
```

#### 핵심 컴포넌트

**1. DataCollector 클래스**
```python
from backend.core.data_collector import DataCollector

# 데이터 수집기 생성
collector = DataCollector()

# 30초간 연속 데이터 수집
activities = collector.collect_data_sample(duration_seconds=30)

# 비동기 연속 수집 (콜백 기반)
def on_data_collected(samples):
    print(f"수집된 샘플: {len(samples)}개")

collector.start_continuous_collection(30, callback=on_data_collected)
```

**2. 수집되는 데이터 유형**
- **윈도우 정보**: 제목, 클래스명, 프로세스명, 위치, 크기, 상태
- **사용자 활동**: 마우스 위치, 키보드 활동, 윈도우 변화
- **시계열 데이터**: 0.1초마다 샘플링하여 30초 시퀀스 구성

#### 데이터 수집 실행
```bash
# 30초간 데이터 수집
python run.py --mode data-collect --duration 30

# 결과: data/window_activity_data_[timestamp].json
```

### 🤖 ML 모델 아키텍처

#### 모델 구조
```
입력: 30초 윈도우 시퀀스 + 활동 시퀀스
  ↓
WindowFeatureExtractor: 윈도우 정보 → 특징 벡터
  ↓
ActivityFeatureExtractor: 사용자 활동 → 특징 벡터
  ↓
Transformer Encoder: 시퀀스 처리
  ↓
출력: 윈도우별 위치/크기 + 존재 여부
```

#### 모델 생성 및 사용
```python
from backend.ml.model import create_model, WindowArrangementPredictor

# 모델 생성
model = create_model({
    'window_feature_dim': 128,
    'activity_feature_dim': 64,
    'hidden_dim': 256,
    'num_heads': 8,
    'num_layers': 6,
    'max_windows': 20
})

# 예측기 생성
predictor = WindowArrangementPredictor("path/to/model.pth")

# 다음 순간 윈도우 배열 예측
predicted_positions = predictor.predict_next_arrangement(
    window_sequence, activity_sequence
)
```

#### 모델 훈련
```bash
# 기본 훈련 (50 에포크)
python run.py --mode train --epochs 50

# 특정 데이터 파일로 훈련
python run.py --mode train --data-file data/my_data.json --epochs 100

# 결과: data/models/best.pth, data/models/final.pth
```

### 🔄 실시간 추론 파이프라인

#### 추론 엔진 동작 원리
```
1. 30초 데이터 버퍼 수집
2. 버퍼가 가득 차면 예측 수행
3. 예측 결과를 실제 윈도우에 적용
4. 1초 후 새로운 데이터로 버퍼 업데이트
5. 반복하여 연속 최적화
```

#### 추론 엔진 사용법
```python
from backend.ml.inference import RealTimeInferenceEngine

# 추론 엔진 생성
engine = RealTimeInferenceEngine(
    model_path="path/to/model.pth",
    prediction_interval=1.0  # 1초마다 예측
)

# 콜백 설정
def on_prediction_complete(predictions, success):
    print(f"예측 완료: {len(predictions)}개 윈도우")

engine.set_callbacks(
    on_prediction_complete=on_prediction_complete
)

# 추론 시작
engine.start_inference()

# 상태 확인
status = engine.get_inference_status()
print(f"버퍼 크기: {status['buffer_size']}/{status['buffer_max_size']}")

# 추론 중지
engine.stop_inference()
```

#### 실시간 추론 실행
```bash
# 60초간 실시간 추론
python run.py --mode inference --duration 60

# 특정 모델로 추론
python run.py --mode inference --model-path data/models/best.pth --duration 120
```

### 🎯 연속 학습 시스템

#### ContinuousLearningEngine
```python
from backend.ml.inference import ContinuousLearningEngine

# 연속 학습 엔진 생성
learning_engine = ContinuousLearningEngine(
    model_path="path/to/model.pth",
    update_interval=300.0  # 5분마다 학습 데이터 수집
)

# 연속 학습 시작
learning_engine.start_continuous_learning()

# 수집된 학습 데이터 확인
data_count = learning_engine.get_training_data_count()
print(f"수집된 학습 데이터: {data_count}개")

# 학습 데이터 내보내기
learning_engine.export_training_data("continuous_learning_data.json")
```

### ⚡ NPU 최적화 파이프라인

#### NPU 변환 과정
```
PyTorch 모델 → ONNX 변환 → NPU 최적화 → Snapdragon NPU 전용 모델
```

#### NPU 변환 실행
```bash
# NPU 변환
python run.py --mode npu-convert

# 특정 모델 변환
python run.py --mode npu-convert --model-path data/models/best.pth

# 결과: data/models/npu_optimized.npu
```

#### NPU 변환기 사용법
```python
from backend.ml.npu_converter import NPUConverter

# NPU 변환기 생성
converter = NPUConverter("path/to/model.pth")

# NPU 변환 실행
success = converter.convert_to_npu()

if success:
    # 벤치마크 실행
    results = converter.benchmark_model()
    print(f"NPU 성능: {results}")
```

## 🎮 사용 방법

### 1. 빠른 시작
```bash
# 전체 시스템 실행 (백엔드 + 프론트엔드)
python run.py --mode all

# 백엔드만 실행
python run.py --mode backend

# 프론트엔드만 실행
python run.py --mode frontend
```

### 2. 단계별 실행

#### 1단계: 데이터 수집
```bash
# 30초간 사용자 활동 데이터 수집
python run.py --mode data-collect --duration 30
```

#### 2단계: 모델 훈련
```bash
# 수집된 데이터로 모델 훈련
python run.py --mode train --epochs 100
```

#### 3단계: 실시간 추론
```bash
# 훈련된 모델로 실시간 윈도우 배열 최적화
python run.py --mode inference --duration 300
```

#### 4단계: NPU 최적화 (선택사항)
```bash
# NPU 전용 모델로 변환
python run.py --mode npu-convert
```

### 3. 고급 사용법

#### 배치 추론
```python
from backend.ml.inference import BatchInferenceEngine

# 배치 추론 엔진
batch_engine = BatchInferenceEngine("path/to/model.pth")

# 여러 데이터 파일에 대해 배치 예측
results = batch_engine.batch_predict(
    data_file="data/batch_data.json",
    output_file="results/batch_predictions.json"
)
```

#### 커스텀 설정
```python
from backend.config.settings import update_settings

# 설정 업데이트
update_settings({
    'ml': {
        'sequence_length': 45,  # 45초 시퀀스
        'prediction_interval': 0.5,  # 0.5초마다 예측
        'max_windows': 25
    }
})
```

## 📈 성능 최적화

### 1. 모델 경량화
- **양자화**: int8 정밀도로 모델 크기 감소
- **프루닝**: 불필요한 가중치 제거
- **지식 증류**: 작은 모델로 성능 유지

### 2. 추론 최적화
- **배치 처리**: 여러 윈도우 동시 예측
- **비동기 처리**: UI 블로킹 방지
- **메모리 관리**: 효율적인 버퍼 관리

### 3. NPU 최적화
- **모델 융합**: 연산 레이어 결합
- **메모리 최적화**: NPU 메모리 효율적 사용
- **병렬 처리**: NPU 병렬 연산 활용

## 🐛 문제 해결

### 일반적인 문제

#### 1. 모듈 import 오류
```bash
# 가상환경 활성화 확인
conda activate zonemakeraiconda

# 의존성 재설치
pip install -r requirements.txt
```

#### 2. Windows API 권한 오류
```bash
# 관리자 권한으로 실행
# 또는 Windows Defender 예외 설정
```

#### 3. 메모리 부족
```bash
# 배치 크기 줄이기
python run.py --mode train --epochs 50  # 기본값 사용

# 시퀀스 길이 줄이기 (설정 파일에서)
```

### 디버깅 팁

#### 1. 로그 레벨 조정
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. 데이터 수집 테스트
```bash
# 짧은 시간으로 테스트
python run.py --mode data-collect --duration 5
```

#### 3. 모델 테스트
```python
# 간단한 입력으로 모델 테스트
python backend/ml/model.py
```

## 🔧 개발 가이드

### 1. 새로운 기능 추가
```python
# 새로운 특징 추출기 추가
class CustomFeatureExtractor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # 구현...
    
    def forward(self, data):
        # 구현...
        return features
```

### 2. 모델 아키텍처 수정
```python
# 모델 설정 변경
config = {
    'window_feature_dim': 256,  # 증가
    'hidden_dim': 512,          # 증가
    'num_layers': 8             # 증가
}

model = create_model(config)
```

### 3. 새로운 추론 방식 추가
```python
# 커스텀 추론 엔진
class CustomInferenceEngine(RealTimeInferenceEngine):
    def _perform_prediction(self):
        # 커스텀 예측 로직
        pass
```

## 📚 API 참조

### 핵심 클래스

#### DataCollector
- `collect_data_sample(duration)`: 지정된 시간 동안 데이터 수집
- `start_continuous_collection(duration, callback)`: 연속 데이터 수집
- `save_data(activities, filename)`: 데이터 저장

#### RealTimeWindowPredictor
- `predict_next_arrangement(window_seq, activity_seq)`: 다음 순간 예측
- `apply_prediction(window_handles, positions)`: 예측 결과 적용

#### RealTimeInferenceEngine
- `start_inference()`: 추론 시작
- `stop_inference()`: 추론 중지
- `get_inference_status()`: 상태 정보 반환

#### ModelTrainer
- `prepare_training_data(data_file, ...)`: 훈련 데이터 준비
- `train(train_loader, val_loader, ...)`: 모델 훈련
- `export_to_onnx(save_path)`: ONNX 모델 내보내기

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 지원

- **Issues**: GitHub Issues 사용
- **Discussions**: GitHub Discussions 참여
- **Wiki**: 프로젝트 Wiki 참조

## 📝 변경 이력

### v1.0.0 (2024-01-XX)
- 초기 ML 파이프라인 구현
- 실시간 연속 예측 시스템
- NPU 최적화 지원
- PySide6 기반 GUI
- FastAPI 백엔드

---

**Zonemaker AI** - AI로 윈도우를 더 스마트하게!