import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import asdict
from backend.core.data_collector import UserActivity, WindowInfo

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """시퀀스 위치 인코딩"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class WindowFeatureExtractor(nn.Module):
    """윈도우 정보를 특징 벡터로 변환"""
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 윈도우 제목 임베딩 (간단한 해시 기반)
        self.title_embedding = nn.Embedding(1000, 32)  # 제한된 어휘 크기
        
        # 윈도우 클래스명 임베딩
        self.class_embedding = nn.Embedding(500, 32)
        
        # 프로세스명 임베딩
        self.process_embedding = nn.Embedding(1000, 32)
        
        # 위치, 크기, 상태 정보 처리
        self.spatial_encoder = nn.Sequential(
            nn.Linear(6, 64),  # rect(4) + is_minimized(1) + is_maximized(1)
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 최종 특징 결합
        self.feature_fusion = nn.Sequential(
            nn.Linear(32 + 32 + 32 + 32, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, window_info: WindowInfo) -> torch.Tensor:
        # 제목 해시 (간단한 방법)
        title_hash = hash(window_info.title) % 1000
        title_emb = self.title_embedding(torch.tensor(title_hash, dtype=torch.long))
        
        # 클래스명 해시
        class_hash = hash(window_info.class_name) % 500
        class_emb = self.class_embedding(torch.tensor(class_hash, dtype=torch.long))
        
        # 프로세스명 해시
        process_hash = hash(window_info.process_name) % 1000
        process_emb = self.process_embedding(torch.tensor(process_hash, dtype=torch.long))
        
        # 공간 정보
        rect = torch.tensor(window_info.rect, dtype=torch.float32)
        spatial_features = torch.cat([
            rect,
            torch.tensor([float(window_info.is_minimized), float(window_info.is_maximized)], dtype=torch.float32)
        ])
        spatial_emb = self.spatial_encoder(spatial_features)
        
        # 특징 결합
        combined = torch.cat([title_emb, class_emb, process_emb, spatial_emb])
        features = self.feature_fusion(combined)
        
        return features

class ActivityFeatureExtractor(nn.Module):
    """사용자 활동 정보를 특징 벡터로 변환"""
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 마우스 위치 정규화
        self.mouse_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # 활동 타입 임베딩
        self.activity_embedding = nn.Embedding(4, 16)  # 4가지 활동 타입
        
        # 키보드 활동
        self.keyboard_encoder = nn.Linear(1, 16)
        
        # 특징 결합
        self.feature_fusion = nn.Sequential(
            nn.Linear(32 + 16 + 16, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, activity: UserActivity) -> torch.Tensor:
        # 마우스 위치 (화면 크기로 정규화)
        mouse_pos = torch.tensor(activity.mouse_position, dtype=torch.float32)
        # 간단한 정규화 (1920x1080 기준)
        mouse_pos = mouse_pos / torch.tensor([1920.0, 1080.0])
        mouse_features = self.mouse_encoder(mouse_pos)
        
        # 활동 타입
        activity_type_map = {'idle': 0, 'click': 1, 'keyboard': 2, 'window_change': 3}
        activity_type = activity_type_map.get(activity.activity_type, 0)
        activity_emb = self.activity_embedding(torch.tensor(activity_type, dtype=torch.long))
        
        # 키보드 활동
        keyboard_active = torch.tensor([float(activity.keyboard_active)], dtype=torch.float32)
        keyboard_features = self.keyboard_encoder(keyboard_active)
        
        # 특징 결합
        combined = torch.cat([mouse_features, activity_emb, keyboard_features])
        features = self.feature_fusion(combined)
        
        return features

class RealTimeWindowPredictor(nn.Module):
    """실시간 윈도우 배열 예측 모델"""
    def __init__(self, 
                 window_feature_dim: int = 128,
                 activity_feature_dim: int = 64,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 max_windows: int = 20):
        super().__init__()
        
        self.max_windows = max_windows
        self.window_feature_dim = window_feature_dim
        self.activity_feature_dim = activity_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 특징 추출기
        self.window_extractor = WindowFeatureExtractor(window_feature_dim)
        self.activity_extractor = ActivityFeatureExtractor(activity_feature_dim)
        
        # 입력 차원 계산
        input_dim = window_feature_dim + activity_feature_dim
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 위치 인코딩
        self.pos_encoding = PositionalEncoding(input_dim, max_len=100)
        
        # 출력 레이어 (윈도우별 위치, 크기 예측)
        self.output_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4)  # x, y, width, height
        )
        
        # 윈도우 존재 여부 예측
        self.window_existence = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                window_sequences: List[List[WindowInfo]], 
                activity_sequences: List[List[UserActivity]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            window_sequences: [batch_size, seq_len, windows] - 각 시퀀스의 윈도우 정보
            activity_sequences: [batch_size, seq_len] - 각 시퀀스의 활동 정보
        
        Returns:
            predicted_positions: [batch_size, max_windows, 4] - 예측된 윈도우 위치/크기
            window_existence: [batch_size, max_windows] - 윈도우 존재 확률
        """
        batch_size = len(window_sequences)
        seq_len = len(window_sequences[0])
        
        # 특징 추출 및 결합
        combined_features = []
        
        for batch_idx in range(batch_size):
            sequence_features = []
            for seq_idx in range(seq_len):
                # 윈도우 특징들 결합
                window_features = []
                for window in window_sequences[batch_idx][seq_idx][:self.max_windows]:
                    window_feat = self.window_extractor(window)
                    window_features.append(window_feat)
                
                # 최대 윈도우 수에 맞춰 패딩
                while len(window_features) < self.max_windows:
                    window_features.append(torch.zeros(self.window_feature_dim))
                
                # 윈도우 특징들을 평균화 (시퀀스 내 윈도우들의 통합된 특징)
                window_features = torch.stack(window_features)
                avg_window_features = torch.mean(window_features, dim=0)
                
                # 활동 특징
                activity_feat = self.activity_extractor(activity_sequences[batch_idx][seq_idx])
                
                # 특징 결합
                combined = torch.cat([avg_window_features, activity_feat])
                sequence_features.append(combined)
            
            # 시퀀스 특징들을 텐서로 변환
            sequence_tensor = torch.stack(sequence_features)
            combined_features.append(sequence_tensor)
        
        # 배치 차원 결합
        combined_features = torch.stack(combined_features)  # [batch_size, seq_len, feature_dim]
        
        # 위치 인코딩 추가
        combined_features = self.pos_encoding(combined_features.transpose(0, 1)).transpose(0, 1)
        
        # Transformer 처리
        transformer_output = self.transformer(combined_features)
        
        # 마지막 시퀀스 요소에서 예측
        last_features = transformer_output[:, -1, :]  # [batch_size, feature_dim]
        
        # 윈도우별 예측
        predicted_positions = []
        window_existence_probs = []
        
        for i in range(self.max_windows):
            # 각 윈도우 슬롯에 대한 특징 생성
            window_slot_features = last_features + torch.randn_like(last_features) * 0.1
            
            # 위치/크기 예측
            positions = self.output_projection(window_slot_features)  # [batch_size, 4]
            predicted_positions.append(positions)
            
            # 존재 여부 예측
            existence = self.window_existence(window_slot_features)  # [batch_size, 1]
            window_existence_probs.append(existence)
        
        predicted_positions = torch.stack(predicted_positions, dim=1)  # [batch_size, max_windows, 4]
        
        # window_existence 차원 올바르게 처리
        window_existence = torch.cat(window_existence_probs, dim=1)  # [batch_size, max_windows, 1]
        window_existence = window_existence.squeeze(-1)  # [batch_size, max_windows]
        
        return predicted_positions, window_existence

def create_model(config: Dict = None) -> RealTimeWindowPredictor:
    """모델 생성"""
    if config is None:
        config = {
            'window_feature_dim': 128,
            'activity_feature_dim': 64,
            'hidden_dim': 256,
            'num_heads': 8,
            'num_layers': 6,
            'max_windows': 20
        }
    
    model = RealTimeWindowPredictor(**config)
    return model

class WindowArrangementPredictor:
    """윈도우 배열 예측 및 적용"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """학습된 모델 로드"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model = create_model(checkpoint.get('config', {}))
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model = create_model()
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"모델 로드 완료: {model_path}")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            self.model = None
    
    def validate_window_handle(self, handle: int) -> bool:
        """윈도우 핸들 유효성 검사"""
        try:
            import win32gui
            return win32gui.IsWindow(handle) and win32gui.IsWindowVisible(handle)
        except Exception:
            return False
    
    def get_window_handles_from_sequence(self, window_sequence: List[List[WindowInfo]]) -> List[int]:
        """윈도우 시퀀스에서 유효한 핸들 추출"""
        valid_handles = []
        seen_handles = set()
        
        for time_step in window_sequence:
            for window in time_step:
                if window.handle and window.handle not in seen_handles:
                    if self.validate_window_handle(window.handle):
                        valid_handles.append(window.handle)
                        seen_handles.add(window.handle)
        
        return valid_handles
    
    def preprocess_input(self, 
                        window_sequence: List[List[WindowInfo]], 
                        activity_sequence: List[UserActivity]) -> Tuple[List[List[WindowInfo]], List[List[UserActivity]]]:
        """입력 데이터 전처리"""
        # 시퀀스 길이 정규화 (30초 관찰 데이터)
        target_length = 30
        
        if len(window_sequence) > target_length:
            # 최근 30개 샘플만 사용
            window_sequence = window_sequence[-target_length:]
            activity_sequence = activity_sequence[-target_length:]
        elif len(window_sequence) < target_length:
            # 부족한 부분은 마지막 샘플로 패딩
            last_windows = window_sequence[-1] if window_sequence else []
            last_activity = activity_sequence[-1] if activity_sequence else None
            
            while len(window_sequence) < target_length:
                window_sequence.append(last_windows)
                if last_activity:
                    activity_sequence.append(last_activity)
        
        return window_sequence, activity_sequence
    
    def predict_next_arrangement(self, 
                                window_sequence: List[List[WindowInfo]], 
                                activity_sequence: List[UserActivity]) -> List[Tuple[int, int, int, int]]:
        """다음 순간의 윈도우 배열 예측"""
        if not self.model:
            logger.error("모델이 로드되지 않았습니다.")
            return []
        
        try:
            # 입력 전처리
            processed_windows, processed_activities = self.preprocess_input(window_sequence, activity_sequence)
            
            # 배치 차원 추가
            window_batch = [processed_windows]
            activity_batch = [processed_activities]
            
            # 예측 수행
            with torch.no_grad():
                predicted_positions, window_existence = self.model(window_batch, activity_batch)
            
            # 결과 후처리
            positions = predicted_positions[0].cpu().numpy()  # [max_windows, 4]
            existence_probs = window_existence[0].cpu().numpy()  # [max_windows]
            
            # 존재 확률이 0.5 이상인 윈도우만 반환
            valid_windows = []
            for i, (pos, prob) in enumerate(zip(positions, existence_probs)):
                if prob > 0.5:
                    x, y, w, h = pos
                    
                    # NaN이나 무한대 값 체크
                    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(w) and np.isfinite(h)):
                        logger.warning(f"윈도우 {i}에 무효한 값: x={x}, y={y}, w={w}, h={h}")
                        continue
                    
                    # 화면 경계 내로 클램핑
                    x = max(0, min(x, 1920 - w))
                    y = max(0, min(y, 1080 - h))
                    w = max(100, min(w, 1920))
                    h = max(100, min(h, 1080))
                    
                    # 정수로 변환
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    
                    valid_windows.append((x, y, w, h))
            
            logger.info(f"윈도우 배열 예측 완료: {len(valid_windows)}개 윈도우")
            return valid_windows
            
        except Exception as e:
            logger.error(f"윈도우 배열 예측 실패: {e}")
            return []
    
    def apply_prediction(self, 
                        window_handles: List[int], 
                        predicted_positions: List[Tuple[int, int, int, int]]) -> bool:
        """예측된 윈도우 배열 적용"""
        try:
            import win32gui
            import win32con
            
            # Windows API 상수 정의 (win32gui에 없는 경우를 대비)
            HWND_TOP = 0
            SWP_SHOWWINDOW = 0x0040
            SWP_NOZORDER = 0x0004
            
            success_count = 0
            
            for i, (handle, (x, y, w, h)) in enumerate(zip(window_handles, predicted_positions)):
                if i >= len(predicted_positions):
                    break
                
                # 윈도우 핸들 유효성 검사
                if not self.validate_window_handle(handle):
                    logger.warning(f"유효하지 않은 윈도우 핸들: {handle}")
                    continue
                
                try:
                    # 윈도우 위치 및 크기 설정
                    # HWND_TOP 대신 0을 사용하고, z-order는 변경하지 않음
                    win32gui.SetWindowPos(
                        handle, 
                        HWND_TOP,  # 0 (최상위)
                        x, y, w, h, 
                        SWP_SHOWWINDOW | SWP_NOZORDER
                    )
                    
                    logger.debug(f"윈도우 {handle} 위치 설정: ({x}, {y}) 크기: {w}x{h}")
                    success_count += 1
                    
                except Exception as window_error:
                    logger.warning(f"윈도우 {handle} 위치 설정 실패: {window_error}")
                    continue
            
            logger.info(f"윈도우 배열 적용 완료: {success_count}/{len(predicted_positions)}개 윈도우")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"윈도우 배열 적용 실패: {e}")
            return False
    
    def predict_and_apply(self, 
                         window_sequence: List[List[WindowInfo]], 
                         activity_sequence: List[UserActivity]) -> bool:
        """예측 및 적용을 한 번에 수행"""
        try:
            # 예측 수행
            predicted_positions = self.predict_next_arrangement(window_sequence, activity_sequence)
            
            if not predicted_positions:
                logger.warning("예측된 윈도우 위치가 없습니다.")
                return False
            
            # 유효한 윈도우 핸들 추출
            valid_handles = self.get_window_handles_from_sequence(window_sequence)
            
            if not valid_handles:
                logger.warning("유효한 윈도우 핸들이 없습니다.")
                return False
            
            # 핸들 수와 예측 위치 수 맞추기
            min_count = min(len(valid_handles), len(predicted_positions))
            valid_handles = valid_handles[:min_count]
            predicted_positions = predicted_positions[:min_count]
            
            # 윈도우 배열 적용
            return self.apply_prediction(valid_handles, predicted_positions)
            
        except Exception as e:
            logger.error(f"예측 및 적용 실패: {e}")
            return False

if __name__ == "__main__":
    # 모델 테스트
    model = create_model()
    print(f"모델 생성 완료: {model}")
    
    # 입력 크기 확인
    dummy_windows = [[WindowInfo(0, "Test", "TestClass", 0, "test.exe", (0,0,100,100), True, False, False, 0, 0.0)]]
    dummy_activities = [UserActivity(0.0, None, [], (0,0), False, 'idle')]
    
    try:
        with torch.no_grad():
            positions, existence = model(dummy_windows, dummy_activities)
        print(f"모델 출력 크기: positions {positions.shape}, existence {existence.shape}")
    except Exception as e:
        print(f"모델 테스트 실패: {e}")
