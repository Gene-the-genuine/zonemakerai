import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import logging
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import matplotlib.pyplot as plt

from backend.core.data_collector import UserActivity, WindowInfo
from backend.ml.model import create_model, RealTimeWindowPredictor

logger = logging.getLogger(__name__)

def custom_collate_fn(batch):
    """배치 데이터를 동일한 크기로 패딩하는 커스텀 collate 함수"""
    if not batch:
        return {
            'window_sequences': [],
            'activity_sequences': [],
            'target_positions': torch.empty(0, 0, 4),
            'sequence_length': 0
        }
    
    try:
        # 배치 내 최대 시퀀스 길이와 최대 윈도우 수 찾기
        max_seq_len = max(len(item['window_sequences']) for item in batch if item and 'window_sequences' in item)
        max_windows = max(
            len(item['window_sequences'][0]) if item and 'window_sequences' in item and item['window_sequences'] else 0 
            for item in batch
        )
        
        # 최소값 보장
        max_seq_len = max(max_seq_len, 1)
        max_windows = max(max_windows, 1)
        
        # 모델의 max_windows와 일치하도록 강제 조정 (20으로 고정)
        target_max_windows = 20
        
        # 배치 데이터 정규화
        normalized_window_sequences = []
        normalized_activity_sequences = []
        normalized_target_positions = []
        
        for item in batch:
            if not item or 'window_sequences' not in item:
                continue
                
            window_seq = item['window_sequences']
            activity_seq = item.get('activity_sequences', [])
            target_pos = item.get('target_positions', [])
            
            # 시퀀스 길이 정규화
            if len(window_seq) > max_seq_len:
                window_seq = window_seq[-max_seq_len:]
                activity_seq = activity_seq[-max_seq_len:]
            elif len(window_seq) < max_seq_len:
                # 부족한 부분은 마지막 샘플로 패딩
                last_windows = window_seq[-1] if window_seq else []
                last_activity = activity_seq[-1] if activity_seq else None
                
                while len(window_seq) < max_seq_len:
                    window_seq.append(last_windows)
                    if last_activity:
                        activity_seq.append(last_activity)
            
            # 윈도우 수를 정확히 target_max_windows로 정규화
            for i in range(len(window_seq)):
                if len(window_seq[i]) > target_max_windows:
                    window_seq[i] = window_seq[i][:target_max_windows]
                elif len(window_seq[i]) < target_max_windows:
                    # 부족한 윈도우는 더미 윈도우로 패딩
                    while len(window_seq[i]) < target_max_windows:
                        dummy_window = WindowInfo(
                            handle=0, title="", class_name="", process_id=0, 
                            process_name="", rect=(0,0,100,100), is_visible=False, 
                            is_minimized=False, is_maximized=False, z_order=0, timestamp=0.0
                        )
                        window_seq[i].append(dummy_window)
            
            # 활동 시퀀스 정규화
            for i in range(len(activity_seq)):
                if activity_seq[i] is None:
                    # 더미 활동 생성
                    activity_seq[i] = UserActivity(
                        timestamp=0.0, active_window=None, all_windows=[], 
                        mouse_position=(0,0), keyboard_active=False, activity_type='idle'
                    )
            
            # 타겟 위치를 정확히 target_max_windows 크기로 정규화
            if isinstance(target_pos, torch.Tensor):
                target_pos = target_pos.tolist()
            
            # 타겟 위치를 정확히 target_max_windows 크기로 맞추기
            if len(target_pos) > target_max_windows:
                target_pos = target_pos[:target_max_windows]
            elif len(target_pos) < target_max_windows:
                while len(target_pos) < target_max_windows:
                    target_pos.append([0, 0, 100, 100])
            
            # 각 위치가 정확히 4개 값(x, y, width, height)을 가지도록 보장
            for i in range(len(target_pos)):
                if len(target_pos[i]) != 4:
                    if len(target_pos[i]) < 4:
                        target_pos[i].extend([0] * (4 - len(target_pos[i])))
                    else:
                        target_pos[i] = target_pos[i][:4]
            
            normalized_window_sequences.append(window_seq)
            normalized_activity_sequences.append(activity_seq)
            normalized_target_positions.append(target_pos)
        
        if not normalized_window_sequences:
            return {
                'window_sequences': [],
                'activity_sequences': [],
                'target_positions': torch.empty(0, 0, 4),
                'sequence_length': 0
            }
        
        # 타겟 위치를 텐서로 변환 - 완벽한 크기 보장
        try:
            target_tensor = torch.tensor(normalized_target_positions, dtype=torch.float32)
            # 텐서 크기 검증 - 정확히 [batch_size, target_max_windows, 4] 형태여야 함
            expected_shape = (len(normalized_window_sequences), target_max_windows, 4)
            
            if target_tensor.shape != expected_shape:
                logger.warning(f"타겟 텐서 크기 불일치: {target_tensor.shape} vs {expected_shape}")
                # 크기 강제 조정
                if len(target_tensor.shape) == 2:
                    target_tensor = target_tensor.unsqueeze(0)
                
                # 배치 차원이 1인 경우 확장
                if target_tensor.shape[0] == 1 and len(normalized_window_sequences) > 1:
                    target_tensor = target_tensor.expand(len(normalized_window_sequences), -1, -1)
                
                # 윈도우 수 차원 조정
                if target_tensor.shape[1] != target_max_windows:
                    if target_tensor.shape[1] > target_max_windows:
                        target_tensor = target_tensor[:, :target_max_windows, :]
                    else:
                        padding = torch.zeros(target_tensor.shape[0], target_max_windows - target_tensor.shape[1], 4)
                        target_tensor = torch.cat([target_tensor, padding], dim=1)
                
                # 위치 차원 조정
                if target_tensor.shape[2] != 4:
                    if target_tensor.shape[2] > 4:
                        target_tensor = target_tensor[:, :, :4]
                    else:
                        padding = torch.zeros(target_tensor.shape[0], target_tensor.shape[1], 4 - target_tensor.shape[2])
                        target_tensor = torch.cat([target_tensor, padding], dim=2)
                
                # 최종 검증
                if target_tensor.shape != expected_shape:
                    logger.error(f"최종 텐서 크기 조정 실패: {target_tensor.shape} vs {expected_shape}")
                    # 강제로 올바른 크기 생성
                    target_tensor = torch.zeros(expected_shape, dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"타겟 텐서 변환 실패: {e}")
            # 기본 텐서 생성 - 정확한 크기로
            target_tensor = torch.zeros(len(normalized_window_sequences), target_max_windows, 4, dtype=torch.float32)
        
        return {
            'window_sequences': normalized_window_sequences,
            'activity_sequences': normalized_activity_sequences,
            'target_positions': target_tensor,
            'sequence_length': max_seq_len
        }
        
    except Exception as e:
        logger.error(f"배치 정규화 중 오류: {e}")
        # 오류 발생 시 빈 배치 반환
        return {
            'window_sequences': [],
            'activity_sequences': [],
            'target_positions': torch.empty(0, 0, 4),
            'sequence_length': 0
        }

class WindowActivityDataset(Dataset):
    """윈도우 활동 데이터셋 (30초 시퀀스 기반)"""
    
    def __init__(self, 
                 window_sequences: List[List[List[WindowInfo]]], 
                 activity_sequences: List[List[UserActivity]],
                 target_positions: List[List[Tuple[int, int, int, int]]],
                 sequence_length: int = 30):
        """
        Args:
            window_sequences: [num_samples, seq_len, windows] - 각 샘플의 윈도우 시퀀스
            activity_sequences: [num_samples, seq_len] - 각 샘플의 활동 시퀀스
            target_positions: [num_samples, num_windows, 4] - 목표 윈도우 위치/크기
            sequence_length: 시퀀스 길이 (기본값: 30초)
        """
        self.window_sequences = window_sequences
        self.activity_sequences = activity_sequences
        self.target_positions = target_positions
        self.sequence_length = sequence_length
        
        # 데이터 검증
        assert len(window_sequences) == len(activity_sequences) == len(target_positions), \
            "데이터 길이가 일치하지 않습니다."
        
        logger.info(f"데이터셋 생성 완료: {len(window_sequences)}개 샘플")
    
    def __len__(self):
        return len(self.window_sequences)
    
    def __getitem__(self, idx):
        window_seq = self.window_sequences[idx]
        activity_seq = self.activity_sequences[idx]
        target_pos = self.target_positions[idx]
        
        # 시퀀스 길이 정규화
        if len(window_seq) > self.sequence_length:
            window_seq = window_seq[-self.sequence_length:]
            activity_seq = activity_seq[-self.sequence_length:]
        elif len(window_seq) < self.sequence_length:
            # 패딩
            last_windows = window_seq[-1] if window_seq else []
            last_activity = activity_seq[-1] if activity_seq else None
            
            while len(window_seq) < self.sequence_length:
                window_seq.append(last_windows)
                if last_activity:
                    activity_seq.append(last_activity)
        
        return {
            'window_sequences': window_seq,
            'activity_sequences': activity_seq,
            'target_positions': torch.tensor(target_pos, dtype=torch.float32),
            'sequence_length': self.sequence_length
        }

class ModelTrainer:
    """실시간 윈도우 배열 예측 모델 훈련기"""
    
    def __init__(self, 
                 model: RealTimeWindowPredictor,
                 device: str = 'auto',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        
        self.model = model
        
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # 옵티마이저 및 손실 함수
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 손실 함수: 위치/크기 예측 + 윈도우 존재 여부 예측
        self.position_criterion = nn.MSELoss()
        self.existence_criterion = nn.BCELoss()
        
        # 학습 상태
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'position_loss': [],
            'existence_loss': []
        }
        
        logger.info(f"모델 훈련기 초기화 완료 (디바이스: {self.device})")
    
    def prepare_training_data(self, 
                            data_file: str,
                            train_split: float = 0.8,
                            batch_size: int = 8,
                            sequence_length: int = 30) -> Tuple[DataLoader, DataLoader]:
        """훈련 데이터 준비"""
        try:
            # 데이터 로드
            from backend.core.data_collector import DataCollector
            data_collector = DataCollector()
            activities = data_collector.load_data(data_file)
            
            if not activities:
                raise ValueError("데이터를 로드할 수 없습니다.")
            
            # 시퀀스 단위로 그룹화
            window_sequences, activity_sequences, target_positions = self._prepare_sequences(
                activities, sequence_length
            )
            
            # 훈련/검증 분할
            num_samples = len(window_sequences)
            train_size = int(num_samples * train_split)
            
            train_windows = window_sequences[:train_size]
            train_activities = activity_sequences[:train_size]
            train_targets = target_positions[:train_size]
            
            val_windows = window_sequences[train_size:]
            val_activities = activity_sequences[train_size:]
            val_targets = target_positions[train_size:]
            
            # 데이터셋 생성
            train_dataset = WindowActivityDataset(
                train_windows, train_activities, train_targets, sequence_length
            )
            val_dataset = WindowActivityDataset(
                val_windows, val_activities, val_targets, sequence_length
            )
            
            # 데이터로더 생성 (커스텀 collate_fn 사용)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=0,  # Windows에서 안정성을 위해 0으로 설정
                collate_fn=custom_collate_fn
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=0,
                collate_fn=custom_collate_fn
            )
            
            logger.info(f"훈련 데이터 준비 완료: 훈련 {len(train_dataset)}개, 검증 {len(val_dataset)}개")
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"훈련 데이터 준비 실패: {e}")
            raise
    
    def _prepare_sequences(self, 
                          activities: List[UserActivity], 
                          sequence_length: int) -> Tuple[List, List, List]:
        """활동 데이터를 시퀀스로 변환"""
        window_sequences = []
        activity_sequences = []
        target_positions = []
        
        # 시퀀스 단위로 그룹화
        for i in range(0, len(activities), sequence_length):
            sequence_activities = activities[i:i + sequence_length]
            
            if len(sequence_activities) < sequence_length:
                continue  # 완전한 시퀀스가 아닌 경우 건너뛰기
            
            # 윈도우 시퀀스
            window_seq = []
            for activity in sequence_activities:
                window_seq.append(activity.all_windows)
            
            # 다음 시퀀스의 윈도우 위치를 목표로 설정
            if i + sequence_length < len(activities):
                next_activity = activities[i + sequence_length]
                target_pos = [(w.rect[0], w.rect[1], w.rect[2] - w.rect[0], w.rect[3] - w.rect[1]) 
                             for w in next_activity.all_windows]
                
                # 최대 윈도우 수에 맞춰 패딩
                max_windows = self.model.max_windows
                while len(target_pos) < max_windows:
                    target_pos.append((0, 0, 100, 100))  # 기본값
                target_pos = target_pos[:max_windows]
                
                window_sequences.append(window_seq)
                activity_sequences.append(sequence_activities)
                target_positions.append(target_pos)
        
        return window_sequences, activity_sequences, target_positions
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0.0
        total_position_loss = 0.0
        total_existence_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            try:
                # 빈 배치 체크
                if not batch or len(batch) == 0:
                    logger.warning("빈 배치 건너뛰기")
                    continue
                
                # 데이터를 디바이스로 이동
                window_sequences = batch['window_sequences']
                activity_sequences = batch['activity_sequences']
                target_positions = batch['target_positions'].to(self.device)
                
                # 텐서 크기 검증 및 수정 - 모델의 max_windows와 정확히 일치
                batch_size = len(window_sequences)
                model_max_windows = self.model.max_windows  # 정확히 20
                
                # 타겟 위치 텐서 크기 검증 및 강제 수정
                if target_positions.shape[1] != model_max_windows:
                    logger.warning(f"훈련 타겟 위치 윈도우 수 불일치: {target_positions.shape[1]} vs {model_max_windows}")
                    if target_positions.shape[1] > model_max_windows:
                        target_positions = target_positions[:, :model_max_windows, :]
                    else:
                        padding = torch.zeros(batch_size, model_max_windows - target_positions.shape[1], 4, device=self.device)
                        target_positions = torch.cat([target_positions, padding], dim=1)
                
                if target_positions.shape[2] != 4:
                    logger.warning(f"훈련 타겟 위치 차원 불일치: {target_positions.shape[2]} vs 4")
                    if target_positions.shape[2] > 4:
                        target_positions = target_positions[:, :, :4]
                    else:
                        padding = torch.zeros(batch_size, model_max_windows, 4 - target_positions.shape[2], device=self.device)
                        target_positions = torch.cat([target_positions, padding], dim=2)
                
                # 최종 크기 검증
                expected_shape = (batch_size, model_max_windows, 4)
                if target_positions.shape != expected_shape:
                    logger.error(f"훈련 최종 타겟 위치 크기 불일치: {target_positions.shape} vs {expected_shape}")
                    # 강제로 올바른 크기 생성
                    target_positions = torch.zeros(expected_shape, device=self.device)
                
                # 그래디언트 초기화
                self.optimizer.zero_grad()
                
                # 순전파
                predicted_positions, window_existence = self.model(
                    window_sequences, activity_sequences
                )
                
                # 예측 결과 크기 검증 및 수정
                if predicted_positions.shape != target_positions.shape:
                    logger.warning(f"훈련 예측 결과 크기 불일치: {predicted_positions.shape} vs {target_positions.shape}")
                    # 크기 맞추기
                    if predicted_positions.shape[1] != target_positions.shape[1]:
                        if predicted_positions.shape[1] > target_positions.shape[1]:
                            predicted_positions = predicted_positions[:, :target_positions.shape[1], :]
                        else:
                            padding = torch.zeros(batch_size, target_positions.shape[1] - predicted_positions.shape[1], 4, device=self.device)
                            predicted_positions = torch.cat([predicted_positions, padding], dim=1)
                
                # 손실 계산
                position_loss = self.position_criterion(predicted_positions, target_positions)
                
                # 윈도우 존재 여부 계산 - 차원 완벽하게 맞추기
                target_existence = (target_positions.sum(dim=-1) > 0).float()  # [batch_size, max_windows]
                
                # window_existence 차원 검증 및 수정
                if window_existence.shape[1] != target_existence.shape[1]:
                    logger.warning(f"훈련 존재 여부 윈도우 수 불일치: {window_existence.shape[1]} vs {target_existence.shape[1]}")
                    if window_existence.shape[1] > target_existence.shape[1]:
                        window_existence = window_existence[:, :target_existence.shape[1]]
                    else:
                        padding = torch.zeros(batch_size, target_existence.shape[1] - window_existence.shape[1], device=self.device)
                        window_existence = torch.cat([window_existence, padding], dim=1)
                
                # 차원이 3차원인 경우 2차원으로 변환
                if len(window_existence.shape) == 3 and window_existence.shape[2] == 1:
                    window_existence = window_existence.squeeze(-1)  # [batch_size, max_windows]
                
                # target_existence도 동일한 차원으로 맞추기
                if len(target_existence.shape) == 3 and target_existence.shape[2] == 1:
                    target_existence = target_existence.squeeze(-1)  # [batch_size, max_windows]
                
                # 최종 차원 검증 및 강제 맞추기
                if window_existence.shape != target_existence.shape:
                    logger.warning(f"훈련 최종 존재 여부 차원 불일치: {window_existence.shape} vs {target_existence.shape}")
                    # 강제로 차원 맞추기
                    min_windows = min(window_existence.shape[1], target_existence.shape[1])
                    window_existence = window_existence[:, :min_windows]
                    target_existence = target_existence[:, :min_windows]
                
                existence_loss = self.existence_criterion(window_existence, target_existence)
                
                # 전체 손실
                loss = position_loss + 0.1 * existence_loss
                
                # 역전파
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # 통계 업데이트
                total_loss += loss.item()
                total_position_loss += position_loss.item()
                total_existence_loss += existence_loss.item()
                num_batches += 1
                
            except Exception as e:
                logger.warning(f"배치 훈련 중 오류: {e}")
                continue
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_position_loss = total_position_loss / num_batches
            avg_existence_loss = total_existence_loss / num_batches
        else:
            avg_loss = avg_position_loss = avg_existence_loss = 0.0
        
        return {
            'loss': avg_loss,
            'position_loss': avg_position_loss,
            'existence_loss': avg_existence_loss
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """한 에포크 검증"""
        self.model.eval()
        total_loss = 0.0
        total_position_loss = 0.0
        total_existence_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    # 빈 배치 체크
                    if not batch or len(batch) == 0:
                        logger.warning("빈 배치 건너뛰기")
                        continue
                    
                    # 데이터를 디바이스로 이동
                    window_sequences = batch['window_sequences']
                    activity_sequences = batch['activity_sequences']
                    target_positions = batch['target_positions'].to(self.device)
                    
                    # 텐서 크기 검증 및 수정 - 모델의 max_windows와 정확히 일치 (검증과 동일)
                    batch_size = len(window_sequences)
                    model_max_windows = self.model.max_windows  # 정확히 20
                    
                    # 타겟 위치 텐서 크기 검증 및 강제 수정
                    if target_positions.shape[1] != model_max_windows:
                        logger.warning(f"검증 타겟 위치 윈도우 수 불일치: {target_positions.shape[1]} vs {model_max_windows}")
                        if target_positions.shape[1] > model_max_windows:
                            target_positions = target_positions[:, :model_max_windows, :]
                        else:
                            padding = torch.zeros(batch_size, model_max_windows - target_positions.shape[1], 4, device=self.device)
                            target_positions = torch.cat([target_positions, padding], dim=1)
                    
                    if target_positions.shape[2] != 4:
                        logger.warning(f"검증 타겟 위치 차원 불일치: {target_positions.shape[2]} vs 4")
                        if target_positions.shape[2] > 4:
                            target_positions = target_positions[:, :, :4]
                        else:
                            padding = torch.zeros(batch_size, model_max_windows, 4 - target_positions.shape[2], device=self.device)
                            target_positions = torch.cat([target_positions, padding], dim=2)
                    
                    # 최종 크기 검증
                    expected_shape = (batch_size, model_max_windows, 4)
                    if target_positions.shape != expected_shape:
                        logger.error(f"검증 최종 타겟 위치 크기 불일치: {target_positions.shape} vs {expected_shape}")
                        # 강제로 올바른 크기 생성
                        target_positions = torch.zeros(expected_shape, device=self.device)
                    
                    # 순전파
                    predicted_positions, window_existence = self.model(
                        window_sequences, activity_sequences
                    )
                    
                    # 예측 결과 크기 검증 및 수정
                    if predicted_positions.shape != target_positions.shape:
                        logger.warning(f"검증 예측 결과 크기 불일치: {predicted_positions.shape} vs {target_positions.shape}")
                        # 크기 맞추기
                        if predicted_positions.shape[1] != target_positions.shape[1]:
                            if predicted_positions.shape[1] > target_positions.shape[1]:
                                predicted_positions = predicted_positions[:, :target_positions.shape[1], :]
                            else:
                                padding = torch.zeros(batch_size, target_positions.shape[1] - predicted_positions.shape[1], 4, device=self.device)
                                predicted_positions = torch.cat([predicted_positions, padding], dim=1)
                    
                    # 손실 계산
                    position_loss = self.position_criterion(predicted_positions, target_positions)
                    
                    # 윈도우 존재 여부 계산 - 차원 완벽하게 맞추기 (검증과 동일)
                    target_existence = (target_positions.sum(dim=-1) > 0).float()  # [batch_size, max_windows]
                    
                    # window_existence 차원 검증 및 수정
                    if window_existence.shape[1] != target_existence.shape[1]:
                        logger.warning(f"검증 존재 여부 윈도우 수 불일치: {window_existence.shape[1]} vs {target_existence.shape[1]}")
                        if window_existence.shape[1] > target_existence.shape[1]:
                            window_existence = window_existence[:, :target_existence.shape[1]]
                        else:
                            padding = torch.zeros(batch_size, target_existence.shape[1] - window_existence.shape[1], device=self.device)
                            window_existence = torch.cat([window_existence, padding], dim=1)
                    
                    # 차원이 3차원인 경우 2차원으로 변환
                    if len(window_existence.shape) == 3 and window_existence.shape[2] == 1:
                        window_existence = window_existence.squeeze(-1)  # [batch_size, max_windows]
                    
                    # target_existence도 동일한 차원으로 맞추기
                    if len(target_existence.shape) == 3 and target_existence.shape[2] == 1:
                        target_existence = target_existence.squeeze(-1)  # [batch_size, max_windows]
                    
                    # 최종 차원 검증 및 강제 맞추기
                    if window_existence.shape != target_existence.shape:
                        logger.warning(f"검증 최종 존재 여부 차원 불일치: {window_existence.shape} vs {target_existence.shape}")
                        # 강제로 차원 맞추기
                        min_windows = min(window_existence.shape[1], target_existence.shape[1])
                        window_existence = window_existence[:, :min_windows]
                        target_existence = target_existence[:, :min_windows]
                    
                    existence_loss = self.existence_criterion(window_existence, target_existence)
                    
                    loss = position_loss + 0.1 * existence_loss
                    
                    # 통계 업데이트
                    total_loss += loss.item()
                    total_position_loss += position_loss.item()
                    total_existence_loss += existence_loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.warning(f"배치 검증 중 오류: {e}")
                    continue
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_position_loss = total_position_loss / num_batches
            avg_existence_loss = total_existence_loss / num_batches
        else:
            avg_loss = avg_position_loss = avg_existence_loss = 0.0
        
        return {
            'loss': avg_loss,
            'position_loss': avg_position_loss,
            'existence_loss': avg_existence_loss
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100,
              patience: int = 10,
              save_dir: str = "data/models") -> Dict:
        """모델 훈련"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 학습률 스케줄러
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5 # verbose=True
        )
        
        # 조기 종료
        early_stopping_counter = 0
        
        logger.info(f"훈련 시작: {num_epochs} 에포크")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # 훈련
            train_metrics = self.train_epoch(train_loader)
            
            # 검증
            val_metrics = self.validate_epoch(val_loader)
            
            # 학습률 조정
            scheduler.step(val_metrics['loss'])
            
            # 히스토리 업데이트
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['position_loss'].append(train_metrics['position_loss'])
            self.training_history['existence_loss'].append(train_metrics['existence_loss'])
            
            # 로그 출력
            logger.info(
                f"Epoch {self.current_epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}"
            )
            
            # 최고 모델 저장
            if val_metrics['loss'] < self.best_loss:
                self.best_loss = val_metrics['loss']
                self.save_model(save_dir, 'best')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # 체크포인트 저장
            if (epoch + 1) % 10 == 0:
                self.save_model(save_dir, f'checkpoint_epoch_{epoch + 1}')
            
            # 조기 종료 체크
            if early_stopping_counter >= patience:
                logger.info(f"조기 종료: {patience} 에포크 동안 개선 없음")
                break
        
        # 최종 모델 저장
        self.save_model(save_dir, 'final')
        
        # 훈련 히스토리 저장
        self.save_training_history(save_dir)
        
        # 손실 곡선 플롯
        self.plot_training_curves(save_dir)
        
        logger.info("훈련 완료!")
        return self.training_history
    
    def save_model(self, save_dir: str, name: str):
        """모델 저장"""
        try:
            # 모델 상태 및 설정 저장
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                'training_history': self.training_history,
                'config': {
                    'window_feature_dim': self.model.window_feature_dim,
                    'activity_feature_dim': self.model.activity_feature_dim,
                    'hidden_dim': self.model.hidden_dim,
                    'num_heads': self.model.num_heads,
                    'num_layers': self.model.num_layers,
                    'max_windows': self.model.max_windows
                }
            }
            
            save_path = os.path.join(save_dir, f'{name}.pth')
            torch.save(checkpoint, save_path)
            
            logger.info(f"모델 저장 완료: {save_path}")
            
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
    
    def save_training_history(self, save_dir: str):
        """훈련 히스토리 저장"""
        try:
            history_path = os.path.join(save_dir, 'training_history.json')
            
            # numpy 배열을 리스트로 변환
            history_data = {}
            for key, values in self.training_history.items():
                if isinstance(values, list):
                    history_data[key] = [float(v) for v in values]
                else:
                    history_data[key] = values
            
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"훈련 히스토리 저장 완료: {history_path}")
            
        except Exception as e:
            logger.error(f"훈련 히스토리 저장 실패: {e}")
    
    def plot_training_curves(self, save_dir: str):
        """훈련 곡선 플롯"""
        try:
            plt.figure(figsize=(15, 5))
            
            # 손실 곡선
            plt.subplot(1, 3, 1)
            plt.plot(self.training_history['train_loss'], label='Train Loss')
            plt.plot(self.training_history['val_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # 위치 손실
            plt.subplot(1, 3, 2)
            plt.plot(self.training_history['position_loss'], label='Position Loss')
            plt.title('Position Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # 존재 여부 손실
            plt.subplot(1, 3, 3)
            plt.plot(self.training_history['existence_loss'], label='Existence Loss')
            plt.title('Existence Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            # 저장
            plot_path = os.path.join(save_dir, 'training_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"훈련 곡선 저장 완료: {plot_path}")
            
        except Exception as e:
            logger.error(f"훈련 곡선 저장 실패: {e}")
    
    def export_to_onnx(self, save_path: str, input_shape: Tuple = None):
        """ONNX 형식으로 모델 내보내기"""
        try:
            self.model.eval()
            
            # 더미 입력 생성
            if input_shape is None:
                batch_size, seq_len = 1, 30
                max_windows = self.model.max_windows
                
                # 더미 윈도우 시퀀스
                dummy_windows = []
                for _ in range(batch_size):
                    seq_windows = []
                    for _ in range(seq_len):
                        windows = [WindowInfo(0, "dummy", "dummy", 0, "dummy.exe", 
                                            (0,0,100,100), True, False, False, 0, 0.0)] * max_windows
                        seq_windows.append(windows)
                    dummy_windows.append(seq_windows)
                
                # 더미 활동 시퀀스
                dummy_activities = []
                for _ in range(batch_size):
                    seq_activities = []
                    for _ in range(seq_len):
                        activity = UserActivity(0.0, None, [], (0,0), False, 'idle')
                        seq_activities.append(activity)
                    dummy_activities.append(seq_activities)
                
                # ONNX 내보내기
                torch.onnx.export(
                    self.model,
                    (dummy_windows, dummy_activities),
                    save_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['window_sequences', 'activity_sequences'],
                    output_names=['predicted_positions', 'window_existence'],
                    dynamic_axes={
                        'window_sequences': {0: 'batch_size', 1: 'sequence_length'},
                        'activity_sequences': {0: 'batch_size', 1: 'sequence_length'},
                        'predicted_positions': {0: 'batch_size'},
                        'window_existence': {0: 'batch_size'}
                    }
                )
            
            logger.info(f"ONNX 모델 내보내기 완료: {save_path}")
            
        except Exception as e:
            logger.error(f"ONNX 모델 내보내기 실패: {e}")

def prepare_training_data(data_file: str,
                         train_split: float = 0.8,
                         batch_size: int = 8,
                         sequence_length: int = 30) -> Tuple[DataLoader, DataLoader]:
    """훈련 데이터 준비 헬퍼 함수"""
    # 더미 모델로 데이터로더 생성 (실제 훈련 시에는 훈련된 모델 사용)
    dummy_model = create_model()
    trainer = ModelTrainer(dummy_model)
    
    return trainer.prepare_training_data(data_file, train_split, batch_size, sequence_length)

if __name__ == "__main__":
    # 훈련기 테스트
    print("모델 훈련기 테스트...")
    
    # 모델 생성
    model = create_model()
    print(f"모델 생성 완료: {model}")
    
    # 훈련기 생성
    trainer = ModelTrainer(model)
    print(f"훈련기 생성 완료")
    
    # 모델 파라미터 수 확인
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 파라미터 수: {total_params:,}")
    print(f"훈련 가능한 파라미터 수: {trainable_params:,}")
    
    print("테스트 완료!")
