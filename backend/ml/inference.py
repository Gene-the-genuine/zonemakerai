import time
import threading
import logging
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import asdict
import json
from datetime import datetime

from backend.core.data_collector import DataCollector, UserActivity, WindowInfo
from backend.ml.model import WindowArrangementPredictor

logger = logging.getLogger(__name__)

class RealTimeInferenceEngine:
    """실시간 윈도우 배열 추론 엔진"""
    
    def __init__(self, model_path: str = None, prediction_interval: float = 1.0):
        self.model_path = model_path
        self.prediction_interval = prediction_interval
        
        # 데이터 수집기 및 예측기
        self.data_collector = DataCollector()
        self.predictor = WindowArrangementPredictor(model_path) if model_path else None
        
        # 실시간 추론 상태
        self.inference_active = False
        self.inference_thread = None
        
        # 데이터 버퍼 (30초 관찰 데이터)
        self.window_buffer: List[List[WindowInfo]] = []
        self.activity_buffer: List[UserActivity] = []
        self.buffer_max_size = 30  # 30초 관찰
        
        # 추론 통계
        self.inference_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_inference_time': 0.0,
            'last_prediction_time': 0.0
        }
        
        # 콜백 함수
        self.on_prediction_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
    
    def start_inference(self):
        """실시간 추론 시작"""
        if self.inference_active:
            logger.warning("이미 추론이 진행 중입니다.")
            return False
        
        if not self.predictor:
            logger.error("모델이 로드되지 않았습니다.")
            return False
        
        self.inference_active = True
        self.inference_thread = threading.Thread(
            target=self._inference_worker,
            daemon=True
        )
        self.inference_thread.start()
        
        logger.info("실시간 추론 엔진 시작됨")
        return True
    
    def stop_inference(self):
        """실시간 추론 중지"""
        try:
            self.inference_active = False
            if self.inference_thread and self.inference_thread.is_alive():
                self.inference_thread.join(timeout=2.0)
            
            logger.info("실시간 추론 엔진 중지됨")
        except Exception as e:
            logger.warning(f"추론 엔진 중지 중 오류: {e}")
    
    def _inference_worker(self):
        """추론 워커 스레드"""
        try:
            while self.inference_active:
                start_time = time.time()
                
                try:
                    # 현재 윈도우 상태 수집
                    current_windows = self.data_collector.get_all_windows()
                    current_activity = self.data_collector.collect_activity_sample()
                    
                    if current_activity:
                        # 버퍼에 데이터 추가
                        self.window_buffer.append(current_windows)
                        self.activity_buffer.append(current_activity)
                        
                        # 버퍼 크기 제한 (30초 관찰)
                        if len(self.window_buffer) > self.buffer_max_size:
                            self.window_buffer.pop(0)
                            self.activity_buffer.pop(0)
                        
                        # 30초 데이터가 수집되면 예측 시작
                        if len(self.window_buffer) >= self.buffer_max_size:
                            self._perform_prediction()
                    
                    # 다음 예측까지 대기
                    elapsed = time.time() - start_time
                    sleep_time = max(0, self.prediction_interval - elapsed)
                    time.sleep(sleep_time)
                    
                except Exception as batch_error:
                    logger.warning(f"추론 배치 처리 중 오류: {batch_error}")
                    time.sleep(1.0)  # 오류 발생 시 잠시 대기
                    continue
                    
        except Exception as e:
            logger.error(f"추론 워커 스레드 오류: {e}")
            if self.on_error:
                try:
                    self.on_error({
                        'success': False,
                        'error': f'워커 스레드 오류: {str(e)}',
                        'inference_time': 0.0
                    })
                except Exception as callback_error:
                    logger.warning(f"오류 콜백 실행 실패: {callback_error}")
    
    def _perform_prediction(self):
        """예측 수행 및 적용"""
        try:
            start_time = time.time()
            
            # 버퍼에서 데이터 추출
            window_sequence = self.window_buffer.copy()
            activity_sequence = self.activity_buffer.copy()
            
            # 예측 및 적용을 한 번에 수행 (개선된 메서드 사용)
            success = self.predictor.predict_and_apply(window_sequence, activity_sequence)
            
            # 통계 업데이트
            inference_time = time.time() - start_time
            self.inference_stats['total_predictions'] += 1
            self.inference_stats['last_prediction_time'] = time.time()
            
            if success:
                self.inference_stats['successful_predictions'] += 1
                logger.info(f"윈도우 배열 예측 및 적용 성공 (소요시간: {inference_time:.3f}초)")
                
                # 성공 콜백 호출
                if self.on_prediction_complete:
                    try:
                        self.on_prediction_complete({
                            'success': True,
                            'inference_time': inference_time,
                            'window_count': len(window_sequence[-1]) if window_sequence else 0
                        })
                    except Exception as callback_error:
                        logger.warning(f"성공 콜백 실행 실패: {callback_error}")
            else:
                self.inference_stats['failed_predictions'] += 1
                logger.warning(f"윈도우 배열 예측 및 적용 실패 (소요시간: {inference_time:.3f}초)")
                
                # 실패 콜백 호출
                if self.on_error:
                    try:
                        self.on_error({
                            'success': False,
                            'error': '예측 및 적용 실패',
                            'inference_time': inference_time
                        })
                    except Exception as callback_error:
                        logger.warning(f"실패 콜백 실행 실패: {callback_error}")
            
            # 평균 추론 시간 업데이트
            current_avg = self.inference_stats['average_inference_time']
            total_preds = self.inference_stats['total_predictions']
            self.inference_stats['average_inference_time'] = (
                (current_avg * (total_preds - 1) + inference_time) / total_preds
            )
            
        except Exception as e:
            self.inference_stats['failed_predictions'] += 1
            logger.error(f"예측 수행 중 오류: {e}")
            
            # 오류 콜백 호출
            if self.on_error:
                try:
                    self.on_error({
                        'success': False,
                        'error': str(e),
                        'inference_time': 0.0
                    })
                except Exception as callback_error:
                    logger.warning(f"오류 콜백 실행 실패: {callback_error}")
    
    def get_inference_stats(self) -> Dict:
        """추론 통계 반환"""
        return self.inference_stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        try:
            self.inference_stats = {
                'total_predictions': 0,
                'successful_predictions': 0,
                'failed_predictions': 0,
                'average_inference_time': 0.0,
                'last_prediction_time': 0.0
            }
            logger.info("추론 통계 초기화 완료")
        except Exception as e:
            logger.warning(f"통계 초기화 중 오류: {e}")
    
    def get_inference_status(self) -> Dict:
        """추론 상태 정보 반환"""
        try:
            return {
                'inference_active': self.inference_active,
                'buffer_size': len(self.window_buffer),
                'buffer_max_size': self.buffer_max_size,
                'thread_alive': self.inference_thread.is_alive() if self.inference_thread else False,
                'stats': self.inference_stats.copy()
            }
        except Exception as e:
            logger.warning(f"상태 정보 반환 중 오류: {e}")
            return {
                'inference_active': False,
                'error': str(e),
                'stats': {}
            }
    
    def set_prediction_interval(self, interval: float):
        """예측 간격 설정"""
        try:
            self.prediction_interval = max(0.1, interval)  # 최소 0.1초
            logger.info(f"예측 간격 설정: {self.prediction_interval}초")
        except Exception as e:
            logger.warning(f"예측 간격 설정 중 오류: {e}")
    
    def set_callbacks(self, 
                     on_prediction_complete: Optional[Callable] = None,
                     on_error: Optional[Callable] = None):
        """콜백 함수 설정"""
        try:
            self.on_prediction_complete = on_prediction_complete
            self.on_error = on_error
            logger.info("콜백 함수 설정 완료")
        except Exception as e:
            logger.warning(f"콜백 함수 설정 중 오류: {e}")
    
    def save_inference_log(self, filename: str = None):
        """추론 로그 저장"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inference_log_{timestamp}.json"
        
        try:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'inference_stats': self.inference_stats,
                'buffer_info': {
                    'window_buffer_size': len(self.window_buffer),
                    'activity_buffer_size': len(self.activity_buffer)
                },
                'system_info': {
                    'inference_active': self.inference_active,
                    'thread_alive': self.inference_thread.is_alive() if self.inference_thread else False
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"추론 로그 저장 완료: {filename}")
            
        except Exception as e:
            logger.error(f"추론 로그 저장 실패: {e}")
    
    def get_detailed_stats(self) -> Dict:
        """상세한 추론 통계 반환"""
        try:
            return {
                'basic_stats': self.inference_stats.copy(),
                'buffer_info': {
                    'window_buffer_size': len(self.window_buffer),
                    'activity_buffer_size': len(self.activity_buffer),
                    'buffer_max_size': self.buffer_max_size
                },
                'system_status': {
                    'inference_active': self.inference_active,
                    'thread_alive': self.inference_thread.is_alive() if self.inference_thread else False,
                    'prediction_interval': self.prediction_interval
                }
            }
        except Exception as e:
            logger.warning(f"상세 통계 반환 중 오류: {e}")
            return {'error': str(e)}

class ContinuousLearningEngine:
    """연속 학습 엔진 (실시간 데이터로 모델 업데이트)"""
    
    def __init__(self, model_path: str, update_interval: float = 300.0):  # 5분마다 업데이트
        self.model_path = model_path
        self.update_interval = update_interval
        
        self.data_collector = DataCollector()
        self.predictor = WindowArrangementPredictor(model_path)
        
        self.learning_active = False
        self.learning_thread = None
        
        # 학습 데이터 버퍼
        self.training_buffer: List[Tuple[List[List[WindowInfo]], List[UserActivity], List[Tuple[int, int, int, int]]]] = []
        self.buffer_max_size = 100  # 최대 100개 학습 샘플
        
        # 실제 사용자 행동과 예측 결과 비교
        self.actual_positions: List[List[Tuple[int, int, int, int]]] = []
        
    def start_continuous_learning(self):
        """연속 학습 시작"""
        if self.learning_active:
            logger.warning("이미 연속 학습이 진행 중입니다.")
            return False
        
        self.learning_active = True
        self.learning_thread = threading.Thread(
            target=self._learning_worker,
            daemon=True
        )
        self.learning_thread.start()
        
        logger.info("연속 학습 엔진 시작됨")
        return True
    
    def stop_continuous_learning(self):
        """연속 학습 중지"""
        self.learning_active = False
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=2.0)
        
        logger.info("연속 학습 엔진 중지됨")
    
    def _learning_worker(self):
        """학습 워커 스레드"""
        try:
            while self.learning_active:
                start_time = time.time()
                
                # 30초간 데이터 수집
                window_sequence = []
                activity_sequence = []
                
                for _ in range(30):  # 30초 관찰
                    if not self.learning_active:
                        break
                    
                    windows = self.data_collector.get_all_windows()
                    activity = self.data_collector.collect_activity_sample()
                    
                    if activity:
                        window_sequence.append(windows)
                        activity_sequence.append(activity)
                    
                    time.sleep(1.0)
                
                if len(window_sequence) >= 30:
                    # 예측 수행
                    predicted_positions = self.predictor.predict_next_arrangement(
                        window_sequence, activity_sequence
                    )
                    
                    # 1초 후 실제 윈도우 위치 수집
                    time.sleep(1.0)
                    actual_windows = self.data_collector.get_all_windows()
                    actual_positions = [(w.rect[0], w.rect[1], w.rect[2] - w.rect[0], w.rect[3] - w.rect[1]) 
                                      for w in actual_windows]
                    
                    # 학습 데이터 버퍼에 추가
                    self.training_buffer.append((window_sequence, activity_sequence, actual_positions))
                    
                    # 버퍼 크기 제한
                    if len(self.training_buffer) > self.buffer_max_size:
                        self.training_buffer.pop(0)
                    
                    logger.info(f"학습 데이터 수집 완료: {len(self.training_buffer)}개 샘플")
                
                # 다음 학습 사이클까지 대기
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"학습 워커 스레드 오류: {e}")
    
    def get_training_data_count(self) -> int:
        """수집된 학습 데이터 수 반환"""
        return len(self.training_buffer)
    
    def export_training_data(self, filename: str = None):
        """학습 데이터 내보내기"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"continuous_training_data_{timestamp}.json"
        
        try:
            export_data = []
            for window_seq, activity_seq, actual_pos in self.training_buffer:
                # 데이터를 JSON 직렬화 가능한 형태로 변환
                export_item = {
                    'window_sequences': [[asdict(w) for w in seq] for seq in window_seq],
                    'activity_sequence': [asdict(a) for a in activity_seq],
                    'actual_positions': actual_pos
                }
                export_data.append(export_item)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"학습 데이터 내보내기 완료: {filename}")
            
        except Exception as e:
            logger.error(f"학습 데이터 내보내기 실패: {e}")

class BatchInferenceEngine:
    """배치 추론 엔진 (오프라인 예측 및 평가)"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.predictor = WindowArrangementPredictor(model_path) if model_path else None
    
    def batch_predict(self, 
                     data_file: str,
                     output_file: str = None) -> List[Dict]:
        """배치 예측 수행"""
        if not self.predictor:
            logger.error("모델이 로드되지 않았습니다.")
            return []
        
        try:
            # 데이터 로드
            data_collector = DataCollector()
            activities = data_collector.load_data(data_file)
            
            if not activities:
                logger.error("데이터를 로드할 수 없습니다.")
                return []
            
            # 시퀀스 단위로 그룹화 (30초씩)
            sequences = self._group_into_sequences(activities, 30)
            
            results = []
            for i, (window_seq, activity_seq) in enumerate(sequences):
                # 예측 수행
                predicted_positions = self.predictor.predict_next_arrangement(
                    window_seq, activity_seq
                )
                
                result = {
                    'sequence_id': i,
                    'input_length': len(window_seq),
                    'predicted_positions': predicted_positions,
                    'prediction_time': time.time()
                }
                results.append(result)
            
            # 결과 저장
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"배치 예측 결과 저장: {output_file}")
            
            logger.info(f"배치 예측 완료: {len(results)}개 시퀀스")
            return results
            
        except Exception as e:
            logger.error(f"배치 예측 실패: {e}")
            return []
    
    def _group_into_sequences(self, 
                             activities: List[UserActivity], 
                             sequence_length: int) -> List[Tuple[List[List[WindowInfo]], List[UserActivity]]]:
        """활동 데이터를 시퀀스로 그룹화"""
        sequences = []
        
        for i in range(0, len(activities), sequence_length):
            sequence_activities = activities[i:i + sequence_length]
            
            # 각 활동에서 윈도우 정보 추출
            window_sequence = []
            for activity in sequence_activities:
                window_sequence.append(activity.all_windows)
            
            sequences.append((window_sequence, sequence_activities))
        
        return sequences

if __name__ == "__main__":
    # 테스트 실행
    print("실시간 추론 엔진 테스트...")
    
    # 엔진 생성
    engine = RealTimeInferenceEngine()
    
    # 상태 확인
    status = engine.get_inference_status()
    print(f"엔진 상태: {status}")
    
    # 연속 학습 엔진 테스트
    print("\n연속 학습 엔진 테스트...")
    learning_engine = ContinuousLearningEngine("dummy_model.pth")
    
    print(f"학습 데이터 수: {learning_engine.get_training_data_count()}")
    
    print("테스트 완료!")
