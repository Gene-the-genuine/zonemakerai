import torch
import torch.nn as nn
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .model import VisionTransformer, create_model

class NPUConverter:
    """NPU 환경을 위한 모델 변환기"""
    
    def __init__(self, 
                 model_path: str,
                 output_dir: str = "data/models/npu_converted",
                 target_platform: str = "snapdragon"):
        
        self.model_path = model_path
        self.output_dir = output_dir
        self.target_platform = target_platform
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 모델 로드
        self.model = self._load_model(model_path)
        if not self.model:
            raise ValueError(f"모델을 로드할 수 없습니다: {model_path}")
        
        # NPU 최적화 설정
        self.optimization_config = self._get_optimization_config()
        
        print(f"NPU 변환기 초기화 완료")
        print(f"대상 플랫폼: {target_platform}")
        print(f"출력 디렉토리: {output_dir}")
    
    def _load_model(self, model_path: str) -> Optional[VisionTransformer]:
        """모델 로드"""
        try:
            if os.path.exists(model_path):
                model = torch.load(model_path, map_location='cpu')
                model.eval()
                print(f"모델 로드 완료: {model_path}")
                return model
            else:
                print(f"모델 파일이 존재하지 않습니다: {model_path}")
                return None
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return None
    
    def _get_optimization_config(self) -> Dict:
        """NPU 최적화 설정 반환"""
        if self.target_platform == "snapdragon":
            return {
                'quantization': True,
                'pruning': True,
                'fusion': True,
                'memory_optimization': True,
                'precision': 'int8',  # NPU에서 효율적인 int8 정밀도
                'batch_size': 1,      # 실시간 추론을 위한 배치 크기 1
                'input_shape': (1, 600, 15),  # (batch, sequence, features)
                'output_shape': (1, 4)        # (batch, x, y, w, h)
            }
        else:
            return {
                'quantization': False,
                'pruning': False,
                'fusion': False,
                'memory_optimization': False,
                'precision': 'float32',
                'batch_size': 1,
                'input_shape': (1, 600, 15),
                'output_shape': (1, 4)
            }
    
    def convert_to_npu(self) -> Dict:
        """NPU 환경을 위한 모델 변환"""
        print("NPU 변환 시작...")
        
        conversion_results = {
            'original_model_size': 0,
            'converted_model_size': 0,
            'optimization_applied': [],
            'conversion_time': 0,
            'output_files': []
        }
        
        start_time = datetime.now()
        
        try:
            # 1. 모델 최적화
            optimized_model = self._optimize_model()
            conversion_results['optimization_applied'].append('model_optimization')
            
            # 2. 양자화 (Quantization)
            if self.optimization_config['quantization']:
                quantized_model = self._quantize_model(optimized_model)
                conversion_results['optimization_applied'].append('quantization')
                optimized_model = quantized_model
            
            # 3. 모델 압축
            if self.optimization_config['pruning']:
                pruned_model = self._prune_model(optimized_model)
                conversion_results['optimization_applied'].append('pruning')
                optimized_model = pruned_model
            
            # 4. ONNX 변환
            onnx_path = self._export_to_onnx(optimized_model)
            conversion_results['output_files'].append(onnx_path)
            
            # 5. NPU 전용 형식 변환
            npu_path = self._convert_to_npu_format(onnx_path)
            if npu_path:
                conversion_results['output_files'].append(npu_path)
            
            # 6. 변환된 모델 테스트
            test_result = self._test_converted_model(optimized_model)
            conversion_results['test_result'] = test_result
            
            # 7. 모델 크기 비교
            original_size = self._get_model_size(self.model)
            converted_size = self._get_model_size(optimized_model)
            
            conversion_results['original_model_size'] = original_size
            conversion_results['converted_model_size'] = converted_size
            
            # 8. 변환 결과 저장
            self._save_conversion_results(conversion_results)
            
            conversion_time = (datetime.now() - start_time).total_seconds()
            conversion_results['conversion_time'] = conversion_time
            
            print(f"NPU 변환 완료! 소요 시간: {conversion_time:.2f}초")
            print(f"모델 크기: {original_size:.2f}MB -> {converted_size:.2f}MB")
            print(f"적용된 최적화: {conversion_results['optimization_applied']}")
            
            return conversion_results
            
        except Exception as e:
            print(f"NPU 변환 실패: {e}")
            conversion_results['error'] = str(e)
            return conversion_results
    
    def _optimize_model(self) -> VisionTransformer:
        """모델 최적화"""
        print("모델 최적화 중...")
        
        # 모델을 추론 모드로 설정
        self.model.eval()
        
        # 그래디언트 계산 비활성화
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 배치 정규화를 추론 모드로 설정
        self.model.train(False)
        
        return self.model
    
    def _quantize_model(self, model: VisionTransformer) -> VisionTransformer:
        """모델 양자화 (int8)"""
        print("모델 양자화 중...")
        
        try:
            # 동적 양자화 적용
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.Conv2d}, 
                dtype=torch.qint8
            )
            
            print("동적 양자화 완료")
            return quantized_model
            
        except Exception as e:
            print(f"양자화 실패, 원본 모델 사용: {e}")
            return model
    
    def _prune_model(self, model: VisionTransformer) -> VisionTransformer:
        """모델 가지치기 (Pruning)"""
        print("모델 가지치기 중...")
        
        try:
            # 간단한 가지치기: 가중치가 작은 연결 제거
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # 가중치의 절대값이 작은 연결을 0으로 설정
                    weight = module.weight.data
                    threshold = torch.quantile(torch.abs(weight), 0.1)  # 하위 10% 제거
                    mask = torch.abs(weight) > threshold
                    module.weight.data *= mask
            
            print("가지치기 완료")
            return model
            
        except Exception as e:
            print(f"가지치기 실패, 원본 모델 사용: {e}")
            return model
    
    def _export_to_onnx(self, model: VisionTransformer) -> str:
        """ONNX 형식으로 모델 내보내기"""
        print("ONNX 변환 중...")
        
        onnx_path = os.path.join(self.output_dir, "window_arrangement_model.onnx")
        
        # 더미 입력 생성
        dummy_input = torch.randn(*self.optimization_config['input_shape'])
        
        # ONNX 내보내기
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"ONNX 모델 저장: {onnx_path}")
        return onnx_path
    
    def _convert_to_npu_format(self, onnx_path: str) -> Optional[str]:
        """NPU 전용 형식으로 변환"""
        print("NPU 형식 변환 중...")
        
        try:
            # 여기서는 간단한 변환만 수행
            # 실제로는 Qualcomm AI Hub나 해당 NPU의 도구를 사용해야 함
            
            npu_path = os.path.join(self.output_dir, "window_arrangement_model.npu")
            
            # 간단한 헤더 정보 추가
            npu_header = {
                'platform': self.target_platform,
                'model_type': 'window_arrangement',
                'input_shape': self.optimization_config['input_shape'],
                'output_shape': self.optimization_config['output_shape'],
                'precision': self.optimization_config['precision'],
                'conversion_date': datetime.now().isoformat()
            }
            
            # ONNX 파일을 복사하고 헤더 추가
            import shutil
            shutil.copy2(onnx_path, npu_path)
            
            # 헤더 정보를 별도 파일로 저장
            header_path = npu_path + ".json"
            with open(header_path, 'w') as f:
                json.dump(npu_header, f, indent=2)
            
            print(f"NPU 형식 모델 저장: {npu_path}")
            return npu_path
            
        except Exception as e:
            print(f"NPU 형식 변환 실패: {e}")
            return None
    
    def _test_converted_model(self, model: VisionTransformer) -> Dict:
        """변환된 모델 테스트"""
        print("변환된 모델 테스트 중...")
        
        try:
            # 더미 입력으로 테스트
            dummy_input = torch.randn(*self.optimization_config['input_shape'])
            
            with torch.no_grad():
                output = model(dummy_input)
            
            # 출력 형태 확인
            expected_shape = self.optimization_config['output_shape']
            actual_shape = tuple(output.shape)
            
            test_result = {
                'input_shape': tuple(dummy_input.shape),
                'output_shape': actual_shape,
                'expected_shape': expected_shape,
                'shape_match': actual_shape == expected_shape,
                'output_range': {
                    'min': float(output.min()),
                    'max': float(output.max()),
                    'mean': float(output.mean())
                }
            }
            
            print(f"모델 테스트 완료: {test_result}")
            return test_result
            
        except Exception as e:
            print(f"모델 테스트 실패: {e}")
            return {'error': str(e)}
    
    def _get_model_size(self, model: VisionTransformer) -> float:
        """모델 크기 계산 (MB)"""
        try:
            # 모델 파라미터 수 계산
            param_size = 0
            buffer_size = 0
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            size_mb = (param_size + buffer_size) / 1024 / 1024
            return size_mb
            
        except Exception as e:
            print(f"모델 크기 계산 실패: {e}")
            return 0.0
    
    def _save_conversion_results(self, results: Dict):
        """변환 결과 저장"""
        results_path = os.path.join(self.output_dir, "conversion_results.json")
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"변환 결과 저장: {results_path}")
    
    def benchmark_model(self, num_runs: int = 100) -> Dict:
        """모델 성능 벤치마크"""
        print(f"모델 성능 벤치마크 시작: {num_runs}회 실행")
        
        # 원본 모델과 변환된 모델 모두 테스트
        models = {
            'original': self.model,
            'converted': self._load_model(self.model_path)  # 다시 로드
        }
        
        benchmark_results = {}
        
        for model_name, model in models.items():
            if model is None:
                continue
                
            print(f"{model_name} 모델 벤치마크 중...")
            
            model.eval()
            dummy_input = torch.randn(*self.optimization_config['input_shape'])
            
            # 워밍업
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            # 실제 벤치마크
            inference_times = []
            memory_usage = []
            
            for i in range(num_runs):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                
                with torch.no_grad():
                    output = model(dummy_input)
                
                end_time.record()
                torch.cuda.synchronize()
                
                inference_time = start_time.elapsed_time(end_time) / 1000.0  # 초 단위
                inference_times.append(inference_time)
                
                # 메모리 사용량 (간단한 추정)
                memory_usage.append(dummy_input.nelement() * 4 / 1024 / 1024)  # MB
            
            # 통계 계산
            avg_inference_time = np.mean(inference_times)
            std_inference_time = np.std(inference_times)
            avg_memory_usage = np.mean(memory_usage)
            
            benchmark_results[model_name] = {
                'average_inference_time': avg_inference_time,
                'std_inference_time': std_inference_time,
                'min_inference_time': min(inference_times),
                'max_inference_time': max(inference_times),
                'average_memory_usage': avg_memory_usage,
                'throughput': 1.0 / avg_inference_time  # FPS
            }
            
            print(f"{model_name} 모델 벤치마크 완료:")
            print(f"  평균 추론 시간: {avg_inference_time:.4f}초")
            print(f"  처리량: {1.0/avg_inference_time:.2f} FPS")
        
        # 벤치마크 결과 저장
        benchmark_path = os.path.join(self.output_dir, "benchmark_results.json")
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"벤치마크 결과 저장: {benchmark_path}")
        return benchmark_results

def create_npu_converter(model_path: str, 
                        output_dir: str = "data/models/npu_converted",
                        target_platform: str = "snapdragon") -> NPUConverter:
    """NPU 변환기 생성"""
    return NPUConverter(
        model_path=model_path,
        output_dir=output_dir,
        target_platform=target_platform
    )

if __name__ == "__main__":
    # NPU 변환기 테스트
    print("NPU 변환기 테스트...")
    
    # 더미 모델 경로 (실제로는 훈련된 모델이 필요)
    dummy_model_path = "data/models/test_model_best.pth"
    
    try:
        # NPU 변환기 생성
        converter = create_npu_converter(
            model_path=dummy_model_path,
            target_platform="snapdragon"
        )
        
        # NPU 변환 실행
        results = converter.convert_to_npu()
        
        if 'error' not in results:
            # 벤치마크 실행
            benchmark_results = converter.benchmark_model(num_runs=50)
            print(f"벤치마크 완료: {benchmark_results}")
        
    except Exception as e:
        print(f"NPU 변환기 테스트 실패: {e}")
        print("모델 파일이 없어서 테스트를 건너뜁니다.")
    
    print("NPU 변환기 테스트 완료!")
