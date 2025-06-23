import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, List
import warnings
warnings.filterwarnings("ignore")


class AdvancedSCMGenerator:
    """고급 구조적 인과 모델 기반 데이터셋 생성기"""
    
    def __init__(self, feature_types=['numerical', 'categorical', 'ordinal']):
        self.feature_types = feature_types
        self.causal_mechanisms = [
            self._linear_mechanism,
            self._polynomial_mechanism,
            self._interaction_mechanism,
            self._threshold_mechanism,
            self._mixture_mechanism
        ]
    
    def _linear_mechanism(self, X: np.ndarray, indices: List[int]) -> np.ndarray:
        """선형 결합 메커니즘"""
        weights = np.random.randn(len(indices))
        return np.sum(X[:, indices] * weights, axis=1)
    
    def _polynomial_mechanism(self, X: np.ndarray, indices: List[int]) -> np.ndarray:
        """다항식 메커니즘"""
        result = np.zeros(X.shape[0])
        for i, idx in enumerate(indices):
            power = np.random.randint(1, 4)
            result += np.random.randn() * (X[:, idx] ** power)
        return result
    
    def _interaction_mechanism(self, X: np.ndarray, indices: List[int]) -> np.ndarray:
        """상호작용 메커니즘"""
        if len(indices) < 2:
            return self._linear_mechanism(X, indices)
        
        result = np.zeros(X.shape[0])
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                result += np.random.randn() * X[:, indices[i]] * X[:, indices[j]]
        return result
    
    def _threshold_mechanism(self, X: np.ndarray, indices: List[int]) -> np.ndarray:
        """임계값 기반 메커니즘"""
        base = self._linear_mechanism(X, indices)
        threshold = np.random.randn()
        return np.where(base > threshold, base * 2, base * 0.5)
    
    def _mixture_mechanism(self, X: np.ndarray, indices: List[int]) -> np.ndarray:
        """혼합 메커니즘"""
        mechanisms = [self._linear_mechanism, self._polynomial_mechanism, self._interaction_mechanism]
        chosen = np.random.choice(mechanisms)
        return chosen(X, indices)
    
    def generate_dataset(self, num_samples: int = 128, num_features: int = 16, 
                        num_classes: int = 3, complexity: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """복잡한 구조적 인과 모델 기반 데이터셋 생성"""
        
        # 피처 생성 (다양한 분포)
        X = np.zeros((num_samples, num_features))
        
        for i in range(num_features):
            dist_type = np.random.choice(['normal', 'uniform', 'exponential', 'beta'])
            if dist_type == 'normal':
                X[:, i] = np.random.normal(0, np.random.uniform(0.5, 2.0), num_samples)
            elif dist_type == 'uniform':
                X[:, i] = np.random.uniform(-2, 2, num_samples)
            elif dist_type == 'exponential':
                X[:, i] = np.random.exponential(1, num_samples) - 1
            else:  # beta
                X[:, i] = np.random.beta(2, 2, num_samples) * 4 - 2
        
        # 인과적 피처 선택
        num_causal = max(2, int(num_features * complexity))
        causal_indices = np.random.choice(num_features, size=num_causal, replace=False)
        
        # 복수 메커니즘 적용
        y_components = []
        for _ in range(np.random.randint(2, 4)):  # 2-3개의 메커니즘 조합
            mechanism = np.random.choice(self.causal_mechanisms)
            subset_size = np.random.randint(2, min(5, len(causal_indices) + 1))
            subset_indices = np.random.choice(causal_indices, size=subset_size, replace=False)
            y_components.append(mechanism(X, subset_indices))
        
        # 최종 타겟 생성
        y_raw = np.sum(y_components, axis=0)
        y_raw += np.random.normal(0, 0.1, num_samples)  # 노이즈 추가
        
        # 클래스 분할 (적응적 임계값)
        if num_classes == 2:
            y = (y_raw > np.median(y_raw)).astype(np.int64)
        else:
            percentiles = np.linspace(0, 100, num_classes + 1)
            thresholds = np.percentile(y_raw, percentiles[1:-1])
            y = np.digitize(y_raw, thresholds).astype(np.int64)
            y = np.clip(y, 0, num_classes - 1)
        
        return X.astype(np.float32), y

class TabPFNV2Dataset(Dataset):
    
    def __init__(self, num_tasks: int = 10000, min_features: int = 8, max_features: int = 32,
                 min_samples: int = 32, max_samples: int = 256, max_classes: int = 5):
        self.num_tasks = num_tasks
        self.min_features = min_features
        self.max_features = max_features
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.max_classes = max_classes
        self.generator = AdvancedSCMGenerator()
    
    def __len__(self):
        return self.num_tasks
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 랜덤 태스크 설정
        num_features = np.random.randint(self.min_features, self.max_features + 1)
        num_samples = np.random.randint(self.min_samples, self.max_samples + 1)
        num_classes = np.random.randint(2, self.max_classes + 1)
        complexity = np.random.uniform(0.3, 0.8)
        
        # 데이터 생성
        X, y = self.generator.generate_dataset(
            num_samples=num_samples,
            num_features=num_features,
            num_classes=num_classes,
            complexity=complexity
        )
        
        # 컨텍스트/테스트 분할
        split_idx = np.random.randint(1, num_samples)
        context_x = torch.tensor(X[:split_idx])
        context_y = torch.tensor(y[:split_idx])
        test_x = torch.tensor(X[split_idx:])
        test_y = torch.tensor(y[split_idx:])
        
        return context_x, context_y, test_x, test_y