# stay-positive.py
import torch
import torch.nn as nn
from typing import Optional, Literal

class StayPositiveTrainer:
    """
    Stay-Positive 방법을 적용하는 트레이너
    논문: Stay-Positive: A Case for Ignoring Real Image Features in Fake Image Detection
    """
    
    def __init__(
        self,
        model: nn.Module,
        method: Literal['clamp', 'sigmoid'] = 'clamp',
        target_layer_name: str = 'fc',
        freeze_backbone: bool = True,
        unfreeze_last_k_blocks: int = 0
    ):
        """
        Args:
            model: 학습할 모델
            method: 'clamp' (0 이상으로 제한) 또는 'sigmoid' (0~1 사이로 제한)
            target_layer_name: Stay-Positive를 적용할 레이어 이름
            freeze_backbone: 백본을 고정할지 여부
            unfreeze_last_k_blocks: 백본에서 학습할 마지막 k개 블록
        """
        self.model = model
        self.method = method
        self.target_layer_name = target_layer_name
        
        if freeze_backbone:
            self._freeze_backbone(unfreeze_last_k_blocks)
    
    def _freeze_backbone(self, unfreeze_last_k: int = 0):
        """백본을 고정하고 마지막 레이어만 학습 가능하게 설정"""
        
        # ResNet 스타일 모델 가정 (layer1, layer2, layer3, layer4)
        backbone_blocks = []
        for name, _ in self.model.named_children():
            if 'layer' in name or 'block' in name:
                backbone_blocks.append(name)
        
        # 언프리즈할 블록 결정
        blocks_to_unfreeze = backbone_blocks[-unfreeze_last_k:] if unfreeze_last_k > 0 else []
        
        # 파라미터 고정/해제
        for name, param in self.model.named_parameters():
            # 타겟 레이어(fc)는 항상 학습
            if self.target_layer_name in name:
                param.requires_grad = True
                # 선택적: FC 레이어 초기화
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            # 언프리즈할 블록에 속하면 학습
            elif any(block in name for block in blocks_to_unfreeze):
                param.requires_grad = True
            # 나머지는 고정
            else:
                param.requires_grad = False
    
    def apply_stay_positive(self):
        """Stay-Positive 제약 적용"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if self.target_layer_name in name and 'weight' in name:
                    if self.method == 'clamp':
                        param.data.clamp_(min=0)
                    elif self.method == 'sigmoid':
                        param.data = torch.sigmoid(param.data)
    
    def train_step(self, inputs, labels, loss_fn, optimizer):
        """단일 학습 스텝 실행"""
        # Forward pass
        outputs = self.model(inputs)
        if outputs.dim() > 2:
            outputs = outputs.squeeze()
        
        # Loss 계산
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Stay-Positive 적용
        self.apply_stay_positive()
        
        return loss.item()

# 간단한 wrapper 함수
def add_stay_positive_to_model(
    model: nn.Module,
    method: str = 'clamp',
    fc_layer_name: str = 'fc'
) -> nn.Module:
    """
    기존 모델에 Stay-Positive를 적용하는 간단한 함수
    
    사용 예:
    model = add_stay_positive_to_model(model, method='clamp')
    """
    
    class StayPositiveWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.method = method
            self.fc_layer_name = fc_layer_name
        
        def forward(self, x):
            return self.base_model(x)
        
        def apply_stay_positive(self):
            with torch.no_grad():
                for name, param in self.base_model.named_parameters():
                    if self.fc_layer_name in name and 'weight' in name:
                        if self.method == 'clamp':
                            param.data.clamp_(min=0)
    
    return StayPositiveWrapper(model)