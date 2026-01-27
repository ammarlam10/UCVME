import torch
from torch import Tensor
import torch.nn as nn
import timm
from typing import Any, Optional


class EfficientNetB0_unc(nn.Module):
    """
    EfficientNetB0 with uncertainty estimation.
    Similar structure to ResNet_unc with dropout applied after feature extraction.
    Uses timm EfficientNetB0 as backbone and adds mean/variance heads.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = False,
        drp_p: float = 0.2,
        **kwargs: Any
    ) -> None:
        super(EfficientNetB0_unc, self).__init__()
        
        # Load EfficientNetB0 from timm without classifier
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
            **kwargs
        )
        
        # Get feature dimension (EfficientNetB0 outputs 1280 features)
        # Access the classifier input features
        if hasattr(self.backbone, 'num_features'):
            self.num_features = self.backbone.num_features
        elif hasattr(self.backbone, 'classifier'):
            # Fallback: get from classifier if available
            if hasattr(self.backbone.classifier, 'in_features'):
                self.num_features = self.backbone.classifier.in_features
            else:
                self.num_features = 1280  # Default for EfficientNetB0
        else:
            self.num_features = 1280  # Default for EfficientNetB0
        
        print("using drp_p = ", drp_p)
        self.drop_rate = drp_p
        
        # Mean and variance prediction heads (similar to ResNet)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_m = nn.Linear(self.num_features, 1)
        self.fc_v = nn.Linear(self.num_features, 1)
        
        # Initialize the new layers
        nn.init.normal_(self.fc_m.weight, 0, 0.01)
        nn.init.constant_(self.fc_m.bias, 0)
        nn.init.normal_(self.fc_v.weight, 0, 0.01)
        nn.init.constant_(self.fc_v.bias, 0)
    
    def _forward_features_with_dropout(self, x: Tensor) -> Tensor:
        """
        Forward pass through EfficientNet with dropout applied after feature extraction.
        Similar to ResNet's approach of applying dropout after layer1, layer2, layer3.
        Note: training=True keeps dropout active for MC Dropout during inference.
        """
        # Extract features using forward_features
        # EfficientNetB0 structure: conv_stem -> blocks -> conv_head
        x = self.backbone.forward_features(x)
        
        # Apply dropout after feature extraction (similar to ResNet applying dropout
        # after layer1, layer2, layer3 - here we apply it after the full feature extraction)
        # Note: training=True keeps dropout active for MC Dropout during inference
        x = nn.functional.dropout(x, p=self.drop_rate, training=True)
        
        return x
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        """
        Forward implementation matching ResNet_unc structure.
        """
        # Extract features with dropout
        x = self._forward_features_with_dropout(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply dropout before heads (same as ResNet)
        x_feat_m = nn.functional.dropout(x, p=self.drop_rate, training=True)
        x_feat_v = nn.functional.dropout(x, p=self.drop_rate, training=True)
        
        # Predict mean and variance
        x_m = self.fc_m(x_feat_m)
        x_v = self.fc_v(x_feat_v)
        
        return x_m, x_v
    
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def efficientnetb0_unc(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNetB0_unc:
    r"""EfficientNetB0 model with uncertainty estimation using timm.
    
    Similar interface to resnet50_unc for drop-in replacement.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        drp_p (float): Dropout probability (default: 0.2)
    """
    return EfficientNetB0_unc(pretrained=pretrained, **kwargs)




