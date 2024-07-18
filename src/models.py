import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class MaxNormLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, max_norm=1.0):
        super().__init__(in_features, out_features, bias)
        self.max_norm = max_norm

    def forward(self, input):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return F.linear(input, self.weight, self.bias)
    
class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        nb_classes: int,
        seq_len: int,
        in_channels: int,
        num_subjects: int,
        dropoutRate: float = 0.5,
        kernLength: int = 16,
        F1: int = 96,
        D: int = 1,
        F2: int = 96,
        max_norm: float = 2,
        dropoutType: str = 'Dropout',
    ) -> None:
        super().__init__()

        self.dropoutType = nn.Dropout if dropoutType == 'Dropout' else nn.Dropout2d

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.GELU()
        )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.GELU(),
            nn.AvgPool2d((1, 2)),
            self.dropoutType(dropoutRate)
        )
        
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.GELU(),
            nn.AvgPool2d((1, 2)),
            self.dropoutType(dropoutRate)
        )

        self.flat_dim = self._get_flat_dim(in_channels, seq_len)

        self.class_head = nn.Sequential(
            MaxNormLinear(self.flat_dim, nb_classes, max_norm=max_norm)
        )
        
        self.subject_head = nn.Sequential(
            nn.Linear(self.flat_dim, num_subjects)
        )

        self._initialize_weights()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X (b, c, t): _description_
        Returns:
            X (b, num_classes): _description_
        """
        X = X.unsqueeze(1)
        X = self.temporal_conv(X)
        X = self.depthwise_conv(X)
        X = self.separable_conv(X)
        X = X.flatten(start_dim=1)

        class_logits = self.class_head(X)
        subject_logits = self.subject_head(X)
        
        return class_logits, subject_logits

    def _get_flat_dim(self, in_channels, seq_len):
        dummy_input = torch.zeros(1, 1, in_channels, seq_len)
        with torch.no_grad():
            dummy_output = self.separable_conv(self.depthwise_conv(self.temporal_conv(dummy_input)))
        return dummy_output.numel()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)