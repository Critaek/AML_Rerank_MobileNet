from .matcher import MatchERT
from .mobilenet import mobilenetv3_large
import torch.nn as nn

class Network(nn.Module):
    def __init__(self,
                 num_global_features,
                 num_local_features,
                 ert_dim_feedforward,
                 ert_nhead,
                 ert_num_encoder_layers, ert_dropout, ert_activation, ert_normalize_before):
        self.backbone = mobilenetv3_large(num_local_features=num_local_features)
        self.transformer = MatchERT(d_global=num_global_features, d_model=num_local_features, 
            nhead=ert_nhead, num_encoder_layers=ert_num_encoder_layers, 
            dim_feedforward=ert_dim_feedforward, dropout=ert_dropout, 
            activation=ert_activation, normalize_before=ert_normalize_before)

    def forward(self, x):
        
