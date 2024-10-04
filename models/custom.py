import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lib.BEiT_v2 import modeling_finetune


class DDBF_BEiT_v2(nn.Module):
    def __init__(self, num_class=2,**kwargs):
        super(DDBF_BEiT_v2,self).__init__()
        self.num_class = num_class
        feature_dim = 768
        self.backbone_model  = modeling_finetune.beit_base_patch16_224(**BEiT_Config)
        self.backbone_model2 = modeling_finetune.beit_base_patch16_224(**BEiT_Config)
        self.backbone_model.head = nn.Identity()
        self.backbone_model2.head = nn.Identity()        
        pretrain_path = kwargs['pretrained_path']
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        checkpoint_model = checkpoint['model'] # module
        msg1 = self.backbone_model.load_state_dict(checkpoint_model, strict=False)
        msg2 = self.backbone_model2.load_state_dict(checkpoint_model, strict=False)
        print('load info', msg1)
        print('load info', msg2)
        self.norm_for_cls = nn.LayerNorm(feature_dim*1)
        self.header = nn.Sequential(nn.Linear(feature_dim*1, num_class))
    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.flatten(0,1)
        clstoken = self.backbone_model(x)
        clstoken2 = self.backbone_model2(x)
        pearson_loss = pearson_correlation_loss(clstoken.detach(), clstoken2)
        pred1 = self.header(self.norm_for_cls(clstoken))
        pred2 = self.header(self.norm_for_cls(clstoken2))
        pred1_un = uncertainty(pred1)
        pred2_un = uncertainty(pred2)
        weight = (-10)*torch.cat([pred1_un,pred2_un], dim=-1)
        w = torch.nn.functional.softmax(weight, dim=-1)
        weighted_output = torch.matmul(w.unsqueeze(1), torch.stack((clstoken, clstoken2), dim=1)).squeeze(1)
        pred = self.header(self.norm_for_cls(weighted_output))
        output  = pred.view(B, T, -1)
        output1 = pred1.view(B, T, -1)
        output2 = pred2.view(B, T, -1)  
        return output, output1, output2, pearson_loss

BEiT_Config = {
    'drop_path_rate': 0.1,
    'use_mean_pooling': False,
    'init_values': 0.1,
    'qkv_bias': True,
    'use_abs_pos_emb': False,
    'use_rel_pos_bias': True,
    'use_shared_rel_pos_bias': False
}


def pearson_correlation_loss(x, y): 
    mean_x = torch.mean(x, dim=-1,keepdim=True) #[32,1]
    mean_y = torch.mean(y, dim=-1,keepdim=True) #[32,1]
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    x_u = xm.unsqueeze(1) #[B,1,768]
    y_u = ym.unsqueeze(2) #[B,768,1]
    Conv_xy  = torch.bmm(x_u, y_u).squeeze()/x.size(-1) #[B]
    std_x    = torch.std(x, dim=-1) #[B]
    std_y    = torch.std(y, dim=-1) #[B]
    pearson_cor = Conv_xy/(std_x*std_y)
    pearson_cor = pearson_cor.mean()
    r = torch.clamp(pearson_cor, -1.0, 1.0)
    return r


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10)) 


def uncertainty(output):
    evidence = exp_evidence(output)
    alpha = evidence + 1
    uncertainty = 2 / torch.sum(alpha, dim=1, keepdim=True)
    return uncertainty
