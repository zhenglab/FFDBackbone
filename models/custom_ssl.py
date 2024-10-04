import torch
import timm
import torch.nn as nn
import torchvision
import torch.nn.functional as F


'''MAE'''
from models.lib.MAE import models_mae

class MAE(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(MAE,self).__init__()
        feature_dim = kwargs['feature_dim']
        self.backbone_model = models_mae.mae_vit_base_patch16()
        checkpoint = torch.load(kwargs['pretrained_path'], map_location='cpu')
        self.backbone_model.load_state_dict(checkpoint['model'], strict=False)
        self.normal = nn.LayerNorm(feature_dim*1)
        self.header = nn.Sequential(nn.Linear(feature_dim*1, num_class))
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        feature, _, _ = self.backbone_model.forward_encoder(x, mask_ratio=0)
        clstoken = feature[: , 0]
        output = self.normal(clstoken)
        output = self.header(output)
        output = output.view(B,T,-1)
        return output


'''BEiT_v2'''
from models.lib.BEiT_v2 import modeling_finetune

class BEiT_v2(nn.Module):
    def __init__(self, 
        pretrained=True, 
        **kwargs):
        super(BEiT_v2,self).__init__()
        feature_dim = 768
        self.backbone_model = modeling_finetune.beit_base_patch16_224(**BEiT_Config)
        self.backbone_model.head = nn.Identity()
        if pretrained:
            checkpoint = torch.load(kwargs['pretrained_path'], map_location='cpu')
            self.backbone_model.load_state_dict(checkpoint['model'], strict=False)
        self.norm_for_cls = nn.LayerNorm(feature_dim*1)
        self.header = nn.Sequential(nn.Linear(feature_dim*1, 2))
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        clstoken = self.backbone_model(x)
        output = self.norm_for_cls(clstoken)
        output = self.header(output)
        output = output.view(B,T,-1)
        return output


'''SIMMIM'''
from models.lib.SIMMIM import build_model
from models.lib.SIMMIM.config import parse_option
import models.lib.SIMMIM.utils as SIMMIM_utils

class SIMMIM(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(SIMMIM,self).__init__()
        feature_dim = kwargs['feature_dim']
        _,config = parse_option()
        self.backbone_model = build_model(config, is_pretrain=False)
        checkpoint = torch.load(kwargs['pretrained_path'], map_location='cpu')
        checkpoint_model = checkpoint['model']
        if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
            checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if k.startswith('encoder.')}
            print('Detect pre-trained model, remove [encoder.] prefix.')
        else:
            print('Detect non-pre-trained model, pass without doing anything.')
        print(f">>>>>>>>>> Remapping pre-trained keys for VIT ..........")
        checkpoint = SIMMIM_utils.remap_pretrained_keys_vit(self.backbone_model, checkpoint_model)
        self.backbone_model.load_state_dict(checkpoint_model, strict=False)
        self.backbone_model.head = nn.Identity()
        self.normal = nn.LayerNorm(feature_dim)
        self.header = nn.Sequential(nn.Linear(feature_dim, num_class))
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        output = self.backbone_model.forward_features(x)
        clstoken = self.normal(output)
        output = self.header(clstoken)
        output = output.view(B,T,-1)
        return output

'''Simsiam'''

class Simsiam(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(Simsiam,self).__init__()
        self.num_class = num_class
        feature_dim = kwargs['feature_dim']
        self.backbone_model = torchvision.models.__dict__['resnet50']()
        print("=> loading checkpoint '{}'".format(self.pretrain_path))
        checkpoint = torch.load(kwargs['pretrained_path'], map_location="cpu")
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
                state_dict[k[len("module.encoder."):]] = state_dict[k]
            del state_dict[k]
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        print("=> loaded pre-trained model '{}'".format(self.pretrain_path))
        self.backbone_model.fc = nn.Identity()
        self.normal = nn.LayerNorm(feature_dim)
        self.header = nn.Sequential(nn.Linear(feature_dim, num_class))
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        output = self.backbone_model(x)
        output = self.normal(output)
        output = self.header(output)
        output = output.view(B,T,-1)
        return output     

'''SimCLR'''
from mmpretrain import get_model
class SimCLR(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(SimCLR,self).__init__()
        self.backbone_model = get_model(kwargs['pretrained_path'], pretrained=False)
        checkpoint = torch.load(kwargs['pretrained_path'], map_location='cpu')
        self.backbone_model.load_state_dict(checkpoint['state_dict'], strict=False)
        feature_dim = kwargs['feature_dim']
        self.normal = nn.LayerNorm(feature_dim*1)
        self.header = nn.Sequential(nn.Linear(feature_dim*1, num_class))
        for name, param in self.backbone_model.neck.named_parameters():
            param.requires_grad = False
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        output = self.backbone_model(x)[0]
        output = F.avg_pool2d(output, kernel_size=7)
        flattened_features = torch.flatten(output, start_dim=1)
        clstoken = self.normal(flattened_features)
        output = self.header(clstoken)
        output = output.view(B,T,-1)
        return output

'''BYOL'''
class BYOL(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(BYOL,self).__init__()
        feature_dim = kwargs['feature_dim']
        self.backbone_model = torchvision.models.resnet50(pretrained=False)
        checkpoint = torch.load(kwargs['pretrained_path'], map_location='cpu')
        new_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('backbone.'):
                new_key = key[len('backbone.'):]
                new_state_dict[new_key] = value
        self.backbone_model.load_state_dict(new_state_dict, strict=False)
        self.backbone_model.fc=nn.Identity()
        self.normal = nn.LayerNorm(feature_dim*1)
        self.header = nn.Sequential(nn.Linear(feature_dim*1, num_class))
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        output = self.backbone_model(x)
        clstoken = self.normal(output)
        output = self.header(clstoken)
        output = output.view(B,T,-1)
        return output

'''VicReg'''
class Vicreg(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(Vicreg,self).__init__()
        self.num_class = num_class
        feature_dim = kwargs['feature_dim']
        self.backbone_model = torchvision.models.__dict__['resnet50']()
        print("=> loading checkpoint '{}'".format(kwargs['pretrained_path']))
        state_dict = torch.load(kwargs['pretrained_path'], map_location="cpu")
        state_dict = {key.replace("module.backbone.", ""): value for (key, value) in state_dict.items()}
        self.backbone_model.load_state_dict(state_dict, strict=False)
        print("=> loaded pre-trained model '{}'".format(self.pretrain_path))  
        self.backbone_model.fc = nn.Identity()
        self.normal = nn.LayerNorm(feature_dim)
        self.header = nn.Sequential(nn.Linear(feature_dim, num_class))
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        output = self.backbone_model(x)
        output = self.normal(output)
        output = self.header(output)
        output = output.view(B,T,-1)
        return output

'''Moco_v2'''
class MoCo_V2(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(MoCo_V2,self).__init__()
        self.num_class = num_class
        feature_dim = kwargs['feature_dim']
        self.backbone_model = torchvision.models.__dict__['resnet50']()
        print("=> loading checkpoint '{}'".format(kwargs['pretrained_path']))
        checkpoint = torch.load(kwargs['pretrained_path'], map_location="cpu")
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith("module.encoder_q") and not k.startswith(
                "module.encoder_q.fc"):
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            del state_dict[k]
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        print("=> loaded pre-trained model '{}'".format(self.pretrain_path))  
        self.backbone_model.fc = nn.Identity()
        self.normal = nn.LayerNorm(feature_dim)
        self.header = nn.Sequential(
            nn.Linear(feature_dim, num_class))
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        output = self.backbone_model(x)
        '''header'''
        output = self.normal(output)
        output = self.header(output)
        output = output.view(B,T,-1)
        return output

import models.lib.MoCoV3.builder as builder
import models.lib.MoCoV3.vits as vits
from functools import partial

class MoCo_V3(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(MoCo_V3,self).__init__()
        self.pretrianed_path = kwargs['pretrained_path']
        self.backbone_model =  builder.MoCo_ViT(partial(vits.__dict__['vit_base'], stop_grad_conv1=True),256, 4096, 1.0).base_encoder
        print('load_checkpoint:', kwargs['pretrained_path'])
        checkpoint = torch.load(kwargs['pretrained_path'])
        new_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('module.base_encoder.'):
                new_key = key[len('module.base_encoder.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        self.backbone_model.load_state_dict(new_state_dict, strict=False)
        self.backbone_model.head_drop == nn.Identity()
        self.backbone_model.head = nn.Identity()
        dim = self.backbone_model.num_features
        self.normal = nn.LayerNorm(dim*1)
        self.header = nn.Sequential(nn.Linear(dim*1, num_class))
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        output = self.backbone_model.forward_features(x)
        clstoken = self.normal(output[:, 0])
        output = self.header(clstoken)
        output = output.view(B,T,-1)
        return output

import models.lib.obow as Obow
class ObOW(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(ObOW,self).__init__()
        feature_dim = kwargs['feature_dim']
        self.pretrained = kwargs['pretrained']
        args = Obow.config.get_arguments()
        self.backbone_model, channels = Obow.feature_extractor.FeatureExtractor(
            'resnet50', opts=args.exp_config['model']['feature_extractor_opts'])
        print(f"Loading pre-trained feature extractor from: {kwargs['pretrained_path']}")
        out_msg = Obow.utils.load_network_params(
            self.backbone_model, kwargs['pretrained_path'], strict=False)
        print(f"Loading output msg: {out_msg}")
        self.backbone_model.head = nn.Identity()
        self.normal = nn.LayerNorm(feature_dim)
        self.header = nn.Sequential(nn.Linear(feature_dim, num_class))
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        output = self.backbone_model(x)
        output = self.normal(output.squeeze())
        output = self.header(output)
        output = output.view(B,T,-1)
        return output

class ConMIM(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(ConMIM,self).__init__()
        self.num_class = num_class
        feature_dim = kwargs['feature_dim']
        self.backbone_model = timm.create_model('beit_base_patch16_224',pretrained=False,num_classes=num_class,
        drop_rate=0.0,drop_path_rate=0.1,attn_drop_rate=0.0,drop_block_rate=None,use_rel_pos_bias=True,use_abs_pos_emb=False,init_values=0.1,)
        '''load pretrained pth'''
        checkpoint = torch.load(kwargs['pretrained_path'], map_location='cpu')
        msg = self.backbone_model.load_state_dict(checkpoint['module'], strict=False)
        print(msg)
        '''define header'''
        self.backbone_model.head = nn.Identity()
        self.backbone_model.fc_norm = nn.Identity()
        self.normal = nn.LayerNorm(feature_dim)
        self.header = nn.Sequential(nn.Linear(feature_dim, num_class))
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        output = self.backbone_model.forward_features(x)
        clstoken = self.normal(output[:, 0])
        output = self.header(clstoken)
        output = output.view(B,T,-1)
        return output

# from models.lib.BEiT_v3 import modeling_finetune
from models.lib.BEiT_v3 import utils

class BEiT_v3(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(BEiT_v3,self).__init__()
        self.pretrain_path = 'SSL_weight/beit3_base_patch16_224.pth'
        feature_dim = kwargs['feature_dim']
        self.backbone_model = modeling_finetune.beit3_base_patch16_224_imageclassification()
        print("=> loading checkpoint '{}'".format(kwargs['pretrained_path']))
        model_key = 'model|module'
        model_prefix = ''
        utils.load_model_and_may_interpolate(kwargs['pretrained_path'], self.backbone_model, model_key, model_prefix) 
        self.remove_module_B(self.backbone_model)
        self.backbone_model.head = nn.Identity()
        self.normal = nn.LayerNorm(feature_dim)
        self.header = nn.Sequential(nn.Linear(feature_dim, num_class))
        self.backbone_model.beit3.text_embed.weight.requires_grad = False
        self.backbone_model.beit3.vision_embed.requires_grad = False
        self.backbone_model.beit3.vision_embed.mask_token.requires_grad = False
        
    def remove_module_B(self, model):
        for name, child in model.named_children():
            if 'B' in name:
                setattr(model, name, None)
            else:
                self.remove_module_B(child)
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        output = self.backbone_model(x)
        clstoken = self.normal(output)
        output = self.header(clstoken)
        output = output.view(B,T,-1)
        return output
     
import models.lib.DINO.vision_transformer as vits
class DINO(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(DINO,self).__init__()
        feature_dim = kwargs['feature_dim']
        self.backbone = vits.__dict__['vit_base'](patch_size=16, num_classes=2)
        checkpoint = torch.load(kwargs['pretrained_path'],map_location='cpu')
        self.backbone.load_state_dict(checkpoint, strict=False)
        self.norm_for_cls = nn.LayerNorm(feature_dim*1)
        self.header = nn.Sequential(nn.Linear(feature_dim*1, num_class))
        self.backbone.head = nn.Identity()
        self.backbone.norm = nn.Identity()
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        x = self.backbone(x)
        pred = self.header(self.norm_for_cls(x))
        output  = pred.view(B, T, -1)
        return output

BEiT_Config = {
    'drop_path_rate': 0.1,
    'use_mean_pooling': False,
    'init_values': 0.1,
    'qkv_bias': True,
    'use_abs_pos_emb': False,
    'use_rel_pos_bias': True,
    'use_shared_rel_pos_bias': False
}