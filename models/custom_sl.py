import torch
import timm
import torch.nn as nn
import torchvision

'''ResNet'''
class ResNet(nn.Module):  # Supervised
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(ResNet,self).__init__()
        feature_dim = kwargs['feature_dim']
        self.backbone_model = timm.create_model(kwargs['network'], pretrained=False)
        checkpoint = torch.load(kwargs['pretrained_path'], map_location="cpu")
        self.backbone_model.load_state_dict(checkpoint, strict=True)
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

'''EfficientNet'''
class EfficientNet(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(EfficientNet,self).__init__()
        feature_dim = kwargs['feature_dim']
        self.backbone_model = timm.create_model(kwargs['network'], pretrained=False)
        checkpoint = torch.load(kwargs['pretrained_path'], map_location="cpu")
        self.backbone_model.load_state_dict(checkpoint, strict=True)
        self.backbone_model.classifier = nn.Identity()
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

'''
b0  efficientnet_b0_ra-3dd342df.pth      1280
b3  efficientnet_b3_ra2-cf984f9c.pth     1536
b4  efficientnet_b4_ra2_320-7eb33cd5.pth 1792
'''

'''Xception'''
class Xception(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(Xception,self).__init__()
        feature_dim = kwargs['feature_dim']
        self.backbone_model = timm.create_model(kwargs['network'], pretrained=False)
        checkpoint = torch.load(kwargs['pretrained_path'], map_location="cpu")
        self.backbone_model.load_state_dict(checkpoint, strict=True)
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

'''ViT'''
class ViT(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(ViT,self).__init__()
        self.backbone_model = timm.create_model(kwargs['network'], pretrained=False)
        timm.models.vision_transformer._load_weights(self.backbone_model, checkpoint_path=kwargs['pretrained_path'])
        self.backbone_model.head_drop == nn.Identity()
        self.backbone_model.head = nn.Identity()
        feature_dim = self.backbone_model.num_features
        self.normal = nn.LayerNorm(feature_dim*1)
        self.header = nn.Sequential(nn.Linear(feature_dim*1, num_class))
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        output = self.backbone_model.forward_features(x)
        clstoken = self.normal(output[:, 0])
        output = self.header(clstoken)
        output = output.view(B,T,-1)
        return output




















class ViT_Model_clip(nn.Module):
    def __init__(self, 
        num_class=2,
        num_segment=8,
        add_softmax=False,
        **kwargs):
        super(ViT_Model_clip,self).__init__()
        self.num_class = num_class
        self.num_segment = num_segment
        self.embed_dim = 768
        self.backbone_model = timm.create_model('vit_base_patch16_clip_224.laion2b_ft_in1k', pretrained=True)
        self.backbone_model.head_drop == nn.Identity()
        self.backbone_model.head = nn.Identity()
        dim = self.backbone_model.num_features
        self.normal = nn.LayerNorm(dim*1)
        self.header = nn.Sequential(nn.Linear(dim*1, num_class))
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        output = self.backbone_model.forward_features(x)  # x shape[32, 3, 224, 224] # output.shape vit [32, 197, 768] 768
        clstoken = self.normal(output[:, 0])
        output = self.header(clstoken)
        output = output.view(B,T,-1)
        return output
    



# class SimCLR(nn.Module):
#     def __init__(self, 
#         num_class=2,
#         num_segment=8,
#         add_softmax=False,
#         **kwargs):
#         super(SimCLR,self).__init__()
#         self.num_class = num_class
#         self.num_segment = num_segment
#         self.backbone_model = get_model('simclr_resnet50_16xb256-coslr-200e_in1k', pretrained=False)
#         pretrain_path = 'ckpts/simclr_resnet50_16xb256-coslr-800e_in1k_20220825-85fcc4de.pth'
#         checkpoint = torch.load(pretrain_path, map_location='cpu')
#         checkpoint_model = checkpoint['state_dict'] # module
#         msg = self.backbone_model.load_state_dict(checkpoint_model, strict=False)
#         print(msg)
#         # self.backbone_model.head = nn.Identity()
#         dim = 2048
#         self.normal = nn.LayerNorm(dim*1)
#         self.header = nn.Sequential(nn.Linear(dim*1, num_class))
#         for name, param in self.backbone_model.neck.named_parameters():
#             param.requires_grad = False
#     def forward(self, x):
#         B, T, _, _, _ = x.size()
#         x = x.flatten(0,1)
#         output = self.backbone_model(x)[0]
#         output = F.avg_pool2d(output, kernel_size=7)
#         flattened_features = torch.flatten(output, start_dim=1)
#         clstoken = self.normal(flattened_features)
#         output = self.header(clstoken)
#         output = output.view(B,T,-1)
#         return output


class BYOL(nn.Module):
    def __init__(self, 
        num_class=2,
        num_segment=8,
        add_softmax=False,
        **kwargs):
        super(BYOL,self).__init__()
        self.num_class = num_class
        self.num_segment = num_segment
        self.backbone_model = torchvision.models.resnet50(pretrained=False)
        pretrain_path = 'ckpts/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth'
        # checkpoint = torch.load(pretrain_path, map_location='cpu')

        # new_state_dict = {}
        # for key, value in checkpoint['state_dict'].items():
        #     if key.startswith('backbone.'):
        #         new_key = key[len('backbone.'):]
        #         new_state_dict[new_key] = value
        # msg = self.backbone_model.load_state_dict(new_state_dict, strict=False)
        # print(msg)
        dim = 2048
        self.backbone_model.fc=nn.Identity()
        self.normal = nn.LayerNorm(dim*1)
        self.header = nn.Sequential(nn.Linear(dim*1, num_class))
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        output = self.backbone_model(x)
        # output = F.avg_pool2d(output, kernel_size=7)
        # flattened_features = torch.flatten(output, start_dim=1)
        clstoken = self.normal(output)
        output = self.header(clstoken)
        output = output.view(B,T,-1)
        return output





class ViT_Model_tiny(nn.Module):
    def __init__(self, 
        num_class=2,
        num_segment=8,
        add_softmax=False,
        **kwargs):
        super(ViT_Model_tiny,self).__init__()
        self.num_class = num_class
        self.num_segment = num_segment
        self.embed_dim = 768
        self.backbone_model = timm.create_model('vit_base_patch16_clip_224.laion2b_ft_in1k', pretrained=True)
        self.backbone_model.head_drop == nn.Identity()
        self.backbone_model.head = nn.Identity()
        dim = self.backbone_model.num_features
        self.normal = nn.LayerNorm(dim*1)
        self.header = nn.Sequential(nn.Linear(dim*1, num_class))
    def forward(self, x):
        B, T, _, _, _ = x.size()
        x = x.flatten(0,1)
        output = self.backbone_model.forward_features(x)  # x shape[32, 3, 224, 224] # output.shape vit [32, 197, 768] 768
        clstoken = self.normal(output[:, 0])
        output = self.header(clstoken)
        output = output.view(B,T,-1)
        return output