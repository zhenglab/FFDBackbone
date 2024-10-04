import os
import torch
from collections import OrderedDict


def convert_weight(pth_path, checkpoint_save_dir):
    checkpoint = torch.load(pth_path, map_location='cpu')
    sd = checkpoint['state_dict']

    for k in list(sd.keys()):
        if k.startswith('module.backbone_model2'):
            del sd[k]
    checkpoint_m = OrderedDict()
    checkpoint_m['state_dict'] = sd
    torch.save(checkpoint_m, checkpoint_save_dir)

convert_weight('Final-BEiT_v2.tar','Final-BEiT_v2.tar')