import models.custom_ssl as custom_ssl
import torch
from PIL import Image
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import csv
import glob
from tqdm import tqdm
import os
from face_process.real_face_process_Frame import process

def get_model():
    model = custom_ssl.BEiT_v2(pretrained=False)
    ckpt_load_path = 'checkpoints/Final_DDBF_BEiT_v2/ckpt/Final-mainbranch.tar'
    checkpoint = torch.load(ckpt_load_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']
    else:
        sd = checkpoint
    new_state_dict = {}    
    for k, v in sd.items():
        if k.startswith('module.'):
            k = k.replace('module.', '')
            new_state_dict[k] = v
    msg = model.load_state_dict(new_state_dict,strict=False)
    print('sdload', msg)

    return model

def Inference_Img(model, path, transfrom):
    img_al = process(path)
    img = Image.open(img_al)
    img = np.asarray(img)
    tmp_imgs = {"image": img}
    input_tensor = transfrom(**tmp_imgs)
    input_tensor = input_tensor['image'].cuda().unsqueeze(0)
    input_tensor = input_tensor.unsqueeze(1)
    output = model(input_tensor).squeeze(1)
    pred = torch.nn.functional.softmax(output, dim=1)[:,1]
    return pred



if __name__ == "__main__":
    model = get_model()
    model = model.cuda()
    model.eval()
    additional_targets = {}
    base_transform = alb.Compose([
        alb.Resize(224, 224),
        alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets=additional_targets)

    csv_file = 'inference_results.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["img_name", "y_pred"])
        while True:
            path = input("Enter the path of the image (e.g., 'test.jpg') or 'exit' to stop: ")
            if path.lower() == 'exit':
                break
            elif os.path.splitext(path)[1].lower() =='.jpg' or os.path.splitext(path)[1].lower() =='.png':
                pred = Inference_Img(model, path, base_transform)
                writer.writerow([path, pred.cpu().detach().numpy()[0]])
                print(path, 'Fake Score:', pred.cpu().detach().numpy()[0])
            else:
                print('Inference all images in:', path)
                paths = glob.glob(path + '/*.jpg', recursive=True)+glob.glob(path + '/*.png', recursive=True)
                for path in tqdm(paths):
                    pred = Inference_Img(model, path, base_transform)
                    writer.writerow([path, pred.cpu().detach().numpy()[0]])
                print('result saved in :', csv_file)
                break