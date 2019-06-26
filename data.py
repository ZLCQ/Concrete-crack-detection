from torch.utils.data import Dataset
import torch
import os
import cv2
from PIL import Image
class CrackDataSet(Dataset):
    def __init__(self,img_dir,imgs,label_dir):
        self.img_dir=img_dir
        self.label_dir=label_dir
        self.imgs=imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img=self.imgs[item]
        img_path=os.path.join(self.img_dir,img)
        label_path=os.path.join(self.label_dir,img)
        im=cv2.imread(img_path)
        im = cv2.resize(im, (224, 224))
        im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im=torch.from_numpy(im).permute(2,0,1)
        
        label=cv2.imread(label_path,)
        label = cv2.resize(label, (224, 224))[:,:,0]
        label[label<10]=0
        label[label>0]=1
        label=torch.from_numpy(label).long()
        return im.float()/255.,label