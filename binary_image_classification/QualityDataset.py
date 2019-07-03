from torch.utils.data import Dataset
import os
import numpy as np
import torchvision.transforms as transforms
import cv2

class QualityDataset(Dataset):
    
    def __init__(self, dataset_dir, width, height):
        self.width = width
        self.height = height
        self.img_dir = []
        self.img_name = []
        self.label = []
        for root, dirs, files in os.walk(dataset_dir):
            for img_name in files:
                self.img_dir.append(os.path.join(root, img_name))
                self.img_name.append(img_name)
                self.label.append(int(os.path.basename(os.path.normpath(root))))
        print("data len: ", len(self.img_dir))
        self.mean_channels = self.get_mean_std("mean")
        self.std_channels = self.get_mean_std("std")
        
        
    
    def __getitem__(self, index):
        img = self.get_img(self.img_dir[index])
        trans = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[self.mean_channels[0], self.mean_channels[1], self.mean_channels[2]],
                                                             std=[self.std_channels[0], self.std_channels[1], self.std_channels[2]])])
        img = trans(img)
        return (img, self.label[index], self.img_name[index])
    
    
    def get_img(self, img_dir):
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height))
        return img


    def __len__(self):
        return len(self.img_dir) 

    
    def get_mean_std(self, select="mean"):
        r_ = 0
        g_ = 0
        b_ = 0
        for img in self.img_dir:
            img = self.get_img(img)
            if select == "mean":
                r, g, b = np.mean(img, axis=(0, 1)) / 255.0
            if select == "std":
                r, g, b = np.mean(img, axis=(0, 1)) / 255.0
            r_ += r
            g_ += g
            b_ += b
        return [r_/len(self.img_dir), 
                g_/len(self.img_dir), 
                b_/len(self.img_dir)]
            
            
        
    """
    def get_mean(img):
        return np.mean(img, axis=(0, 1)) / 255.0

    def get_std(img):
        return np.std(img, axis=(0, 1)) / 255.0
    """
    
    
    
    