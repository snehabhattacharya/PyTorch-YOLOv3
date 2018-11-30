import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

import sys

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        print(folder_path,"foldeer")
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        print (self.files)
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)
    
    #self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):
        print(list_path)
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        full_path = "/home/snehabhattac/PyTorch-YOLOv3/data/test_data/lwir/"
        self.img_files = [full_path+i.strip()+".jpg" for i in self.img_files]
        self.label_files = [path.replace('test_data/lwir/', 'kaist_data/annotations/set05/V000/').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        print(self.label_files)
        #/home/snehabhattac/PyTorch-YOLOv3/data/kaist_data/annotations/set05/V000
        self.img_shape = (img_size, img_size)
        self.max_objects = 10
        self.classes = ('__background__', # always index 0
                         'person')
        self.class_to_ind = dict(zip(self.classes, range(self.max_objects)))


    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        #print(img_path)
        #print(self.img_files)
        img = np.array(Image.open(img_path))
        #/pprint("hereeeeeeeeeeee")
        # Handles images with less than three channels
        while len(img.shape) != 3:
            #print("here222222222222")
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        #print(input_img.shape,"shape")

        #---------
        #  Label
        #---------
        #print("label shit")
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        #print(label_path)
        
        if os.path.exists(label_path):
            with open(label_path) as f:
                lines = f.readlines()
            filled_labels = np.zeros((self.max_objects, 5))
            labels = np.zeros((len(lines)-1, 5))
            if len(lines)==1:
                labels = np.zeros((len(lines), 5))

            #print(lines, "lines")
            ix = 0
            for obj in lines:
                info = obj.split()
                 
                if info[0] == "%":
                    continue
            
                #labels = np.loadtxt(label_path).reshape(-1, 5)
                
                
                #print(labels,"labels")
                # Extract coordinates for unpadded + unscaled image
                x = float(info[1]) 
                y = float(info[2])
                w_ = float(info[3])
                h_ = float(info[4])
                #print (x,y, w_, h_,label_path)
                x = x +  float(w_/2)
                y = y +  float(h_/2) 
                x = x/padded_w
                y = y/padded_h
                w_ = w_/padded_w
                h_ = h_/padded_h
                print(x,y,w_,h_) 
                # Adjust for added padding
                #x1 += pad[1][0]
                #y1 += pad[0][0]
                #x2 += pad[1][0]
                #y2 += pad[0][0]
                # Calculate ratios from coordinates
                #print(x1,x2,y1,y2)
                #print(((1 + x2) / 2) / padded_w, "label 1 ")
                labels[ix,0] = self.class_to_ind['person']
                labels[ix, 1] = x
                labels[ix, 2] = y
                labels[ix, 3] = w_
                labels[ix, 4] = h_
                ix += 1
                #print (x,y, w_, h_,label_path)
                # Fill matrix
                #filled_labels = np.zeros((self.max_objects, 5))
                #if labels is not None:
                    #filled_labels[range(len(labels))[:len(lines)]] = labels[:len(lines)]
                #    filled_labels = labels
                #filled_labels = torch.from_numpy(filled_labels)
        #print(labels, "labels")
        filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)
        #print(filled_labels, "filled_labels")        
        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
