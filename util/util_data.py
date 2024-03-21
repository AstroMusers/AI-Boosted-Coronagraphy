import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from itertools import product
from PIL import Image
import numpy as np
import glob
import cv2


class SynDataset(Dataset):

    def __init__(self, image_paths):

        self.image_paths = image_paths
        self.transform  = transforms.Compose([
        transforms.ToTensor(),
        ])

    def __len__(self,):

        return len(self.image_paths)
    
    def __getitem__(self, index):
        
        image = self.image_paths[index]
        image = np.load(image)
        image = self.transform(image)

        return image



class SynDatasetLabel(Dataset):

    def __init__(self, image_paths, args):

        self.image_paths = image_paths
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        ])

        self.args = args

    def __len__(self):

        return len(self.image_paths)

    def low_pass_filter(self, array):

        r = 50
        ham = np.hamming(80)[:,None] 
        ham2d = np.sqrt(np.dot(ham, ham.T)) ** r 

        f = cv2.dft(array.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        f_shifted = np.fft.fftshift(f)
        f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
        f_filtered = ham2d * f_complex
        f_filtered_shifted = np.fft.fftshift(f_filtered)
        inv_img = np.fft.ifft2(f_filtered_shifted) 
        filtered_img = np.abs(inv_img)
        filtered_img -= filtered_img.min()
        filtered_img = filtered_img*255 / filtered_img.max()
        filtered_img = filtered_img.astype(np.uint8)
        output = filtered_img

        return output

    def apply_low_pass(self, array):
        
        if len(array.shape) == 2:
            output = self.low_pass_filter(array)

        elif len(array.shape) == 3:

            for i in range(array.shape[0]):
                filtered_img = self.low_pass_filter(array[i])
                output.append(filtered_img)
            
            output = np.concatenate(np.expand_dims(output, axis=0))

        return output


    def __getitem__(self, index):
        
        image_path = self.image_paths[index]
        
        if 'fc' in image_path.split('/')[-1]:
            label = torch.squeeze(torch.Tensor([1]))
        else:
            label = torch.squeeze(torch.Tensor([0]))

        image = np.load(image_path).astype(np.float32)

        if self.args.apply_lowpass:
            filtered_img = self.apply_low_pass(image)

        image = self.transform(image)

        batch = (image, label, self.transform(filtered_img) if self.args.apply_lowpass else 0, image_path)

        return batch


