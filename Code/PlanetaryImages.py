# Custom Image Dataset & Pipeline

import os
import PIL
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

torch.manual_seed(42) #set global seed


# PlanetaryDataloader
class PlanetaryImages(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        
        ####---- NEED MANUALLY MAKE GREYSCALE IMAGES RGB ----####

        # Basic Read in of Image
        #image = read_image(img_path)
        image = PIL.Image.open(img_path).convert("RGB") #alt load in mechanism

        # MANUAL ATTEMPT TO FIX GRAYSCALE IMAGES (trying logic step in transforms)
        # try:
        #     # Apply Transformation -- If throw error, is GreyScale
        #     self.transform(image)
        # except:
        #     # Method 1 for Converting to RGB
        #     # img = PIL.Image.open(img_path)
        #     # rgb_img = PIL.Image.new("RGBA", img.size)
        #     # image = rgb_img.paste(img)

        #     # ALT APPROACH FOR CONVERTING G-Scale to RGB: https://stackoverflow.com/questions/18522295/python-pil-change-greyscale-tif-to-rgb
        #     img = PIL.Image.open(img_path)
        #     img.point(lambda p: p*0.0039063096, mode='RGB')
        #     image = img.convert('RGB')

        #image = read_image(img_path, mode=ImageReadMode.RGB) #can force to RGB instead
        #image = read_image(img_path, mode=ImageReadMode.GRAY) #had to convert all images to GrayScale

        #image = PIL.Image.open(img_path) #experimental read in, handles greyscales in `transform` step (need uncomment extra step in transforms to make live)


        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# Deal w GreyScale Images
class NoneTransform(object):
    ''' Does nothing to the image. To be used instead of None '''
    
    def __call__(self, image):       
        return image




# Pipeline
input_size = 224 #GhostNet Required Size

transform = transforms.Compose([
        #transforms.ToPILImage(), #sneaky bastard -- Converts to Image Data, not tensor
        transforms.ToTensor(),
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(), #used these two originally, images are slightly smaller so switched to center crop
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x), #Convert G-Scale to RGB- https://discuss.pytorch.org/t/convert-grayscale-images-to-rgb/113422
        #transforms.Lambda(lambda x: x.repeat(3, 1, 1))  if image.mode!='RGB'  else NoneTransform(), #experimenting alt greyscale img handler
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #hard-coded normalization values 
    ])

# Call Custom Dataset & Apply Transforms
planetary_data = PlanetaryImages("../PlanetaryImages.csv", "../PlanetaryImages/", transform=transform)

# Get Train & Test Sets
train_set, test_set = random_split(planetary_data, [6500, 872])


# Define DataLoaders & Get Dict for em
trainloader = DataLoader(train_set, batch_size=64, shuffle=True)
testloader = DataLoader(test_set, batch_size=64, shuffle=True)

dataloaders = {'train':trainloader, 'test':testloader}

