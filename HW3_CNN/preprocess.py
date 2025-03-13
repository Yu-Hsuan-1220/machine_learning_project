import torchvision.transforms as transforms 
from torch.utils.data import Dataset
import os
from PIL import Image


test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(128, ),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=45, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

])


class FoodDataset(Dataset):
    def __init__(self,path,tfm=test_tfm,files = None):
        super(FoodDataset, self).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx]
        fname = self.files[idx].replace("\\", "/")
        
        im = Image.open(fname) 
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
        #im = Image.fromarray(im)  
        im = self.transform(im)

        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1

        return im,label