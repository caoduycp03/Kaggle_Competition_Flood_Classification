from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
import torchvision.transforms.functional as F
import os
import pandas as pd
from PIL import Image

class FloodDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.train = train
        if train:
            dir = os.path.join(root, 'devset_images\\devset_images')
            paths = os.listdir(dir)
            paths = sorted(paths, key=lambda x: int(x.split('.')[0])) 
            image_files = [os.path.join(dir, path) for path in paths]
            label_file = 'data\devset_images_gt.csv'
        else:
            dir = os.path.join(root, 'testset_images\\testset_images')
            paths = os.listdir(dir)
            paths = sorted(paths, key=lambda x: int(x.split('.')[0]))
            image_files = [os.path.join(dir, path) for path in paths]
            id_file = r'data\test.csv'
        
        self.images_path = image_files
        if train:
            data = pd.read_csv(label_file)
            data = data.sort_values(by='id', ascending=True)
            self.labels = list(data['label'])
        else:
            data = pd.read_csv(id_file)
            data = data.sort_values(by='image_id', ascending=True)
            self.id = list(data['image_id'])

    def __len__(self):
        return len(self.images_path)
    def __getitem__(self, idx):      
        image_path = self.images_path[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        if self.train:
            label = self.labels[idx]
            return image, label
        else:
            id = self.id[idx]
            return image
        

if __name__ == '__main__':
    dataset = FloodDataset(root='data', train=0)
    image = dataset.__getitem__(193)
    image.show()
