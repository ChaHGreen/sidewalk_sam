from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class ObjectDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        prepare dataset to train the autoencoder
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
