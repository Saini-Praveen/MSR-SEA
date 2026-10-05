import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UnderwaterDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        """
        Args:
            input_dir (string): Directory with enhanced images from the algorithm.
            target_dir (string): Directory with ground truth clean images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.input_images = sorted(os.listdir(input_dir))
        self.target_images = sorted(os.listdir(target_dir))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = Image.open(os.path.join(self.input_dir, self.input_images[idx]))
        target_image = Image.open(os.path.join(self.target_dir, self.target_images[idx]))

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image
