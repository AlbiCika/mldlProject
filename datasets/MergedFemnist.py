import numpy as np
import os
import sys
path = os.getcwd()
if 'kaggle' not in path:
    import datasets.ss_transforms as tr
else:
    sys.path.append('datasets')
    import ss_transforms as tr



#from torchvision import transforms

#from torch import from_numpy
from PIL import Image
IMAGE_SIZE = 28

from torch.utils.data import Dataset

IMAGE_SIZE = 28

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

#convert_tensor = transforms.ToTensor()


class MergedFemnist(Dataset):
    def __init__(self, femnist_objects):
        super().__init__()
        self.merged_samples = []
        # Collect samples from each Femnist object
        for femnist_obj in femnist_objects:
            self.merged_samples.extend(femnist_obj.samples)
        # Assuming that all Femnist objects have the same transform and client_name
        self.transform = femnist_objects[0].transform if femnist_objects else None
        self.client_name = femnist_objects[0].client_name if femnist_objects else None


    def __getitem__(self, index: int):
        sample = self.merged_samples[index]
        image = Image.fromarray(np.uint8(np.array(sample[0]).reshape(28, 28) * 255))
        label = sample[1]
        if self.transform is not None:
            image = self.transform(np.array(image))
        return image, label


    def __len__(self) -> int:
        return len(self.merged_samples)

