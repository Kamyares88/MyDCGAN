import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import glob


class MyDCGANDataset(Dataset):
    def __init__(self, config, transform=None):
        #super(MyDCGANDataset, self).__init__()
        # making path_df
        img_list = glob.glob(f"{config.dataroot}/*/*.jpg")
        self.path_df = pd.DataFrame(
            {'path':img_list}
        )
        self.transform = transform

    def __len__(self):
        return self.path_df.shape[0]

    def __getitem__(self, idx):
        # image_path
        img_path = self.path_df.loc[idx,'path']

        # opening the image
        #img = np.asarray(Image.open(img_path))
        img = Image.open(img_path)

        # augmenting the image
        if self.transform:
            img = self.transform(img)

        return img, 1

