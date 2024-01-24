import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
from natsort import natsorted
import random

random.seed(10)

TRANSFORMS = {
    "imagenet": {
        "transform": T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Resize((c.cropsize, c.cropsize), antialias=None),
            T.ToTensor()]),
        "tranform_val": T.Compose([
            T.Resize((c.cropsize, c.cropsize), antialias=None),
            T.ToTensor()])
    },
    "div2k" : {
        "transform_imagenet": T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomCrop(c.cropsize),
            T.ToTensor()]),
        "transform_val": T.Compose([
            T.CenterCrop(c.cropsize_val),
            T.ToTensor()]),
    }
}


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageNetDataset(Dataset):
    def __init__(self, 
                transforms_=None, 
                mode = "train"
            ):
        self.transforms = transforms_
        #
        # Init dataset, triggers, and responses
        #
        print(c.TRAIN_PATH)
        if mode == 'train':
            # train
            self.dataset = natsorted(sorted(glob.glob(c.TRAIN_PATH + "/*." + c.format_train)))
            if c.num_train is not None:
                num_images = min(c.num_train, len(self.dataset))
                random.shuffle(self.dataset)
                self.dataset = self.dataset[:num_images]

        else:
            # test
            self.dataset = sorted(glob.glob(c.VAL_PATH + "/*." + c.format_val))
            if c.num_val is not None:
                num_images = min(c.num_val, len(self.dataset))
                random.shuffle(self.dataset)
                self.dataset = self.dataset[:num_images]

    def __len__(self):
        """
        Get length of dataset
        """
        return len(self.dataset)


    def __getitem__(self, idx):
        """
        Get image, trigger, and trigger response
        """
        image_name = self.dataset[idx]
        image = Image.open(image_name)

        if self.transforms:
            image = self.transforms(image)
        
        return image



class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            # train
            self.files = natsorted(sorted(glob.glob(c.TRAIN_PATH + "/*." + c.format_train)))
        else:
            # test
            self.files = sorted(glob.glob(c.VAL_PATH + "/*." + c.format_val))



    def __getitem__(self, index):
        try:
            img = Image.open(self.files[index])
            img = to_rgb(img)
            item = self.transform(img)
            return item

        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        if self.mode == 'shuffle':
            return max(len(self.files_cover), len(self.files_secret))

        else:
            return len(self.files)


if c.dataset == "imagenet":
    trainloader = DataLoader(
        ImageNetDataset(transforms_=TRANSFORMS["imagenet"]["transform"], mode="train"),
        batch_size=c.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True
    )

    testloader = DataLoader(
        Hinet_Dataset(transforms_=TRANSFORMS["imagenet"]["transform_val"], mode="val"),
        batch_size=c.batchsize_val,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        drop_last=True
    )

elif c.dataset == "div2k":
    trainloader = DataLoader(
        Hinet_Dataset(transforms_=TRANSFORMS["div2k"]["transform_val"], mode="train"),
        batch_size=c.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True
    )
    # Test data loader
    testloader = DataLoader(
        Hinet_Dataset(transforms_=TRANSFORMS["div2k"]["transform_val"], mode="val"),
        batch_size=c.batchsize_val,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        drop_last=True

    )
