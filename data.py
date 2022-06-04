import cv2
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(Dataset):
    def __init__(self, root, kind='train'):
        self.root = root
        if kind == "train":
            self.transform = A.Compose([
                A.RandomCrop(width=224, height=224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1),
                ToTensorV2(),
            ])
            self.img_list = [x for x in Path(self.root).glob('**/*.jpg') if x.with_suffix(".png").exists()]
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1),
                ToTensorV2(),
            ])
            self.img_list = list(Path(self.root).glob('**/*.jpg'))
        self.kind = kind

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.img_list[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.kind == "train":
            label = cv2.imread(str(self.img_list[idx].with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
            transformed = self.transform(image=img, mask=label)
            transformed['mask'] = (transformed['mask'] / 255).float()
        else:
            transformed = self.transform(image=img)
        return transformed
