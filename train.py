import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from data import SegmentationDataset
from module import SegmenterModule

if __name__ == "__main__":
    module = SegmenterModule()
    # !gdown --id 1d3wU8KNjPL4EqMCIEO_rO-O3-REpG82T
    # module.load_pretrained_weights("mit_b3.pth")
    train_dataset = SegmentationDataset(root='/Users/george/Library/Mobile Documents/com~apple~CloudDocs/Projects/0529_dog_segmentation/ClickSEG/')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    print(len(train_loader))
    for batch in train_loader:
        print(batch['image'].shape, batch['image'].dtype)
        print(batch['mask'].shape)
        x = module.forward(batch['image'])