import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from mix_transformer import mit_b3
from segformer_head import SegFormerHead


def iou(preds: torch.Tensor, labels: torch.Tensor):
    """
    IoU metric suitable for loss
    """
    preds = preds.view(-1)
    labels = labels.view(-1)
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum()
    return intersection / union


class SegmenterModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.feature_extractor = mit_b3()
        in_channels = [64, 128, 320, 512]
        in_index = [0, 1, 2, 3]
        feature_strides = [4, 8, 16, 32]
        dropout_ratio = 0.1
        num_classes = 1
        decoder_params = dict(embed_dim=512)
        self.decoder_head = SegFormerHead(feature_strides=feature_strides, in_channels=in_channels,
                                          num_classes=num_classes, in_index=in_index, dropout_ratio=dropout_ratio,
                                          decoder_params=decoder_params)

    def load_pretrained_weights(self, path_to_weights=' '):
        backbone_state_dict = self.feature_extractor.state_dict()
        pretrained_state_dict = torch.load(path_to_weights, map_location='cpu')
        ckpt_keys = set(pretrained_state_dict.keys())
        own_keys = set(backbone_state_dict.keys())
        missing_keys = own_keys - ckpt_keys
        unexpected_keys = ckpt_keys - own_keys
        print('Missing Keys: ', missing_keys)
        print('Unexpected Keys: ', unexpected_keys)
        backbone_state_dict.update(pretrained_state_dict)
        self.backbone.load_state_dict(backbone_state_dict, strict=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.decoder_head.parameters(), lr=0.001)
        # diffrent lr for feature extractor
        optimizer_feature_extractor = torch.optim.Adam(self.feature_extractor.parameters(), lr=0.0001)
        return [optimizer, optimizer_feature_extractor]

    def forward(self, x):
        features = self.feature_extractor(x)
        mask, _ = self.decoder_head(features)
        mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return mask

    def compute_iou_loss(self, y, y_hat):
        iou_loss = 1 - iou(y, y_hat)
        return iou_loss

    def compute_loss(self, y, y_hat):
        bce_loss = F.binary_cross_entropy_with_logits(y_hat, y)
        iou_loss = self.compute_iou_loss(y_hat, y)
        return bce_loss * 0.5 + iou_loss * 0.5

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_hat = self(x)
        loss = self.compute_loss(y, y_hat)
        iou_value = iou(y, y_hat).item()
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        self.log('train_iou', iou_value, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_hat = self(x)
        loss = self.compute_loss(y, y_hat)
        iou_value = iou(y, y_hat).item()
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_iou', iou_value, on_step=False, on_epoch=True)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_iou = torch.stack([x['val_iou'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, on_step=False, on_epoch=True)
        self.log('val_iou', avg_iou, on_step=False, on_epoch=True)
        return {'val_loss': avg_loss, 'val_iou': avg_iou}
