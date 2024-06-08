from dataset import LIDCDataset, XrayLIDCDataset, DEFAULTDataset, XrayCTPADataset, CTPADataset, RSPECTDataset
from torch.utils.data import WeightedRandomSampler
from params import TRAIN_LABELS, VALID_LABELS, RSPECT_TRAIN_LABELS, RSPECT_VALID_LABELS, LIDC_TRAIN_LABELS, LIDC_TEST_LABELS

def get_dataset(cfg):

    if cfg.dataset.name == 'LIDC':
        train_dataset = LIDCDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        val_dataset = LIDCDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'XRAY_LIDC':
        train_dataset = XrayLIDCDataset(root_dir=cfg.dataset.root_dir, target=LIDC_TRAIN_LABELS, mode="train", augmentation=True)
        val_dataset = XrayLIDCDataset(root_dir=cfg.dataset.root_dir, target=LIDC_TEST_LABELS, mode="test", augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'XRAY_CTPA':
        train_dataset = XrayCTPADataset(root=cfg.dataset.root_dir, target=TRAIN_LABELS, mode="train", augmentation=True)
        val_dataset = XrayCTPADataset(root=cfg.dataset.root_dir, target=VALID_LABELS, mode="test", augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'CTPA':
        train_dataset = CTPADataset(root=cfg.dataset.root_dir, target=TRAIN_LABELS, mode="train", augmentation=True)
        val_dataset = CTPADataset(root=cfg.dataset.root_dir, target=VALID_LABELS, mode="test", augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'RSPECT':
        train_dataset = RSPECTDataset(root_dir=cfg.dataset.root_dir, target=RSPECT_TRAIN_LABELS, mode="train", augmentation=True)
        val_dataset = RSPECTDataset(root_dir=cfg.dataset.root_dir, target=RSPECT_VALID_LABELS, mode="test", augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'DEFAULT':
        train_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir)
        val_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None
    raise ValueError(f'{cfg.dataset.name} Dataset is not available')
