from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class convOut_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super(convOut_Dataset, self).__init__()
        meta = torch.load(data_path)
        self.target = meta['target']
        self.pred1 = meta['pred1']
        self.pred2 = meta['pred2']
        self.convOut1 = meta['convOut1']
        self.convOut2 = meta['convOut2']
        del meta

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        convOut1 = self.convOut1[idx]
        convOut2 = self.convOut2[idx]

        pack = {'convOut1': convOut1, 'convOut2': convOut2, 'idx': idx}

        return pack


def test():
    train_dataset = convOut_Dataset('/data/HaoChen/knowledge_distillation/FeatureFactorization/convOut_CUB200_vgg16_bn_L37_.pkl')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    for step, batch in enumerate(train_dataloader):
        for k in batch:
            batch[k] = batch[k].cuda(1)
        print(batch['convOut1'].shape)
        print(batch['convOut2'].shape)

# test()
