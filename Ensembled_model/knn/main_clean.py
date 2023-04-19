import os
import csv
import argparse

from utils.resnet18_32x32 import ResNet18_32x32
from utils.base_evaluator import BaseEvaluator
from utils.metrics import compute_all_metrics
from utils.knn_postprocessor import KNNPostprocessor, BasePostprocessor
from utils.resnet_supcon import resnet18
from utils.resnet_ss import resnet18_cifar
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='name of the dataset')
    parser.add_argument('--mix', type=bool, default=False, help='whether to use mixing dataset')
    parser.add_argument('--filter', type=bool, default=False, help='whether to use filtering dataset')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA visible devices')
    parser.add_argument('--k', type=int, default=50, help='number of nearest neighbors to use for KNN postprocessing')
    return parser.parse_args()



def save_csv(metrics, dataset_name):
    [fpr, auroc, aupr_in, aupr_out,
        ccr_4, ccr_3, ccr_2, ccr_1, accuracy] \
        = metrics
    if args.mix:
        filename = f"{args.dataset}_mix_k{args.k}.csv"
    elif args.filter:
        filename = f"{args.dataset}_filter_k{args.k}.csv"
    else:
        filename = f"{args.dataset}_k{args.k}.csv"
    write_content = {
        'dataset': dataset_name,
        'FPR@95': '{:.2f}'.format(100 * fpr),
        'AUROC': '{:.2f}'.format(100 * auroc),
        'AUPR_IN': '{:.2f}'.format(100 * aupr_in),
        'AUPR_OUT': '{:.2f}'.format(100 * aupr_out),
        'CCR_4': '{:.2f}'.format(100 * ccr_4),
        'CCR_3': '{:.2f}'.format(100 * ccr_3),
        'CCR_2': '{:.2f}'.format(100 * ccr_2),
        'CCR_1': '{:.2f}'.format(100 * ccr_1),
        'ACC': '{:.2f}'.format(100 * accuracy)
    }

    fieldnames = list(write_content.keys())

    # print ood metric results
    print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
            end=' ',
            flush=True)
    print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
        100 * aupr_in, 100 * aupr_out),
            flush=True)
    print('CCR: {:.2f}, {:.2f}, {:.2f}, {:.2f},'.format(
        ccr_4 * 100, ccr_3 * 100, ccr_2 * 100, ccr_1 * 100),
            end=' ',
            flush=True)
    print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
    print(u'\u2500' * 70, flush=True)

    csv_path = os.path.join(filename)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(write_content)
    else:
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(write_content)


class TxtDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.data = []
        with open(txt_file, 'r') as f:
            for line in f:
                img_path, label = line.strip().split()
                img_path = os.path.join('data/', img_path)
                self.data.append((img_path, int(label)))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

args = parse_args()
dataset_name = args.dataset
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

txt_file_train = f'data/{dataset_name}/train_{dataset_name}.txt'
txt_file_test = f'data/{dataset_name}/test_{dataset_name}.txt'
output_file = f'data/{dataset_name}/train_{dataset_name}_mix.txt'
filtered_file = f'data/{dataset_name}/train_{dataset_name}_filtered.txt'

net = resnet18_cifar(num_classes=10)
checkpoint = torch.load("checkpoints/resnet18-supcon/checkpoint_500.pth.tar", map_location='cpu')
checkpoint = {'state_dict': {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}}
net.load_state_dict(checkpoint['state_dict'])
net.cuda()

if args.mix:
    id_train_dataset = TxtDataset(txt_file=output_file, transform=transforms.ToTensor())
elif args.filter:
    id_train_dataset = TxtDataset(txt_file=filtered_file, transform=transforms.ToTensor())
else:
    id_train_dataset = TxtDataset(txt_file=txt_file_train, transform=transforms.ToTensor())
id_test_dataset = TxtDataset(txt_file=txt_file_test, transform=transforms.ToTensor())

args = parse_args()
postprocessor = KNNPostprocessor(args)
id_data_loader = { 'train': DataLoader(id_train_dataset, batch_size=128, shuffle=True, num_workers=4),
                     'test': DataLoader(id_test_dataset, batch_size=128, shuffle=True, num_workers=4)}

ood_dataset_names = ['texture','places365','mnist','svhn']
ood_data_loaders = {}
for ood_dataset_name in ood_dataset_names:
    # ood_dataset = TxtDataset(txt_file=f'./data/{dataset_name}/test_{ood_dataset_name}.txt', transform=transforms.ToTensor())
    ood_dataset = ImageFolder(root=f'./data/{ood_dataset_name}', transform=transforms.Compose([
                                            transforms.Resize((32, 32)),
                                            transforms.ToTensor(),
                                            ]))
    ood_data_loaders[ood_dataset_name] = DataLoader(ood_dataset, batch_size=128, shuffle=True, num_workers=4)
postprocessor.setup(net, id_data_loader, ood_data_loaders)
print(f'Performing inference on {dataset_name} dataset...', flush=True)
id_pred, id_conf, id_gt = postprocessor.inference(
    net, id_data_loader['test'])


for ood_dataset_name, ood_data_loader in ood_data_loaders.items():
    metrics_list = []
    ood_metrics = None
    print(f'Performing inference on {ood_dataset_name} dataset...', flush=True)
    ood_pred, ood_conf, ood_gt = postprocessor.inference(net, ood_data_loader)
    ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
    ood_pred = -1 * np.ones_like(ood_pred)  # hard set to -1 as ood

    pred = np.concatenate([id_pred, ood_pred])
    conf = np.concatenate([id_conf, ood_conf])
    label = np.concatenate([id_gt, ood_gt])

    ood_metrics = compute_all_metrics(conf, label, pred)
    metrics_list.append(ood_metrics)

    metrics_list = np.array(metrics_list)
    metrics_mean = np.mean(metrics_list, axis=0)
    save_csv(metrics_mean, dataset_name=ood_dataset_name)
