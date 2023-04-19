from typing import Any
import argparse

import faiss
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F


class BasePostprocessor:
    def __init__(self, args):
        self.args = args

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        for batch in tqdm(data_loader, disable=not progress):

            data = batch[0].cuda()
            label = batch[1].cuda()
            pred, conf = self.postprocess(net, data)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list


normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


class KNNPostprocessor(BasePostprocessor):
    def __init__(self, args):
        super(KNNPostprocessor, self).__init__(args)
        self.K = args.k
        self.activation_log = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        activation_log = []
        net.eval()
        with torch.no_grad():
            for batch in tqdm(id_loader_dict['train'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch[0].cuda()
                data = data.float()

                batch_size = data.shape[0]

                # _, features = net(data, return_feature_list=True)
                score, features = net.feature_list(data)
                feature = features[-1]
                dim = feature.shape[1]
                # print(dim)
                activation_log.append(
                    normalizer(feature.data.cpu().numpy().reshape(
                        batch_size, dim, -1).mean(2)))

        self.activation_log = np.concatenate(activation_log, axis=0)
        self.index = faiss.IndexFlatL2(feature.shape[1])
        self.index.add(self.activation_log)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # output, feature = net(data, return_feature=True)
        output, features =  net.feature_list(data)
        # out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature], dim=1)
        feature = features[-1]
        # print(feature.shape)
        feature_normed = normalizer(feature.data.cpu().numpy().reshape(
            feature.shape[0], feature.shape[1], -1).mean(2))
        # print(feature_normed[1].shape)
        # feature_normed = normalizer([feature[i].cpu().numpy() for i in range(len(feature))])
        # feature_normed = normalizer(feature.data.cpu().numpy())
        D, _ = self.index.search(
            feature_normed,
            self.K,
        )
        kth_dist = -D[:, -1]
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return pred, torch.from_numpy(kth_dist)

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=5,
                        help='Number of nearest neighbors to use for KNN postprocessing')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # instantiate
    nn_postprocessor = KNNPostprocessor(args)