## Intro
This directory is for feature extraction using Bayessian/Adam optimizer.

## Pre-knowledge: structure of LeNet5 with batch normalization
We extract features from LeNet5. Below is the network structure
```
class LeNet5BatchNorm(nn.Module):
    def __init__(self, num_classes=10, affine=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6, affine=affine)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16, affine=affine)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120, affine=affine)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84, affine=affine)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        feature1 = out.view(out.size(0), -1)
        feature2 = F.relu(self.bn3(self.fc1(feature1)))
        feature3 = F.relu(self.bn4(self.fc2(feature2)))
        out = self.fc3(feature3)
        return out,feature1,feature2,feature3
```

## Intro to folders
ckpts: store checkpoints with each optimizer-driven method (only select the one of last epoch)

configs: store configure files of each optimizer-driven method

models: store models (LeNet5, ResNet...)

## How to use this directory
**Choices of parameters:**
- optimizer: ['Bayessian','Adam'], default == Bayessian
- dataset:['DATASET_CIFAR10','DATASET_SVHN','DATASET_MNIST','DATASET_Places365','DATASET_Texture'], default == 'DATASET_CIFAR10' 
- data_augmentation:[True, False], default == False
- download: [True,False], default == True
**Explanation:**
download: need to download dataset or not
**Code:**
```
features = feature_extraction(optimizer='Bayessian',dataset='DATASET_CIFAR10',data_augmentation = False,download = True)
```

## Anything else
1. data is not uploaded due to its large size
2. If you have any problem, please contact Yike Guo
