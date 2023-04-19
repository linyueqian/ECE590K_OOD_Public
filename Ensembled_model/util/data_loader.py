import os
import torch
import torchvision
from torchvision import transforms
from easydict import EasyDict

imagesize = 32

transform_test = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_MNIST = torchvision.transforms.Compose([
    transforms.Resize(32),
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.ToTensor(),
])

transform_train = transforms.Compose([
    transforms.RandomCrop(imagesize, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

kwargs = {'num_workers': 2, 'pin_memory': True}

def get_loader_in(args, config_type='default', split=('train', 'val')):
    config = EasyDict({
        "default": {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'batch_size': args.batch_size,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
        },
    })[config_type]

    train_loader, val_loader, lr_schedule, num_classes = None, None, [50, 75, 90], 0
    if args.in_dataset == "CIFAR-10":
        # Data loading code
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=config.transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 10
    elif args.in_dataset == "CIFAR-100":
        # Data loading code
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=config.transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=config.transform_test)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 100
    # elif args.in_dataset == "imagenet":
    #     root = 'datasets/id_data/imagenet'
    #     # Data loading code
    #     if 'train' in split:
    #         train_loader = torch.utils.data.DataLoader(
    #             torchvision.datasets.ImageFolder(os.path.join(root, 'train'), config.transform_train_largescale),
    #             batch_size=config.batch_size, shuffle=True, **kwargs)
    #     if 'val' in split:
    #         val_loader = torch.utils.data.DataLoader(
    #             torchvision.datasets.ImageFolder(os.path.join(root, 'val'), config.transform_test_largescale),
    #             batch_size=config.batch_size, shuffle=True, **kwargs)
    #     num_classes = 1000

    return EasyDict({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "lr_schedule": lr_schedule,
        "num_classes": num_classes,
    })

def get_loader_out(args, dataset=(''), config_type='default', split=('train', 'val')):

    config = EasyDict({
        "default": {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
            'batch_size': args.batch_size
        },
    })[config_type]
    train_ood_loader, val_ood_loader = None, None

    if 'train' in split:
        if dataset[0] == 'CIFAR-10':
            train_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root='./data/cifar10/train', train=True, download=True, transform=transform_train),
                                                       batch_size=batch_size, shuffle=True, num_workers=2)
        elif dataset[0] == 'CIFAR-100':
            train_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train),
                                                       batch_size=batch_size, shuffle=True, num_workers=2)

    if 'val' in split:
        val_dataset = dataset[1]
        batch_size = args.batch_size
        if val_dataset == 'SVHN':
            # from util.svhn_loader import SVHN
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.SVHN(root='./data', download=True, transform=transform_test),
                                                       batch_size=batch_size, shuffle=False, num_workers=2)
        elif val_dataset == 'MNIST':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_MNIST),
                                                       batch_size=batch_size, shuffle=False, num_workers=2)
        elif val_dataset == 'Texture':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("./data/texture", transform=transform_test),
                                                         batch_size=batch_size, shuffle=False, num_workers=2)
        
        # elif val_dataset == 'Places365':
        #     val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.Places365(root='./data/places365', download=True, split='val', transform=transform_test), 
        #                                                  batch_size=batch_size, shuffle=False, num_workers=2)
        
        
        # elif val_dataset == 'inat':
        #     val_ood_loader = torch.utils.data.DataLoader(
        #         torchvision.datasets.ImageFolder("./datasets/ood_data/iNaturalist".format(val_dataset),
        #                                          transform=config.transform_test_largescale), batch_size=batch_size, shuffle=False,
        #         num_workers=2)
        # elif val_dataset == 'imagenet':
        #     val_ood_loader = torch.utils.data.DataLoader(
        #         torchvision.datasets.ImageFolder(os.path.join('./datasets/id_data/imagenet', 'val'), config.transform_test_largescale),
        #         batch_size=config.batch_size, shuffle=True, **kwargs)
        # else:
        #     val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("./datasets/ood_data/{}".format(val_dataset),
        #                                                   transform=transform_test), batch_size=batch_size, shuffle=False, num_workers=2)
        
        #torchvision.datasets.ImageFolder("./data/places365", transform=transform_test)
    return EasyDict({
        "train_ood_loader": train_ood_loader,
        "val_ood_loader": val_ood_loader,
    })