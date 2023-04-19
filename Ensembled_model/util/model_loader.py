import os

from importlib import import_module
import json

import torch

def get_model(args, num_classes, load_ckpt=False, optimizer=None):
    if args.in_dataset == 'imagenet':
        if args.model_arch == 'resnet18':
            from models.resnet import resnet18
            model = resnet18(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'resnet50':
            from models.resnet import resnet50
            model = resnet50(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'mobilenet':
            from models.mobilenet import mobilenet_v2
            model = mobilenet_v2(num_classes=num_classes, pretrained=True)
    else:
        if args.model_arch == 'resnet18':
            from models.resnet import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes, method=args.method)
            if load_ckpt:
                checkpoint = torch.load("./checkpoint/CIFAR-10/{model_arch}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, model_arch=args.name,epochs=args.epochs))
                model.load_state_dict(checkpoint['state_dict'])
        # elif args.model_arch == 'resnet50':
        #     from models.resnet import resnet50_cifar
        #     model = resnet50_cifar(num_classes=num_classes, method=args.method)
        elif args.model_arch == 'lenet':
                if optimizer == 'Bayessian':
                    configpath = 'configs/lenet_vogn_feature.json'
                    ckptpath = './checkpoint/CIFAR-10/lenet/bay_epoch30_ckp.ckpt'
                elif optimizer == 'Adam':
                    configpath = 'configs/lenet_adam_feature.json'
                    ckptpath = './checkpoint/CIFAR-10/lenet/adam_epoch30_ckp.ckpt'
                else:
                    print('Not Available Optimizer')
                    
                with open(configpath) as f:
                    config = json.load(f)
                    
                device = ("cuda" if torch.cuda.is_available() else "cpu")
                batchsize = config['batch_size']
                _, ext = os.path.splitext(config['arch_file'])
                dirname = os.path.dirname(config['arch_file'])
                module_path = '.'.join(os.path.split(config['arch_file'])).replace(ext, '')
                module = import_module(module_path)
                arch_class = getattr(module, config['arch_name'])
                arch_kwargs = {} if config['arch_args'] == 'None' else config['arch_args']
                arch_kwargs['num_classes'] = num_classes
                model = arch_class(**arch_kwargs)
                setattr(model, 'num_classes', num_classes)
                model.to(device)
                print('Model Loading Done')
                if load_ckpt:
                    checkpoint = torch.load(ckptpath)
                    print('checkpoint path: ' + ckptpath)
                    model.load_state_dict(checkpoint['model'])
        else:
            assert False, 'Not supported model arch: {}'.format(args.model_arch)

        

    model.cuda()
    model.eval()
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    return model