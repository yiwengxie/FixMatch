import torch.nn as nn
import logging
import torch
import torchvision.models as models

logger = logging.getLogger(__name__)

def create_model(args):
    if args.arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_classes)
    elif args.arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_classes)
    elif args.arch == 'wideresnet':
        model == models.wide_resnet50_2(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_classes)
    elif args.arch == 'resnext':
        model == models.resnext50_32x4d(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_classes)

    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters())/1e6))
    
    device = torch.device(args.device, args.gpu)
    # print(device)
    model.to(device)
    # print(f"device:", device, "before:", model)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)

    return model