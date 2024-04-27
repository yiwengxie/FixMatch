import argparse
import torch
import os
import math
from utils.seed_utils import set_seed
from torch.utils.tensorboard import SummaryWriter

def get_config():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training', add_help=False)
    parser.add_argument('--gpu-id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--dataset', default='flowers102', type=str, choices=['cifar10', 'cifar100', 'flowers102'], help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=612, help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='resnet50', type=str, choices=['resnet18', 'resnet50', 'wideresnet', 'resnext'], help='backbone architecture')
    # parser.add_argument('--total-steps', default=100000, type=int, help='number of total steps to run')
    # parser.add_argument('--eval-step', default=1224, type=int, help='number of eval steps to run')
    parser.add_argument('--epochs', default=1000, type=int, help='number of epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=8, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
    #  Attention
    parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=0.001, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    # something trouble 分布式模型load没有解决
    parser.add_argument('--use-ema', action='store_true', default=False, help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
    parser.add_argument('--out', default='results', help='directory to output the result')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int, help="random seed")
    parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt-level", type=str, default="O1", help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']." "See details at https://nvidia.github.io/apex/amp.html")
    # attention
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', -1), help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    # wwwwwwwwwc 这sb东西不能要
    # parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--train', action='store_true', default=False, help='train the model')
    parser.add_argument('--test', action='store_true', default=False, help='test the model')
    parser.add_argument('--semi', action='store_true', default=False, help='train the model with unlabeled data')
    parser.add_argument('--pretrained-model', default='results_test/model_best.pth.tar', type=str, help='path to pretrained model')
    parser.add_argument('--one', action='store_true', default=False, help='train the model with one data')

    return parser

def check_args(args):
    if args.seed is not None:
        set_seed(args)

    # if (args.distributed == False) or (args.rank == 0):
    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
    
    elif args.dataset == 'flowers102':
        args.num_classes = 102
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    # 罪魁祸首！！！！！！！！！！！！！！ 我草泥马你他妈能写出这种代码？？？？？？？？？？？？？？？？？？？？
    # if (args.distributed == True) and (args.rank != 0):
    if args.distributed == True:
        torch.distributed.barrier()

    return


def load_and_initialize_model(args, model, optimizer, scheduler, best_acc):
    # Calculate epochs based on total_steps and eval_step
    # args.epochs = math.ceil(args.total_steps / args.eval_step)
    
    # Exponential Moving Average (EMA) model initialization
    from models.ema import ModelEMA
    ema_model = ModelEMA(args, model, args.ema_decay)

    # Initialize starting epoch
    args.start_epoch = 0
    
    # Resume training from a checkpoint
    if args.resume:
        print("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Automatic Mixed Precision (AMP) initialization
    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    return model, optimizer, scheduler, ema_model, best_acc

