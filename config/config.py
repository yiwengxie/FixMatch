import argparse
import torch
import os
import math
from utils.seed_utils import set_seed
from torch.utils.tensorboard import SummaryWriter

def get_config():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--dataset', default='flowers102', type=str, choices=['cifar10', 'cifar100', 'flowers102'], help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=204, help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str, choices=['wideresnet', 'resnext'], help='dataset name')
    parser.add_argument('--total-steps', default=10000, type=int, help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int, help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=4, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True, help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
    parser.add_argument('--out', default='results', help='directory to output the result')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int, help="random seed")
    parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1", help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']." "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")

    return parser

def check_args(args):
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1
    args.device = device

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
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

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    return args


def load_and_initialize_model(args, model, optimizer, scheduler):
    # Calculate epochs based on total_steps and eval_step
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    
    # Exponential Moving Average (EMA) model initialization
    if args.use_ema:
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
    
    # Distributed training initialization
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
    
    return model, optimizer, scheduler, ema_model

