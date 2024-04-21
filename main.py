import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from dataset.dataset import DATASET_GETTERS
from models.create_model import create_model
from utils.samplers import RASampler
from utils.metrics import accuracy
from utils.average_meter import AverageMeter
from utils.checkpoint_utils import save_checkpoint
from utils.lr_scheduler import get_cosine_schedule_with_warmup
from utils.tensor_utils import interleave, de_interleave
from utils.distributed_utils import init_distributed_model
from config.config import check_args, load_and_initialize_model
import utils.distributed_utils

logger = logging.getLogger(__name__)

def initialize_logging(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
        level=logging.INFO)
    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.gpu}, "
        f"distributed training: {args.distributed}, "
        f"16-bits training: {args.amp}")
    logger.info(dict(args._get_kwargs()))

def load_datasets(args):
    labeled_dataset, unlabeled_dataset, test_dataset, valid_dataset = DATASET_GETTERS[args.dataset](args, './data')
    return labeled_dataset, unlabeled_dataset, test_dataset, valid_dataset

def create_data_loaders(args, labeled_dataset, unlabeled_dataset, test_dataset, valid_dataset):
    if args.distributed:
        num_tasks = utils.distributed_utils.get_world_size()
        global_rank = utils.distributed_utils.get_rank()
        if args.repeated_aug:
            sampler_train_labeled = RASampler(
                labeled_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            sampler_train_unlabeled = RASampler(
                unlabeled_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train_labeled = torch.utils.data.DistributedSampler(
                labeled_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            sampler_train_unlabeled = torch.utils.data.DistributedSampler(
                unlabeled_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(test_dataset) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_test = torch.utils.data.SequentialSampler(test_dataset)
    else:
        sampler_train_labeled = torch.utils.data.RandomSampler(labeled_dataset)
        sampler_train_unlabeled = torch.utils.data.RandomSampler(unlabeled_dataset)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)
    
    sampler_valid = torch.utils.data.SequentialSampler(valid_dataset)
    
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=sampler_train_labeled,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=sampler_train_unlabeled,
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    
    valid_loader = DataLoader(
        valid_dataset,
        sampler=sampler_valid,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    return labeled_trainloader, unlabeled_trainloader, test_loader, valid_loader

def create_optimizer(args, model):
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)
    return optimizer

def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, best_acc):
    if args.amp:
        from apex import amp
    test_accs = []
    end = time.time()

    if args.distributed:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    # 坑死爹啦！！！！！！！！！！！！
    # model.train()
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.distributed and args.rank != 0)
        for batch_idx in range(args.eval_step):
            
            try:
                inputs_x, targets_x = next(iter(labeled_iter))
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(iter(labeled_iter))
            try:
                # week and strong
                (inputs_u_w, inputs_u_s), _ = next(iter(unlabeled_iter))
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = next(iter(unlabeled_iter))

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            # ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            loss = Lx + args.lambda_u * Lu

            optimizer.zero_grad()

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if (epoch+1) % 5 == 0:
            if args.distributed == False or args.rank == 0:
                test_loss, test_acc = test(args, test_loader, test_model)

                args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
                args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
                args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
                args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
                args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
                args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

                is_best = test_acc > best_acc
                best_acc = max(test_acc, best_acc)

                model_to_save = model.module if hasattr(model, "module") else model
                if args.use_ema:
                    ema_to_save = ema_model.ema.module if hasattr(
                        ema_model.ema, "module") else ema_model.ema
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, is_best, args.out)

                test_accs.append(test_acc)
                logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
                logger.info('Mean top-1 acc: {:.2f}\n'.format(
                    np.mean(test_accs[-20:])))

    if args.distributed == False or args.rank == 0:
        args.writer.close()


def test(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader, disable=args.distributed == True and args.rank != 0)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            model.eval()

            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg

def valid(args, valid_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        valid_loader = tqdm(valid_loader, disable=args.distributed == True and args.rank != 0)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            model.eval()

            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                valid_loader.set_description("Valid Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(valid_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            valid_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))


def main(args):
    init_distributed_model(args)
    check_args(args)
    initialize_logging(args)
    model = create_model(args)
    labeled_dataset, unlabeled_dataset, test_dataset, valid_dataset = load_datasets(args)
    labeled_trainloader, unlabeled_trainloader, test_loader, valid_loader = create_data_loaders(args, labeled_dataset, unlabeled_dataset, test_dataset, valid_dataset)
    optimizer = create_optimizer(args, model)
    args.total_steps = args.epochs * len(labeled_dataset)
    args.eval_step = len(labeled_dataset) // (args.batch_size*args.world_size)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)
    best_acc = 0
    model, optimizer, scheduler, ema_model, best_acc = load_and_initialize_model(args, model, optimizer, scheduler, best_acc)

    if args.train:
        logger.info("****************** Running training ******************")
        logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
        logger.info(f"  Num Epochs = {args.epochs}")
        logger.info(f"  Batch size per GPU = {args.batch_size}")
        logger.info(f"  Total train batch size = {args.batch_size*args.world_size}")
        logger.info(f"  Total optimization steps = {args.total_steps}") 
        train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, best_acc)
    if args.eval:
        logger.info("****************** Running validation ******************")
        valid(args, valid_loader, model)

if __name__ == '__main__':
    # $tensorboard --logdir=results
    main()