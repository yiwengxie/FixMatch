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
from dataset.dataloader import create_data_loaders
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
    labeled_dataset, unlabeled_dataset, generated_dataset, valid_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, './data')
    return labeled_dataset, unlabeled_dataset, generated_dataset, valid_dataset, test_dataset

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

def train(args, labeled_trainloader, unlabeled_trainloader, generated_trainloader, valid_loader,
          model, optimizer, ema_model, scheduler, best_acc):
    if args.amp:
        from apex import amp
    valid_accs = []
    end = time.time()

    if args.distributed:
        labeled_epoch = 0
        unlabeled_epoch = 0
        generated_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
        generated_trainloader.sampler.set_epoch(generated_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    generated_iter = iter(generated_trainloader)

    # 坑死爹啦！！！！！！！！！！！！ Lead to loss = nan !!!!!!!!!!
    # model.train()
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_g = AverageMeter()
        mask_probs_u = AverageMeter()
        mask_probs_g = AverageMeter()
        error_rate_u = AverageMeter()
        error_rate_g = AverageMeter()

        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step))
        
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(iter(labeled_iter))
            except:
                if args.distributed:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(iter(labeled_iter))

            if args.semi and args.generated:
                try:
                    # week and strong
                    (inputs_u_w, inputs_u_s), targets_u = next(iter(unlabeled_iter))
                except:
                    if args.distributed:
                        unlabeled_epoch += 1
                        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                    unlabeled_iter = iter(unlabeled_trainloader)
                    (inputs_u_w, inputs_u_s), targets_u = next(iter(unlabeled_iter))

                try:
                    inputs_g , targets_g = next(iter(generated_iter))
                except:
                    if args.distributed:
                        generated_epoch += 1
                        generated_trainloader.sampler.set_epoch(generated_epoch)
                    generated_iter = iter(generated_trainloader)
                    inputs_g , targets_g = next(iter(generated_iter))

                data_time.update(time.time() - end)
                batch_size = inputs_x.shape[0]
                inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s, inputs_g)), 1 + 2*args.mu + 3).to(args.device)
                targets_x = targets_x.to(args.device)
                logits = model(inputs)
                logits = de_interleave(logits, 1 + 2*args.mu + 3)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:(1 + 2*args.mu)*batch_size].chunk(2)
                logits_g = logits[(1 + 2*args.mu)*batch_size:]
                del logits

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                pseudo_label_u = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
                max_probs_u, targets_pseudo_u = torch.max(pseudo_label_u, dim=-1)
                # 一个标签限定
                mask_u = max_probs_u.ge(args.threshold).float()
                Lu = (F.cross_entropy(logits_u_s, targets_pseudo_u, reduction='none') * mask_u).mean()

                pseudo_label_g = torch.softmax(logits_g.detach()/args.T, dim=-1)
                max_probs_g, targets_pseudo_g = torch.max(pseudo_label_g, dim=-1)
                mask_g = max_probs_g.ge(args.threshold).float()
                targets_g = targets_g.to(args.device)
                mask_g_real = mask_g * (targets_pseudo_g == targets_g).float()
                Lg = (F.cross_entropy(logits_g, targets_g, reduction='none') * mask_g_real).mean()

                targets_u = targets_u.to(args.device)
                mask_u_temp = mask_u.bool()
                error_u = (targets_pseudo_u[mask_u_temp] != targets_u[mask_u_temp]).float()

                targets_g = targets_g.to(args.device)
                mask_g_temp = mask_g.bool()
                error_g = (targets_pseudo_g[mask_g_temp] != targets_g[mask_g_temp]).float()

                if error_g.numel() > 0:
                    beta = args.lambda_g + mask_g_real.mean().item()
                else:
                    beta = args.lambda_g
                
                if error_u.numel() > 0:
                    alpha = args.lambda_u + mask_u.mean().item()
                else:
                    alpha = args.lambda_u
                
                loss = Lx + alpha * Lu + beta * Lg

            elif args.semi:                
                try:
                    # week and strong
                    (inputs_u_w, inputs_u_s), targets_u = next(iter(unlabeled_iter))
                except:
                    if args.distributed:
                        unlabeled_epoch += 1
                        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                    unlabeled_iter = iter(unlabeled_trainloader)
                    (inputs_u_w, inputs_u_s), targets_u = next(iter(unlabeled_iter))

                data_time.update(time.time() - end)
                batch_size = inputs_x.shape[0]
                inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 1 + 2*args.mu).to(args.device)
                targets_x = targets_x.to(args.device)
                logits = model(inputs)
                logits = de_interleave(logits, 1 + 2*args.mu)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:(1 + 2*args.mu)*batch_size].chunk(2)
                del logits

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                pseudo_label_u = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
                max_probs_u, targets_pseudo_u = torch.max(pseudo_label_u, dim=-1)
                # 一个标签限定
                mask_u = max_probs_u.ge(args.threshold).float()
                Lu = (F.cross_entropy(logits_u_s, targets_pseudo_u, reduction='none') * mask_u).mean()

                targets_u = targets_u.to(args.device)
                mask_u_temp = mask_u.bool()
                error_u = (targets_pseudo_u[mask_u_temp] != targets_u[mask_u_temp]).float()
                
                if error_u.numel() > 0:
                    alpha = args.lambda_u + mask_u.mean().item()
                else:
                    alpha = args.lambda_u

                Lg = torch.empty((0, 0))
                mask_g = torch.empty((0, 0))
                error_g = torch.empty((0, 0))
                
                loss = Lx + alpha * Lu
                
            elif args.generated:
                try:
                    inputs_g , targets_g = next(iter(generated_iter))
                except:
                    if args.distributed:
                        generated_epoch += 1
                        generated_trainloader.sampler.set_epoch(generated_epoch)
                    generated_iter = iter(generated_trainloader)
                    inputs_g , targets_g = next(iter(generated_iter))

                data_time.update(time.time() - end)
                batch_size = inputs_x.shape[0]
                inputs = interleave(torch.cat((inputs_x, inputs_g)), 1 + 3).to(args.device)
                targets_x = targets_x.to(args.device)
                logits = model(inputs)
                logits = de_interleave(logits, 1 + 3)
                logits_x = logits[:batch_size]
                logits_g = logits[batch_size:]
                del logits

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                pseudo_label_g = torch.softmax(logits_g.detach()/args.T, dim=-1)
                max_probs_g, targets_pseudo_g = torch.max(pseudo_label_g, dim=-1)
                mask_g = max_probs_g.ge(args.threshold).float()
                targets_g = targets_g.to(args.device)
                mask_g_real = mask_g * (targets_pseudo_g == targets_g).float()
                Lg = (F.cross_entropy(logits_g, targets_g, reduction='none') * mask_g_real).mean()

                targets_g = targets_g.to(args.device)
                mask_g_temp = mask_g.bool()
                error_g = (targets_pseudo_g[mask_g_temp] != targets_g[mask_g_temp]).float()

                if error_g.numel() > 0:
                    beta = args.lambda_g + mask_g_real.mean().item()
                else:
                    beta = args.lambda_g

                Lu = torch.empty((0, 0))
                mask_u = torch.empty((0, 0))
                error_u = torch.empty((0, 0))
                
                loss = Lx + beta * Lg

            else:
                data_time.update(time.time() - end)
                # inputs_x = interleave(inputs_x, 1).to(args.device)
                inputs_x = inputs_x.to(args.device)
                targets_x = targets_x.to(args.device)
                logits_x = model(inputs_x)

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
                Lu = torch.empty((0, 0))
                Lg = torch.empty((0, 0))
                mask_u = torch.empty((0, 0))
                mask_g = torch.empty((0, 0))
                error_u = torch.empty((0, 0))
                error_g = torch.empty((0, 0))
                loss = Lx

            optimizer.zero_grad()

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            if Lx.numel() > 0: losses_x.update(Lx.item())
            if Lu.numel() > 0: losses_u.update(Lu.item())
            if Lg.numel() > 0: losses_g.update(Lg.item())
            if error_u.numel() > 0: error_rate_u.update(error_u.mean().item())
            if error_g.numel() > 0: error_rate_g.update(error_g.mean().item())
            if mask_u.numel() > 0: mask_probs_u.update(mask_u.mean().item())
            if mask_g.numel() > 0: mask_probs_g.update(mask_g_real.mean().item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask_u: {mask_u:.4f}. Error_rate: {error_rate_u:.4f}. Loss_g: {loss_g:.4f}. Mask_g: {mask_g:.4f}. Error_rate_g: {error_rate_g:.4f}. ".format(
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
                    loss_g=losses_g.avg,
                    mask_u=mask_probs_u.avg,
                    error_rate_u=error_rate_u.avg,
                    mask_g=mask_probs_g.avg,
                    error_rate_g=error_rate_g.avg,))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            valid_model = ema_model.ema
        else:
            valid_model = model

        if (epoch+1) % 2 == 0:
            if args.distributed == False or args.rank == 0:
                valid_loss, valid_acc = valid(args, valid_loader, valid_model)

                args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
                args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
                args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
                args.writer.add_scalar('train/4.error_rate_u', error_rate_u.avg, epoch)
                args.writer.add_scalar('train/5.mask_u', mask_probs_u.avg, epoch)
                args.writer.add_scalar('train/6.mask_g', mask_probs_g.avg, epoch)
                args.writer.add_scalar('valid/1.valid_acc', valid_acc, epoch)
                args.writer.add_scalar('valid/2.valid_loss', valid_loss, epoch)

                is_best = valid_acc > best_acc
                best_acc = max(valid_acc, best_acc)

                model_to_save = model.module if hasattr(model, "module") else model
                if args.use_ema:
                    ema_to_save = ema_model.ema.module if hasattr(
                        ema_model.ema, "module") else ema_model.ema
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                    'acc': valid_acc,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, is_best, args.out)

                valid_accs.append(valid_acc)
                logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
                logger.info('Mean top-1 acc: {:.2f}\n'.format(
                    np.mean(valid_accs[-20:])))

    if args.distributed == False or args.rank == 0:
        args.writer.close()


def valid(args, valid_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        valid_loader = tqdm(valid_loader)

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
                valid_loader.set_description("valid Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
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
    return losses.avg, top1.avg

def test(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

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
                test_loader.set_description("test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
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

def load_pretrained_model(args, model):
    checkpoint = torch.load(args.pretrained_model, map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"Pre-trained model loaded from {args.pretrained_model}")
    return model

def main(args):
    init_distributed_model(args)
    check_args(args)
    initialize_logging(args)
    model = create_model(args)
    labeled_dataset, unlabeled_dataset, generated_dataset, valid_dataset, test_dataset = load_datasets(args)
    labeled_trainloader, unlabeled_trainloader, generated_trainloader, valid_loader, test_loader = create_data_loaders(args, labeled_dataset, unlabeled_dataset, generated_dataset, valid_dataset, test_dataset)
    optimizer = create_optimizer(args, model)
    args.total_steps = args.epochs * len(labeled_dataset)
    args.eval_step = len(labeled_dataset) // (args.batch_size*args.world_size)
    # ！！！！！！！！！！！！！
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
        train(args, labeled_trainloader, unlabeled_trainloader, generated_trainloader, valid_loader, model, optimizer, ema_model, scheduler, best_acc)
        logger.info("****************** Running testation ******************")
        logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
        logger.info(f"  Num Epochs = {args.epochs}")
        logger.info(f"  Batch size per GPU = {args.batch_size}")
        logger.info(f"  Total train batch size = {args.batch_size*args.world_size}")
        test(args, test_loader, model)
    if args.test:
        logger.info("****************** Running testation ******************")
        logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
        logger.info(f"  Num Epochs = {args.epochs}")
        logger.info(f"  Batch size per GPU = {args.batch_size}")
        logger.info(f"  Total train batch size = {args.batch_size*args.world_size}")
        model = load_pretrained_model(args, model)
        test(args, test_loader, model)

if __name__ == '__main__':
    # $tensorboard --logdir=results
    main()