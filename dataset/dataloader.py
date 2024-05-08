import torch
from torch.utils.data import DataLoader
from utils.samplers import RASampler
import utils.distributed_utils

def create_data_loaders(args, labeled_dataset, unlabeled_dataset, generated_dataset, valid_dataset, test_dataset):
    if args.distributed:
        num_tasks = utils.distributed_utils.get_world_size()
        global_rank = utils.distributed_utils.get_rank()
        if args.repeated_aug:
            sampler_train_labeled = RASampler(
                labeled_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            sampler_train_generated = RASampler(
                generated_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
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
            sampler_train_generated = torch.utils.data.DistributedSampler(
                generated_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(valid_dataset) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter testation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_valid = torch.utils.data.DistributedSampler(
                valid_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_valid = torch.utils.data.SequentialSampler(valid_dataset)
    else:
        sampler_train_labeled = torch.utils.data.RandomSampler(labeled_dataset)
        sampler_train_unlabeled = torch.utils.data.RandomSampler(unlabeled_dataset)
        sampler_train_generated = torch.utils.data.RandomSampler(generated_dataset)
        sampler_valid = torch.utils.data.SequentialSampler(valid_dataset)
    
    sampler_test = torch.utils.data.SequentialSampler(test_dataset)
    
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=sampler_train_labeled,
        batch_size=args.batch_size,
        #???????????????????????????????????????????????????????????
        # AssertionError: can only test a child process
        # num_workers=0,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=sampler_train_unlabeled,
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    generated_trainloader = DataLoader(
        generated_dataset,
        sampler=sampler_train_generated,
        batch_size=args.batch_size*3,
        num_workers=args.num_workers,
        drop_last=True)

    valid_loader = DataLoader(
        valid_dataset,
        sampler=sampler_valid,
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    
    test_loader = DataLoader(
        test_dataset,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    return labeled_trainloader, unlabeled_trainloader, generated_trainloader, valid_loader, test_loader