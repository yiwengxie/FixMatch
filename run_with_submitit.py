"""
A script to run multinode training with submitit.
"""
import argparse
import os
import uuid
from pathlib import Path

import main as classification
import submitit
from config.config import get_config

def parse_args():
    classification_parser = get_config()
    parser = argparse.ArgumentParser("Submitit for FixMatch", parents=[classification_parser])
    parser.add_argument("--ngpus", default=1, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=2800, type=int, help="Duration of the job(mins)")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--partition", default="fvl", type=str, help="Partition where to submit")
    parser.add_argument("--qos", default="default", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument('--comment', default="", type=str, help='Comment to pass to scheduler, e.g. priority message')

    args = parser.parse_args()
    return args


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/share/test/").is_dir():
        p = Path(f"/share/test/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        self._setup_gpu_args()
        classification.main(self.args)

    def checkpoint(self):
        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)
    # executor = submitit.LocalExecutor(folder=args.job_dir)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout
    slurm_qos = args.qos

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node, # 8 gpus
        gpus_per_node=num_gpus_per_node, # 8 gpus
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=4,
        nodes=nodes, # 1 node
        slurm_qos=slurm_qos, # high
        timeout_min=timeout_min,  # max is 60 * 48
        slurm_partition=partition, # fvl
        slurm_signal_delay_s=120,
        slurm_exclude="fvl09", # 废物节点
        # slurm_nodelist="fvl15", # 老子就要用这个
        **kwargs)

    executor.update_parameters(name="FixMatch")

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()