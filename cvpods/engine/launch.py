# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import logging
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cvpods.utils import comm

__all__ = ["launch"]


def _get_local_ip():
    """
    Find an ip address of current machine / node.
    """
    import ifcfg

    # ip = ifcfg.default_interface()["inet4"][0]
    ip = ifcfg.interfaces()["ib0"]["inet"]  # valid for ShanghaiTech AI/G/P40-Cluster
    return ip


def _find_free_port():
    """
    Find an available port of current machine / node.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(main_func, num_gpus_per_machine, num_machines=1, machine_rank=0, dist_url=None, args=()):
    """
    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine (one per machine)
        dist_url (str): url to connect to for distributed training, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to auto to automatically select a free port on localhost
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            if machine_rank > 0:
                try:
                    for _ in range(600):
                        if os.path.exists(".ip_dist_url"):
                            break
                        time.sleep(1)
                    with open(".ip_dist_url", "r") as f:
                        dist_url = f.readline()
                except Exception:
                    assert num_machines == 1, "dist_url=auto cannot work with distributed training."
            else:
                port = _find_free_port()
                dist_url = f"tcp://127.0.0.1:{port}"
                local_ip = _get_local_ip()
                ip_dist_url = f"tcp://{local_ip}:{port}"
                with open(".ip_dist_url", "w") as f:
                    f.writelines([ip_dist_url])

        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(main_func, world_size, num_gpus_per_machine, machine_rank, dist_url, args),
            daemon=False,
        )
    else:
        main_func(*args)


def _distributed_worker(
    local_rank, main_func, world_size, num_gpus_per_machine, machine_rank, dist_url, args
):
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL", init_method=dist_url, world_size=world_size, rank=global_rank
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e
    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    if os.path.exists(".ip_dist_url") and dist.get_rank() == 0:
        os.remove(".ip_dist_url")

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    main_func(*args)
