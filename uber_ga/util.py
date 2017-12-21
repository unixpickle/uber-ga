"""
Utilities for using the GA in practice.
"""

import os
import platform
import shutil
import subprocess

# pylint: disable=E1101
from mpi4py import MPI
import tensorflow as tf

def make_session():
    """
    Make a TensorFlow session for this MPI worker.

    Sessions on different workers may share GPUs.
    """
    _setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def _setup_mpi_gpus():
    """
    Set CUDA_VISIBLE_DEVICES using MPI.
    """
    num_gpus = _gpu_count()
    if num_gpus == 0:
        return
    node_id = platform.node()
    nodes = MPI.COMM_WORLD.allgather(node_id)
    local_rank = len([n for n in nodes[:MPI.COMM_WORLD.Get_rank()] if n == node_id])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank % num_gpus)

def _gpu_count():
    """
    Count the GPUs on this machine.
    """
    if shutil.which('nvidia-smi') is None:
        return 0
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv'])
    return max(0, len(output.split(b'\n')) - 2)
