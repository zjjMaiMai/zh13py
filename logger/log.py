import sys
import time
import logging
import builtins

import tensorflow_io
import tensorflow as tf

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


__all__ = [
    "init_logging",
    "save_checkpoint",
    "load_checkpoint",
]


class TFIOFileHandler(logging.StreamHandler):
    def __init__(self, remote_stream):
        super().__init__(remote_stream)
        self.remote_stream = remote_stream

    def close(self):
        self.acquire()
        self.remote_stream.flush()
        self.remote_stream.close()
        self.release()


def do_nothing(*args, **kwargs):
    pass


def init_logging(root, filename="output.log") -> SummaryWriter:
    if dist.is_initialized() and dist.get_rank() != 0:
        builtins.print = do_nothing

        class FakeSummaryWriter:
            def __getattr__(self, name):
                return do_nothing

        return FakeSummaryWriter()
    else:
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

        tf.io.gfile.makedirs(root)
        fp = tf.io.gfile.GFile(tf.io.gfile.join(root, filename), mode="a+")
        logging.getLogger().addHandler(TFIOFileHandler(fp))
        builtins.print = lambda *tup: logging.info(str(", ".join([str(x) for x in tup])))

        # tensorboard
        return SummaryWriter(tf.io.gfile.join(root, "tb"), flush_secs=15)


def save_checkpoint(
    root,
    model,
    global_step,
    optimizer=None,
    scheduler=None,
    max_keep=5,
    verbose=False,
):
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    tf.io.gfile.makedirs(root)
    filename = "{:.0f}.pth".format(time.time())
    filename = tf.io.gfile.join(root, filename)

    if verbose:
        print("ckpt save to {}".format(filename))

    with tf.io.gfile.GFile(filename, mode="wb") as fp:
        torch.save(
            {
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            },
            fp,
        )

    if max_keep:
        filelist = tf.io.gfile.glob(tf.io.gfile.join(root, "*.pth"))
        filelist.sort()
        if len(filelist) > max_keep:
            for path in filelist[:-max_keep]:
                tf.io.gfile.remove(path)
    return


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    verbose=True,
):
    if tf.io.gfile.isdir(path):
        filelist = tf.io.gfile.glob(tf.io.gfile.join(path, "*.pth"))
        if len(filelist) == 0:
            return None

        filelist.sort()
        filename = filelist[-1]
    else:
        filename = path

    if verbose:
        print("ckpt load from {}".format(filename))

    device = next(model.parameters()).device
    with tf.io.gfile.GFile(filename, mode="rb") as fp:
        ckpt = torch.load(
            fp,
            map_location=device,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and ckpt["optimizer_state_dict"]:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt["scheduler_state_dict"]:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt
