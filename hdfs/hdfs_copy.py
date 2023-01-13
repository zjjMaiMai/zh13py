import os
import time
import subprocess
import concurrent.futures
import tensorflow as tf
import tensorflow_io as tfio

__all__ = ["hdfs_cp"]


def hdfs_cp(src_list, dst, worker=16, cli=False):
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker) as executor:
        tasks = []
        for src in src_list:
            if os.path.basename(src) == "":
                src = os.path.split(src)[0]
            if cli and tf.io.gfile.isdir(dst):
                dst_ = tf.io.gfile.join(dst, os.path.basename(src))
            if tf.io.gfile.isdir(src):
                _copytree(src, dst_, executor, tasks)
            else:
                _copy(src, dst_, executor, tasks)
            print(f"get [{len(tasks)}] jobs from {src} to {dst_}!")
        concurrent.futures.wait(tasks)
    print(f"use time: {time.time() - start_time:.2f} sec")
    return dst


def _copytree(src, dst, executor, tasks):
    names = tf.io.gfile.listdir(src)
    tf.io.gfile.makedirs(dst)

    for name in names:
        srcname = tf.io.gfile.join(src, name)
        dstname = tf.io.gfile.join(dst, name)

        if tf.io.gfile.isdir(srcname):
            _copytree(srcname, dstname, executor, tasks)
        else:
            _copy(srcname, dstname, executor, tasks)
    return


def _copy(src, dst, executor, tasks):
    if tf.io.gfile.isdir(dst):
        dst = tf.io.gfile.join(dst, os.path.basename(src))
    if src == dst:
        raise OSError("{!r} and {!r} are the same file".format(src, dst))

    if src.startswith("hdfs://"):
        tasks.append(executor.submit(_copy_to_local, src, dst))
    else:
        tasks.append(executor.submit(_copy_from_local, src, dst))
    return


def _copy_to_local(src, dst):
    subprocess.run(["hdfs", "dfs", "-copyToLocal", src, dst])


def _copy_from_local(src, dst):
    subprocess.run(["hdfs", "dfs", "-copyFromLocal", src, dst])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("SRC", nargs="+")
    parser.add_argument("DST")
    parser.add_argument("-p", default=16, help="number of threads")
    args = parser.parse_args()

    hdfs_cp(args.SRC, args.DST, args.p, True)
