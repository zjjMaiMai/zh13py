import shutil
import pathlib
from .hdfs_copy import hdfs_cp

__all__ = ["remove_all_cache", "download_to_local_cache", "LOCAL_CACHE_DIR"]

LOCAL_CACHE_DIR = pathlib.Path.home() / ".cache" / "HDFSCache"


def remove_all_cache():
    shutil.rmtree(LOCAL_CACHE_DIR)


def download_to_local_cache(path: str):
    if not path.startswith("hdfs://"):
        return path
    import fcntl

    LOCAL_CACHE_DIR.mkdir(exist_ok=True, parents=True)
    dst = LOCAL_CACHE_DIR / path.replace("hdfs://", "")
    with open(LOCAL_CACHE_DIR / "lock", "w") as fp:
        fcntl.flock(fp, fcntl.LOCK_EX)
        if not dst.exists():
            hdfs_cp(path, str(dst))
        fcntl.flock(fp, fcntl.LOCK_UN)
    return str(dst)
