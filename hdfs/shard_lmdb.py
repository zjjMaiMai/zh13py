import os
import lmdb

O_0755 = int("0755", 8)
O_0111 = int("0111", 8)


class ShardEnvironment(object):
    def __init__(
        self,
        path,
        shard=32,
        map_size=10485760,
        readonly=False,
        metasync=True,
        sync=True,
        map_async=False,
        mode=O_0755,
        create=True,
        readahead=True,
        writemap=False,
        meminit=True,
        max_readers=126,
        lock=True,
    ):
        super().__init__()
        if create and not readonly:
            os.mkdir(path, mode)
        else:
            if os.path.exists(os.path.join(path, "data.mdb")):
                shard = 0
            else:
                shard = len(os.listdir(path)) - 1

        self.shard = shard
        if self.shard == 0:
            self.envs = [
                lmdb.open(
                    path,
                    map_size=map_size,
                    readonly=readonly,
                    metasync=metasync,
                    sync=sync,
                    map_async=map_async,
                    mode=mode,
                    create=create,
                    readahead=readahead,
                    writemap=writemap,
                    meminit=meminit,
                    max_readers=max_readers,
                    lock=lock,
                    subdir=True,
                )
            ]
        else:
            self.envs = [
                lmdb.open(
                    os.path.join(path, f"{i:04d}"),
                    map_size=map_size // self.shard,
                    readonly=readonly,
                    metasync=metasync,
                    sync=sync,
                    map_async=map_async,
                    mode=mode,
                    create=create,
                    readahead=readahead,
                    writemap=writemap,
                    meminit=meminit,
                    max_readers=max_readers,
                    lock=lock,
                    subdir=False,
                )
                for i in range(shard)
            ]
            self.envs.append(
                lmdb.open(
                    os.path.join(path, "keys_db"),
                    map_size=map_size // self.shard,
                    readonly=readonly,
                    metasync=metasync,
                    sync=sync,
                    map_async=map_async,
                    mode=mode,
                    create=create,
                    readahead=readahead,
                    writemap=writemap,
                    meminit=meminit,
                    max_readers=max_readers,
                    lock=lock,
                    subdir=False,
                )
            )

    def __enter__(self):
        return self

    def __exit__(self, _1, _2, _3):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        for env in self.envs:
            env.close()

    def begin(self, db=None, parent=None, write=False, buffers=False):
        return ShardTransaction(self, db, parent, write, buffers)


open = ShardEnvironment


class ShardTransaction(object):
    def __init__(self, shard_env, db=None, parent=None, write=False, buffers=False):
        super().__init__()
        self.shard_env = shard_env
        self.shard_txn = [
            env.begin(db, parent, write, buffers) for env in shard_env.envs
        ]

    def __del__(self):
        self.abort()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            self.abort()
        else:
            self.commit()

    def commit(self):
        for txn in self.shard_txn:
            txn.commit()

    def abort(self):
        for txn in self.shard_txn:
            txn.abort()

    def get(self, key, default=None):
        if len(self.shard_txn) == 1:
            return self.shard_txn[0].get(key, default)

        ids = self.shard_txn[-1].get(key)
        if ids is None:
            return default
        ids = int.from_bytes(ids, byteorder="little", signed=False)
        return self.shard_txn[ids].get(key, default)

    def put(self, key, value, dupdata=True, overwrite=True, append=False):
        if len(self.shard_txn) == 1:
            return self.shard_txn[0].put(key, value, dupdata, overwrite, append)

        db_size = [txn.stat()["entries"] for txn in self.shard_txn[:-1]]
        ids = min(range(len(db_size)), key=lambda i: db_size[i])

        self.shard_txn[-1].put(
            key,
            ids.to_bytes(8, byteorder="little", signed=False),
        )
        return self.shard_txn[ids].put(key, value, dupdata, overwrite, append)
