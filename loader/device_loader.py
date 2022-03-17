import torch

__all__ = [
    "DeviceLoader"
]


class DeviceLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        self.device = kwargs.pop("device", "cpu")
        super().__init__(*args, **kwargs)

    def __iter__(self):
        def move_to_device(batch):
            if isinstance(batch, dict):
                return {
                    k: v.to(self.device, non_blocking=True) for k, v in batch.items()
                }
            elif issubclass(batch, (list, tuple)):
                return [v.to(self.device, non_blocking=True) for v in batch]
            else:
                return batch.to(self.device, non_blocking=True)

        return map(move_to_device, super().__iter__())

    def infinity_dpp_loader_wapper(self):
        while True:
            for d in self:
                yield d
