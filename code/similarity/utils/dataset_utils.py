import torch


def get_split_loaders(ds, batch_size, shuffle, num_workers):
    ds1, ds2 = torch.utils.data.random_split(ds, [len(
        ds) // 2, (len(ds) // 2) + len(ds) % 2], generator=torch.Generator().manual_seed(42))

    ds1_loader = get_loader(ds1, batch_size, shuffle, num_workers)
    ds2_loader = get_loader(ds2, batch_size, shuffle, num_workers)

    return ds1_loader, ds2_loader


def get_loader(ds, batch_size, shuffle, num_workers):
    return torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers)
