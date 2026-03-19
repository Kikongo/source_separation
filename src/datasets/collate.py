import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = {}
    batch_size = len(dataset_items)
    # item['spectogram'].shape = (1, 128, T)
    for key in dataset_items[0].keys():
        if isinstance(dataset_items[0][key], torch.Tensor):
            # pad tensors to the max length in the batch
            if key == "spectrogram":
                max_len = max([item[key].shape[2] for item in dataset_items])
            else:
                max_len = max([item[key].shape[1] for item in dataset_items])
            padded_tensors = []
            for item in dataset_items:
                tensor = item[key]
                if key == "spectrogram":
                    pad_size = max_len - tensor.shape[2]
                else:
                    pad_size = max_len - tensor.shape[1]
                if pad_size > 0:
                    pad_tensor = torch.nn.functional.pad(
                        tensor, (0, pad_size), mode="constant", value=0
                    )
                else:
                    pad_tensor = tensor
                padded_tensors.append(pad_tensor)

            result_batch[key] = torch.stack(padded_tensors)
        else:
            # non-tensor fields are collected into a list
            result_batch[key] = [item[key] for item in dataset_items]

    return result_batch