# backend/recognition/utils_ctc.py
import torch

def encode_batch(labels, char_to_idx):
    targets = []
    lengths = []
    for s in labels:
        idxs = [char_to_idx[c] for c in s.replace(" ", "")]
        targets.extend(idxs)
        lengths.append(len(idxs))
    return torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

def collate_fn(batch, char_to_idx):
    images, labels = zip(*batch)
    images = torch.stack(images)
    targets, lengths = encode_batch(labels, char_to_idx)
    return images, targets, lengths, labels
