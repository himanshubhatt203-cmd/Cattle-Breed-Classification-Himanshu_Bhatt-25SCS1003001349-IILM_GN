from collections import defaultdict

def stratified_split_by_index(dataset, train_ratio=0.8, val_ratio=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    idxs_by_class = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):  # use sample labels (fast)
        idxs_by_class[label].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for lbl, idxs in idxs_by_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(idxs[:n_train].tolist())
        val_idx.extend(idxs[n_train:n_train + n_val].tolist())
        test_idx.extend(idxs[n_train + n_val:].tolist())
    return train_idx, val_idx, test_idx

train_indices, val_indices, test_indices = stratified_split_by_index(base_dataset)