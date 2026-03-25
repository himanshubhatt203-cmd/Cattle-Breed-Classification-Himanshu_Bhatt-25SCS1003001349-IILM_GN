class SubsetWithTransform(Dataset):
    def __init__(self, base: datasets.ImageFolder, indices, transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
        self.loader = base.loader
        self.samples = base.samples  # list of (path, target)
        self.targets = [self.samples[i][1] for i in self.indices]  # convenience

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, target = self.samples[real_idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

train_dataset = SubsetWithTransform(base_dataset, train_indices, train_transform)
val_dataset   = SubsetWithTransform(base_dataset, val_indices,   test_transform)
test_dataset  = SubsetWithTransform(base_dataset, test_indices,  test_transform)

print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")



batch_size = 16
num_workers = max(2, os.cpu_count() // 2)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=PIN, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=PIN)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=PIN)