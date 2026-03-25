def denorm(img_tensor):
    """Convert normalized CHW tensor -> HWC [0,1] numpy."""
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * np.array([0.229, 0.224, 0.225]) +
                  np.array([0.485, 0.456, 0.406]), 0, 1)
    return img

def visualize_dataset_enhanced(dataset, class_names, save_path=f"{WORK_DIR}/dataset_samples_enhanced.png"):
    # show up to 6 samples per class from the dataset
    targets = dataset.targets  # valid for ImageFolder
    class_to_idx = dataset.class_to_idx

    ncols = 6
    fig, axes = plt.subplots(nrows=len(class_names), ncols=ncols, figsize=(18, 3*len(class_names)))
    fig.suptitle('Enhanced Dataset Visualization - 6 Samples per Class', fontsize=16, fontweight='bold')

    for i, cname in enumerate(class_names):
        cidx = class_to_idx[cname]
        sample_idxs = [idx for idx, t in enumerate(targets) if t == cidx][:ncols]
        for j, sidx in enumerate(sample_idxs):
            img, _ = dataset[sidx]
            axes[i, j].imshow(denorm(img))
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(cname, fontsize=12, fontweight='bold', rotation=0, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.show(); plt.close()
    print(f"Enhanced dataset visualization saved at {save_path}")

def plot_class_distribution(dataset, class_names, save_path=f"{WORK_DIR}/class_distribution.png"):
    counts = [dataset.targets.count(i) for i in range(len(class_names))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    bars = ax1.bar(class_names, counts, color=plt.cm.Set3(np.linspace(0, 1, len(class_names))))
    ax1.set_title('Class Distribution (Bar)', fontweight='bold')
    ax1.set_ylabel('Images'); ax1.tick_params(axis='x', rotation=45)
    for bar, c in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, str(c), ha='center', va='bottom', fontweight='bold')

    ax2.pie(counts, labels=class_names, autopct='%1.1f%%', startangle=90,
            colors=plt.cm.Set3(np.linspace(0, 1, len(class_names))))
    ax2.set_title('Class Distribution (Pie)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.show(); plt.close()
    print(f"Class distribution visualization saved at {save_path}")

visualize_dataset_enhanced(base_dataset, class_names)
plot_class_distribution(base_dataset, class_names)