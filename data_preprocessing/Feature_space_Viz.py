def visualize_features_tsne_pca(model, loader, class_names, max_samples=1000):
    model.eval()
    feats_list, labs_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=PIN)
            f = model.get_features(x).cpu().numpy()
            feats_list.append(f); labs_list.append(y.numpy())
            if sum(len(a) for a in labs_list) >= max_samples:
                break

    feats = np.vstack(feats_list)[:max_samples]
    labs  = np.concatenate(labs_list)[:max_samples]

    pca = PCA(n_components=min(50, feats.shape[1]), random_state=SEED)
    feats_pca = pca.fit_transform(feats)

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000, init='pca')
    feats_tsne = tsne.fit_transform(feats_pca)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))

    for i, (cname, col) in enumerate(zip(class_names, colors)):
        mask = labs == i
        ax1.scatter(feats_tsne[mask, 0], feats_tsne[mask, 1], label=cname, alpha=0.6, s=30)
    ax1.set_title('t-SNE of Features'); ax1.legend(); ax1.grid(True, alpha=0.3)

    for i, (cname, col) in enumerate(zip(class_names, colors)):
        mask = labs == i
        ax2.scatter(feats_pca[mask, 0], feats_pca[mask, 1], label=cname, alpha=0.6, s=30)
    ax2.set_title('PCA (first 2)'); ax2.legend(); ax2.grid(True, alpha=0.3)

    evr = pca.explained_variance_ratio_[:10]
    ax3.bar(range(1, len(evr)+1), evr); ax3.set_title('PCA Explained Variance (Top 10)'); ax3.grid(True, alpha=0.3)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax4.plot(range(1, len(cumsum)+1), cumsum); ax4.axhline(0.95, ls='--', c='r', label='95%')
    ax4.set_title('Cumulative Explained Variance'); ax4.legend(); ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{WORK_DIR}/feature_analysis.png"
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.show(); plt.close()
    return feats, labs



def grid_from_feature_maps(tensor, max_channels=16, normalize=True):
    """tensor: (C,H,W) -> grid image"""
    C, H, W = tensor.shape
    k = min(max_channels, C)
    idxs = np.linspace(0, C-1, k, dtype=int)
    maps = tensor[idxs].detach().cpu().numpy()

    # normalize each map to [0,1]
    if normalize:
        maps = (maps - maps.min(axis=(1,2), keepdims=True)) / (maps.ptp(axis=(1,2), keepdims=True) + 1e-6)

    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    gap = 2
    grid = np.ones((rows*H + (rows-1)*gap, cols*W + (cols-1)*gap))

    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= k: break
            y0 = r*(H+gap); x0 = c*(W+gap)
            grid[y0:y0+H, x0:x0+W] = maps[i]
            i += 1
    return grid

@torch.no_grad()
def visualize_enhanced_feature_maps(model, image, class_name, save_prefix="sample"):
    """
    Captures feature maps from several backbone stages and saves montages.
    Works with EfficientNet (features Sequential) and ResNet (layer1..4).
    """
    model.eval()
    image = image.unsqueeze(0).to(device, non_blocking=PIN)

    activations = {}
    hooks = []

    def save_act(name):
        def fn(_m, _i, o):
            activations[name] = o.detach()
        return fn

    layer_names = []
    if hasattr(model.backbone, "features"):  # EfficientNet
        feats = model.backbone.features
        candidate_idx = list(range(len(feats)))  # capture many; we’ll sample a few
        # pick representative blocks (early/mid/late) safely within range
        pick = sorted(set([0, 1, 2, 3, 4, 5, 6, len(feats)-2, len(feats)-1]))
        pick = [i for i in pick if 0 <= i < len(feats)]
        for i in pick:
            name = f"features_{i}"
            layer_names.append(name)
            hooks.append(feats[i].register_forward_hook(save_act(name)))
    else:  # ResNet
        picks = [("layer1", model.backbone.layer1),
                 ("layer2", model.backbone.layer2),
                 ("layer3", model.backbone.layer3),
                 ("layer4", model.backbone.layer4)]
        for n, m in picks:
            layer_names.append(n)
            hooks.append(m.register_forward_hook(save_act(n)))

    _ = model(image)

    for h in hooks: h.remove()

    # Save montages
    paths = []
    for name in layer_names:
        if name not in activations: continue
        feat = activations[name][0]  # (C,H,W)
        # Adaptive to manageable size for display
        if feat.shape[-1] > 112:
            feat_small = nn.functional.interpolate(activations[name], size=(112,112), mode='bilinear', align_corners=False)[0]
        else:
            feat_small = feat
        grid = grid_from_feature_maps(feat_small, max_channels=16)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid, cmap='viridis'); plt.axis('off')
        plt.title(f"{name} | {class_name}")
        out = f"{WORK_DIR}/{save_prefix}_{name}_feature_maps.png"
        plt.savefig(out, dpi=300, bbox_inches='tight'); plt.show(); plt.close()
        paths.append(out)
    return paths



class GradCAM:
    """
    Stable Grad-CAM using forward+full_backward hooks.
    """
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        modules = dict(self.model.named_modules())
        if target_layer_name not in modules:
            raise ValueError(f"Layer '{target_layer_name}' not found in model.")
        self.target_layer = modules[target_layer_name]
        self.fh = self.target_layer.register_forward_hook(self._save_activation)
        self.bh = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, image_tensor, class_idx):
        self.gradients = None
        self.activations = None

        img = image_tensor.unsqueeze(0).to(device, non_blocking=PIN).requires_grad_(True)
        out = self.model(img)
        score = out[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return None

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP over H,W
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        cam = nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        return cam.squeeze(0).squeeze(0).detach().cpu().numpy()

    def close(self):
        if hasattr(self, "fh") and self.fh is not None: self.fh.remove()
        if hasattr(self, "bh") and self.bh is not None: self.bh.remove()

def analyze_model_comprehensive(model, loader, class_names, num_samples=5):
    # collect first N images from test loader
    samples = []
    for imgs, labels in loader:
        for i in range(imgs.size(0)):
            samples.append((imgs[i], labels[i].item()))
            if len(samples) >= num_samples: break
        if len(samples) >= num_samples: break

    # choose target layer for Grad-CAM
    target_layer_name = "backbone.features.7" if hasattr(model.backbone, "features") else "backbone.layer4"
    cam = GradCAM(model, target_layer_name)

    for idx, (img_t, y) in enumerate(samples, 1):
        img_np = denorm(img_t)
        with torch.no_grad():
            prob = torch.softmax(model(img_t.unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
            y_pred = int(np.argmax(prob))

        fig = plt.figure(figsize=(20, 12))

        # Original
        ax = plt.subplot(2, 4, 1); ax.imshow(img_np); ax.axis('off')
        ax.set_title(f'Original\nTrue: {class_names[y]}\nPred: {class_names[y_pred]}', fontweight='bold')

        # Probabilities
        ax = plt.subplot(2, 4, 2)
        bars = ax.bar(range(len(class_names)), prob, color=plt.cm.Set3(np.linspace(0,1,len(class_names))))
        bars[y_pred].set_color('red'); bars[y].set_edgecolor('green'); bars[y].set_linewidth(3)
        ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45)
        ax.set_title('Prediction Probabilities', fontweight='bold')

        # Grad-CAM (pred class)
        ax = plt.subplot(2, 4, 3)
        cam_map = cam(img_t.clone(), y_pred)
        ax.imshow(img_np)
        if cam_map is not None:
            ax.imshow(cam_map, cmap='jet', alpha=0.4)
            ax.set_title('Grad-CAM (Pred Class)', fontweight='bold')
        else:
            ax.set_title('Grad-CAM (N/A)', fontweight='bold')
        ax.axis('off')

        # Feature vector (first 100 dims)
        ax = plt.subplot(2, 4, 4)
        with torch.no_grad():
            fv = model.get_features(img_t.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        ax.plot(fv[:100]); ax.set_title('Feature Vector (first 100)'); ax.grid(True, alpha=0.3)

        # Confidence / Entropy
        ax = plt.subplot(2, 4, 5)
        confidence = float(prob[y_pred])
        p = prob + 1e-8
        entropy = float(-(p*np.log(p)).sum()/np.log(len(class_names)))
        metrics = ['Confidence', '1-Entropy', 'Correct?']
        values = [confidence, 1.0 - entropy, 1.0 if y == y_pred else 0.0]
        cols = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in values]
        bb = ax.bar(metrics, values, color=cols); ax.set_ylim(0,1); ax.set_title('Prediction Quality', fontweight='bold')
        for b, v in zip(bb, values): ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.3f}', ha='center')

        # Top-3
        ax = plt.subplot(2, 4, 6)
        top3_idx = np.argsort(-prob)[:3]; top3_vals = prob[top3_idx]
        bb = ax.bar(range(3), top3_vals, color=['gold','silver','#CD7F32'])
        ax.set_xticks(range(3)); ax.set_xticklabels([class_names[i] for i in top3_idx], rotation=45)
        ax.set_ylim(0,1); ax.set_title('Top-3 Predictions', fontweight='bold')
        for b, v in zip(bb, top3_vals): ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}', ha='center')

        # Correct/Incorrect pie
        ax = plt.subplot(2, 4, 7)
        ax.pie([1,0] if y==y_pred else [0,1], labels=['Correct','Incorrect'], colors=['green','red'], autopct='%1.0f%%', startangle=90)
        ax.set_title('Prediction Result', fontweight='bold')

        # Input grad importance
        ax = plt.subplot(2, 4, 8)
        img_var = img_t.unsqueeze(0).to(device).requires_grad_(True)
        out = model(img_var)
        target = out[0, y_pred]
        model.zero_grad(set_to_none=True)
        target.backward()
        imp = img_var.grad.detach().abs().mean(dim=1)[0].cpu().numpy()
        ax.imshow(imp, cmap='hot'); ax.axis('off'); ax.set_title('Input Importance (grad)', fontweight='bold')

        plt.suptitle(f'Comprehensive Analysis - Sample {idx}', fontsize=16, fontweight='bold')
        outp = f"{WORK_DIR}/comprehensive_analysis_sample_{idx}.png"
        plt.savefig(outp, dpi=300, bbox_inches='tight'); plt.show(); plt.close()

    cam.close()



    print("Starting Enhanced Cattle Breed Classification...")
print("="*60)

model = EnhancedCattleClassifier(num_classes=num_classes, backbone='efficientnet').to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model Architecture: Enhanced {model.backbone_name.title()}")
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print("="*60)

print("Starting training...")
trained_model, train_losses, val_losses, train_accs, val_accs = train_enhanced_model(
    model, train_loader, val_loader, num_epochs=100, patience=15
)

print("Starting comprehensive evaluation...")
acc, prec, rec, f1, true_labels, pred_labels, pred_probs = evaluate_comprehensive(
    trained_model, test_loader, class_names
)

print("Performing feature analysis (PCA + t-SNE)...")
features, labels = visualize_features_tsne_pca(trained_model, test_loader, class_names)



print("Saving per-layer feature maps for a few test samples...")
# pick one representative sample per class (if possible), else first N
picked = {}
test_iter = iter(test_loader)
while len(picked) < min(len(class_names), 4):  # limit to 4 to keep it light
    try:
        xb, yb = next(test_iter)
    except StopIteration:
        break
    for i in range(len(yb)):
        c = int(yb[i].item())
        if c not in picked:
            picked[c] = xb[i]
        if len(picked) >= min(len(class_names), 4):
            break

for c, img in picked.items():
    _paths = visualize_enhanced_feature_maps(trained_model, img, class_names[c], save_prefix=f"example_{class_names[c]}")



print("Performing comprehensive model analysis on a few samples (with Grad-CAM)...")
analyze_model_comprehensive(trained_model, test_loader, class_names)

# Save final model
torch.save({
    'model_state_dict': trained_model.state_dict(),
    'class_names': class_names,
    'num_classes': num_classes,
    'accuracy': acc,
    'architecture': f'Enhanced{model.backbone_name.title()}'
}, f"{WORK_DIR}/final_enhanced_cattle_classifier.pth")

print("="*60)
print("TRAINING AND EVALUATION COMPLETED!")
print(f"Final Test Accuracy: {acc*100:.2f}%")
print(f"Target Accuracy: 95.00%")
print(f"Achievement: {'✅ TARGET REACHED!' if acc >= 0.95 else '❌ TARGET NOT REACHED'}")
print("="*60)
print("All visualizations and model saved successfully!")
print("Generated files:")
print("- /kaggle/working/dataset_samples_enhanced.png")
print("- /kaggle/working/class_distribution.png")
print("- /kaggle/working/training_history.png")
print("- /kaggle/working/confusion_matrices.png")
print("- /kaggle/working/roc_pr_curves.png")
print("- /kaggle/working/per_class_metrics.png")
print("- /kaggle/working/feature_analysis.png")
print("- /kaggle/working/comprehensive_analysis_sample_X.png (X = 1..5)")
print("- /kaggle/working/example_<CLASS>_features_*.png (per-layer feature maps)")
print("- /kaggle/working/final_enhanced_cattle_classifier.pth")