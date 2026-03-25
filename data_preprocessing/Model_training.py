def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label='Train'); ax1.plot(epochs, val_losses, label='Val')
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, label='Train'); ax2.plot(epochs, val_accs, label='Val')
    ax2.set_title('Accuracy (%)'); ax2.legend(); ax2.grid(True, alpha=0.3)

    loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
    ax3.plot(epochs, loss_diff); ax3.set_title('Loss Gap'); ax3.grid(True, alpha=0.3)

    acc_diff = [abs(t - v) for t, v in zip(train_accs, val_accs)]
    ax4.plot(epochs, acc_diff); ax4.set_title('Accuracy Gap'); ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{WORK_DIR}/training_history.png"
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.show(); plt.close()

def train_enhanced_model(model, train_loader, val_loader, num_epochs=100, patience=15):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_acc = 0.0
    epochs_no_improve = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    start = time.time()
    for epoch in range(num_epochs):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=PIN), labels.to(device, non_blocking=PIN)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_loss += loss.item()
            tr_total += labels.size(0)
            tr_correct += (outputs.argmax(1) == labels).sum().item()

        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=PIN), labels.to(device, non_blocking=PIN)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                va_loss += loss.item()
                va_total += labels.size(0)
                va_correct += (outputs.argmax(1) == labels).sum().item()

        tr_loss /= len(train_loader)
        va_loss /= len(val_loader)
        tr_acc = 100.0 * tr_correct / tr_total
        va_acc = 100.0 * va_correct / va_total

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc);    val_accs.append(va_acc)

        scheduler.step(epoch + 1)
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss {tr_loss:.4f} Acc {tr_acc:.2f}% | "
              f"Val Loss {va_loss:.4f} Acc {va_acc:.2f}% | LR {lr:.6f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": va_acc
            }, f"{WORK_DIR}/best_enhanced_model.pth")
            print(f"  -> New best model saved (Val Acc {va_acc:.2f}%)")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Total training time: {time.time()-start:.2f}s")
    # Load best
    ckpt = torch.load(f"{WORK_DIR}/best_enhanced_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    return model, train_losses, val_losses, train_accs, val_accs



def plot_confusion_matrices(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Counts)', fontweight='bold')
    ax1.set_xlabel('Predicted'); ax1.set_ylabel('True')

    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Reds', ax=ax2,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    ax2.set_title('Confusion Matrix (Percent)', fontweight='bold')
    ax2.set_xlabel('Predicted'); ax2.set_ylabel('True')

    plt.tight_layout()
    plt.savefig(f"{WORK_DIR}/confusion_matrices.png", dpi=300, bbox_inches='tight'); plt.show(); plt.close()

def plot_roc_and_pr_curves(true_labels, pred_probs, class_names):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize

    y_true = label_binarize(true_labels, classes=list(range(len(class_names))))
    probs = np.array(pred_probs)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
    auc_scores, ap_scores = [], []

    # ROC
    for i, (cname, col) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        ax1.plot(fpr, tpr, label=f'{cname} (AUC {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax1.set_title('ROC (OvR)'); ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR'); ax1.legend(); ax1.grid(True, alpha=0.3)

    # PR
    for i, (cname, col) in enumerate(zip(class_names, colors)):
        precision, recall, _ = precision_recall_curve(y_true[:, i], probs[:, i])
        ap = average_precision_score(y_true[:, i], probs[:, i])
        ap_scores.append(ap)
        ax2.plot(recall, precision, label=f'{cname} (AP {ap:.3f})')
    ax2.set_title('Precision-Recall'); ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.legend(); ax2.grid(True, alpha=0.3)

    bars = ax3.bar(class_names, auc_scores)
    ax3.set_title('AUC by Class'); ax3.set_ylim(0, 1); ax3.tick_params(axis='x', rotation=45)
    for b, s in zip(bars, auc_scores): ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{s:.3f}', ha='center')

    bars = ax4.bar(class_names, ap_scores)
    ax4.set_title('Average Precision by Class'); ax4.set_ylim(0, 1); ax4.tick_params(axis='x', rotation=45)
    for b, s in zip(bars, ap_scores): ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{s:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig(f"{WORK_DIR}/roc_pr_curves.png", dpi=300, bbox_inches='tight'); plt.show(); plt.close()

def plot_per_class_metrics(true_labels, pred_labels, class_names):
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, pred_labels, average=None, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels)
    per_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    x = np.arange(len(class_names)); w = 0.2

    ax1.bar(x - 1.5*w, precision, w, label='Precision')
    ax1.bar(x - 0.5*w, recall,   w, label='Recall')
    ax1.bar(x + 0.5*w, f1,       w, label='F1')
    ax1.bar(x + 1.5*w, per_acc,  w, label='Accuracy')
    ax1.set_title('Per-Class Metrics', fontweight='bold'); ax1.set_xticks(x); ax1.set_xticklabels(class_names, rotation=45); ax1.set_ylim(0, 1); ax1.legend(); ax1.grid(True, alpha=0.3)

    # Precision heat
    prec_mat = precision.reshape(1, -1)
    im = ax2.imshow(prec_mat, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(class_names))); ax2.set_xticklabels(class_names, rotation=45); ax2.set_yticks([0]); ax2.set_yticklabels(['Precision'])
    for i, v in enumerate(precision): ax2.text(i, 0, f'{v:.3f}', ha='center', va='center', fontweight='bold')
    ax2.set_title('Precision Heatmap', fontweight='bold')

    bars = ax3.bar(class_names, support)
    ax3.set_title('Support'); ax3.tick_params(axis='x', rotation=45)
    for b, s in zip(bars, support): ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, str(s), ha='center')

    # Radar-like polar plot (compact)
    angles = np.linspace(0, 2*np.pi, len(class_names), endpoint=False).tolist()
    angles += angles[:1]
    ax4 = plt.subplot(2,2,4, projection='polar')
    def _close(vals): return np.concatenate([vals, vals[:1]])
    for name, vals in [('Precision', precision), ('Recall', recall), ('F1', f1), ('Acc', per_acc)]:
        ax4.plot(angles, _close(vals), linewidth=2, label=name)
        ax4.fill(angles, _close(vals), alpha=0.15)
    ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(class_names); ax4.set_ylim(0, 1); ax4.set_title('Polar Metrics', pad=20); ax4.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

    plt.tight_layout()
    plt.savefig(f"{WORK_DIR}/per_class_metrics.png", dpi=300, bbox_inches='tight'); plt.show(); plt.close()

def evaluate_comprehensive(model, test_loader, class_names):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=PIN)
            y = y.to(device, non_blocking=PIN)
            out = model(x)
            prob = torch.softmax(out, dim=1)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("="*60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall:    {rec:.4f}")
    print(f"Macro F1 Score:  {f1:.4f}")
    print("\nDetailed Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    plot_confusion_matrices(all_labels, all_preds, class_names)
    plot_roc_and_pr_curves(all_labels, all_probs, class_names)
    plot_per_class_metrics(all_labels, all_preds, class_names)
    return acc, prec, rec, f1, np.array(all_labels), np.array(all_preds), np.array(all_probs)