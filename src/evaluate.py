import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(model, test_loader, class_names, device, model_path="models/best_model.pth"):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print("Test Accuracy: {:.2f}%".format(acc * 100))
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))