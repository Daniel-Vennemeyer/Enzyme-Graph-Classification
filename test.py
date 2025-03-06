import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from model import GCN, ENZYMESDataset

def test_model():

    # Define your dataset and test loader
    num_classes = 6  # For the ENZYMES dataset
    dataset = ENZYMESDataset(root='ENZYMES')
    test_dataset = dataset[int(len(dataset) * 0.8):]
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Path to the saved model
    saved_model_path = "gcn_model_142.pth"

    # Load the saved model
    model = GCN(num_node_features=test_loader.dataset[0].x.size(1), num_classes=num_classes)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    auc = roc_auc_score(
        np.eye(num_classes)[all_labels], all_probs, average='macro', multi_class='ovr'
    )

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    return all_labels, all_probs, accuracy, f1, auc


def plot_auc(all_labels, all_probs, num_classes=6):
    # One-hot encode labels
    one_hot_labels = np.eye(num_classes)[all_labels]

    # Compute ROC curves
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(one_hot_labels[:, i], np.array(all_probs)[:, i])
        plt.plot(fpr, tpr, label=f"Class {i + 1} (AUC)")

    # Plot the random guess line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC-ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Test the model
all_labels, all_probs, accuracy, f1, auc = test_model()
# Plot the AUC-ROC curve
plot_auc(all_labels, all_probs)