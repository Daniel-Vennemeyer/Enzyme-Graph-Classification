import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Data, Dataset
import os
import numpy as np

class ENZYMESDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ENZYMESDataset, self).__init__(root, transform, pre_transform)
        self.process()

    @property
    def raw_file_names(self):
        return [
            "ENZYMES_A.txt",
            "ENZYMES_graph_indicator.txt",
            "ENZYMES_graph_labels.txt",
            "ENZYMES_node_labels.txt",
            "ENZYMES_node_attributes.txt"
        ]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        if not os.path.isfile("ENZYMES/processed/data.pt"):
            # Load raw files
            edges = np.loadtxt(os.path.join(self.raw_dir, "ENZYMES_A.txt"), delimiter=",", dtype=int)
            graph_indicators = np.loadtxt(os.path.join(self.raw_dir, "ENZYMES_graph_indicator.txt"), dtype=int)
            graph_labels = np.loadtxt(os.path.join(self.raw_dir, "ENZYMES_graph_labels.txt"), dtype=int)
            node_labels = np.loadtxt(os.path.join(self.raw_dir, "ENZYMES_node_labels.txt"), dtype=int)
            node_attributes = np.loadtxt(os.path.join(self.raw_dir, "ENZYMES_node_attributes.txt"), delimiter=",")

            # Preprocess data
            data_list = []
            num_graphs = graph_labels.shape[0]

            for graph_id in range(1, num_graphs + 1):
                # Get node indices for the current graph
                node_indices = np.where(graph_indicators == graph_id)[0]
                graph_node_labels = node_labels[node_indices]
                graph_node_attributes = node_attributes[node_indices]

                # Get edges for the current graph
                graph_edges = edges[np.isin(edges[:, 0], node_indices) & np.isin(edges[:, 1], node_indices)]
                # Get edges for the current graph
                graph_edges = edges[np.isin(edges[:, 0], node_indices) & np.isin(edges[:, 1], node_indices)]

                if graph_edges.size > 0:  # Check if edges exist
                    graph_edges -= graph_edges.min()  # Re-index edges starting from 0  # Re-index edges starting from 0

                # Create PyTorch Geometric Data object
                x = torch.tensor(graph_node_attributes, dtype=torch.float)
                edge_index = torch.tensor(graph_edges.T, dtype=torch.long)
                y = torch.tensor([graph_labels[graph_id - 1] - 1], dtype=torch.long)  # Graph label (0-indexed)

                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)
            np.random.shuffle(data_list)

            # Save processed data
            torch.save(data_list, os.path.join(self.processed_dir, "data.pt"))

    def len(self):
        return len(torch.load(os.path.join(self.processed_dir, "data.pt")))

    def get(self, idx):
        data_list = torch.load(os.path.join(self.processed_dir, "data.pt"))
        return data_list[idx]


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph convolutional layers with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global mean pooling to obtain graph-level features
        x = global_mean_pool(x, batch)

        # Fully connected layer for classification
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)


def main():
    # Load ENZYMES dataset
    dataset = ENZYMESDataset(root='ENZYMES')

    # Split the dataset into training and testing
    train_dataset = dataset[:int(len(dataset) * 0.8)]
    test_dataset = dataset[int(len(dataset) * 0.8):]

    # Load the data into loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = GCN(num_node_features=dataset[0].x.size(1), num_classes=6).to(device)

    # Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(200):
        train_loss = train(model, train_loader, optimizer, criterion)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
        if (epoch+1) % 2 == 0:
            torch.save(model.state_dict(), f'saved_models/gcn_model_{epoch+1}.pth')

if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()