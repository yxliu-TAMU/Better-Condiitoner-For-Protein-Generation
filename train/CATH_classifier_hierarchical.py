import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import json
from collections import defaultdict
import numpy as np
from argparse import ArgumentParser

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# Class-Balanced Loss weight computation
def compute_class_weights(labels, num_classes, beta=0.9999):
    counts = np.bincount(labels, minlength=num_classes)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum()  
    return torch.tensor(weights, dtype=torch.float32)

# Dataset with hierarchical labels
class EmbeddingDataset(Dataset):
    def __init__(self, embedding_file, tasks):
        data = torch.load(embedding_file)
        self.embeddings = data["embeddings"]
        self.labels = {task: data["labels"][task] for task in tasks}
        self.tasks = tasks

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        labels = {task: self.labels[task][idx] for task in self.tasks}
        return self.embeddings[idx], labels

# Custom collate function for multi-level labels
def collect_fn(batch):
    embeddings, labels_list = zip(*batch)
    embeddings = torch.stack(embeddings)
    labels = {task: torch.stack([label[task] for label in labels_list]) for task in labels_list[0].keys()}
    return embeddings, labels

# Hierarchical MLP Header
class HierarchicalMLPHeader(nn.Module):
    def __init__(self, input_dim, num_classes_per_level):
        super(HierarchicalMLPHeader, self).__init__()
        self.num_levels = len(num_classes_per_level)
        self.mlp_layers = nn.ModuleList()
        self.hierarchy_hidden = nn.ModuleList()
        self.output_mlp = nn.ModuleList()
        self.prefix_dim = 32

        for i, num_classes in enumerate(num_classes_per_level):
            if i == 0:
                current_dim = input_dim
            else:
                current_dim = input_dim + self.prefix_dim
            
            self.mlp_layers.append(
                nn.Sequential(
                    nn.Linear(current_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128)
                )
            )
            
            if i != 0:
                self.hierarchy_hidden.append(
                    nn.Sequential(
                        nn.Linear(num_classes_per_level[i-1], self.prefix_dim),
                        nn.ReLU()
                    )
                )
            
            self.output_mlp.append(nn.Linear(128, num_classes))

    def forward(self, x):
        pred_outputs = []
        for i in range(self.num_levels):
            if i == 0:
                current_input = x
            else:
                prefix = self.hierarchy_hidden[i-1](pred_outputs[-1])
                current_input = torch.cat([x, prefix], dim=1)
            
            hidden = self.mlp_layers[i](current_input)
            pred_output = self.output_mlp[i](hidden)
            pred_outputs.append(pred_output)
        
        return pred_outputs

# Training epoch function
def train_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for embeddings, labels in tqdm(dataloader, desc="Training"):
        embeddings = embeddings.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings)
        
        loss = sum(criterion[i](outputs[i], labels[f"{task}"].to(device)) for i, task in enumerate(labels.keys()))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, device, criterion, num_classes_per_level):
    model.eval()
    total_loss = 0
    correct_per_level = [0] * len(num_classes_per_level)
    total_samples = len(dataloader.dataset)

    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            outputs = model(embeddings)
            
            loss = sum(criterion[i](outputs[i], labels[f"{task}"].to(device)) 
                       for i, task in enumerate(labels.keys()))
            total_loss += loss.item()

            for i, output in enumerate(outputs):
                preds = output.argmax(1)
                correct_per_level[i] += (preds == labels[f"{list(labels.keys())[i]}"].to(device)).sum().item()
    
    accuracy_per_level = [correct / total_samples for correct in correct_per_level]
    return total_loss / len(dataloader), accuracy_per_level

# Main training script
if __name__ == "__main__":
    parser = ArgumentParser(description="Train a hierarchical CATH classifier")
    parser.add_argument("--loss_type", type=str, choices=["cross_entropy", "focal", "class_balanced"], default="cross_entropy",
                        help="Loss function: cross_entropy, focal, or class_balanced")
    args = parser.parse_args()

    with open("../data/CATH_DATA/lab2idx.json", "r") as f:
        label2idx = json.load(f)
    num_classes_per_level = [len(labels) for labels in label2idx.values()]
    print(f"Number of classes per level: {num_classes_per_level}")
    #num_classes_per_level = [5, 43, 1470, 6576]
    task_names = ["c", "a", "t", "h"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hierarchical_model = HierarchicalMLPHeader(input_dim=512, num_classes_per_level=num_classes_per_level).to(device)

    if args.loss_type == "class_balanced":
        labels = [label for _, label in train_dataset]
        class_weights = [compute_class_weights([lbl[task] for _, lbl in train_dataset], num_classes)
                         for task, num_classes in zip(task_names, num_classes_per_level)]
        criterion = [nn.CrossEntropyLoss(weight=weights.to(device)) for weights in class_weights]
    elif args.loss_type == "cross_entropy":
        criterion = [nn.CrossEntropyLoss() for _ in num_classes_per_level]
    elif args.loss_type == "focal":
        criterion = [FocalLoss(alpha=1, gamma=2) for _ in num_classes_per_level]

    optimizer = torch.optim.Adam(hierarchical_model.parameters(), lr=1e-4)
    checkpoint_dir = f"./checkpoints_hierarchical_{args.loss_type}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    batch_size = 8
    train_dataset = EmbeddingDataset("train_embeddings.pt", task_names)
    valid_dataset = EmbeddingDataset("valid_embeddings.pt", task_names)
    test_dataset = EmbeddingDataset("test_embeddings.pt", task_names)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collect_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collect_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collect_fn)

    epochs = 200
    best_accuracy = 0
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    last_model_path = os.path.join(checkpoint_dir, "last_model.pth")

    for epoch in range(epochs):
        train_loss = train_epoch(hierarchical_model, train_loader, optimizer, device, criterion)
        valid_loss, valid_accuracy = evaluate(hierarchical_model, valid_loader, device, criterion, num_classes_per_level)

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        print("Per-Level Validation Accuracy:")
        for i, acc in enumerate(valid_accuracy):
            print(f"  Level {i+1} (Classes: {num_classes_per_level[i]}): {acc:.4f}")

        if sum(valid_accuracy) > best_accuracy:
            best_accuracy = sum(valid_accuracy)
            torch.save(hierarchical_model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")

    torch.save(hierarchical_model.state_dict(), last_model_path)
    print(f"Saved last model to {last_model_path}")

    test_loss, test_accuracy = evaluate(hierarchical_model, test_loader, device, criterion, num_classes_per_level)
    print(f"Test Loss: {test_loss:.4f}")
    print("Per-Level Test Accuracy:")
    for i, acc in enumerate(test_accuracy):
        print(f"  Level {i+1} (Classes: {num_classes_per_level[i]}): {acc:.4f}")
