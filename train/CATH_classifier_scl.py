import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import os
import warnings
import esm
from esm.inverse_folding.util import CoordBatchConverter, load_structure, extract_coords_from_structure
from biotite.structure import get_chains
import torch.multiprocessing as mp
from collections import defaultdict, Counter
import numpy as np
from argparse import ArgumentParser

# Suppress warnings from biotite
warnings.filterwarnings("ignore", category=UserWarning, module="biotite")
class CATHDataset(Dataset):
    def __init__(self, split_file, label_file, data_dir, alphabet, lab2idx, split='train'):
        with open(split_file) as f:
            self.split_data = json.load(f)[split]
        with open(label_file) as f:
            self.labels = json.load(f)
        with open(lab2idx) as f:
            self.lab2idx = json.load(f)
        self.data_dir = data_dir
        self.batch_converter = CoordBatchConverter(alphabet)

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, idx):
        sample = self.split_data[idx]
        fpath = os.path.join(self.data_dir, f"{sample}.pdb")
        structure = load_structure(fpath)
        chain_id = get_chains(structure)
        
        if len(chain_id) != 1:
            return self.__getitem__((idx + 1) % len(self.split_data))
        
        chain_id = chain_id[0]
        
        try:
            coords, _ = extract_coords_from_structure(structure)
        except KeyError as e:
            return self.__getitem__((idx + 1) % len(self.split_data))

        sample_labels = {task: self.lab2idx[task][self.labels[sample][task]] for task in self.lab2idx.keys()}
        
        return coords, sample_labels

    def collect_fn(self, batch):
        coords, labels_list = zip(*batch)
        coord_list = [(coord, None, None) for coord in coords]
        coords, confidence, strs, tokens, padding_mask = self.batch_converter(coord_list, device="cuda")
        
        labels = {task: torch.tensor([label[task] for label in labels_list]) for task in labels_list[0]}
        
        return coords, labels, padding_mask, confidence

class EmbeddingDataset(Dataset):
    def __init__(self, embedding_file, task):
        data = torch.load(embedding_file)
        self.embeddings = data["embeddings"]
        self.labels = data["labels"][task]
        self.task = task

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Suppress warnings from biotite
warnings.filterwarnings("ignore", category=UserWarning, module="biotite")

# Supervised Contrastive Loss
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask = label_matrix.float() - torch.eye(labels.size(0)).to(labels.device)

        positives = similarity_matrix * mask
        negatives = similarity_matrix * (1 - mask)

        pos_exp = torch.exp(positives)
        neg_exp = torch.exp(negatives)

        log_prob = pos_exp / (pos_exp + neg_exp.sum(dim=1, keepdim=True))
        scl_loss = -torch.log(log_prob + 1e-7).mean()  # Avoid log(0)
        return scl_loss

class ProjectedClassifierModel(nn.Module):
    def __init__(self, input_dim, projection_dim, num_classes):
        super(ProjectedClassifierModel, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, projection_dim)
        )
        self.classifier = nn.Linear(projection_dim, num_classes)

    def forward(self, embeddings):
        projected_embeddings = self.projector(embeddings)
        class_output = self.classifier(projected_embeddings)
        return projected_embeddings, class_output

def train_epoch(model, dataloader, optimizer, device, criterion_scl, criterion_ce):
    model.train()
    total_loss, scl_loss_total, ce_loss_total = 0, 0, 0
    for embeddings, labels in tqdm(dataloader, desc="Training"):
        embeddings, labels = embeddings.to(device), labels.to(device)
        
        optimizer.zero_grad()

        projected_embeddings, class_output = model(embeddings)

        scl_loss = criterion_scl(projected_embeddings, labels)
        ce_loss = criterion_ce(class_output, labels)
        loss = scl_loss + ce_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        scl_loss_total += scl_loss.item()
        ce_loss_total += ce_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_scl_loss = scl_loss_total / len(dataloader)
    avg_ce_loss = ce_loss_total / len(dataloader)

    return avg_loss, avg_scl_loss, avg_ce_loss

def evaluate(model, dataloader, device, criterion_ce, num_classes):
    model.eval()
    total_loss, correct = 0, 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            _, class_output = model(embeddings)

            ce_loss = criterion_ce(class_output, labels)
            total_loss += ce_loss.item()

            preds = class_output.argmax(1)
            correct += (preds == labels).sum().item()
            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1

    accuracy = correct / len(dataloader.dataset)
    class_accuracy = {cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0 
                      for cls in range(num_classes)}
    return total_loss / len(dataloader), accuracy, class_accuracy

def save_projected_embeddings(model, dataloader, device, save_path="projected_embeddings.pt"):
    model.eval()
    projected_embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            projected_embeddings, _ = model(embeddings)
            projected_embeddings_list.append(projected_embeddings.cpu())
            labels_list.append(labels)
    
    torch.save({
        "projected_embeddings": torch.cat(projected_embeddings_list),
        "labels": torch.cat(labels_list)
    }, save_path)
    print(f"Projected embeddings saved to {save_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = ArgumentParser(description="Train a CATH classifier with SCL and CE losses")
    parser.add_argument("--task", type=str, choices=["c", "a", "t", "h"], required=True, 
                        help="Select task: c, a, t, or h for different classification levels")
    args = parser.parse_args()

    task_to_num_classes = {"c": 5, "a": 43, "t": 1472, "h": 6631}
    num_classes = task_to_num_classes[args.task]
    task = args.task

    device = "cuda"
    input_dim = 512  # Adjust if needed based on your embeddings
    projection_dim = 128  # Dimension for projected embeddings

    train_embeddings_path = "train_embeddings.pt"
    valid_embeddings_path = "valid_embeddings.pt"
    test_embeddings_path = "test_embeddings.pt"

    # Load precomputed embeddings for training, validation, and test sets
    train_dataset = EmbeddingDataset(train_embeddings_path, task=task)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_dataset = EmbeddingDataset(valid_embeddings_path, task=task)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
    test_dataset = EmbeddingDataset(test_embeddings_path, task=task)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Initialize projector + classifier model, optimizer, and loss functions
    model = ProjectedClassifierModel(input_dim=input_dim, projection_dim=projection_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion_scl = SupervisedContrastiveLoss(temperature=0.1)
    criterion_ce = nn.CrossEntropyLoss()

    # Checkpoint setup
    checkpoint_dir = f"./checkpoints_{task}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"best_model_{task}.pth")
    last_model_path = os.path.join(checkpoint_dir, f"last_model_{task}.pth")
    
    # Training loop with validation
    best_accuracy = 0
    epochs = 200
    for epoch in range(epochs):
        avg_loss, avg_scl_loss, avg_ce_loss = train_epoch(
            model, train_loader, optimizer, device, criterion_scl, criterion_ce
        )
        valid_loss, valid_accuracy, class_accuracy = evaluate(
            model, valid_loader, device, criterion_ce, num_classes=num_classes
        )

        print(f"Epoch {epoch}, Total Loss: {avg_loss:.4f}, SCL Loss: {avg_scl_loss:.4f}, CE Loss: {avg_ce_loss:.4f}")
        print(f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}")
        print("Per-Class Accuracy:")
        for cls, acc in class_accuracy.items():
            print(f"  Class {cls}: {acc:.4f}")

        # Save best model based on validation accuracy
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")

    # Save last model
    torch.save(model.state_dict(), last_model_path)
    print(f"Saved last model to {last_model_path}")

    # Final evaluation on test set with the best model
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_accuracy, test_class_accuracy = evaluate(
        model, test_loader, device, criterion_ce, num_classes=num_classes
    )
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print("Per-Class Accuracy (Test):")
    for cls, acc in test_class_accuracy.items():
        print(f"  Class {cls}: {acc:.4f}")

    # Save projected embeddings for visualization
    save_projected_embeddings(model, train_loader, device, save_path="projected_train_embeddings.pt")
