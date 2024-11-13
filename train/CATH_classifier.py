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
    """
    Computes class-balanced weights using the effective number of samples for each class.
    
    Args:
        labels (list or array): List of labels for each sample.
        num_classes (int): Total number of classes.
        beta (float): Hyperparameter for smoothing. Set to close to 1 (e.g., 0.9999).
    
    Returns:
        weights (torch.Tensor): Tensor of weights for each class.
    """
    counts = np.bincount(labels, minlength=num_classes)  # Get counts for each class
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum()  # Normalize weights to sum to 1
    
    return torch.tensor(weights, dtype=torch.float32)

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

def generate_and_save_embeddings(esm_model, dataloader, save_path="embeddings.pt"):
    esm_model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for coords, labels, padding_mask, confidence in tqdm(dataloader, desc="Generating embeddings"):
            coords = coords.to("cuda")
            padding_mask = padding_mask.to("cuda")
            confidence = confidence.to("cuda")
            
            encoder_out = esm_model.encoder(coords, padding_mask, confidence, return_all_hiddens=False)
            embeddings = encoder_out['encoder_out'][0].transpose(0, 1)
            embeddings = embeddings * (~padding_mask.unsqueeze(-1))
            non_padding_count = (~padding_mask).sum(dim=1, keepdim=True)
            mean_embeddings = embeddings.sum(dim=1) / non_padding_count

            embeddings_list.append(mean_embeddings.cpu())
            labels_list.append(labels)

    torch.save({
        "embeddings": torch.cat(embeddings_list),
        "labels": {task: torch.cat([lbl[task] for lbl in labels_list]) for task in labels_list[0]}
    }, save_path)
    print(f"Embeddings saved to {save_path}")

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

class MLPHeader(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPHeader, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

def train_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for embeddings, labels in tqdm(dataloader, desc="Training"):
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(embeddings)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, criterion, num_classes):
    model.eval()
    total_loss, correct = 0, 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            output = model(embeddings)
            loss = criterion(output, labels)
            total_loss += loss.item()

            preds = output.argmax(1)
            correct += (preds == labels).sum().item()
            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1

    accuracy = correct / len(dataloader.dataset)
    class_accuracy = {cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0 
                      for cls in range(num_classes)}
    return total_loss / len(dataloader), accuracy, class_accuracy

if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Argument parser for selecting loss type and task
    parser = ArgumentParser(description="Train a CATH classifier with selectable loss functions and tasks")
    parser.add_argument("--loss_type", type=str, choices=["cross_entropy", "focal", "class_balanced"], default="cross_entropy", 
                        help="Select loss function: cross_entropy, focal, or class_balanced")
    parser.add_argument("--task", type=str, choices=["c", "a", "t", "h"], required=True, 
                        help="Select task: c, a, t, or h for different classification levels")
    args = parser.parse_args()

    # Define number of classes based on task
    task_to_num_classes = {"c": 5, "a": 43, "t": 1472, "h": 6631}
    num_classes = task_to_num_classes[args.task]
    task = args.task

    # Load ESM model and set up paths
    device = "cuda"
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.to(device)

    split_file = "../data/CATH_DATA/split.json"
    label_file = "../data/CATH_DATA/all.json"
    data_dir = "../data/CATH_DATA/dompdb"
    lab2idx = "../data/CATH_DATA/lab2idx.json"
    batch_size = 8

    # Load or generate embeddings as before
    train_embeddings_path = f"train_embeddings.pt"
    valid_embeddings_path = f"valid_embeddings.pt"
    test_embeddings_path = f"test_embeddings.pt"

    if not os.path.exists(train_embeddings_path):
        print("Generating training embeddings...")
        train_dataset = CATHDataset(split_file, label_file, data_dir, alphabet, lab2idx, split="train")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collect_fn, shuffle=False)
        generate_and_save_embeddings(model, train_loader, save_path=train_embeddings_path)
    else:
        print(f"Training embeddings found at {train_embeddings_path}, loading directly.")

    if not os.path.exists(valid_embeddings_path):
        print("Generating validation embeddings...")
        valid_dataset = CATHDataset(split_file, label_file, data_dir, alphabet, lab2idx, split="valid")
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collect_fn, shuffle=False)
        generate_and_save_embeddings(model, valid_loader, save_path=valid_embeddings_path)
    else:
        print(f"Validation embeddings found at {valid_embeddings_path}, loading directly.")

    if not os.path.exists(test_embeddings_path):
        print("Generating test embeddings...")
        test_dataset = CATHDataset(split_file, label_file, data_dir, alphabet, lab2idx, split="test")
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collect_fn, shuffle=False)
        generate_and_save_embeddings(model, test_loader, save_path=test_embeddings_path)
    else:
        print(f"Test embeddings found at {test_embeddings_path}, loading directly.")

    # Load precomputed embeddings for training, validation, and test
    train_dataset = EmbeddingDataset(train_embeddings_path, task=task)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = EmbeddingDataset(valid_embeddings_path, task=task)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = EmbeddingDataset(test_embeddings_path, task=task)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute class weights if using Class-Balanced Loss
    if args.loss_type == "class_balanced":
        labels = [label for _, label in train_dataset]  # Collect all labels from train dataset
        class_weights = compute_class_weights(labels, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == "focal":
        criterion = FocalLoss(alpha=1, gamma=2)
    
    checkpoint_dir = f"./checkpoints_{args.loss_type}_{task}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    mlp_header = MLPHeader(input_dim=512, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(mlp_header.parameters(), lr=1e-4)
    best_model_path = os.path.join(checkpoint_dir, f"best_model_{args.loss_type}_{task}.pth")
    last_model_path = os.path.join(checkpoint_dir, f"last_model_{args.loss_type}_{task}.pth")

    # Training loop with validation
    epochs = 200
    best_accuracy = 0

    for epoch in range(epochs):
        train_loss = train_epoch(mlp_header, train_loader, optimizer, device, criterion)
        valid_loss, valid_accuracy, class_accuracy = evaluate(mlp_header, valid_loader, device, criterion, num_classes=num_classes)

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")
        #print("Per-Class Accuracy:")
        #for cls, acc in class_accuracy.items():
            #print(f"  Class {cls}: {acc:.4f}")

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(mlp_header.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")

    torch.save(mlp_header.state_dict(), last_model_path)
    print(f"Saved last model to {last_model_path}")

    test_loss, test_accuracy, test_class_accuracy = evaluate(mlp_header, test_loader, device, criterion, num_classes=num_classes)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    #print("Per-Class Accuracy:")
    #for cls, acc in test_class_accuracy.items():
        #print(f"  Class {cls}: {acc:.4f}")
