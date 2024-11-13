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
from collections import defaultdict
import numpy as np
from argparse import ArgumentParser

# Suppress warnings from biotite
warnings.filterwarnings("ignore", category=UserWarning, module="biotite")

# Hierarchical Supervised Contrastive Loss with Conditional Masking
class HierarchicalSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, layer_weights=None):
        super(HierarchicalSupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.layer_weights = layer_weights if layer_weights else [1.0, 0.5, 0.25, 0.125]

    def forward(self, embeddings, *label_layers):
        embeddings = F.normalize(embeddings, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        total_loss = 0.0

        for layer, labels in enumerate(label_layers):
            # Generate mask that respects hierarchical structure
            mask = self._create_conditional_mask(labels, label_layers[:layer])
            positives = similarity_matrix * mask
            negatives = similarity_matrix * (1 - mask)

            pos_exp = torch.exp(positives)
            neg_exp = torch.exp(negatives)

            log_prob = pos_exp / (pos_exp + neg_exp.sum(dim=1, keepdim=True))
            layer_loss = -torch.log(log_prob + 1e-7).mean()
            total_loss += self.layer_weights[layer] * layer_loss
        return total_loss

    def _create_conditional_mask(self, labels, parent_labels):
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        for parent_label in parent_labels:
            mask *= torch.eq(parent_label.unsqueeze(1), parent_label.unsqueeze(0)).float()
        mask -= torch.eye(labels.size(0)).to(labels.device)
        return mask

class MultiLevelClassifierModel(nn.Module):
    def __init__(self, projection_dim, num_classes_dict):
        super(MultiLevelClassifierModel, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, projection_dim)
        )
        self.classifier_c = nn.Linear(projection_dim, num_classes_dict["c"])
        self.classifier_a = nn.Linear(projection_dim, num_classes_dict["a"])
        self.classifier_t = nn.Linear(projection_dim, num_classes_dict["t"])
        self.classifier_h = nn.Linear(projection_dim, num_classes_dict["h"])

    def forward(self, embeddings):
        # Project embeddings
        projected_embeddings = self.projector(embeddings)
        
        # Generate classification outputs for each hierarchical level
        class_output_c = self.classifier_c(projected_embeddings)
        class_output_a = self.classifier_a(projected_embeddings)
        class_output_t = self.classifier_t(projected_embeddings)
        class_output_h = self.classifier_h(projected_embeddings)
        
        return projected_embeddings, class_output_c, class_output_a, class_output_t, class_output_h


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
    def __init__(self, embedding_file):
        data = torch.load(embedding_file)
        self.embeddings = data["embeddings"]
        self.labels = {task: data["labels"][task] for task in ["c", "a", "t", "h"]}

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], {task: self.labels[task][idx] for task in ["c", "a", "t", "h"]}

def train_epoch(model, dataloader, optimizer, device, criterion_scl, criterion_ce):
    model.train()
    total_loss, scl_loss_total, ce_loss_total = 0, 0, 0
    for embeddings, labels in tqdm(dataloader, desc="Training"):
        embeddings = embeddings.to(device)
        label_layers = [labels["c"].to(device), labels["a"].to(device), labels["t"].to(device), labels["h"].to(device)]
        
        optimizer.zero_grad()

        projected_embeddings, output_c, output_a, output_t, output_h = model(embeddings)

        # Hierarchical SCL loss across all 4 levels
        scl_loss = criterion_scl(projected_embeddings, *label_layers)
        ce_loss_c = criterion_ce(output_c, labels["c"].to(device))
        ce_loss_a = criterion_ce(output_a, labels["a"].to(device))
        ce_loss_t = criterion_ce(output_t, labels["t"].to(device))
        ce_loss_h = criterion_ce(output_h, labels["h"].to(device))
        ce_loss = ce_loss_c + ce_loss_a + ce_loss_t + ce_loss_h

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

def evaluate(model, dataloader, device, criterion_ce, num_classes_dict):
    model.eval()
    total_loss, correct_c, correct_a, correct_t, correct_h = 0, 0, 0, 0, 0
    class_correct = {key: defaultdict(int) for key in num_classes_dict}
    class_total = {key: defaultdict(int) for key in num_classes_dict}

    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            label_c = labels["c"].to(device)
            label_a = labels["a"].to(device)
            label_t = labels["t"].to(device)
            label_h = labels["h"].to(device)
            _, output_c, output_a, output_t, output_h = model(embeddings)

            ce_loss = (
                criterion_ce(output_c, label_c) +
                criterion_ce(output_a, label_a) +
                criterion_ce(output_t, label_t) +
                criterion_ce(output_h, label_h)
            )
            total_loss += ce_loss.item()

            # Calculate accuracies
            preds_c = output_c.argmax(1)
            preds_a = output_a.argmax(1)
            preds_t = output_t.argmax(1)
            preds_h = output_h.argmax(1)

            correct_c += (preds_c == label_c).sum().item()
            correct_a += (preds_a == label_a).sum().item()
            correct_t += (preds_t == label_t).sum().item()
            correct_h += (preds_h == label_h).sum().item()

            for label, pred, task in zip([label_c, label_a, label_t, label_h],
                                         [preds_c, preds_a, preds_t, preds_h],
                                         ["c", "a", "t", "h"]):
                for lbl, prd in zip(label, pred):
                    class_total[task][lbl.item()] += 1
                    if lbl == prd:
                        class_correct[task][lbl.item()] += 1

    accuracy_c = correct_c / len(dataloader.dataset)
    accuracy_a = correct_a / len(dataloader.dataset)
    accuracy_t = correct_t / len(dataloader.dataset)
    accuracy_h = correct_h / len(dataloader.dataset)

    class_accuracy = {task: {cls: class_correct[task][cls] / class_total[task][cls] if class_total[task][cls] > 0 else 0 
                             for cls in range(num_classes_dict[task])} for task in num_classes_dict}
    
    return (total_loss / len(dataloader), 
            {"c": accuracy_c, "a": accuracy_a, "t": accuracy_t, "h": accuracy_h}, 
            class_accuracy)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = ArgumentParser(description="Train a hierarchical CATH classifier with SCL and CE losses")
    args = parser.parse_args()
    batch_size=8
    num_classes_dict = {"c": 5, "a": 43, "t": 1472, "h": 6631}

    device = "cuda"
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.to(device)
    model.eval()

    train_embeddings_path = "train_embeddings.pt"
    valid_embeddings_path = "valid_embeddings.pt"
    test_embeddings_path = "test_embeddings.pt"

    if not os.path.exists(train_embeddings_path):
        train_dataset = CATHDataset(split_file, label_file, data_dir, alphabet, lab2idx, split="train")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collect_fn, shuffle=False)
        generate_and_save_embeddings(model, train_loader, save_path=train_embeddings_path)

    train_dataset = EmbeddingDataset(train_embeddings_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = EmbeddingDataset(valid_embeddings_path)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = EmbeddingDataset(test_embeddings_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    hierarchical_model = MultiLevelClassifierModel(projection_dim=128, num_classes_dict=num_classes_dict).to(device)
    optimizer = torch.optim.Adam(hierarchical_model.parameters(), lr=1e-4)
    criterion_scl = HierarchicalSupervisedContrastiveLoss(temperature=0.1)
    criterion_ce = nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(200):
        avg_loss, avg_scl_loss, avg_ce_loss = train_epoch(
            hierarchical_model, train_loader, optimizer, device, criterion_scl, criterion_ce
        )
        valid_loss, valid_accuracy, class_accuracy = evaluate(
            hierarchical_model, valid_loader, device, criterion_ce, num_classes_dict
        )

        print(f"Epoch {epoch}, Total Loss: {avg_loss:.4f}, SCL Loss: {avg_scl_loss:.4f}, CE Loss: {avg_ce_loss:.4f}")
        print("Validation Accuracies:", valid_accuracy)

    print("Final test evaluation with best model")
    test_loss, test_accuracy, test_class_accuracy = evaluate(
        hierarchical_model, test_loader, device, criterion_ce, num_classes_dict
    )
    print("Test Accuracies:", test_accuracy)
