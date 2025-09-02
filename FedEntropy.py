#Lenet MNIST alpha =0.1
#CNN    FashionMNIST alpha = 0.2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import random
import math
from statistics import median, mean
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch.nn.utils as nn_utils
from scipy.optimize import linear_sum_assignment
from scipy import stats
from torch.utils.data import Subset
import logging
import datetime
import os

# Configure logging
log_dir = "outcomes"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, f"federated_learning_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the VGG9 model 
class VGG9(nn.Module):
    def __init__(self):
        super(VGG9, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Define the LeNet5 model 
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class VGG9_Ma(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG9_Ma, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 47, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(47)
        self.conv1_2 = nn.Conv2d(47, 79, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(79)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(79, 143, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(143)
        self.conv2_2 = nn.Conv2d(143, 143, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(143)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(143, 271, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(271)
        self.conv3_2 = nn.Conv2d(271, 271, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(271)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(271 * 4 * 4, 4336)
        self.bn_fc1 = nn.BatchNorm1d(4336)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4336, 527)
        self.bn_fc2 = nn.BatchNorm1d(527)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(527, 527)
        self.bn_fc3 = nn.BatchNorm1d(527)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(527, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

def evaluate_global_model(model, test_loader, criterion, top_k=(1, 5)):
    model.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred_top1 = output.argmax(dim=1, keepdim=True)
            correct_top1 += pred_top1.eq(target.view_as(pred_top1)).sum().item()
            if 5 in top_k:
                _, pred_top5 = output.topk(5, dim=1, largest=True, sorted=True)
                correct_top5 += pred_top5.eq(target.view(-1, 1).expand_as(pred_top5)).sum().item()
            all_preds.extend(pred_top1.squeeze().cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    top1_accuracy = 100.0 * correct_top1 / len(test_loader.dataset)
    top5_accuracy = 100.0 * correct_top5 / len(test_loader.dataset) if 5 in top_k else None
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=1)
    cm = confusion_matrix(all_targets, all_preds)

    return test_loss, top1_accuracy, top5_accuracy, precision, recall, f1, cm

def non_iid_partition_dirichlet(dataset, arr_number_of_client, partition="hetero", alpha=0.3):
    num_clients = len(arr_number_of_client)
    y_train = np.array(dataset.targets)
    N = y_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(N)
        batch_idxs = np.array_split(idxs, num_clients)
        clients_data = {arr_number_of_client[i]: batch_idxs[i].tolist() for i in range(num_clients)}
        proportions = [1.0 / num_clients] * num_clients
    elif partition == "hetero":
        K = len(np.unique(y_train))
        min_size = 0
        while min_size < 10:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, split_points))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
        clients_data = {arr_number_of_client[i]: idx_batch[i] for i in range(num_clients)}
        proportions = [len(client_data) / N for client_data in idx_batch]

    return clients_data, proportions

def get_mnist_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_data, test_data

def get_fashion_mnist_data():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    return train_data, test_data

def get_dataloaders(train_data, clients_data, batch_size):
    dataloaders = {}
    for client_id, data_idxs in clients_data.items():
        client_dataset = torch.utils.data.Subset(train_data, data_idxs)
        dataloaders[client_id] = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
    return dataloaders

def get_cifar10_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    return train_data, test_data

def train_local_model_avg(model, dataloader, criterion, optimizer, epochs=1):
    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.size(0)
        total_loss += epoch_loss
    avg_loss = total_loss / (len(dataloader.dataset) * epochs)
    return model, avg_loss

def train_local_model_prox(global_model, model, dataloader, criterion, optimizer, epochs=1, mu=0.001):
    model.train()
    total_loss = 0.0
    global_params = {name: param.detach().clone().to(device) for name, param in global_model.named_parameters()}
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            prox_term = 0.0
            for name, param in model.named_parameters():
                prox_term += ((param - global_params[name]) ** 2).sum()
            loss += (mu / 2) * prox_term
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.size(0)
        total_loss += epoch_loss
    avg_loss = total_loss / (len(dataloader.dataset) * epochs)
    return model, avg_loss

def aggregate_avg(global_model, client_models, client_data_sizes):
    client_data_sizes = np.array(client_data_sizes, dtype=np.float64)
    total_client_data_sizes = client_data_sizes.sum()
    weighted_sum = {key: torch.zeros_like(global_model.state_dict()[key], dtype=torch.float) for key in global_model.state_dict().keys()}
    for key in global_model.state_dict().keys():
        for i in range(len(client_models)):
            weighted_sum[key] += client_models[i].state_dict()[key].float() * client_data_sizes[i] / total_client_data_sizes
    global_model.load_state_dict(weighted_sum)
    return global_model

def aggregate_prox(global_model, client_models):
    weighted_sum = {key: torch.zeros_like(global_model.state_dict()[key], dtype=torch.float) for key in global_model.state_dict().keys()}
    for key in global_model.state_dict().keys():
        for i in range(len(client_models)):
            weighted_sum[key] += client_models[i].state_dict()[key].float()
        weighted_sum[key] = weighted_sum[key] / len(client_models)
    global_model.load_state_dict(weighted_sum)
    return global_model

def aggregate_fedma(global_model, client_models):
    global_state = global_model.state_dict()
    new_state = {}
    num_clients = len(client_models)
    for key in global_state.keys():
        client_params = [client_models[i].state_dict()[key].float() for i in range(num_clients)]
        if ("fc3" in key or "classifier" in key) or (client_params[0].dim() < 2):
            avg = sum(client_params) / num_clients
            new_state[key] = avg
        else:
            ref = client_params[0].clone().detach()
            n = ref.shape[0]
            ref_flat = ref.view(n, -1)
            aggregated = ref_flat.clone()
            count = 1
            for candidate_tensor in client_params[1:]:
                candidate_flat = candidate_tensor.view(n, -1)
                cost = torch.cdist(candidate_flat, ref_flat, p=2).cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost)
                permuted = torch.zeros_like(candidate_flat)
                for j in range(n):
                    idx = np.where(col_ind == j)[0]
                    if len(idx) > 0:
                        candidate_idx = row_ind[idx[0]]
                        permuted[j] = candidate_flat[candidate_idx]
                    else:
                        permuted[j] = candidate_flat[j]
                aggregated += permuted
                count += 1
            aggregated = aggregated / count
            new_state[key] = aggregated.view(client_params[0].shape)
    global_model.load_state_dict(new_state)
    return global_model

def aggregate_asl(global_model, client_models, client_train_losses, alpha=0.5, beta=0.2):
    losses = np.array(client_train_losses, dtype=np.float64)
    med_loss = np.median(losses)
    sigma = np.std(losses)
    if sigma == 0:
        weights = np.ones(len(losses)) / len(losses)
    else:
        d = []
        for L in losses:
            if (L >= med_loss - alpha * sigma) and (L <= med_loss + alpha * sigma):
                d.append(beta * sigma)
            else:
                d.append(abs(med_loss - L))
        logger.info(f"FedASL distances: {d}")
        d = np.array(d, dtype=np.float64)
        d = d / d.sum()
        inv_d = 1.0 / d
        weights = inv_d / inv_d.sum()
    logger.info(f"FedASL weights: {weights}")
    weighted_sum = {key: torch.zeros_like(global_model.state_dict()[key], dtype=torch.float) for key in global_model.state_dict().keys()}
    for key in global_model.state_dict().keys():
        for i in range(len(client_models)):
            weighted_sum[key] += client_models[i].state_dict()[key].float() * weights[i]
    global_model.load_state_dict(weighted_sum)
    return global_model

def aggregate_nolowe(global_model, client_models, feedback_train_losses, clients_data_loaders, criterion):
    logger.info(f"Clients train loss: {feedback_train_losses}")
    feedback_train_losses = np.array(feedback_train_losses, dtype=np.float64)
    if feedback_train_losses.sum() == 0:
        logger.warning("Clients_train_loss sum to 0. Assigning equal weights.")
        feedback_train_losses = np.ones(len(feedback_train_losses)) / len(feedback_train_losses)
    else:
        feedback_train_losses /= feedback_train_losses.sum()
    feedback_train_losses = 1 - feedback_train_losses
    feedback_train_losses /= feedback_train_losses.sum()
    logger.info(f"Normalized clients train loss: {feedback_train_losses}")
    client_grad_vectors = []
    for model, dataloader in zip(client_models, clients_data_loaders):
        grad_vector = get_gradient_vector(model, dataloader, criterion)
        client_grad_vectors.append(grad_vector)
    global_grad_vector = sum([client_grad_vectors[i] * feedback_train_losses[i] for i in range(len(client_models))])
    weighted_sum = {key: torch.zeros_like(global_model.state_dict()[key], dtype=torch.float) for key in global_model.state_dict().keys()}
    for key in global_model.state_dict().keys():
        for i in range(len(client_models)):
            weighted_sum[key] += client_models[i].state_dict()[key].float() * feedback_train_losses[i]
    global_model.load_state_dict(weighted_sum)
    cosine_similarities = []
    for grad_vector in client_grad_vectors:
        cos_sim = F.cosine_similarity(grad_vector.unsqueeze(0), global_grad_vector.unsqueeze(0), dim=1).item()
        cosine_similarities.append(cos_sim)
    logger.info(f"Cosine similarities: {cosine_similarities}")
    return global_model, mean(cosine_similarities)

def get_gradient_vector(model, dataloader, criterion):
    model.zero_grad()
    model.train()
    data, target = next(iter(dataloader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    grad_vector = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).detach().to(device)
    return grad_vector

def aggregate_fedentropy(global_model, client_models):
    # Collect all weights to determine bin edges
    all_weights = []
    for m in [global_model] + client_models:
        weights = torch.cat([p.data.flatten() for p in m.parameters()]).cpu().numpy()
        all_weights.extend(weights)
    min_w = np.min(all_weights)
    max_w = np.max(all_weights)
    bin_edges = np.linspace(min_w, max_w, num=101)  # 100 bins

    def get_probs(model):
        weights = torch.cat([p.data.flatten() for p in model.parameters()]).cpu().numpy()
        counts, _ = np.histogram(weights, bins=bin_edges)
        probs = counts / counts.sum().astype(float)
        probs = np.where(probs == 0, 1e-12, probs)
        probs /= probs.sum()
        return probs

    probs_global = get_probs(global_model)
    kl_divs = []
    for cm in client_models:
        probs_c = get_probs(cm)
        kl = stats.entropy(probs_c, probs_global)
        kl_divs.append(kl)

    kl_divs = np.array(kl_divs)
    kl_divs[np.isinf(kl_divs)] = np.max(kl_divs[~np.isinf(kl_divs)]) * 10 if np.any(~np.isinf(kl_divs)) else 1.0

    if np.all(kl_divs == 0):
        weights = np.ones(len(client_models)) / len(client_models)
    else:
        #inv_kl = 1 / (1 + kl_divs)
        #inv_kl = 1 - kl_divs
        #weights = inv_kl / inv_kl.sum()
        normalized_kl = kl_divs / kl_divs.sum()
        inv_kl = 1-normalized_kl
        weights = inv_kl / inv_kl.sum()

    weighted_sum = {key: torch.zeros_like(global_model.state_dict()[key], dtype=torch.float) for key in global_model.state_dict()}
    for key in global_model.state_dict():
        for i, cm in enumerate(client_models):
            weighted_sum[key] += cm.state_dict()[key].float() * weights[i]
    global_model.load_state_dict(weighted_sum)
    logger.info(f"FedEntropy weights: {weights}")
    return global_model

def compute_cosine_similarity(model1, model2):
    params1 = torch.cat([p.flatten() for p in model1.parameters()]).detach().to(device)
    params2 = torch.cat([p.flatten() for p in model2.parameters()]).detach().to(device)
    cos_sim = torch.nn.functional.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0), dim=1)
    return cos_sim.item()

def split_test_dataset_evenly(dataset, num_clients):
    labels = np.array(dataset.targets)
    class_indices = {}
    for class_label in np.unique(labels):
        class_indices[class_label] = np.where(labels == class_label)[0]
    clients_data = [[] for _ in range(num_clients)]
    for class_label, indices in class_indices.items():
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, num_clients)
        for client_idx in range(num_clients):
            clients_data[client_idx].extend(split_indices[client_idx].tolist())
    clients_subsets = [Subset(dataset, client_indices) for client_indices in clients_data]
    return clients_subsets

def federated_learning(model_name='lenet5', algorithm='fedavg', num_clients=5, num_rounds=10, epochs=1, batch_size=32):
    set_seed(42)
    logger.info(f"Starting federated learning with model: {model_name}, algorithm: {algorithm}, clients: {num_clients}, rounds: {num_rounds}")
    
    if model_name == 'vgg9':
        train_data, test_data = get_cifar10_data()
    elif model_name == 'cnn':
        train_data, test_data = get_fashion_mnist_data()
    elif model_name == 'lenet5':
        train_data, test_data = get_mnist_data()
    
    global_test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    clients_data, proportions = non_iid_partition_dirichlet(train_data, list(range(num_clients)), alpha=0.2)
    logger.info(f"Initial data proportions assigned to clients: {proportions}")
    
    clients_data_loaders = get_dataloaders(train_data, clients_data, batch_size)
    
    clients_subsets = split_test_dataset_evenly(test_data, num_clients)
    clients_test_loaders = [DataLoader(subset, batch_size=32, shuffle=False) for subset in clients_subsets]
    
    if model_name == 'vgg9':
        global_model = VGG9().to(device)
        client_models = [VGG9().to(device) for _ in range(num_clients)]
    elif model_name == 'cnn':
        global_model = CNN().to(device)
        client_models = [CNN().to(device) for _ in range(num_clients)]
    elif model_name == 'lenet5':
        global_model = LeNet5().to(device)
        client_models = [LeNet5().to(device) for _ in range(num_clients)]
    
    client_optimizers = [optim.SGD(client_models[i].parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3) for i in range(num_clients)]
    criterion = nn.CrossEntropyLoss()
    
    client_data_sizes = [len(clients_data[i]) for i in range(num_clients)]
    feedback_train_loss = [0.0 for _ in range(num_clients)]
    
    round_cosine_similarities = []
    global_train_losses = []
    global_validation_losses = []
    global_top1_accuracies = []
    global_top5_accuracies = []
    global_precisions = []
    global_recalls = []
    global_f1_scores = []
    number_of_participants = []
    clients_id_per_round = []
    
    for round_num in range(num_rounds):
        logger.info(f"Starting Round {round_num + 1}/{num_rounds} with algorithm {algorithm} and model {model_name}")
        
        selected_clients = random.sample(range(num_clients), 5)
        logger.info(f"Selected clients for Round {round_num + 1}: {selected_clients}")
        number_of_participants.append(len(selected_clients))
        clients_id_per_round.append(selected_clients)
        
        for client_idx in selected_clients:
            client_models[client_idx].load_state_dict(global_model.state_dict())
        
        total_train_loss = 0
        client_gradients = []
        
        for client_idx in selected_clients:
            client_optimizer = client_optimizers[client_idx]
            
            if algorithm in ['fedavg', 'fedma', 'fedasl', 'fedentropy']:
                client_models[client_idx], client_train_loss = train_local_model_avg(
                    client_models[client_idx], clients_data_loaders[client_idx], criterion, client_optimizer, epochs)
                if algorithm in ['fednolowe', 'fedasl']:
                    feedback_train_loss[client_idx] = client_train_loss
            elif algorithm == 'fedprox':
                client_models[client_idx], client_train_loss = train_local_model_prox(
                    global_model, client_models[client_idx], clients_data_loaders[client_idx], criterion, client_optimizer, epochs)
            elif algorithm == 'fednolowe':
                client_models[client_idx], client_train_loss = train_local_model_avg(
                    client_models[client_idx], clients_data_loaders[client_idx], criterion, client_optimizer, epochs)
                feedback_train_loss[client_idx] = client_train_loss
            
            logger.info(f"Client {client_idx} Train Loss: {client_train_loss:.4f}")
            total_train_loss += client_train_loss
        
        avg_train_loss = total_train_loss / len(selected_clients)
        global_train_losses.append(avg_train_loss)
        logger.info(f"Global Average Train Loss: {avg_train_loss:.4f}")
        
        if algorithm == 'fedavg':
            global_model = aggregate_avg(global_model, [client_models[i] for i in selected_clients],
                                        [client_data_sizes[j] for j in selected_clients])
            logger.info("Aggregated using FedAvg")
        elif algorithm == 'fednolowe':
            global_model, cosine = aggregate_nolowe(
                global_model,
                [client_models[i] for i in selected_clients],
                [feedback_train_loss[j] for j in selected_clients],
                [clients_data_loaders[i] for i in selected_clients],
                criterion
            )
            round_cosine_similarities.append(cosine)
            logger.info(f"Round cosine similarity between mean clients gradient and global gradient: {cosine:.4f}")
        elif algorithm == 'fedprox':
            global_model = aggregate_prox(global_model, [client_models[i] for i in selected_clients])
            logger.info("Aggregated using FedProx")
        elif algorithm == 'fedma':
            global_model = aggregate_fedma(global_model, [client_models[i] for i in selected_clients])
            logger.info("Aggregated using FedMa")
        elif algorithm == 'fedasl':
            global_model = aggregate_asl(global_model, [client_models[i] for i in selected_clients],
                                        [feedback_train_loss[j] for j in selected_clients])
            logger.info("Aggregated using FedASL")
        elif algorithm == 'fedentropy':
            global_model = aggregate_fedentropy(global_model, [client_models[i] for i in selected_clients])
            logger.info("Aggregated using FedEntropy")
        
        val_loss, top1_accuracy, top5_accuracy, precision, recall, f1, cm = evaluate_global_model(
            global_model, global_test_loader, criterion)
        
        global_validation_losses.append(val_loss)
        global_top1_accuracies.append(top1_accuracy)
        global_top5_accuracies.append(top5_accuracy)
        global_precisions.append(precision)
        global_recalls.append(recall)
        global_f1_scores.append(f1)
        
        logger.info(f"Global Model Validation Loss: {val_loss:.4f}, Top1_Accuracy: {top1_accuracy:.2f}%, Top5_Accuracy: {top5_accuracy:.2f}%")
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")

    if algorithm == 'fednolowe' and round_cosine_similarities:
        cos_values = round_cosine_similarities
        mean_cos = np.mean(cos_values) if cos_values else 0.0
        std_cos = np.std(cos_values) if cos_values else 0.0
        logger.info(f"Average Cosine Similarity: {mean_cos:.4f} ± {std_cos:.4f}")
    
    rounds = list(range(num_rounds))
    df_metrics = pd.DataFrame({
        'Round': rounds,
        'Global Train Loss': global_train_losses,
        'Global Validation Loss': global_validation_losses,
        'Global Top1 Accuracy (%)': global_top1_accuracies,
        'Global Top5 Accuracy (%)': global_top5_accuracies,
        'Precision': global_precisions,
        'Recall': global_recalls,
        'F1-Score': global_f1_scores,
        'Participants': number_of_participants,
        'Clients': clients_id_per_round,
    })
    
    csv_name = f"outcomes/metrics_{model_name}_{algorithm}_dong_clients{num_clients}_rounds{num_rounds}_clientsperround{len(selected_clients)}.csv"
    df_metrics.to_csv(csv_name, index=False)
    logger.info(f"Metrics saved to {csv_name}")
    
    plt.figure(figsize=(12, 8))
    plt.plot(rounds, global_train_losses, label='Client Average Train Loss', linewidth=3)
    plt.plot(rounds, global_validation_losses, label='Global Validation Loss', linewidth=3)
    plt.title(f"{model_name}_{algorithm}_Global Train Loss vs Validation Loss", fontsize=25)
    plt.xlabel("Rounds", fontsize=25)
    plt.ylabel("Loss", fontsize=25)
    plt.xticks(rounds[::10], fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    fig_name = f"outcomes/Loss_{model_name}_{algorithm}_clients{num_clients}_rounds{num_rounds}_clientsperround{len(selected_clients)}.png"
    plt.savefig(fig_name)
    plt.close()
    logger.info(f"Loss plot saved to {fig_name}")
    
    plt.figure(figsize=(12, 8))
    plt.plot(rounds, global_top1_accuracies, label='Global Top1 Accuracy', linewidth=3)
    plt.plot(rounds, global_top5_accuracies, label='Global Top5 Accuracy', linewidth=3)
    plt.title(f"{model_name}_{algorithm}_Global Accuracy Comparation", fontsize=25)
    plt.xlabel("Rounds", fontsize=25)
    plt.ylabel("Accuracy (%)", fontsize=25)
    plt.xticks(rounds[::10], fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    fig_name = f"outcomes/Accuracy_{model_name}_{algorithm}_clients{num_clients}_rounds{num_rounds}_clientsperround{len(selected_clients)}.png"
    plt.savefig(fig_name)
    plt.close()
    logger.info(f"Accuracy plot saved to {fig_name}")

model_name = input("Enter model name (lenet5, cnn, vgg9): ").strip().lower()
algorithm = input("Enter algorithm (fedavg, fedprox, fednolowe, fedma, fedasl, fedentropy): ").strip().lower()

if model_name not in ["cnn", "lenet5", "vgg9"]:
    raise ValueError("Invalid model name. Choose 'cnn', 'lenet5', 'vgg9'.")
if algorithm not in ["fedavg", "fedprox", "fednolowe", "fedma", "fedasl", "fedentropy"]:
    raise ValueError("Invalid algorithm. Choose 'fedavg', 'fedprox', 'fednolowe', 'fedma', 'fedasl', 'fedentropy'.")

federated_learning(model_name=model_name, algorithm=algorithm, num_clients=50, num_rounds=100, epochs=2, batch_size=32)