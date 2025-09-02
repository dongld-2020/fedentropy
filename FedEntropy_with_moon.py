#MNIST ALPHA=0.1
#FASHION MNIST ALHPA=0.3
# MỖI VÒNG CHỌN 5 CLIENTS

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import random
import logging
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import copy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data loading functions
def get_cifar10_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

def get_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

def get_fashion_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

# Non-IID data partitioning using Dirichlet distribution
def non_iid_partition_dirichlet(dataset, client_ids, alpha=0.1):
    num_classes = len(dataset.classes)
    num_clients = len(client_ids)
    class2data = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset):
        class2data[label].append(idx)
    
    for c in range(num_classes):
        random.shuffle(class2data[c])
    
    proportions = np.random.dirichlet([alpha] * num_clients, num_classes)
    client2data = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        num_samples = len(class2data[c])
        start_idx = 0
        for client_idx in range(num_clients):
            num_assigned = int(proportions[c, client_idx] * num_samples)
            client2data[client_idx].extend(class2data[c][start_idx:start_idx + num_assigned])
            start_idx += num_assigned
        if start_idx < num_samples:
            client2data[-1].extend(class2data[c][start_idx:])
    
    for client_idx in range(num_clients):
        random.shuffle(client2data[client_idx])
    
    return client2data, proportions.tolist()

# Split test dataset evenly among clients
def split_test_dataset_evenly(dataset, num_clients):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_size = len(dataset) // num_clients
    client_subsets = [Subset(dataset, indices[i * subset_size:(i + 1) * subset_size]) for i in range(num_clients)]
    if len(dataset) % num_clients:
        client_subsets[-1] = Subset(dataset, indices[(num_clients - 1) * subset_size:])
    return client_subsets

# Create dataloaders for clients
def get_dataloaders(dataset, client2data, batch_size):
    return [DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True) for indices in client2data]

# Model definitions
class VGG9(nn.Module):
    def __init__(self, num_classes=10):
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
        )
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)  # Output dimension 256 for MOON
        )
        self.output_layer = nn.Linear(512, num_classes)

    def forward(self, x, return_representation=False):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        if return_representation:
            return self.projection_head(x)
        return self.output_layer(x)

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.projection_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256)  # Output dimension 256 for MOON
        )
        self.output_layer = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, return_representation=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if return_representation:
            return self.projection_head(x)
        return self.output_layer(x)

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)  # Output dimension 256 for MOON
        )
        self.output_layer = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, return_representation=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        if return_representation:
            return self.projection_head(x)
        return self.output_layer(x)

# Local training functions
def train_local_model_avg(model, dataloader, criterion, optimizer, epochs=1):
    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data, target in dataloader:
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

def train_local_model_prox(global_model, model, dataloader, criterion, optimizer, epochs=1, mu=0.01):
    model.train()
    total_loss = 0.0
    global_state = global_model.state_dict()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            prox_term = 0.0
            for p, p_global in zip(model.parameters(), global_model.parameters()):
                prox_term += (p - p_global).norm(2)
            loss = loss + (mu / 2) * prox_term
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.size(0)
        total_loss += epoch_loss
    avg_loss = total_loss / (len(dataloader.dataset) * epochs)
    return model, avg_loss

def train_local_model_moon(global_model, prev_local_model, model, dataloader, criterion, optimizer, epochs=1, mu=1.0, tau=0.5):
    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            z = model(data, return_representation=True)
            with torch.no_grad():
                z_glob = global_model(data, return_representation=True)
                z_prev = prev_local_model(data, return_representation=True) if prev_local_model is not None else z_glob
            output = model(data)
            loss_sup = criterion(output, target)
            loss_con = model_contrastive_loss(z, z_glob, z_prev, tau=tau)
            loss = loss_sup + mu * loss_con
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.size(0)
        total_loss += epoch_loss
    avg_loss = total_loss / (len(dataloader.dataset) * epochs)
    return model, avg_loss

def model_contrastive_loss(z, z_glob, z_prev, tau=0.5):
    sim_z_zglob = F.cosine_similarity(z, z_glob, dim=1) / tau
    sim_z_zprev = F.cosine_similarity(z, z_prev, dim=1) / tau
    exp_sim_z_zglob = torch.exp(sim_z_zglob)
    exp_sim_z_zprev = torch.exp(sim_z_zprev)
    loss_con = -torch.log(exp_sim_z_zglob / (exp_sim_z_zglob + exp_sim_z_zprev))
    return loss_con.mean()

# Aggregation functions
def aggregate_avg(global_model, client_models, client_data_sizes):
    total_size = sum(client_data_sizes)
    global_state = global_model.state_dict()
    for key in global_state:
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float)
        for i, client_model in enumerate(client_models):
            global_state[key] += client_model.state_dict()[key].float() * (client_data_sizes[i] / total_size)
    global_model.load_state_dict(global_state)
    return global_model

def aggregate_prox(global_model, client_models):
    global_state = global_model.state_dict()
    for key in global_state:
        global_state[key] = torch.mean(torch.stack([client_model.state_dict()[key].float() for client_model in client_models]), dim=0)
    global_model.load_state_dict(global_state)
    return global_model

def aggregate_fedma(global_model, client_models):
    global_state = global_model.state_dict()
    for key in global_state:
        client_params = [client_model.state_dict()[key].float() for client_model in client_models]
        param_shape = client_params[0].shape
        flattened_params = [param.flatten() for param in client_params]
        param_matrix = torch.stack(flattened_params)
        mean_params = torch.mean(param_matrix, dim=0)
        global_state[key] = mean_params.view(param_shape)
    global_model.load_state_dict(global_state)
    return global_model

def aggregate_nolowe(global_model, client_models, client_losses, client_dataloaders, criterion):
    weights = 1 / (np.array(client_losses) + 1e-8)
    weights = weights / np.sum(weights)
    global_state = global_model.state_dict()
    for key in global_state:
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float)
        for i, client_model in enumerate(client_models):
            global_state[key] += client_model.state_dict()[key].float() * weights[i]
    global_model.load_state_dict(global_state)
    client_grads = []
    for idx, dataloader in enumerate(client_dataloaders):
        model = client_models[idx]
        model.eval()
        grad_norm = 0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)
            grad_norm += sum(g.norm().item() for g in grad if g is not None)
        client_grads.append(grad_norm / len(dataloader))
    global_grad = 0
    for data, target in client_dataloaders[0]:
        data, target = data.to(device), target.to(device)
        output = global_model(data)
        loss = criterion(output, target)
        grad = torch.autograd.grad(loss, global_model.parameters(), create_graph=False)
        global_grad += sum(g.norm().item() for g in grad if g is not None)
    global_grad /= len(client_dataloaders[0])
    mean_client_grad = np.mean(client_grads)
    cosine = (mean_client_grad * global_grad) / (np.linalg.norm(mean_client_grad) * np.linalg.norm(global_grad) + 1e-8)
    return global_model, cosine

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

def aggregate_fedentropy(global_model, client_models):
    all_weights = []
    for m in [global_model] + client_models:
        weights = torch.cat([p.data.flatten() for p in m.parameters()]).cpu().numpy()
        all_weights.extend(weights)
    min_w = np.min(all_weights)
    max_w = np.max(all_weights)
    bin_edges = np.linspace(min_w, max_w, num=101)
    
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
        inv_kl = 1 / (1 + kl_divs)
        weights = inv_kl / inv_kl.sum()
    
    weighted_sum = {key: torch.zeros_like(global_model.state_dict()[key], dtype=torch.float) for key in global_model.state_dict()}
    for key in global_model.state_dict():
        for i, cm in enumerate(client_models):
            weighted_sum[key] += cm.state_dict()[key].float() * weights[i]
    global_model.load_state_dict(weighted_sum)
    logger.info(f"FedEntropy weights: {weights}")
    return global_model

# Evaluation function
def evaluate_global_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, pred = output.topk(5, dim=1, largest=True, sorted=True)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:, 0].sum().item()
            correct5 += correct.sum().item()
            total += target.size(0)
            all_preds.extend(pred[:, 0].cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / total
    top1_accuracy = 100. * correct1 / total
    top5_accuracy = 100. * correct5 / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    cm = confusion_matrix(all_targets, all_preds)
    return avg_loss, top1_accuracy, top5_accuracy, precision, recall, f1, cm

# Main federated learning function
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
    
    clients_data, proportions = non_iid_partition_dirichlet(train_data, list(range(num_clients)), alpha=0.1)
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
    prev_client_models = [None] * num_clients  # Store previous round's local models for MOON
    
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
            elif algorithm == 'moon':
                # Ensure prev_local_model is on the correct device and in eval mode
                prev_model = prev_client_models[client_idx]
                if prev_model is not None:
                    prev_model = copy.deepcopy(prev_model).to(device)
                    prev_model.eval()
                client_models[client_idx], client_train_loss = train_local_model_moon(
                    global_model, prev_model, client_models[client_idx],
                    clients_data_loaders[client_idx], criterion, client_optimizer, epochs, mu=0.01, tau=0.5)
                feedback_train_loss[client_idx] = client_train_loss
            
            logger.info(f"Client {client_idx} Train Loss: {client_train_loss:.4f}")
            total_train_loss += client_train_loss
            
            # Store the local model for the next round (for MOON)
            prev_client_models[client_idx] = copy.deepcopy(client_models[client_idx])
        
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
        elif algorithm == 'moon':
            global_model = aggregate_avg(global_model, [client_models[i] for i in selected_clients],
                                        [client_data_sizes[j] for j in selected_clients])
            logger.info("Aggregated using MOON (FedAvg aggregation)")
        
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

if __name__ == "__main__":
    model_name = input("Enter model name (lenet5, cnn, vgg9): ").strip().lower()
    algorithm = input("Enter algorithm (fedavg, fedprox, fednolowe, fedma, fedasl, fedentropy, moon): ").strip().lower()

    if model_name not in ["cnn", "lenet5", "vgg9"]:
        raise ValueError("Invalid model name. Choose 'cnn', 'lenet5', 'vgg9'.")
    if algorithm not in ["fedavg", "fedprox", "fednolowe", "fedma", "fedasl", "fedentropy", "moon"]:
        raise ValueError("Invalid algorithm. Choose 'fedavg', 'fedprox', 'fednolowe', 'fedma', 'fedasl', 'fedentropy', 'moon'.")

    os.makedirs("outcomes", exist_ok=True)
    federated_learning(model_name=model_name, algorithm=algorithm, num_clients=50, num_rounds=50, epochs=2, batch_size=32)