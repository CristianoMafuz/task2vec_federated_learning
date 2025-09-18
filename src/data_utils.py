"""
Script principal para experimentos Task2Vec + Federated Learning
"""

import torch
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms
import medmnist
from medmnist import INFO
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import random
from utils import calculate_entropy

class DirichletPartitioner:
    """Particiona dados usando distribuição de Dirichlet para simular non-IID."""
    
    def __init__(self, alpha: float, num_clients: int, seed: int = 42):
        self.alpha = alpha
        self.num_clients = num_clients
        self.seed = seed
        
    def partition(self, dataset: Dataset, labels: Optional[np.ndarray] = None) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Particiona o dataset usando Dirichlet.
        
        Returns:
            client_indices: Lista de índices para cada cliente
            metadata: Metadados sobre o particionamento
        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # Extrair labels se não fornecidos
        if labels is None:
            # A linha abaixo pode ser lenta para datasets grandes.
            # O ideal é que o dataset já retorne o label no formato correto.
            raw_labels = [dataset[i][1] for i in range(len(dataset))]

            # Converte qualquer formato (tensor, numpy array, etc.) para int
            labels = np.array([
                lbl.item() if hasattr(lbl, 'item') else int(lbl) for lbl in raw_labels
            ])
        
        num_classes = len(np.unique(labels))
        num_samples = len(labels)
        
        # Agrupar índices por classe
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[int(label)].append(idx)
        
        # Gerar distribuição de Dirichlet para cada cliente
        client_distributions = np.random.dirichlet([self.alpha] * num_classes, self.num_clients)
        
        # Alocar amostras para cada cliente
        client_indices = [[] for _ in range(self.num_clients)]
        client_class_counts = [defaultdict(int) for _ in range(self.num_clients)]
        
        for class_id in range(num_classes):
            class_idx = class_indices[class_id]
            np.random.shuffle(class_idx)
            
            # Calcular quantas amostras cada cliente recebe desta classe
            proportions = client_distributions[:, class_id]
            proportions = proportions / proportions.sum()
            client_samples = (proportions * len(class_idx)).astype(int)
            
            # Ajustar para garantir que todas as amostras sejam alocadas
            diff = len(class_idx) - client_samples.sum()
            if diff > 0:
                # Adicionar amostras extras aos clientes com mais proporção
                top_clients = np.argsort(proportions)[-diff:]
                for c in top_clients:
                    client_samples[c] += 1
            elif diff < 0:
                # Remover amostras dos clientes com menos proporção
                bottom_clients = np.argsort(proportions)[:abs(diff)]
                for c in bottom_clients:
                    if client_samples[c] > 0:
                        client_samples[c] -= 1
            
            # Distribuir amostras
            start_idx = 0
            for client_id, num_samples_client in enumerate(client_samples):
                if num_samples_client > 0:
                    end_idx = start_idx + num_samples_client
                    selected_indices = class_idx[start_idx:end_idx]
                    client_indices[client_id].extend(selected_indices)
                    client_class_counts[client_id][class_id] = num_samples_client
                    start_idx = end_idx
        
        # Embaralhar índices de cada cliente
        for indices in client_indices:
            np.random.shuffle(indices)
        
        # Calcular metadados
        metadata = {
            "alpha": self.alpha,
            "num_clients": self.num_clients,
            "num_classes": num_classes,
            "total_samples": num_samples,
            "seed": self.seed,
            "client_stats": []
        }
        
        for client_id, indices in enumerate(client_indices):
            client_labels = labels[indices]
            unique_classes, counts = np.unique(client_labels, return_counts=True)
            
            client_stat = {
                "client_id": client_id,
                "num_samples": len(indices),
                "num_classes": len(unique_classes),
                "class_distribution": dict(zip(unique_classes.tolist(), counts.tolist())),
                "entropy": calculate_entropy(client_labels),
                "dominant_class": int(unique_classes[np.argmax(counts)]),
                "dominant_class_ratio": float(np.max(counts) / len(indices)) if len(indices) > 0 else 0
            }
            metadata["client_stats"].append(client_stat)
        
        return client_indices, metadata



class MedMNISTWrapper(Dataset):
    """
    Wrapper para datasets MedMNIST para garantir que os rótulos 
    sejam retornados como escalares (inteiros), e não arrays.
    """
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        # Pega a imagem e o rótulo original
        image, label = self.original_dataset[index]
        
        # Converte o rótulo de array (ex: array([6])) para um inteiro (ex: 6)
        if isinstance(label, np.ndarray):
            label = label.item()
        
        # Garante que seja um tensor long, que é o padrão esperado pelo PyTorch
        return image, torch.tensor(label, dtype=torch.long)


class DatasetManager:
    """Gerencia o carregamento e preparação de diferentes datasets."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        
    def load_dataset(self, dataset_name: str, split: str = "train") -> Tuple[Dataset, transforms.Compose]:
        """
        Carrega um dataset específico com as transformações apropriadas.
        
        Args:
            dataset_name: Nome do dataset (cifar10, femnist, pathmnist, etc.)
            split: 'train' ou 'test'
            
        Returns:
            dataset: O dataset carregado
            transform: As transformações aplicadas
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name == "cifar10":
            return self._load_cifar10(split)
        elif dataset_name == "femnist":
            return self._load_femnist(split)
        elif dataset_name in ["pathmnist", "bloodmnist", "organamnist", "dermamnist"]:
            return self._load_medmnist(dataset_name, split)
        else:
            raise ValueError(f"Dataset {dataset_name} não suportado")
    
    def _load_cifar10(self, split: str) -> Tuple[Dataset, transforms.Compose]:
        """Carrega CIFAR-10."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        is_train = split == "train"
        dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=is_train,
            download=True,
            transform=transform
        )
        
        return dataset, transform
    
    def _load_femnist(self, split: str) -> Tuple[Dataset, transforms.Compose]:
        """Carrega FEMNIST (usando EMNIST como proxy)."""
        # FEMNIST é similar ao EMNIST, usaremos EMNIST-letters como aproximação
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Converter para 3 canais
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        is_train = split == "train"
        dataset = datasets.EMNIST(
            root=self.data_dir,
            split='letters',
            train=is_train,
            download=True,
            transform=transform
        )
        
        return dataset, transform
    
    def _load_medmnist(self, dataset_name: str, split: str) -> Tuple[Dataset, transforms.Compose]:
        """Carrega um dataset do MedMNIST v2."""
        # Mapear nome para classe MedMNIST
        dataset_map = {
            "pathmnist": medmnist.PathMNIST,
            "bloodmnist": medmnist.BloodMNIST,
            "organamnist": medmnist.OrganAMNIST,
            "dermamnist": medmnist.DermaMNIST
        }
        
        DataClass = dataset_map[dataset_name]
        info = INFO[dataset_name]
        
        # Transformações para MedMNIST
        if info['n_channels'] == 1:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Converter para 3 canais
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        dataset = DataClass(
            split=split,
            transform=transform,
            download=True,
            root=self.data_dir
        )
        
        return dataset, transform
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Retorna informações sobre o dataset."""
        dataset_name = dataset_name.lower()
        
        if dataset_name == "cifar10":
            return {
                "num_classes": 10,
                "input_channels": 3,
                "image_size": (32, 32),
                "target_metric": "accuracy"
            }
        elif dataset_name == "femnist":
            return {
                "num_classes": 62,  # Letters A-Z
                "input_channels": 3,  # Convertemos para 3
                "image_size": (28, 28),
                "target_metric": "accuracy"
            }
        elif dataset_name in ["pathmnist", "bloodmnist", "organamnist", "dermamnist"]:
            info = INFO[dataset_name]
            return {
                "num_classes": len(info['label']),
                "input_channels": 3,  # Sempre convertemos para 3
                "image_size": (28, 28),
                "target_metric": "auc_macro",
                "task": info['task']
            }
        else:
            raise ValueError(f"Dataset {dataset_name} não suportado")

class FederatedDataManager:
    """Gerencia dados federados para múltiplas federações."""
    
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        
    def create_federated_datasets(
        self,
        dataset_name: str,
        num_clients: int,
        alpha: float,
        num_federations: int,
        base_seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Cria múltiplas federações independentes.
        
        Returns:
            Lista de federações, cada uma contendo:
            - client_datasets: Lista de Subsets para cada cliente
            - metadata: Metadados do particionamento
            - federation_seed: Seed usado para esta federação
        """
        # Carregar dataset base
        train_dataset, _ = self.dataset_manager.load_dataset(dataset_name, "train")
        test_dataset, _ = self.dataset_manager.load_dataset(dataset_name, "test")
        
        # Se for um dataset MedMNIST, aplica o wrapper para corrigir os rótulos
        if dataset_name.lower() in ["pathmnist", "bloodmnist", "organamnist", "dermamnist"]:
            train_dataset = MedMNISTWrapper(train_dataset)
            test_dataset = MedMNISTWrapper(test_dataset)

        federations = []
        
        for fed_id in range(num_federations):
            # Seed única para cada federação
            federation_seed = base_seed + fed_id * 1000
            
            # Particionar dados
            partitioner = DirichletPartitioner(alpha, num_clients, federation_seed)
            client_indices, partition_metadata = partitioner.partition(train_dataset)
                        
            federation = {
                "federation_id": fed_id,
                "federation_seed": federation_seed,
                "client_indices": client_indices,
                "train_dataset_ref": train_dataset, # Referência para o dataset completo
                "test_dataset": test_dataset,
                "partition_metadata": partition_metadata,
                "dataset_info": self.dataset_manager.get_dataset_info(dataset_name)
            }
            
            federations.append(federation)
        
        return federations
    
    def create_dataloaders(
        self,
        client_datasets: List[Subset],
        test_dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 2
    ) -> Tuple[List[DataLoader], DataLoader]:
        """
        Cria DataLoaders para treinamento federado.
        
        Returns:
            train_loaders: Lista de DataLoaders para cada cliente
            test_loader: DataLoader para teste global
        """
        train_loaders = [
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            for dataset in client_datasets
        ]
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 2,  # Batch maior para teste
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loaders, test_loader