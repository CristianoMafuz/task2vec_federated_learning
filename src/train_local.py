#!/usr/bin/env python3
"""
Script de Treinamento Local para ResNet34 em Múltiplos Datasets
================================================================

Este script treina um modelo ResNet34 localmente em quatro datasets:
- CIFAR-10
- FEMNIST
- PathMNIST
- BloodMNIST

Instalação de Dependências:
---------------------------
pip install torch torchvision medmnist numpy tqdm scikit-learn

Como Executar:
-------------
# Treinar em todos os datasets
python train_local.py --datasets all

# Treinar em dataset específico
python train_local.py --datasets cifar10
python train_local.py --datasets pathmnist
python train_local.py --datasets bloodmnist

# Treinar em múltiplos datasets específicos
python train_local.py --datasets cifar10,pathmnist

Autor: Baseado em código de aprendizado federado
Data: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from typing import Dict, Tuple, List, Any
import argparse
import os
import logging
from tqdm import tqdm
import warnings
from datetime import datetime

# Tentar importar medmnist
try:
    import medmnist
    from medmnist import INFO
    MEDMNIST_AVAILABLE = True
except ImportError:
    MEDMNIST_AVAILABLE = False
    warnings.warn("medmnist não está instalado. PathMNIST e BloodMNIST não estarão disponíveis.")

from models import get_model

# Configurar reprodutibilidade
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Detectar dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🔧 Dispositivo detectado: {DEVICE}")


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
            split: 'train', 'test', ou 'val'
            
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
        if not MEDMNIST_AVAILABLE:
            raise ImportError(f"medmnist não está instalado. Instale com: pip install medmnist")
            
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
        
        # Aplicar wrapper para corrigir rótulos MedMNIST
        dataset = MedMNISTWrapper(dataset)
        
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
            if not MEDMNIST_AVAILABLE:
                raise ImportError(f"medmnist não está instalado para obter info do {dataset_name}")
            
            info = INFO[dataset_name]
            return {
                "num_classes": len(info['label']),
                "input_channels": 3,  # Sempre convertemos para 3
                "image_size": (28, 28),
                "target_metric": "accuracy",  # Simplificado para treinamento local
                "task": info['task']
            }
        else:
            raise ValueError(f"Dataset {dataset_name} não suportado")


def get_dataloaders(
    dataset_name: str,
    batch_size: int = 32,
    data_root: str = './data',
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Carrega e prepara os dataloaders para um dataset específico usando DatasetManager.
    
    Args:
        dataset_name: Nome do dataset ('cifar10', 'femnist', 'pathmnist', 'bloodmnist')
        batch_size: Tamanho do batch
        data_root: Diretório raiz para armazenar dados
        val_split: Proporção de dados para validação
        
    Returns:
        Tupla (train_loader, val_loader, test_loader)
    """
    logger.info(f"📊 Carregando dataset: {dataset_name}")
    
    # Criar dataset manager
    dataset_manager = DatasetManager(data_root)
    
    # Carregar datasets
    if dataset_name.lower() in ["pathmnist", "bloodmnist", "organamnist", "dermamnist"]:
        # MedMNIST já tem split de validação
        train_dataset, _ = dataset_manager.load_dataset(dataset_name, "train")
        val_dataset, _ = dataset_manager.load_dataset(dataset_name, "val")
        test_dataset, _ = dataset_manager.load_dataset(dataset_name, "test")
        
        # Criar DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"  ✅ Carregado: {len(train_dataset)} treino, {len(val_dataset)} val, {len(test_dataset)} teste")
        return train_loader, val_loader, test_loader
        
    else:
        # Para CIFAR-10 e FEMNIST, fazer split manual
        train_dataset, _ = dataset_manager.load_dataset(dataset_name, "train")
        test_dataset, _ = dataset_manager.load_dataset(dataset_name, "test")
        
        # Dividir dataset de treino em treino/validação
        train_size = int((1 - val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_subset, val_subset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(RANDOM_SEED)
        )
        
        # Criar DataLoaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"  ✅ Carregado: {train_size} treino, {val_size} val, {len(test_dataset)} teste")
        return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Treina o modelo por uma época.
    
    Args:
        model: Modelo a treinar
        loader: DataLoader de treino
        optimizer: Otimizador
        criterion: Função de loss
        device: Dispositivo (CPU/GPU)
        epoch: Número da época atual
        
    Returns:
        Tupla (loss médio, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Usar tqdm para barra de progresso
    pbar = tqdm(loader, desc=f'Época {epoch:02d} - Treino', leave=False)
    
    for batch_idx, (data, target) in enumerate(pbar):
        # Mover dados para dispositivo
        data, target = data.to(device), target.to(device)
        
        # Para MedMNIST, targets podem vir como (batch_size, 1)
        if len(target.shape) > 1:
            target = target.squeeze()
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Estatísticas
        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Atualizar barra de progresso
        current_loss = running_loss / total
        current_acc = 100. * correct / total
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Avaliação"
) -> Tuple[float, float]:
    """
    Avalia o modelo em um conjunto de dados.
    
    Args:
        model: Modelo a avaliar
        loader: DataLoader
        criterion: Função de loss
        device: Dispositivo (CPU/GPU)
        desc: Descrição para a barra de progresso
        
    Returns:
        Tupla (loss médio, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # Para MedMNIST, targets podem vir como (batch_size, 1)
            if len(target.shape) > 1:
                target = target.squeeze()
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Atualizar barra de progresso
            current_loss = running_loss / total
            current_acc = 100. * correct / total
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


class Trainer:
    """Classe para encapsular o loop de treinamento e avaliação."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.01,
        num_epochs: int = 20
    ):
        """
        Inicializa o Trainer.
        
        Args:
            model: Modelo a treinar
            device: Dispositivo (CPU/GPU)
            learning_rate: Taxa de aprendizado
            num_epochs: Número de épocas
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Configurar otimizador e critério
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Para rastrear melhor modelo
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dataset_name: str
    ) -> Dict[str, List[float]]:
        """
        Treina o modelo completo.
        
        Args:
            train_loader: DataLoader de treino
            val_loader: DataLoader de validação
            dataset_name: Nome do dataset (para salvar modelo)
            
        Returns:
            Dicionário com histórico de métricas
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        logger.info(f"\n🚀 Iniciando treinamento para {dataset_name}")
        logger.info(f"   Configuração: {self.num_epochs} épocas, LR={self.learning_rate}")
        
        for epoch in range(1, self.num_epochs + 1):
            # Treinar por uma época
            train_loss, train_acc = train_one_epoch(
                self.model,
                train_loader,
                self.optimizer,
                self.criterion,
                self.device,
                epoch
            )
            
            # Avaliar no conjunto de validação
            val_loss, val_acc = evaluate(
                self.model,
                val_loader,
                self.criterion,
                self.device,
                f"Época {epoch:02d} - Validação"
            )
            
            # Armazenar histórico
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Log de métricas
            logger.info(
                f"  Época {epoch:02d}/{self.num_epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
            
            # Salvar melhor modelo
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_model(dataset_name)
                logger.info(f"  💾 Novo melhor modelo salvo! (Val Acc: {val_acc:.4f})")
        
        logger.info(
            f"\n✅ Treinamento concluído! Melhor validação: {self.best_val_acc:.4f} "
            f"(época {self.best_epoch})"
        )
        
        return history
    
    def save_model(self, dataset_name: str):
        """Salva o state_dict do modelo."""
        filename = f"best_{dataset_name}.pth"
        torch.save(self.model.state_dict(), filename)
        logger.info(f"  💾 Modelo salvo em: {filename}")
    
    def test(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Testa o modelo no conjunto de teste.
        
        Args:
            test_loader: DataLoader de teste
            
        Returns:
            Tupla (test_loss, test_accuracy)
        """
        test_loss, test_acc = evaluate(
            self.model,
            test_loader,
            self.criterion,
            self.device,
            "Teste Final"
        )
        
        logger.info(f"\n📊 Resultados no conjunto de teste:")
        logger.info(f"   Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        
        return test_loss, test_acc


def main():
    """Função principal do script."""
    parser = argparse.ArgumentParser(
        description='Treina ResNet34 em múltiplos datasets'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default='all',
        help='Datasets para treinar (all|cifar10|femnist|pathmnist|bloodmnist ou lista separada por vírgula)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Número de épocas (padrão: 20)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamanho do batch (padrão: 32)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Taxa de aprendizado (padrão: 0.01)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data',
        help='Diretório raiz para dados (padrão: ./data)'
    )
    
    args = parser.parse_args()
    
    # Determinar quais datasets treinar
    available_datasets = ['cifar10', 'femnist']
    if MEDMNIST_AVAILABLE:
        available_datasets.extend(['pathmnist', 'bloodmnist'])
    
    if args.datasets == 'all':
        datasets_to_train = available_datasets
        if not MEDMNIST_AVAILABLE:
            logger.warning("⚠️  MedMNIST datasets não disponíveis (medmnist não instalado)")
    else:
        datasets_to_train = [d.strip() for d in args.datasets.split(',')]
        # Verificar se todos os datasets estão disponíveis
        for dataset in datasets_to_train:
            if dataset not in available_datasets:
                if dataset in ['pathmnist', 'bloodmnist'] and not MEDMNIST_AVAILABLE:
                    logger.error(f"❌ Dataset {dataset} requer medmnist. Instale com: pip install medmnist")
                else:
                    logger.error(f"❌ Dataset {dataset} não suportado")
                datasets_to_train.remove(dataset)
    
    # Criar diretório para dados se não existir
    os.makedirs(args.data_root, exist_ok=True)
    
    # Criar dataset manager
    dataset_manager = DatasetManager(args.data_root)
    
    # Treinar em cada dataset
    results = {}
    
    for dataset_name in datasets_to_train:
        logger.info(f"\n{'='*60}")
        logger.info(f"DATASET: {dataset_name.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            # Obter informações do dataset
            dataset_info = dataset_manager.get_dataset_info(dataset_name)
            
            # Carregar dados
            train_loader, val_loader, test_loader = get_dataloaders(
                dataset_name,
                batch_size=args.batch_size,
                data_root=args.data_root
            )
            
            # Criar modelo
            model = get_model(
                model_name="resnet34",
                num_classes=dataset_info['num_classes']
            )
            
            # Criar trainer
            trainer = Trainer(
                model=model,
                device=DEVICE,
                learning_rate=args.lr,
                num_epochs=args.epochs
            )
            
            # Treinar
            history = trainer.train(train_loader, val_loader, dataset_name)
            
            # Testar
            test_loss, test_acc = trainer.test(test_loader)
            
            # Armazenar resultados
            results[dataset_name] = {
                'history': history,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'best_val_acc': trainer.best_val_acc,
                'best_epoch': trainer.best_epoch,
                'dataset_info': dataset_info
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao treinar {dataset_name}: {str(e)}")
            continue
    
    # Resumo final
    logger.info(f"\n{'='*60}")
    logger.info("RESUMO FINAL")
    logger.info(f"{'='*60}")
    
    for dataset_name, result in results.items():
        logger.info(f"\n📊 {dataset_name.upper()}:")
        logger.info(f"   Melhor Val Acc: {result['best_val_acc']:.4f} (época {result['best_epoch']})")
        logger.info(f"   Test Acc: {result['test_acc']:.4f}")
        logger.info(f"   Modelo salvo: best_{dataset_name}.pth")
        logger.info(f"   Classes: {result['dataset_info']['num_classes']}")
    
    logger.info(f"\n✨ Treinamento completo! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()