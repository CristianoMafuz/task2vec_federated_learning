"""
Script principal para experimentos Task2Vec + Federated Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import copy
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import sys
warnings.filterwarnings('ignore')

try:
    from task2vec import Task2Vec
except ImportError:
    print("Warning: task2vec não encontrado. Usando implementação simplificada.")
    sys.exit(1)

class Task2VecEmbedding:
    """Wrapper para embeddings Task2Vec com funcionalidades adicionais."""
    
    def __init__(self, embedding_vector: np.ndarray, metadata: Dict[str, Any] = None):
        self.vector = embedding_vector
        self.metadata = metadata or {}
        
    def distance_to(self, other: 'Task2VecEmbedding', metric: str = 'cosine') -> float:
        """Calcula distância para outro embedding."""
        if metric == 'cosine':
            return cosine(self.vector, other.vector)
        elif metric == 'euclidean':
            return np.linalg.norm(self.vector - other.vector)
        else:
            raise ValueError(f"Métrica {metric} não suportada")
    
    def similarity_to(self, other: 'Task2VecEmbedding') -> float:
        """Calcula similaridade de cosseno com outro embedding."""
        return 1 - self.distance_to(other, 'cosine')

class Task2VecAnalyzer:
    """Analisa tarefas usando Task2Vec."""
    
    def __init__(
        self,
        probe_network: nn.Module,
        max_samples: int = 1000,
        skip_layers: int = 6,
        device: Optional[torch.device] = None
    ):
        self.probe_network = probe_network
        self.max_samples = max_samples
        self.skip_layers = skip_layers
        self.device = device or torch.device('cpu')
        
    def extract_embedding(
        self,
        dataset: Dataset,
        client_id: Optional[int] = None
    ) -> Task2VecEmbedding:
        """
        Extrai embedding Task2Vec de um dataset.
        
        Args:
            dataset: Dataset do cliente
            client_id: ID do cliente (para metadados)
            
        Returns:
            Task2VecEmbedding com vetor e metadados
        """
        # Preparar dataset para Task2Vec (precisa de imagens 224x224 com 3 canais)
        transformed_dataset = self._prepare_dataset_for_task2vec(dataset)
        
        # Criar cópia da rede probe para este cliente
        probe = copy.deepcopy(self.probe_network)
        probe = probe.to(self.device)
        
        # Usar implementação oficial do Task2Vec
        task2vec = Task2Vec(probe, max_samples=self.max_samples, skip_layers=self.skip_layers)
        embedding = task2vec.embed(transformed_dataset)
        embedding_vector = embedding.hessian
        
        # Coletar metadados
        metadata = self._collect_metadata(dataset, client_id)
        
        return Task2VecEmbedding(embedding_vector, metadata)
    
    def _prepare_dataset_for_task2vec(self, dataset: Dataset) -> Dataset:
        """Prepara dataset com transformações necessárias para Task2Vec."""
        
        class TransformedDataset(Dataset):
            def __init__(self, original_dataset):
                self.dataset = original_dataset
                
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                img, label = self.dataset[idx]
                
                # Garantir tensor
                if not isinstance(img, torch.Tensor):
                    img = torch.tensor(img)
                
                # Ajustar canais
                if img.dim() == 2:
                    img = img.unsqueeze(0)
                if img.size(0) == 1:
                    img = img.repeat(3, 1, 1)
                elif img.size(0) > 3:
                    img = img[:3]
                
                # Redimensionar para 224x224
                if img.shape[1:] != (224, 224):
                    img = F.interpolate(
                        img.unsqueeze(0),
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                
                return img, label
        
        return TransformedDataset(dataset)
    
    def _compute_fisher_embedding(self, model: nn.Module, dataset: Dataset) -> np.ndarray:
        """
        Implementação simplificada de embedding baseado em informação de Fisher.
        """
        model.eval()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Coletar gradientes
        gradients = []
        num_samples = min(self.max_samples, len(dataset))
        samples_processed = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            if samples_processed >= num_samples:
                break
                
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Coletar gradientes
            batch_gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    batch_gradients.append(param.grad.data.cpu().numpy().flatten())
            
            if batch_gradients:
                gradients.append(np.concatenate(batch_gradients))
            
            samples_processed += data.size(0)
        
        if not gradients:
            # Retornar vetor aleatório se não houver gradientes
            return np.random.randn(1000)
        
        # Calcular diagonal de Fisher (variância dos gradientes)
        gradients = np.array(gradients)
        fisher_diagonal = np.var(gradients, axis=0)
        
        # Reduzir dimensionalidade se necessário
        if len(fisher_diagonal) > 1000:
            # Selecionar as 1000 dimensões com maior variância
            top_indices = np.argsort(fisher_diagonal)[-1000:]
            fisher_diagonal = fisher_diagonal[top_indices]
        elif len(fisher_diagonal) < 1000:
            # Padding com zeros
            fisher_diagonal = np.pad(fisher_diagonal, (0, 1000 - len(fisher_diagonal)))
        
        return fisher_diagonal
    
    def _collect_metadata(self, dataset: Dataset, client_id: Optional[int] = None) -> Dict[str, Any]:
        """Coleta metadados sobre o dataset."""
        # Extrair labels
        labels = []
        for i in range(min(len(dataset), 1000)):  # Limitar para eficiência
            _, label = dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            labels.append(label)
        
        labels = np.array(labels)
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        metadata = {
            'client_id': client_id,
            'num_samples': len(dataset),
            'num_classes': len(unique_labels),
            'class_distribution': dict(zip(unique_labels.tolist(), counts.tolist())),
            'label_entropy': entropy(counts),
            'dominant_class': int(unique_labels[np.argmax(counts)]),
            'dominant_class_ratio': float(np.max(counts) / len(labels))
        }
        
        return metadata

class ReadinessCalculator:
    """Calcula o índice de readiness R(F) para federações."""
    
    def __init__(self, variant: str = "cohesion_dispersion"):
        """
        Args:
            variant: Variante do cálculo ('cohesion_dispersion', 'weighted_sum', 'density')
        """
        self.variant = variant
        
    def calculate(
        self,
        embeddings: List[Task2VecEmbedding],
        dataset_info: Dict[str, Any],
        use_bootstrap: bool = True,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calcula R(F) para uma federação.
        
        Args:
            embeddings: Lista de embeddings dos clientes
            use_bootstrap: Se deve calcular intervalo de confiança
            n_bootstrap: Número de amostras bootstrap
            confidence_level: Nível de confiança para o intervalo
            
        Returns:
            Dicionário com R(F), componentes e opcionalmente intervalo de confiança
        """
        if len(embeddings) < 2:
            return {'readiness': 0.0, 'error': 'Federação muito pequena'}
        
        # Extrair vetores
        vectors = np.array([emb.vector for emb in embeddings])
        
        # Calcular componentes
        components = self._calculate_components(vectors, embeddings, dataset_info)
        
        # Calcular R(F) baseado na variante
        if self.variant == "cohesion_dispersion":
            readiness = components['cohesion'] - components['dispersion']
        elif self.variant == "weighted_sum":
            # Pesos podem ser ajustados baseado em experimentos
            readiness = (
                0.4 * components['cohesion'] +
                0.2 * components['coverage'] +
                0.2 * components['size_factor'] -
                0.2 * components['dispersion']
            )
        elif self.variant == "density":
            readiness = components['density']
        else:
            raise ValueError(f"Variante {self.variant} não suportada")
        
        result = {
            'readiness': float(readiness),
            'components': components,
            'variant': self.variant,
            'num_clients': len(embeddings)
        }
        
        # Calcular intervalo de confiança via bootstrap
        if use_bootstrap and len(embeddings) > 3:
            ci_lower, ci_upper = self._bootstrap_confidence_interval(
                embeddings, dataset_info, n_bootstrap, confidence_level
            )
            result['confidence_interval'] = {
                'lower': float(ci_lower),
                'upper': float(ci_upper),
                'level': confidence_level
            }
        
        return result
    
    def _calculate_components(
        self,
        vectors: np.ndarray,
        embeddings: List[Task2VecEmbedding],
        dataset_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calcula componentes do índice de readiness."""
        n_clients = len(vectors)
        
        # Coesão: similaridade média entre pares
        similarities = cosine_similarity(vectors)
        np.fill_diagonal(similarities, 0)  # Ignorar diagonal
        cohesion = similarities.sum() / (n_clients * (n_clients - 1))
        
        # Dispersão: distância média do centroide
        centroid = vectors.mean(axis=0)
        dispersions = [np.linalg.norm(v - centroid) for v in vectors]
        dispersion = np.mean(dispersions)
        
        # Normalizar dispersão
        max_dispersion = np.sqrt(vectors.shape[1])  # Máximo teórico
        dispersion = dispersion / max_dispersion
        
        # Cobertura de classes
        all_classes = set()
        for emb in embeddings:
            if 'class_distribution' in emb.metadata:
                all_classes.update(emb.metadata['class_distribution'].keys())
        
        
        num_total_classes = dataset_info.get('num_classes', 10) # Pega do dataset_info
        coverage = len(all_classes) / float(num_total_classes)
        
        # Entropia média
        entropies = [
            emb.metadata.get('label_entropy', 0) for emb in embeddings
        ]
        avg_entropy = np.mean(entropies) if entropies else 0
        
        # Fator de tamanho (normalizado por log)
        total_samples = sum(
            emb.metadata.get('num_samples', 0) for emb in embeddings
        )
        
        total_training_samples = dataset_info.get('total_train_samples', 60000) # Pega do dataset_info
        size_factor = np.log1p(total_samples) / np.log1p(total_training_samples)
        
        # Densidade (baseada em kernel RBF)
        bandwidth = np.median(dispersions) if dispersions else 1.0
        density = self._calculate_density(vectors, bandwidth)
        
        return {
            'cohesion': float(cohesion),
            'dispersion': float(dispersion),
            'coverage': float(coverage),
            'avg_entropy': float(avg_entropy),
            'size_factor': float(size_factor),
            'density': float(density)
        }
    
    def _calculate_density(self, vectors: np.ndarray, bandwidth: float) -> float:
        """Calcula densidade usando kernel RBF."""
        n = len(vectors)
        if n < 2 or bandwidth == 0:
            return 0.0
        
        # Calcular matriz de distâncias
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(vectors[i] - vectors[j])
                distances[i, j] = distances[j, i] = dist
        
        # Aplicar kernel RBF
        kernel_values = np.exp(-distances**2 / (2 * bandwidth**2))
        np.fill_diagonal(kernel_values, 0)
        
        # Densidade média
        density = kernel_values.sum() / (n * (n - 1))
        
        return density
    
    def _bootstrap_confidence_interval(
        self,
        embeddings: List[Task2VecEmbedding],
        dataset_info: Dict[str, Any],
        n_bootstrap: int,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calcula intervalo de confiança via bootstrap."""
        n_clients = len(embeddings)
        bootstrap_readiness = []
        
        for _ in range(n_bootstrap):
            # Reamostrar com reposição
            indices = np.random.choice(n_clients, n_clients, replace=True)
            resampled = [embeddings[i] for i in indices]
            
            # Recalcular R(F)
            result = self.calculate(resampled, dataset_info, use_bootstrap=False)
            bootstrap_readiness.append(result['readiness'])
        
        # Calcular percentis
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_readiness, lower_percentile)
        ci_upper = np.percentile(bootstrap_readiness, upper_percentile)
        
        return ci_lower, ci_upper

class FederationAnalyzer:
    """Análise completa de federações usando Task2Vec."""
    
    def __init__(
        self,
        probe_network: nn.Module,
        readiness_variant: str = "cohesion_dispersion",
        device: Optional[torch.device] = None
    ):
        self.task2vec_analyzer = Task2VecAnalyzer(
            probe_network=probe_network,
            device=device
        )
        self.readiness_calculator = ReadinessCalculator(variant=readiness_variant)
        
    def analyze_federation(
        self,
        client_datasets: List[Dataset],
        dataset_info: Dict[str, Any],
        calculate_readiness: bool = True,
        use_bootstrap: bool = True
    ) -> Dict[str, Any]:
        """
        Análise completa de uma federação.
        
        Returns:
            Dicionário com embeddings, readiness e estatísticas
        """
        # Extrair embeddings de cada cliente
        embeddings = []
        for client_id, dataset in enumerate(client_datasets):
            print(f"  Extraindo embedding do cliente {client_id+1}/{len(client_datasets)}...")
            embedding = self.task2vec_analyzer.extract_embedding(dataset, client_id)
            embeddings.append(embedding)
        
        result = {
            'embeddings': embeddings,
            'num_clients': len(embeddings)
        }
        
        # Calcular readiness
        if calculate_readiness:
            readiness_result = self.readiness_calculator.calculate(
                embeddings,
                dataset_info,
                use_bootstrap=use_bootstrap
            )
            result['readiness'] = readiness_result
        
        # Estatísticas agregadas
        result['federation_stats'] = self._compute_federation_statistics(embeddings)
        
        return result
    
    def _compute_federation_statistics(
        self,
        embeddings: List[Task2VecEmbedding]
    ) -> Dict[str, Any]:
        """Calcula estatísticas agregadas da federação."""
        total_samples = sum(e.metadata.get('num_samples', 0) for e in embeddings)
        all_classes = set()
        entropies = []
        
        for emb in embeddings:
            if 'class_distribution' in emb.metadata:
                all_classes.update(emb.metadata['class_distribution'].keys())
            if 'label_entropy' in emb.metadata:
                entropies.append(emb.metadata['label_entropy'])
        
        return {
            'total_samples': total_samples,
            'total_classes': len(all_classes),
            'avg_entropy': float(np.mean(entropies)) if entropies else 0,
            'std_entropy': float(np.std(entropies)) if entropies else 0,
            'min_client_samples': min(e.metadata.get('num_samples', 0) for e in embeddings),
            'max_client_samples': max(e.metadata.get('num_samples', 0) for e in embeddings)
        }