"""
Módulo de análise de federações usando Task2Vec.

Implementa extração de embeddings Task2Vec e cálculo do índice de readiness
para avaliar a "prontidão" de uma federação para aprendizado colaborativo.
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
from sys import exit
warnings.filterwarnings('ignore')

try:
    from task2vec import Task2Vec
except ImportError:
    print("Warning: task2vec não encontrado. Usando implementação simplificada.")
    Task2Vec = None

class Task2VecEmbedding:
    """
    Encapsula um embedding Task2Vec com metadados associados.
    
    Fornece métodos para calcular distâncias e similaridades entre tarefas,
    facilitando análises comparativas entre clientes em uma federação.
    """
    
    def __init__(self, embedding_vector: np.ndarray, metadata: Dict[str, Any] = None):
        self.vector = embedding_vector
        self.metadata = metadata or {}
        
    def distance_to(self, other: 'Task2VecEmbedding', metric: str = 'cosine') -> float:
        """
        Calcula distância entre embeddings usando métricas diferentes.
        
        Args:
            other: Outro embedding para comparação
            metric: 'cosine' ou 'euclidean'
            
        Returns:
            float: Distância entre os embeddings (0 = idênticos)
        """
        if metric == 'cosine':
            return cosine(self.vector, other.vector)
        elif metric == 'euclidean':
            return np.linalg.norm(self.vector - other.vector)
        else:
            raise ValueError(f"Métrica {metric} não suportada")
    
    def similarity_to(self, other: 'Task2VecEmbedding') -> float:
        """
        Calcula similaridade de cosseno normalizada [0, 1].
        
        Returns:
            float: 1.0 = idênticos, 0.0 = ortogonais
        """
        return 1 - self.distance_to(other, 'cosine')

class Task2VecAnalyzer:
    """
    Extrai e analisa embeddings Task2Vec de datasets.
    
    Usa uma rede neural pré-treinada como "probe network" para
    caracterizar tarefas através da diagonal da matriz de Fisher Information.
    """
    
    def __init__(
        self,
        probe_network: nn.Module,
        max_samples: int = 1000,
        skip_layers: int = 6,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            probe_network: Rede pré-treinada (geralmente ResNet) para extração
            max_samples: Limite de amostras para cálculo eficiente do embedding
            skip_layers: Camadas iniciais a ignorar (features mais genéricas)
            device: Dispositivo de computação (CPU/GPU)
        """
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
        Extrai representação vetorial de uma tarefa/dataset.
        
        O embedding captura características da distribuição de dados
        através da sensibilidade dos parâmetros da rede aos dados.
        
        Args:
            dataset: Dataset do cliente para caracterização
            client_id: Identificador do cliente (para rastreabilidade)
            
        Returns:
            Task2VecEmbedding: Embedding com vetor e metadados estatísticos
        """
        # Adapta imagens para formato esperado pelo Task2Vec (224x224 RGB)
        transformed_dataset = self._prepare_dataset_for_task2vec(dataset)
        
        # Cópia independente para evitar interferência entre clientes
        probe = copy.deepcopy(self.probe_network)
        probe = probe.to(self.device)
        
        if Task2Vec is not None:
            # Implementação oficial: usa diagonal da Hessiana como embedding
            task2vec = Task2Vec(probe, max_samples=self.max_samples, skip_layers=self.skip_layers)
            embedding = task2vec.embed(transformed_dataset)
            embedding_vector = embedding.hessian
        else:
            # Fallback crítico se biblioteca não disponível
            exit(1)
            
        # Enriquece embedding com estatísticas da distribuição
        metadata = self._collect_metadata(dataset, client_id)
        
        return Task2VecEmbedding(embedding_vector, metadata)
    
    def _prepare_dataset_for_task2vec(self, dataset: Dataset) -> Dataset:
        """
        Aplica transformações necessárias para compatibilidade com Task2Vec.
        
        Task2Vec espera imagens RGB 224x224 (padrão ImageNet).
        """
        
        class TransformedDataset(Dataset):
            def __init__(self, original_dataset):
                self.dataset = original_dataset
                
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                img, label = self.dataset[idx]
                
                # Converte para tensor se necessário
                if not isinstance(img, torch.Tensor):
                    img = torch.tensor(img)
                
                # Padroniza número de canais
                if img.dim() == 2:  # Grayscale sem canal
                    img = img.unsqueeze(0)
                if img.size(0) == 1:  # Grayscale com 1 canal
                    img = img.repeat(3, 1, 1)  # Replica para RGB
                elif img.size(0) > 3:  # Mais de 3 canais
                    img = img[:3]  # Usa apenas RGB
                
                # Redimensiona para tamanho padrão ImageNet
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
        [FUNÇÃO DEPRECADA - mantida para referência]
        
        Implementação alternativa baseada em informação de Fisher.
        Calcula sensibilidade dos parâmetros através da variância dos gradientes.
        """
        model.eval()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Acumula gradientes ao longo das amostras
        gradients = []
        num_samples = min(self.max_samples, len(dataset))
        samples_processed = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            if samples_processed >= num_samples:
                break
                
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward + backward para obter gradientes
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            model.zero_grad()
            loss.backward()
            
            # Coleta e achata gradientes de todos os parâmetros
            batch_gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    batch_gradients.append(param.grad.data.cpu().numpy().flatten())
            
            if batch_gradients:
                gradients.append(np.concatenate(batch_gradients))
            
            samples_processed += data.size(0)
        
        if not gradients:
            # Embedding aleatório como fallback
            return np.random.randn(1000)
        
        # Diagonal de Fisher = variância dos gradientes
        gradients = np.array(gradients)
        fisher_diagonal = np.var(gradients, axis=0)
        
        # Trunca ou padroniza para dimensão fixa
        if len(fisher_diagonal) > 1000:
            # Seleciona dimensões mais informativas
            top_indices = np.argsort(fisher_diagonal)[-1000:]
            fisher_diagonal = fisher_diagonal[top_indices]
        elif len(fisher_diagonal) < 1000:
            # Padding com zeros
            fisher_diagonal = np.pad(fisher_diagonal, (0, 1000 - len(fisher_diagonal)))
        
        return fisher_diagonal
    
    def _collect_metadata(self, dataset: Dataset, client_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Extrai estatísticas descritivas do dataset do cliente.
        
        Metadados incluem distribuição de classes, entropia e 
        métricas de desbalanceamento importantes para análise non-IID.
        """
        # Amostra labels para análise estatística
        labels = []
        for i in range(min(len(dataset), 1000)):  # Limita para eficiência
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
            'label_entropy': entropy(counts),  # Medida de uniformidade
            'dominant_class': int(unique_labels[np.argmax(counts)]),
            'dominant_class_ratio': float(np.max(counts) / len(labels))  # Grau de desbalanceamento
        }
        
        return metadata

class ReadinessCalculator:
    """
    Calcula índice de "readiness" R(F) para avaliar prontidão de federações.
    
    O índice combina múltiplos componentes que caracterizam a
    heterogeneidade e compatibilidade dos clientes para FL.
    """
    
    def __init__(self, variant: str = "cohesion_dispersion"):
        """
        Args:
            variant: Fórmula de cálculo do índice
                - 'cohesion_dispersion': Foco em similaridade vs dispersão
                - 'weighted_sum': Combinação ponderada de múltiplos fatores
                - 'density': Baseado em densidade no espaço de embeddings
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
        Calcula R(F) e seus componentes para uma federação.
        
        Args:
            embeddings: Embeddings Task2Vec de todos os clientes
            dataset_info: Informações globais do dataset
            use_bootstrap: Se deve calcular intervalo de confiança
            n_bootstrap: Número de reamostragens para bootstrap
            confidence_level: Nível de confiança (ex: 0.95 para 95%)
            
        Returns:
            Dict contendo:
                - readiness: Valor do índice R(F)
                - components: Decomposição em componentes
                - confidence_interval: IC via bootstrap (se solicitado)
        """
        if len(embeddings) < 2:
            return {'readiness': 0.0, 'error': 'Federação muito pequena'}
        
        # Matriz de embeddings para cálculos vetorizados
        vectors = np.array([emb.vector for emb in embeddings])
        
        # Calcula componentes individuais do índice
        components = self._calculate_components(vectors, embeddings, dataset_info)
        
        # Combina componentes de acordo com a variante escolhida
        if self.variant == "cohesion_dispersion":
            # Alta coesão e baixa dispersão = melhor federação
            readiness = components['cohesion'] - components['dispersion']
        elif self.variant == "weighted_sum":
            # Combinação linear com pesos heurísticos
            readiness = (
                0.4 * components['cohesion'] +      # Similaridade entre clientes
                0.2 * components['coverage'] +       # Cobertura de classes
                0.2 * components['size_factor'] -    # Volume de dados
                0.2 * components['dispersion']       # Penaliza dispersão
            )
        elif self.variant == "density":
            # Densidade no espaço de embeddings
            readiness = components['density']
        else:
            raise ValueError(f"Variante {self.variant} não suportada")
        
        result = {
            'readiness': float(readiness),
            'components': components,
            'variant': self.variant,
            'num_clients': len(embeddings)
        }
        
        # Estimativa de incerteza via bootstrap
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
        """
        Calcula componentes individuais que compõem o índice de readiness.
        
        Cada componente captura um aspecto diferente da federação:
        - Coesão: Quão similares são os clientes
        - Dispersão: Quão espalhados estão no espaço de embeddings
        - Cobertura: Proporção de classes representadas
        - Entropia: Uniformidade das distribuições locais
        - Size factor: Volume total de dados (log-normalizado)
        - Densidade: Concentração local no espaço de embeddings
        """
        n_clients = len(vectors)
        
        # Coesão: similaridade média entre todos os pares de clientes
        similarities = cosine_similarity(vectors)
        np.fill_diagonal(similarities, 0)  # Exclui auto-similaridade
        cohesion = similarities.sum() / (n_clients * (n_clients - 1))
        
        # Dispersão: distância média de cada cliente ao centroide global
        centroid = vectors.mean(axis=0)
        dispersions = [np.linalg.norm(v - centroid) for v in vectors]
        dispersion = np.mean(dispersions)
        
        # Normaliza dispersão pelo máximo teórico
        max_dispersion = np.sqrt(vectors.shape[1])  # Norma máxima em espaço unitário
        dispersion = dispersion / max_dispersion
        
        # Cobertura: fração de classes do dataset presentes na federação
        all_classes = set()
        for emb in embeddings:
            if 'class_distribution' in emb.metadata:
                all_classes.update(emb.metadata['class_distribution'].keys())
        
        # Usa informação global do dataset para normalização
        num_total_classes = dataset_info.get('num_classes', 10)
        coverage = len(all_classes) / float(num_total_classes)
        
        # Entropia média: medida de uniformidade das distribuições locais
        entropies = [
            emb.metadata.get('label_entropy', 0) for emb in embeddings
        ]
        avg_entropy = np.mean(entropies) if entropies else 0
        
        # Fator de tamanho: importância do volume de dados (log-scale)
        total_samples = sum(
            emb.metadata.get('num_samples', 0) for emb in embeddings
        )
        
        # Normaliza pelo tamanho total do dataset de treino
        total_training_samples = dataset_info.get('total_train_samples', 60000)
        size_factor = np.log1p(total_samples) / np.log1p(total_training_samples)
        
        # Densidade: concentração local usando kernel RBF
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
        """
        Calcula densidade no espaço de embeddings usando kernel RBF.
        
        Alta densidade indica clientes agrupados, sugerindo tarefas similares.
        
        Args:
            vectors: Matriz de embeddings dos clientes
            bandwidth: Parâmetro de largura do kernel (baseado na dispersão)
            
        Returns:
            float: Densidade média normalizada [0, 1]
        """
        n = len(vectors)
        if n < 2 or bandwidth == 0:
            return 0.0
        
        # Matriz de distâncias par-a-par
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(vectors[i] - vectors[j])
                distances[i, j] = distances[j, i] = dist
        
        # Kernel RBF (Gaussiano) para suavização
        kernel_values = np.exp(-distances**2 / (2 * bandwidth**2))
        np.fill_diagonal(kernel_values, 0)  # Remove auto-kernel
        
        # Densidade média como proxy de agrupamento
        density = kernel_values.sum() / (n * (n - 1))
        
        return density
    
    def _bootstrap_confidence_interval(
        self,
        embeddings: List[Task2VecEmbedding],
        dataset_info: Dict[str, Any],
        n_bootstrap: int,
        confidence_level: float
    ) -> Tuple[float, float]:
        """
        Estima intervalo de confiança para R(F) via bootstrap.
        
        Reamostra clientes com reposição para avaliar variabilidade
        do índice de readiness.
        
        Returns:
            Tuple[float, float]: (limite_inferior, limite_superior) do IC
        """
        n_clients = len(embeddings)
        bootstrap_readiness = []
        
        for _ in range(n_bootstrap):
            # Reamostragem com reposição
            indices = np.random.choice(n_clients, n_clients, replace=True)
            resampled = [embeddings[i] for i in indices]
            
            # Recalcula R(F) para amostra bootstrap
            result = self.calculate(resampled, dataset_info, use_bootstrap=False)
            bootstrap_readiness.append(result['readiness'])
        
        # Percentis para intervalo de confiança
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_readiness, lower_percentile)
        ci_upper = np.percentile(bootstrap_readiness, upper_percentile)
        
        return ci_lower, ci_upper

class FederationAnalyzer:
    """
    Interface de alto nível para análise completa de federações.
    
    Combina extração de embeddings Task2Vec e cálculo de readiness
    em um pipeline unificado.
    """
    
    def __init__(
        self,
        probe_network: nn.Module,
        readiness_variant: str = "cohesion_dispersion",
        device: Optional[torch.device] = None
    ):
        """
        Args:
            probe_network: Rede pré-treinada para Task2Vec
            readiness_variant: Variante do cálculo de readiness
            device: Dispositivo de computação
        """
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
        Pipeline completo de análise de uma federação.
        
        Args:
            client_datasets: Datasets individuais de cada cliente
            dataset_info: Metadados globais do dataset
            calculate_readiness: Se deve calcular índice R(F)
            use_bootstrap: Se deve incluir intervalo de confiança
            
        Returns:
            Dict com embeddings, readiness e estatísticas agregadas
        """
        # Fase 1: Extração de embeddings para cada cliente
        embeddings = []
        for client_id, dataset in enumerate(client_datasets):
            print(f"  Extraindo embedding do cliente {client_id+1}/{len(client_datasets)}...")
            embedding = self.task2vec_analyzer.extract_embedding(dataset, client_id)
            embeddings.append(embedding)
        
        result = {
            'embeddings': embeddings,
            'num_clients': len(embeddings)
        }
        
        # Fase 2: Cálculo do índice de readiness
        if calculate_readiness:
            readiness_result = self.readiness_calculator.calculate(
                embeddings,
                dataset_info,
                use_bootstrap=use_bootstrap
            )
            result['readiness'] = readiness_result
        
        # Fase 3: Estatísticas agregadas da federação
        result['federation_stats'] = self._compute_federation_statistics(embeddings)
        
        return result
    
    def _compute_federation_statistics(
        self,
        embeddings: List[Task2VecEmbedding]
    ) -> Dict[str, Any]:
        """
        Agrega estatísticas descritivas de toda a federação.
        
        Útil para caracterização geral e comparação entre federações.
        """
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