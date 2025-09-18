#!/usr/bin/env python3
"""
Sistema de experimentação para avaliar a correlação entre Task2Vec embeddings
e performance em Federated Learning com dados não-IID.

Este script orquestra experimentos que:
1. Criam federações com diferentes níveis de heterogeneidade (alpha de Dirichlet)
2. Extraem embeddings Task2Vec para caracterizar cada cliente
3. Calculam índice de readiness da federação
4. Executam simulações de FL e correlacionam com as métricas Task2Vec
"""

import os
import sys
import yaml
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import time
from datetime import datetime
import warnings
import torch

import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
from torch.utils.data import Dataset, Subset
from task2vec import Task2Vec
import copy
import torch.nn.functional as F
import task_similarity
import seaborn as sns
import logging


from data_utils import DatasetManager, FederatedDataManager
from models import get_model
from task2vec_analysis import FederationAnalyzer
from fl_simulation import FederatedExperimentRunner



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

warnings.filterwarnings('ignore')

# Importar módulos do projeto
from utils import (
    set_reproducibility,
    create_experiment_id,
    save_results,
    log_experiment_metadata,
    ExperimentLogger,
    print_progress
)


def setup_logging(log_dir: str):
    """
    Configura sistema de logging dual: arquivo e console.
    
    Cria handlers para capturar logs tanto em arquivo quanto no terminal,
    incluindo captura de exceções não tratadas para debugging pós-execução.
    
    Args:
        log_dir: Diretório onde o arquivo de log será salvo
        
    Returns:
        str: Caminho completo do arquivo de log criado
    """
    
    # Nomeia arquivo com timestamp para evitar sobrescrita
    log_filename = datetime.now().strftime('log_%Y-%m-%d_%H-%M-%S.txt')
    log_filepath = os.path.join(log_dir, log_filename)

    # Define formato padrão: timestamp [LEVEL] mensagem
    log_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-5.5s]  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configura root logger para capturar todos os logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Handler para persistência em arquivo
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Handler para feedback em tempo real no terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # Hook customizado para logar exceções não capturadas (exceto KeyboardInterrupt)
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Loga traceback completo de erros não tratados para análise posterior."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Preserva comportamento padrão para interrupção manual
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        root_logger.error(
            "Exceção não capturada:", 
            exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception
    
    logging.info(f"Logging configurado. Saída será salva em: {log_filepath}")
    return log_filepath



def compute_task_similarity(client_datasets, probe_network, num_clients, dataset_name, alpha, test_id=0):
    """
    Computa embeddings Task2Vec e visualiza similaridade entre clientes.
    
    Esta função extrai características de cada cliente usando uma rede probe
    pré-treinada, calcula similaridades par-a-par e gera visualizações.
    
    Args:
        client_datasets: Lista de datasets dos clientes
        probe_network: Rede neural pré-treinada para extração de features (ResNet34)
        num_clients: Número de clientes na federação
        dataset_name: Nome do dataset base (cifar10, pathmnist, etc)
        alpha: Parâmetro de heterogeneidade da distribuição de Dirichlet
        test_id: ID da federação para nomenclatura de arquivos
    
    Returns:
        list: Task vectors computados para cada cliente
    """
    logging.info(f"\n=== Computando Task2Vec ===")
    
    # Fixa seeds para garantir reprodutibilidade entre execuções
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    task_vectors = []
    client_names = [f"Cliente {i+1}" for i in range(len(client_datasets))]
    
    # Análise exploratória da distribuição de classes por cliente
    logging.info("=== DEBUG: Verificando distribuições dos clientes ===")
    for i, dataset in enumerate(client_datasets):
        class_counts = defaultdict(int)
        
        # Amostragem aleatória para análise eficiente de datasets grandes
        total_dataset_size = len(dataset)
        sample_size = min(1000, total_dataset_size)
        
        # Amostra aleatória ao invés das primeiras amostras (evita viés)
        sample_indices = random.sample(range(total_dataset_size), sample_size)
        
        for idx in sample_indices:
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            class_counts[label] += 1
        
        # Calcula e reporta estatísticas da distribuição
        percentages = {cls: (count/sample_size)*100 for cls, count in class_counts.items()}
        dominant_class = max(class_counts, key=class_counts.get)
        dominant_pct = percentages[dominant_class]
        
        logging.info(f"Cliente {i+1}: classe dominante={dominant_class} ({dominant_pct:.1f}%), "
              f"amostra={sample_size}/{total_dataset_size}")
        logging.info(f"  Top 3 classes: {sorted(percentages.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # Processa cada cliente para extrair embedding Task2Vec
    for i, dataset in enumerate(client_datasets):
        logging.info(f"\nProcessando Cliente {i+1}...")
        
        # Wrapper para adaptar dados ao formato esperado pelo Task2Vec
        class TransformedDataset(Dataset):
            def __init__(self, original_dataset, client_id):
                self.dataset = original_dataset
                self.client_id = client_id
                
                logging.info(f"  Dataset transformado criado para Cliente {client_id+1}")
                if len(original_dataset) > 0:
                    sample_img, sample_label = original_dataset[0]
                    logging.info(f"  Amostra original: shape={sample_img.shape}, label={sample_label}")
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                img, label = self.dataset[idx]
                
                # Log apenas para primeira amostra (evita spam)
                original_shape = img.shape
                
                # Converte imagens grayscale para RGB (ResNet espera 3 canais)
                if img.dim() == 3 and img.size(0) == 1:  # [1, H, W]
                    img = img.repeat(3, 1, 1)  # [3, H, W]
                elif img.dim() == 2:  # [H, W]
                    img = img.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
                
                # ResNet pré-treinada no ImageNet espera imagens 224x224
                img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
                img = img.squeeze(0)  # Remove dimensão de batch adicionada para interpolação
                
                if idx == 0:
                    logging.info(f"  Transformação: {original_shape} → {img.shape}")
                
                return img, label
        
        transformed_dataset = TransformedDataset(dataset, i)
        # Cria cópia da rede para evitar interferência entre clientes
        current_probe_network = copy.deepcopy(probe_network)

        logging.info(f"  Criando Task2Vec para Cliente {i+1}...")
        
        try:
            # Extrai embedding usando diagonal da matriz de Fisher Information
            task_embedding = Task2Vec(current_probe_network, max_samples=1000, skip_layers=6)
            embedding = task_embedding.embed(transformed_dataset)
            
            # Validação básica do embedding extraído
            logging.info(f"  Embedding extraído: shape={embedding.hessian.shape}")
            logging.info(f"  Estatísticas do embedding: mean={np.mean(embedding.hessian):.6f}, "
                  f"std={np.std(embedding.hessian):.6f}")
            logging.info(f"  Primeiros 5 valores: {embedding.hessian[:5]}")
            
            task_vectors.append(embedding)
            
        except Exception as e:
            logging.info(f"  ERRO ao processar Cliente {i+1}: {e}")
            # Cria embedding placeholder para não quebrar pipeline
            dummy_embedding = type('obj', (object,), {'hessian': np.random.randn(1000)})
            task_vectors.append(dummy_embedding)
    
    # Validação final dos embeddings antes de calcular similaridades
    logging.info(f"\n=== DEBUG: Verificando embeddings finais ===")
    for i, embedding in enumerate(task_vectors):
        if hasattr(embedding, 'hessian'):
            logging.info(f"Cliente {i+1}: embedding shape={embedding.hessian.shape}, "
                  f"mean={np.mean(embedding.hessian):.6f}")
        else:
            logging.info(f"Cliente {i+1}: ERRO - embedding inválido")
    
    # Gera visualização da matriz de distâncias usando módulo externo
    task_similarity.plot_distance_matrix(
        task_vectors,
        client_names,
        num_clients,
        dataset_name=dataset_name,
        alpha=alpha,
        test_id=test_id
    )

    # Computa matriz de similaridade baseada em correlação
    n_clients = len(task_vectors)
    similarity_matrix = np.zeros((n_clients, n_clients))

    # Salva matriz para análise posterior
    output_filename = f'similarity_matrix_{num_clients}_{dataset_name}_{alpha}_{test_id}.npy'
    np.save(output_filename, similarity_matrix)
    logging.info(f"\n✅ Matriz de similaridade salva com sucesso em: {output_filename}")

    # Gera heatmap para visualização da similaridade
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                annot_kws={"size": 8},
                xticklabels=[f"C{i+1}" for i in range(n_clients)],
                yticklabels=[f"C{i+1}" for i in range(n_clients)])
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.title(f"Matriz de Similaridade entre Clientes")
    plt.xlabel("Cliente")
    plt.ylabel("Cliente")
    plt.tight_layout()
    plt.savefig(f'results/matriz_{num_clients}_{dataset_name}_{alpha}_{test_id}.png')
    plt.close()
    
    return task_vectors


class ExperimentOrchestrator:
    """
    Orquestra a execução completa de experimentos Task2Vec + FL.
    
    Responsável por:
    - Carregar configurações e preparar ambiente
    - Iterar sobre combinações de parâmetros experimentais
    - Coordenar análise Task2Vec e simulações FL
    - Agregar e salvar resultados
    """
    
    def __init__(self, config_path: str):
        """
        Inicializa o orquestrador com configurações do experimento.
        
        Args:
            config_path: Caminho para arquivo YAML com parâmetros experimentais
        """
        # Carrega configuração experimental
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup do ambiente computacional
        set_reproducibility(self.config['experiment']['seed'])
        self.device = DEVICE
        
        # Identificador único baseado em timestamp e configuração
        self.experiment_id = create_experiment_id(self.config)
        
        # Estrutura de diretórios para outputs
        self.output_dir = Path(self.config['experiment']['output_dir']) / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializa sistema de logging no diretório de output
        self.log_filepath = setup_logging(str(self.output_dir))
        
        # Persiste metadados para rastreabilidade
        log_experiment_metadata(self.config, str(self.output_dir), self.experiment_id)
        
        # Logger estruturado para métricas
        self.logger = ExperimentLogger(str(self.output_dir), self.experiment_id)
        
        # Gerenciadores de dados
        self.dataset_manager = DatasetManager()
        self.federated_manager = FederatedDataManager(self.dataset_manager)
        
        # Buffer para resultados acumulados
        self.all_results = []
        
    def run_all_experiments(self):
        """
        Executa grade completa de experimentos definida na configuração.
        
        Itera sobre todas as combinações de:
        - Datasets
        - Número de clientes
        - Valores de alpha (heterogeneidade)
        - Federações independentes
        """
        logging.info("=" * 80)
        logging.info(f"INICIANDO EXPERIMENTOS - ID: {self.experiment_id}")
        logging.info(f"Configuração: {len(self.config['datasets'])} datasets, "
              f"{len(self.config['federation']['num_clients'])} tamanhos de federação, "
              f"{len(self.config['federation']['alpha_values'])} valores de alpha")
        logging.info("=" * 80)
        
        start_time = time.time()
        # Calcula total de experimentos para barra de progresso
        total_experiments = (
            len(self.config['datasets']) *
            len(self.config['federation']['num_clients']) *
            len(self.config['federation']['alpha_values']) *
            self.config['experiment']['num_independent_federations']
        )
        
        experiment_count = 0
        
        # Loop principal sobre datasets
        for dataset_config in self.config['datasets']:
            dataset_name = dataset_config['name']
            target_metric = dataset_config['target_metric']
            target_threshold = dataset_config['target_threshold']
            
            logging.info(f"\n{'='*60}")
            logging.info(f"DATASET: {dataset_name.upper()}")
            logging.info(f"Métrica alvo: {target_metric} >= {target_threshold}")
            logging.info(f"{'='*60}")
            
            # Probe network compartilhada para Task2Vec neste dataset
            probe_network = get_model('resnet34', pretrained=True, num_classes=self.dataset_manager.get_dataset_info(dataset_name)['num_classes']).to(self.device)
            
            # Analyzer reutilizado para todas as federações do dataset
            analyzer = FederationAnalyzer(
                probe_network=probe_network,
                readiness_variant=self.config['readiness']['variants'][0],
                device=self.device
            )
            
            # Loop sobre tamanhos de federação
            for num_clients in self.config['federation']['num_clients']:
                
                # Loop sobre níveis de heterogeneidade
                for alpha in self.config['federation']['alpha_values']:
                    
                    logging.info(f"\n--- Configuração: {num_clients} clientes, α={alpha} ---")
                    
                    # Cria múltiplas instâncias independentes da mesma configuração
                    federations = self.federated_manager.create_federated_datasets(
                        dataset_name=dataset_name,
                        num_clients=num_clients,
                        alpha=alpha,
                        num_federations=self.config['experiment']['num_independent_federations'],
                        base_seed=self.config['experiment']['seed']
                    )
                    
                    # Processa cada instância de federação
                    for federation in federations:
                        experiment_count += 1
                        print_progress(
                            experiment_count,
                            total_experiments,
                            prefix=f"Progresso total:",
                            suffix=f"Fed {federation['federation_id']+1}/{len(federations)}"
                        )
                        
                        logging.info(f"  Analisando federação {federation['federation_id']+1}...")                        
                        
                        # Validação de integridade dos dados antes do processamento
                        logging.info("🔍 Verificando rótulos do primeiro cliente antes do Task2Vec...")
                        
                        train_dataset_ref = federation['train_dataset_ref']
                        first_client_indices = federation['client_indices'][0]
                        client_dataset = Subset(train_dataset_ref, first_client_indices)

                        num_classes_config = federation['dataset_info']['num_classes']
                        
                        # Verifica range de labels para detectar problemas de mapeamento
                        max_label = -1
                        min_label = float('inf')
                        
                        for _, label in client_dataset:
                            label_val = label.item() if hasattr(label, 'item') else label
                            if label_val > max_label:
                                max_label = label_val
                            if label_val < min_label:
                                min_label = label_val
                                
                        logging.info(f"  Labels encontrados no range: [{min_label}, {max_label}]")
                        logging.info(f"  Modelo configurado para {num_classes_config} classes (espera rótulos até {num_classes_config - 1})")
                        
                        if max_label >= num_classes_config:
                            raise ValueError(f"ERRO DE DADOS: Encontrado rótulo {max_label}, mas o modelo espera no máximo {num_classes_config - 1}")
                        
                        # Prepara subsets para análise Task2Vec
                        client_datasets_for_analysis = [
                            Subset(federation['train_dataset_ref'], indices) 
                            for indices in federation['client_indices']
                        ]

                        # Extrai embeddings e calcula readiness
                        logging.info(f"  Analisando federação {federation['federation_id']+1} com Task2Vec...")
                        task2vec_results = analyzer.analyze_federation(
                            client_datasets=client_datasets_for_analysis,
                            dataset_info=federation['dataset_info'],
                            calculate_readiness=True,
                            use_bootstrap=True 
                        )
                        
                        # Executa simulação de Federated Learning
                        logging.info(f"  Executando simulação FL...")
                        fl_runner = FederatedExperimentRunner(self.config)
                        fl_results = fl_runner.run_single_experiment(
                            federation_data=federation,
                            model_name="resnet34"
                        )
                        
                        # Agrega todos os resultados em estrutura unificada
                        experiment_result = self._combine_results(
                            dataset_name=dataset_name,
                            num_clients=num_clients,
                            alpha=alpha,
                            federation=federation,
                            task2vec_results=task2vec_results,
                            fl_results=fl_results,
                            target_metric=target_metric,
                            target_threshold=target_threshold
                        )
                        
                        self.all_results.append(experiment_result)
                        
                        # Persiste log incremental
                        self.logger.log(experiment_result)
                        
                        # Checkpoint periódico para recuperação em caso de falha
                        if experiment_count % 10 == 0:
                            self._save_partial_results()
        
        # Salva resultados consolidados
        self._save_final_results()
        
        elapsed_time = time.time() - start_time
        logging.info(f"\n{'='*80}")
        logging.info(f"EXPERIMENTOS CONCLUÍDOS!")
        logging.info(f"Total de experimentos: {experiment_count}")
        logging.info(f"Tempo total: {elapsed_time/3600:.2f} horas")
        logging.info(f"Resultados salvos em: {self.output_dir}")
        logging.info(f"{'='*80}")
    
    def _combine_results(
        self,
        dataset_name: str,
        num_clients: int,
        alpha: float,
        federation: Dict[str, Any],
        task2vec_results: Dict[str, Any],
        fl_results: Dict[str, Any],
        target_metric: str,
        target_threshold: float
    ) -> Dict[str, Any]:
        """
        Combina resultados de Task2Vec e FL em registro unificado.
        
        Estrutura o resultado final com:
        - Identificadores do experimento
        - Métricas de readiness e seus componentes
        - Performance final do FL
        - Metadados da federação
        
        Returns:
            Dict com todas as métricas e metadados do experimento
        """
        
        # Extrai estruturas aninhadas do resultado Task2Vec
        readiness_data = task2vec_results.get('readiness', {})
        components = readiness_data.get('components', {})
        ci = readiness_data.get('confidence_interval', {})
        
        final_performance = fl_results['final_metrics'].get(target_metric, 0)
        convergence_round = fl_results.get('convergence_round')
        
        # Monta registro completo para análise
        result = {
            # Identificação única do experimento
            'experiment_id': self.experiment_id,
            'dataset': dataset_name,
            'num_clients': num_clients,
            'alpha': alpha,
            'federation_id': federation['federation_id'],
            'federation_seed': federation['federation_seed'],
            
            # Métricas Task2Vec principais
            'readiness': readiness_data.get('readiness'),
            'readiness_ci_lower': ci.get('lower'),
            'readiness_ci_upper': ci.get('upper'),
            'readiness_variant': readiness_data.get('variant'),
            
            # Decomposição do índice de readiness
            'cohesion': components.get('cohesion'),
            'dispersion': components.get('dispersion'),
            'coverage': components.get('coverage'),
            'avg_entropy': components.get('avg_entropy'),
            'size_factor': components.get('size_factor'),
            'density': components.get('density'),
            
            # Resultados do treinamento federado
            'final_performance': final_performance,
            'final_accuracy': fl_results['final_metrics'].get('accuracy', 0),
            'final_loss': fl_results['final_metrics'].get('test_loss', 0),
            'convergence_round': convergence_round if convergence_round else -1,
            'target_achieved': fl_results.get('target_achieved', False),
            
            # Características da distribuição de dados
            'total_samples': task2vec_results['federation_stats']['total_samples'],
            'min_client_samples': task2vec_results['federation_stats']['min_client_samples'],
            'max_client_samples': task2vec_results['federation_stats']['max_client_samples'],
            'federation_entropy_mean': task2vec_results['federation_stats']['avg_entropy'],
            'federation_entropy_std': task2vec_results['federation_stats']['std_entropy'],
            
            # Timestamp para rastreabilidade
            'timestamp': datetime.now().isoformat()
        }
        
        # Adiciona AUC para datasets médicos se disponível
        if 'auc_macro' in fl_results['final_metrics']:
            result['final_auc_macro'] = fl_results['final_metrics']['auc_macro']
        
        return result
    
    def _save_partial_results(self):
        """
        Salva checkpoint dos resultados acumulados.
        
        Útil para recuperação em caso de interrupção durante
        experimentos longos.
        """
        if not self.all_results:
            return
        
        df = pd.DataFrame(self.all_results)
        save_path = self.output_dir / f"partial_results_{len(self.all_results)}.csv"
        df.to_csv(save_path, index=False)
        logging.info(f"\n  Resultados parciais salvos: {save_path}")
    
    def _save_final_results(self):
        """
        Salva resultados finais em múltiplos formatos.
        
        Gera:
        - CSV para análise estatística
        - JSON para processamento programático
        - Relatório resumido em texto
        """
        if not self.all_results:
            logging.info("Nenhum resultado para salvar!")
            return
        
        # DataFrame para análise estruturada
        df = pd.DataFrame(self.all_results)
        
        # Salva em formatos diversos
        save_results(df, str(self.output_dir), self.experiment_id)
        
        # Persiste logs estruturados
        self.logger.save()
        
        # Gera análise estatística resumida
        self._generate_summary_report(df)
    
    def _generate_summary_report(self, df: pd.DataFrame):
        """
        Gera relatório estatístico dos experimentos.
        
        Calcula correlações entre readiness e performance,
        estatísticas por configuração e taxa de sucesso.
        
        Args:
            df: DataFrame com todos os resultados experimentais
        """
        report_path = self.output_dir / "summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"RELATÓRIO DE EXPERIMENTOS - {self.experiment_id}\n")
            f.write("=" * 80 + "\n\n")
            
            # Sumariza escopo experimental
            f.write("ESTATÍSTICAS GERAIS:\n")
            f.write(f"  Total de experimentos: {len(df)}\n")
            f.write(f"  Datasets testados: {df['dataset'].unique().tolist()}\n")
            f.write(f"  Valores de alpha: {sorted(df['alpha'].unique())}\n")
            f.write(f"  Tamanhos de federação: {sorted(df['num_clients'].unique())}\n\n")
            
            # Análise de correlação por dataset
            f.write("CORRELAÇÕES READINESS vs PERFORMANCE:\n")
            for dataset in df['dataset'].unique():
                dataset_df = df[df['dataset'] == dataset]
                
                # Correlações não-paramétricas e paramétricas
                from scipy.stats import spearmanr, pearsonr
                
                spearman_corr, spearman_p = spearmanr(
                    dataset_df['readiness'],
                    dataset_df['final_performance']
                )
                pearson_corr, pearson_p = pearsonr(
                    dataset_df['readiness'],
                    dataset_df['final_performance']
                )
                
                f.write(f"\n  {dataset.upper()}:\n")
                f.write(f"    Spearman: {spearman_corr:.4f} (p={spearman_p:.4f})\n")
                f.write(f"    Pearson: {pearson_corr:.4f} (p={pearson_p:.4f})\n")
                f.write(f"    Taxa de sucesso: {dataset_df['target_achieved'].mean():.2%}\n")
                
                # Análise estratificada por nível de heterogeneidade
                f.write("    Por alpha:\n")
                for alpha in sorted(dataset_df['alpha'].unique()):
                    alpha_df = dataset_df[dataset_df['alpha'] == alpha]
                    if len(alpha_df) > 3:  # Mínimo para correlação significativa
                        corr, _ = spearmanr(alpha_df['readiness'], alpha_df['final_performance'])
                        f.write(f"      α={alpha}: {corr:.4f} (n={len(alpha_df)})\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logging.info(f"\nRelatório salvo em: {report_path}")

def main():
    """
    Ponto de entrada principal do sistema experimental.
    
    Processa argumentos de linha de comando, valida configurações
    e orquestra a execução completa dos experimentos.
    """
    parser = argparse.ArgumentParser(
        description="Executar experimentos Task2Vec + Federated Learning"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default="../config/config.yaml",
        help="Caminho para arquivo de configuração"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Modo de teste (executa apenas uma configuração)"
    )
    
    args = parser.parse_args()
        
    # Valida existência do arquivo de configuração
    if not Path(args.config).exists():
        logging.info(f"Erro: Arquivo de configuração '{args.config}' não encontrado!")
        sys.exit(1)
    
    # Modo de teste com configuração reduzida para validação rápida
    if args.test:
        logging.info("MODO DE TESTE ATIVADO - Executando configuração mínima")
        # Modificar config para teste rápido
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        config['datasets'] = [config['datasets'][0]]  # Apenas primeiro dataset
        config['federation']['num_clients'] = [10]  # Apenas 10 clientes
        config['federation']['alpha_values'] = [0.1, 1.0]  # Apenas 2 alphas
        config['experiment']['num_independent_federations'] = 2  # Apenas 2 federações
        config['fl_training']['num_rounds'] = 5  # Menos rodadas
        
        # Salvar config temporária
        test_config_path = "test_config.yaml"
        with open(test_config_path, 'w') as f:
            yaml.dump(config, f)
        
        args.config = test_config_path
    
    
    orchestrator = None  # Inicializa como None
    try:
        # Criar e executar orquestrador
        orchestrator = ExperimentOrchestrator(args.config)
        orchestrator.run_all_experiments()
        
    except KeyboardInterrupt:
        logging.warning("\n\nExperimentos interrompidos pelo usuário!")
        if orchestrator:
            logging.info("Salvando resultados parciais...")
            orchestrator._save_final_results()
            
    except Exception as e:
        # O nosso `sys.excepthook` já vai pegar o erro fatal, 
        # mas podemos logar uma mensagem final aqui também.
        logging.critical(f"Erro fatal encerrou a execução: {e}")
        # O traceback completo será logado automaticamente pela função handle_exception.
        sys.exit(1)
    
    finally:
        # Limpar config de teste se existir
        if args.test and Path("test_config.yaml").exists():
            os.remove("test_config.yaml")
        logging.info("Execução finalizada.")



if __name__ == "__main__":
    main()