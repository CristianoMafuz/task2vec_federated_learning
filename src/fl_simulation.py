"""
Módulo de simulação de Federated Learning usando Flower Framework.

Implementa clientes, servidor e estratégias de agregação para
experimentos de aprendizado federado com dados heterogêneos.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict
import flwr as fl
from flwr.common import Context, Metrics, NDArrays, FitRes, parameters_to_ndarrays 
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from sklearn.metrics import roc_auc_score
from models import get_model
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FederatedClient(NumPyClient):
    """
    Cliente base para Federated Learning usando Flower.
    
    Implementa o protocolo de treinamento local e avaliação
    seguindo o paradigma FedAvg.
    """
    
    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        epochs: int = 3,
        learning_rate: float = 0.01,
        num_classes: int = None
    ):
        """
        Args:
            model: Modelo neural local do cliente
            trainloader: DataLoader para treinamento local
            valloader: DataLoader para validação local
            device: Dispositivo de computação (CPU/GPU)
            epochs: Épocas de treinamento por rodada FL
            learning_rate: Taxa de aprendizado do otimizador local
            num_classes: Número de classes esperadas (para validação)
        """
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_classes = num_classes
    
    def _validate_dataset_classes(self):
        """
        Valida compatibilidade entre labels do dataset e arquitetura do modelo.
        
        Previne erros de índice out-of-bounds durante treinamento
        verificando se todos os labels estão no range esperado [0, num_classes).
        """
        print(f"🔍 Validando classes do dataset...")
        
        # Coleta todos os labels únicos
        train_labels = []
        for _, labels in self.trainloader:
            train_labels.extend(labels.tolist())
        
        val_labels = []
        for _, labels in self.valloader:
            val_labels.extend(labels.tolist())
        
        all_labels = train_labels + val_labels
        unique_labels = set(all_labels)
        
        print(f"  📊 Classes encontradas: {sorted(unique_labels)}")
        print(f"  📈 Range de classes: {min(unique_labels)} a {max(unique_labels)}")
        print(f"  🎯 Modelo espera: 0 a {self.num_classes - 1}")
        
        # Validações críticas
        if max(unique_labels) >= self.num_classes:
            raise ValueError(
                f"❌ ERRO: Dataset contém classe {max(unique_labels)}, "
                f"mas modelo só suporta classes 0-{self.num_classes-1}"
            )
        
        if min(unique_labels) < 0:
            raise ValueError(
                f"❌ ERRO: Dataset contém classe negativa: {min(unique_labels)}"
            )
        
        print(f"  ✅ Validação passou! Dataset compatível com modelo.")
    
    
    def get_parameters(self, config: Dict[str, Any]) -> NDArrays:
        """
        Serializa parâmetros do modelo para envio ao servidor.
        
        Converte tensores PyTorch para arrays NumPy para transmissão
        eficiente via protocolo Flower.
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays):
        """
        Atualiza modelo local com parâmetros recebidos do servidor.
        
        Realiza validação de shapes para garantir compatibilidade
        entre modelo local e parâmetros globais.
        """
        model_keys = list(self.model.state_dict().keys())

        # Validação de integridade estrutural
        for (name, param), received in zip(self.model.state_dict().items(), parameters):
            if list(param.shape) != list(received.shape):
                print(f"SHAPE MISMATCH in {name}: expected {param.shape}, got {received.shape}")
                raise ValueError("Shape mismatch detected!")

        # Reconstrói state dict com novos parâmetros
        params_dict = zip(model_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Any]) -> Tuple[NDArrays, int, Dict]:
        """
        Executa uma rodada de treinamento local.
        
        Implementa o lado cliente do algoritmo FedAvg:
        1. Recebe modelo global
        2. Treina localmente
        3. Retorna modelo atualizado e métricas
        
        Returns:
            Tuple contendo:
            - Parâmetros atualizados do modelo
            - Número de amostras usadas (para agregação ponderada)
            - Métricas de treinamento/validação
        """
        self.set_parameters(parameters)
        
        # Treinamento local com dados do cliente
        train_loss = self._train()
        
        # Validação no conjunto local (não no teste global)
        val_loss, val_accuracy = self._evaluate(self.valloader)
        
        metrics = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy)
        }
        
        return self.get_parameters({}), len(self.trainloader.dataset), metrics
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Any]) -> Tuple[float, int, Dict]:
        """
        Avalia modelo global no conjunto de validação local.
        
        Usado para monitoramento distribuído da performance.
        """
        self.set_parameters(parameters)
        loss, accuracy = self._evaluate(self.valloader)
        
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}
    
    def _train(self) -> float:
        """
        Executa épocas de treinamento local com SGD.
        
        Returns:
            float: Loss médio das épocas de treinamento
        """
        # Proteção contra datasets muito pequenos
        if len(self.trainloader.dataset) < 2:
            print(f"⚠️ Cliente com dataset muito pequeno ({len(self.trainloader.dataset)} amostras). Pulando treino.")
            return 0.0

        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Validação em tempo de execução para detectar problemas de mapeamento
                unique_targets = torch.unique(target)
                if torch.max(unique_targets) >= self.model.fc.out_features:  # Assumindo ResNet
                    print(f"❌ ERRO DETECTADO no batch {batch_idx}:")
                    print(f"  Classes no batch: {unique_targets.tolist()}")
                    print(f"  Modelo suporta: 0-{self.model.fc.out_features-1}")
                    raise ValueError("Label fora do range suportado pelo modelo!")
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Avalia modelo em modo de inferência (sem gradientes).
        
        Returns:
            Tuple[float, float]: (loss_médio, acurácia)
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy

class MedMNISTClient(FederatedClient):
    """
    Cliente especializado para datasets médicos MedMNIST.
    
    Estende cliente base com métricas AUC, mais apropriadas
    para tarefas médicas multi-classe desbalanceadas.
    """
    
    def _evaluate_with_auc(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """
        Avaliação estendida com AUC macro para classificação multi-classe.
        
        AUC é métrica preferida em domínio médico por ser
        robusta a desbalanceamento de classes.
        
        Returns:
            Tuple[float, float, float]: (loss, accuracy, auc_macro)
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        all_outputs = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                
                # Coleta probabilidades para cálculo de AUC
                probs = torch.softmax(output, dim=1)
                all_outputs.append(probs.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # Agregação de predições
        all_outputs = np.vstack(all_outputs)
        all_targets = np.concatenate(all_targets)
        
        avg_loss = total_loss / len(all_targets) if len(all_targets) > 0 else 0.0
        
        # Acurácia padrão
        predictions = np.argmax(all_outputs, axis=1)
        accuracy = np.mean(predictions == all_targets)
        
        # AUC macro-averaged (média entre classes)
        try:
            # One-hot encoding para cálculo multi-classe
            num_classes = all_outputs.shape[1]
            targets_onehot = np.zeros((len(all_targets), num_classes))
            targets_onehot[np.arange(len(all_targets)), all_targets] = 1
            
            auc_macro = roc_auc_score(targets_onehot, all_outputs, average='macro')
        except:
            auc_macro = 0.5  # Baseline para classificador aleatório
        
        return avg_loss, accuracy, auc_macro
    
    def fit(self, parameters: NDArrays, config: Dict[str, Any]) -> Tuple[NDArrays, int, Dict]:
        """
        Treino local com métricas estendidas para domínio médico.
        """
        self.set_parameters(parameters)
        
        train_loss = self._train()
        
        # Avaliação com AUC adicional
        val_loss, val_accuracy, val_auc = self._evaluate_with_auc(self.valloader)
        
        metrics = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
            "val_auc": float(val_auc)
        }
        
        return self.get_parameters({}), len(self.trainloader.dataset), metrics
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Any]) -> Tuple[float, int, Dict]:
        """
        Avaliação com AUC para monitoramento médico.
        """
        self.set_parameters(parameters)
        loss, accuracy, auc = self._evaluate_with_auc(self.valloader)
        
        return float(loss), len(self.valloader.dataset), {
            "accuracy": float(accuracy),
            "auc_macro": float(auc)
        }

class CustomFedAvg(FedAvg):
    """
    Estratégia FedAvg customizada com logging e persistência de estado.
    
    Estende FedAvg padrão para capturar métricas detalhadas
    e manter parâmetros finais para avaliação pós-treinamento.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_metrics = []
        self.final_parameters = None  # Cache dos parâmetros agregados mais recentes
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException]
    ):
        """
        Agrega modelos locais e captura métricas de rodada.
        
        Implementa FedAvg: média ponderada pelo número de amostras
        de cada cliente.
        """
        # Agregação padrão do FedAvg
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Mantém referência aos parâmetros mais recentes
            self.final_parameters = aggregated_parameters
        
        # Coleta e agrega métricas dos clientes para monitoramento
        if results:
            metrics = {
                "round": server_round,
                "num_clients": len(results),
                "avg_train_loss": np.mean([r.metrics.get("train_loss", 0) for _, r in results]),
                "avg_val_loss": np.mean([r.metrics.get("val_loss", 0) for _, r in results]),
                "avg_val_accuracy": np.mean([r.metrics.get("val_accuracy", 0) for _, r in results])
            }
            
            # Métricas específicas se disponíveis
            auc_values = [r.metrics.get("val_auc", None) for _, r in results]
            auc_values = [v for v in auc_values if v is not None]
            if auc_values:
                metrics["avg_val_auc"] = np.mean(auc_values)
            
            self.round_metrics.append(metrics)
        
        return aggregated_parameters, aggregated_metrics

class FederatedLearningSimulator:
    """
    Orquestra simulações de Federated Learning usando Flower.
    
    Gerencia criação de clientes, execução de rodadas FL
    e coleta de métricas para análise experimental.
    """
    
    def __init__(
        self,
        model_name: str,
        dataset_info: Dict[str, Any],
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model_name: Arquitetura do modelo (ignorado, usa ResNet34)
            dataset_info: Metadados do dataset (classes, métricas alvo)
            config: Configurações de treinamento FL
            device: Dispositivo de computação
        """
        # Força ResNet34 para consistência experimental
        self.model_name = "resnet34"
        self.dataset_info = dataset_info
        self.config = config
        self.device = device or DEVICE
        
    def run_federation(
        self,
        federation_data: Dict[str, Any],
        num_rounds: int = 50,
        target_metric: str = "accuracy",
        target_threshold: float = 0.75,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.5
    ) -> Dict[str, Any]:
        """
        Executa simulação completa de FL para uma federação.
        
        Args:
            federation_data: Dados particionados da federação
            num_rounds: Número de rodadas de comunicação
            target_metric: Métrica para monitoramento de convergência
            target_threshold: Limiar de performance desejado
            fraction_fit: Fração de clientes para treino por rodada
            fraction_evaluate: Fração de clientes para avaliação
            
        Returns:
            Dict com métricas finais, histórico e análise de convergência
        """
        
        # Extrai estruturas de dados da federação
        client_indices = federation_data['client_indices']
        train_dataset_ref = federation_data['train_dataset_ref']
        test_dataset = federation_data['test_dataset']
        
        print(f"Iniciando simulação FL com {len(client_indices)} clientes...")
        
        # DataLoader global para teste final
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 32) * 2,
            shuffle=False
        )
        
        def client_fn(context: Context) -> Client:
            """
            Factory function para criar clientes sob demanda.
            
            Cada cliente recebe seus dados específicos baseado
            no partition_id atribuído pelo Flower.
            """
            
            # Identifica qual cliente está sendo criado
            partition_id = context.node_config["partition-id"]
            
            # Cria Subset específico deste cliente
            indices_do_cliente = client_indices[partition_id]
            dataset_do_cliente = Subset(train_dataset_ref, indices_do_cliente)

            # Split treino/validação local (80/20)
            train_size = int(0.8 * len(dataset_do_cliente))
            val_size = len(dataset_do_cliente) - train_size
            
            # Proteção contra clientes com pouquíssimos dados
            if train_size < 1 or val_size < 1:
                train_subset, val_subset = dataset_do_cliente, Subset(train_dataset_ref, [])
            else:
                train_subset, val_subset = torch.utils.data.random_split(
                    dataset_do_cliente, [train_size, val_size]
                )
            
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_subset, batch_size=32)

            # Instancia modelo e cliente
            model = get_model(
                self.model_name,
                num_classes=self.dataset_info['num_classes']
            ).to(self.device)
            
            # Escolhe tipo de cliente baseado no dataset
            client = FederatedClient(  # ou MedMNISTClient para datasets médicos
                model=model,
                trainloader=train_loader,
                valloader=val_loader,
                device=self.device,
                epochs=self.config['epochs_per_round'],
                learning_rate=self.config['learning_rate'],
                num_classes=self.dataset_info['num_classes']
            )
            
            return client.to_client()

        # Configuração das aplicações cliente/servidor
        client_app = ClientApp(client_fn=client_fn)
        
        # Recursos computacionais por cliente
        backend_config = {
            "client_resources": {
                "num_cpus": self.config.get('client_cpus', 20),
                "num_gpus": self.config.get('client_gpus', 1)
            }
        }
        
        # Estratégia de agregação com persistência de estado
        strategy = CustomFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=int(len(client_indices) * fraction_fit),
            min_evaluate_clients=max(1, int(len(client_indices) * fraction_evaluate)),
            min_available_clients=len(client_indices),
            evaluate_metrics_aggregation_fn=self._weighted_average
        )
        
        # Wrapper para injetar estratégia no servidor
        def server_fn_wrapper(strategy_instance):
            def server_fn(context: Context):
                config = ServerConfig(num_rounds=num_rounds)
                return ServerAppComponents(strategy=strategy_instance, config=config)
            return server_fn

        server_app = ServerApp(server_fn=server_fn_wrapper(strategy))
        
        # Executa simulação FL
        history = run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=len(client_indices),
            backend_config=backend_config
        )
        
        # Recupera parâmetros finais da estratégia
        final_parameters_obj = strategy.final_parameters
        
        # Converte formato Flower para arrays NumPy
        final_parameters_ndarrays = parameters_to_ndarrays(final_parameters_obj)
        
        # Avalia modelo final no teste global
        final_metrics = self._evaluate_global_model(
            final_parameters_ndarrays,
            test_loader, 
            target_metric
        )
        
        # Analisa histórico para detectar convergência
        processed_history = self._process_history(
            history, target_metric, target_threshold
        )
        
        return {
            "final_metrics": final_metrics,
            "history": processed_history,
            "convergence_round": processed_history.get("convergence_round", None),
            "target_achieved": final_metrics.get(target_metric, 0) >= target_threshold
        }
    
    def _validate_all_datasets(self, client_datasets, test_dataset):
        """
        Validação pré-simulação para detectar incompatibilidades de dados.
        
        Verifica se todos os datasets (clientes + teste) têm labels
        compatíveis com a arquitetura do modelo.
        """
        print(f"🔍 Validando compatibilidade de todos os datasets...")
        
        all_labels = set()
        
        # Amostra labels de cada cliente
        for i, dataset in enumerate(client_datasets):
            client_labels = set()
            sample_count = 0
            
            for data, label in dataset:
                label_val = label.item() if hasattr(label, 'item') else label
                client_labels.add(label_val)
                all_labels.add(label_val)
                sample_count += 1
                
                # Amostragem para eficiência
                if sample_count > 100:
                    break
            
            print(f"  👤 Cliente {i+1}: {len(client_labels)} classes únicas, "
                  f"range: {min(client_labels)}-{max(client_labels)}")
        
        # Verifica dataset de teste
        test_labels = set()
        for data, label in test_dataset:
            label_val = label.item() if hasattr(label, 'item') else label
            test_labels.add(label_val)
            all_labels.add(label_val)
        
        print(f"  🧪 Teste: {len(test_labels)} classes únicas")
        print(f"  📊 TOTAL: {len(all_labels)} classes únicas no dataset completo")
        print(f"  📈 Range global: {min(all_labels)} a {max(all_labels)}")
        
        # Validação crítica
        expected_classes = self.dataset_info['num_classes']
        if max(all_labels) >= expected_classes:
            raise ValueError(
                f"❌ CRÍTICO: Dataset contém classe {max(all_labels)}, "
                f"mas modelo foi configurado para {expected_classes} classes (0-{expected_classes-1}). "
                f"Verifique self.dataset_info['num_classes'] ou normalize os labels."
            )
        
        print(f"  ✅ Validação passou! Datasets compatíveis com modelo ({expected_classes} classes)")


    def _weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """
        Agregação ponderada de métricas dos clientes.
        
        Pondera pela quantidade de amostras de cada cliente
        para refletir contribuição proporcional.
        """
        accuracies = []
        aucs = []
        weights = []
        
        for num_examples, m in metrics:
            weights.append(num_examples)
            if "accuracy" in m:
                accuracies.append(m["accuracy"] * num_examples)
            if "auc_macro" in m:
                aucs.append(m["auc_macro"] * num_examples)
        
        total_weight = sum(weights)
        aggregated = {}
        
        if accuracies:
            aggregated["accuracy"] = sum(accuracies) / total_weight
        if aucs:
            aggregated["auc_macro"] = sum(aucs) / total_weight
        
        return aggregated
    
    def _evaluate_global_model(
        self,
        parameters: Optional[NDArrays],
        test_loader: DataLoader,
        target_metric: str
    ) -> Dict[str, float]:
        """
        Avalia modelo global agregado no conjunto de teste.
        
        Calcula métricas finais para análise de performance
        do modelo federado resultante.
        """
        # Instancia modelo limpo para avaliação
        model = get_model(
            self.model_name,
            num_classes=self.dataset_info['num_classes']
        ).to(self.device)
        
        # Carrega parâmetros finais se disponíveis
        if parameters is not None:
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
        else:
            print("AVISO: Parâmetros finais não encontrados. Avaliando modelo não treinado.")
        
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        all_outputs = []
        all_targets = []
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                
                # Métricas de classificação
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Coleta para AUC se necessário
                if target_metric == "auc_macro":
                    probs = torch.softmax(output, dim=1)
                    all_outputs.append(probs.cpu().numpy())
                    all_targets.append(target.cpu().numpy())
        
        metrics = {
            "test_loss": total_loss / total if total > 0 else 0,
            "accuracy": correct / total if total > 0 else 0
        }
        
        # Cálculo de AUC para datasets médicos
        if target_metric == "auc_macro" and all_outputs:
            all_outputs = np.vstack(all_outputs)
            all_targets = np.concatenate(all_targets)
            
            try:
                num_classes = all_outputs.shape[1]
                targets_onehot = np.zeros((len(all_targets), num_classes))
                targets_onehot[np.arange(len(all_targets)), all_targets] = 1
                
                auc_macro = roc_auc_score(targets_onehot, all_outputs, average='macro')
                metrics["auc_macro"] = auc_macro
            except:
                metrics["auc_macro"] = 0.5
        
        return metrics
    
    def _process_history(
        self,
        history: Any,
        target_metric: str,
        target_threshold: float
    ) -> Dict[str, Any]:
        """
        Processa histórico de treinamento para análise de convergência.
        
        Identifica rodada de convergência (primeira vez que atinge threshold)
        e melhor performance alcançada.
        """
        processed = {
            "rounds": [],
            "convergence_round": None,
            "best_round": None,
            "best_metric": 0.0
        }
        
        # Extrai métricas por rodada do histórico Flower
        if hasattr(history, 'metrics_distributed'):
            for round_num, metrics in enumerate(history.metrics_distributed.items(), 1):
                round_metrics = {
                    "round": round_num,
                    "accuracy": metrics.get("accuracy", 0),
                }
                
                if "auc_macro" in metrics:
                    round_metrics["auc_macro"] = metrics["auc_macro"]
                
                processed["rounds"].append(round_metrics)
                
                # Detecta primeira convergência ao threshold
                metric_value = round_metrics.get(target_metric, 0)
                if metric_value >= target_threshold and processed["convergence_round"] is None:
                    processed["convergence_round"] = round_num
                
                # Rastreia melhor performance
                if metric_value > processed["best_metric"]:
                    processed["best_metric"] = metric_value
                    processed["best_round"] = round_num
        
        return processed

class FederatedExperimentRunner:
    """
    Interface de alto nível para execução de experimentos FL.
    
    Gerencia múltiplas simulações e agrega resultados para análise.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = DEVICE
        self.results = []
        
    def run_single_experiment(
        self,
        federation_data: Dict[str, Any],
        model_name: str = "resnet34"
    ) -> Dict[str, Any]:
        """
        Executa um experimento FL individual.
        
        Args:
            federation_data: Dados particionados de uma federação
            model_name: Arquitetura do modelo (ignorado, usa ResNet34)
            
        Returns:
            Resultados completos do experimento FL
        """
        # Configura simulador com parâmetros do experimento
        simulator = FederatedLearningSimulator(
            model_name=model_name,
            dataset_info=federation_data['dataset_info'],
            config=self.config['fl_training'],
            device=self.device
        )
        
        # Executa simulação FL completa
        fl_results = simulator.run_federation(
            federation_data=federation_data,
            num_rounds=self.config['fl_training']['num_rounds'],
            target_metric=federation_data['dataset_info']['target_metric'],
            target_threshold=self.config.get('target_threshold', 0.75),
            fraction_fit=self.config['fl_training']['fraction_fit'],
            fraction_evaluate=self.config['fl_training']['fraction_evaluate']
        )
        
        return fl_results
    
    def run_multiple_experiments(
        self,
        federations: List[Dict[str, Any]],
        experiment_name: str
    ) -> List[Dict[str, Any]]:
        """
        Executa batch de experimentos para múltiplas federações.
        
        Útil para análise estatística com múltiplas repetições
        da mesma configuração.
        """
        results = []
        
        for i, federation in enumerate(federations):
            print(f"\n--- Experimento {i+1}/{len(federations)} ---")
            
            result = self.run_single_experiment(federation)
            
            # Adiciona metadados para rastreabilidade
            result['experiment_name'] = experiment_name
            result['federation_id'] = federation['federation_id']
            result['partition_metadata'] = federation['partition_metadata']
            
            results.append(result)
            
            # Log de progresso
            final_metric = result['final_metrics'].get(
                federation['dataset_info']['target_metric'], 0
            )
            print(f"  Métrica final: {final_metric:.4f}")
            
            if result.get('convergence_round'):
                print(f"  Convergência na rodada: {result['convergence_round']}")
        
        return results