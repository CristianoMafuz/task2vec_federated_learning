"""
Script principal para experimentos Task2Vec + Federated Learning
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FederatedClient(NumPyClient):
    """Cliente Flower para aprendizado federado."""
    
    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        epochs: int = 3,
        learning_rate: float = 0.01,
        num_classes: int = None  # NOVO PARÂMETRO
    ):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        # # VERIFICAÇÃO: Validar classes nos dados
        # self._validate_dataset_classes()
    
    
    def _validate_dataset_classes(self):
        """Valida se as classes nos dados são compatíveis com o modelo."""
        print(f"🔍 Validando classes do dataset...")
        
        # Verificar classes no dataset de treino
        train_labels = []
        for _, labels in self.trainloader:
            train_labels.extend(labels.tolist())
        
        val_labels = []
        for _, labels in self.valloader:
            val_labels.extend(labels.tolist())
        
        all_labels = train_labels + val_labels
        unique_labels = set(all_labels)
        
        print(f"  Classes encontradas: {sorted(unique_labels)}")
        print(f"  Range de classes: {min(unique_labels)} a {max(unique_labels)}")
        print(f"  Modelo espera: 0 a {self.num_classes - 1}")
        
        # Verificar se há problemas
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
        """Retorna os parâmetros do modelo."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays):
        """Seta o os parametros."""
        model_keys = list(self.model.state_dict().keys())

        # Garantir que shapes batem
        for (name, param), received in zip(self.model.state_dict().items(), parameters):
            if list(param.shape) != list(received.shape):
                print(f"SHAPE MISMATCH in {name}: expected {param.shape}, got {received.shape}")
                raise ValueError("Shape mismatch detected!")

        # Se estiver tudo certo, cria o OrderedDict normalmente
        params_dict = zip(model_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Any]) -> Tuple[NDArrays, int, Dict]:
        """Treina o modelo local."""
        self.set_parameters(parameters)
        
        # Treinar modelo
        train_loss = self._train()
        
        # Avaliar no conjunto de validação local
        val_loss, val_accuracy = self._evaluate(self.valloader)
        
        metrics = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy)
        }
        
        return self.get_parameters({}), len(self.trainloader.dataset), metrics
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Any]) -> Tuple[float, int, Dict]:
        """Avalia o modelo."""
        self.set_parameters(parameters)
        loss, accuracy = self._evaluate(self.valloader)
        
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}
    
    def _train(self) -> float:
        """Executa o treinamento local."""
        if len(self.trainloader.dataset) < 2:
            print(f"⚠️ Cliente com dataset muito pequeno ({len(self.trainloader.dataset)} amostras). Pulando treino.")
            return  # ou continue
        
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                
                # DEBUGGING: Verificar batch antes do forward pass
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
        """Avalia o modelo em um conjunto de dados."""
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
    """Cliente especializado para datasets MedMNIST (usa AUC como métrica)."""
    
    def _evaluate_with_auc(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Avalia com AUC para tarefas multi-classe."""
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
                
                # Coletar probabilidades para AUC
                probs = torch.softmax(output, dim=1)
                all_outputs.append(probs.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # Calcular métricas
        all_outputs = np.vstack(all_outputs)
        all_targets = np.concatenate(all_targets)
        
        avg_loss = total_loss / len(all_targets) if len(all_targets) > 0 else 0.0
        
        # Calcular accuracy
        predictions = np.argmax(all_outputs, axis=1)
        accuracy = np.mean(predictions == all_targets)
        
        # Calcular AUC macro
        try:
            # One-hot encode targets
            num_classes = all_outputs.shape[1]
            targets_onehot = np.zeros((len(all_targets), num_classes))
            targets_onehot[np.arange(len(all_targets)), all_targets] = 1
            
            auc_macro = roc_auc_score(targets_onehot, all_outputs, average='macro')
        except:
            auc_macro = 0.5  # Valor padrão se AUC não puder ser calculado
        
        return avg_loss, accuracy, auc_macro
    
    def fit(self, parameters: NDArrays, config: Dict[str, Any]) -> Tuple[NDArrays, int, Dict]:
        """Treina o modelo local com métricas AUC."""
        self.set_parameters(parameters)
        
        # Treinar modelo
        train_loss = self._train()
        
        # Avaliar com AUC
        val_loss, val_accuracy, val_auc = self._evaluate_with_auc(self.valloader)
        
        metrics = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
            "val_auc": float(val_auc)
        }
        
        return self.get_parameters({}), len(self.trainloader.dataset), metrics
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Any]) -> Tuple[float, int, Dict]:
        """Avalia o modelo com AUC."""
        self.set_parameters(parameters)
        loss, accuracy, auc = self._evaluate_with_auc(self.valloader)
        
        return float(loss), len(self.valloader.dataset), {
            "accuracy": float(accuracy),
            "auc_macro": float(auc)
        }

class CustomFedAvg(FedAvg):
    """Estratégia FedAvg customizada com logging adicional."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_metrics = []
        self.final_parameters = None # <<< NOVA LINHA
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException]
    ):
        """Agrega resultados de fit com logging."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Salva os parâmetros agregados mais recentes
            self.final_parameters = aggregated_parameters
        
        # Coletar métricas dos clientes
        if results:
            metrics = {
                "round": server_round,
                "num_clients": len(results),
                "avg_train_loss": np.mean([r.metrics.get("train_loss", 0) for _, r in results]),
                "avg_val_loss": np.mean([r.metrics.get("val_loss", 0) for _, r in results]),
                "avg_val_accuracy": np.mean([r.metrics.get("val_accuracy", 0) for _, r in results])
            }
            
            # Adicionar AUC se disponível
            auc_values = [r.metrics.get("val_auc", None) for _, r in results]
            auc_values = [v for v in auc_values if v is not None]
            if auc_values:
                metrics["avg_val_auc"] = np.mean(auc_values)
            
            self.round_metrics.append(metrics)
        
        # return aggregated
        return aggregated_parameters, aggregated_metrics

class FederatedLearningSimulator:
    """Simulador de aprendizado federado."""
    
    def __init__(
        self,
        model_name: str,
        dataset_info: Dict[str, Any],
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        # self.model_name = model_name
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
        Executa simulação de aprendizado federado.
        
        Returns:
            Dicionário com métricas finais e histórico
        """
        
        # Extrai as informações necessárias do dicionário
        client_indices = federation_data['client_indices']
        train_dataset_ref = federation_data['train_dataset_ref']
        test_dataset = federation_data['test_dataset']
        
        print(f"Iniciando simulação FL com {len(client_indices)} clientes...")
        
        # DataLoader de teste global (isso pode continuar aqui)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 32) * 2,
            shuffle=False
        )
        
        
        def client_fn(context: Context) -> Client:
            """Cria um cliente Flower e SEUS PRÓPRIOS DADOS sob demanda."""
            
            # 1. Obter ID do cliente que está sendo criado
            partition_id = context.node_config["partition-id"]
            
            # 2. Criar o Subset DESTE cliente específico
            indices_do_cliente = client_indices[partition_id]
            dataset_do_cliente = Subset(train_dataset_ref, indices_do_cliente)

            # 3. Dividir em treino/val e criar DataLoaders SÓ PARA ESTE CLIENTE
            train_size = int(0.8 * len(dataset_do_cliente))
            val_size = len(dataset_do_cliente) - train_size
            
            # Adiciona uma verificação para clientes com pouquíssimas amostras
            if train_size < 1 or val_size < 1:
                # Se não houver dados suficientes, usa o dataset inteiro para treino e um placeholder para validação
                 train_subset, val_subset = dataset_do_cliente, Subset(train_dataset_ref, [])
            else:
                train_subset, val_subset = torch.utils.data.random_split(
                    dataset_do_cliente, [train_size, val_size]
                )
            
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_subset, batch_size=32)

            # 4. Criar o modelo e o cliente (como antes, mas com os loaders locais)
            model = get_model(
                self.model_name,
                num_classes=self.dataset_info['num_classes']
            ).to(self.device)
            
            client = FederatedClient( # ou MedMNISTClient
                model=model,
                trainloader=train_loader,
                valloader=val_loader,
                device=self.device,
                epochs=self.config['epochs_per_round'],
                learning_rate=self.config['learning_rate'],
                num_classes=self.dataset_info['num_classes']
            )
            
            return client.to_client()

        # Criar apps
        client_app = ClientApp(client_fn=client_fn)
        
        # Configuração de recursos
        backend_config = {
            "client_resources": {
                "num_cpus": self.config.get('client_cpus', 20),
                "num_gpus": self.config.get('client_gpus', 1)
            }
        }
        
        strategy = CustomFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=int(len(client_indices) * fraction_fit),
            min_evaluate_clients=max(1, int(len(client_indices) * fraction_evaluate)),
            min_available_clients=len(client_indices),
            evaluate_metrics_aggregation_fn=self._weighted_average
        )
        
        # E passá-la para o server_fn
        def server_fn_wrapper(strategy_instance):
            def server_fn(context: Context):
                config = ServerConfig(num_rounds=num_rounds)
                return ServerAppComponents(strategy=strategy_instance, config=config)
            return server_fn

        server_app = ServerApp(server_fn=server_fn_wrapper(strategy))
        
        history = run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=len(client_indices),
            backend_config=backend_config
        )
        
        # 2. Agora, a `strategy` contém os parâmetros finais
        final_parameters_obj = strategy.final_parameters
        
        # Convertendo o objeto Parameters do Flower para uma lista de arrays NumPy
        final_parameters_ndarrays = parameters_to_ndarrays(final_parameters_obj)
        
        # 3. Avalie o modelo com os pesos corretos
        final_metrics = self._evaluate_global_model(
            final_parameters_ndarrays,
            test_loader, 
            target_metric
        )
        
        # Processar histórico e encontrar convergência
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
        """Valida todos os datasets antes de começar o treinamento."""
        print(f"🔍 Validando compatibilidade de todos os datasets...")
        
        all_labels = set()
        
        # Verificar datasets dos clientes
        for i, dataset in enumerate(client_datasets):
            client_labels = set()
            sample_count = 0
            
            for data, label in dataset:
                label_val = label.item() if hasattr(label, 'item') else label
                client_labels.add(label_val)
                all_labels.add(label_val)
                sample_count += 1
                
                # Verificar apenas algumas amostras por eficiência
                if sample_count > 100:
                    break
            
            print(f"  Cliente {i+1}: {len(client_labels)} classes únicas, "
                  f"range: {min(client_labels)}-{max(client_labels)}")
        
        # Verificar dataset de teste
        test_labels = set()
        for data, label in test_dataset:
            label_val = label.item() if hasattr(label, 'item') else label
            test_labels.add(label_val)
            all_labels.add(label_val)
        
        print(f"  Teste: {len(test_labels)} classes únicas")
        print(f"  TOTAL: {len(all_labels)} classes únicas no dataset completo")
        print(f"  Range global: {min(all_labels)} a {max(all_labels)}")
        
        # Verificar compatibilidade com modelo
        expected_classes = self.dataset_info['num_classes']
        if max(all_labels) >= expected_classes:
            raise ValueError(
                f"❌ CRÍTICO: Dataset contém classe {max(all_labels)}, "
                f"mas modelo foi configurado para {expected_classes} classes (0-{expected_classes-1}). "
                f"Verifique self.dataset_info['num_classes'] ou normalize os labels."
            )
        
        print(f"  ✅ Validação passou! Datasets compatíveis com modelo ({expected_classes} classes)")


    def _weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """Agregação ponderada de métricas."""
        # Separar métricas por tipo
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
        """Avalia o modelo global final."""
        # Criar modelo para avaliação
        model = get_model(
            self.model_name,
            num_classes=self.dataset_info['num_classes']
        ).to(self.device)
        
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
                
                # Accuracy
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Para AUC
                if target_metric == "auc_macro":
                    probs = torch.softmax(output, dim=1)
                    all_outputs.append(probs.cpu().numpy())
                    all_targets.append(target.cpu().numpy())
        
        metrics = {
            "test_loss": total_loss / total if total > 0 else 0,
            "accuracy": correct / total if total > 0 else 0
        }
        
        # Calcular AUC se necessário
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
        """Processa o histórico da simulação."""
        processed = {
            "rounds": [],
            "convergence_round": None,
            "best_round": None,
            "best_metric": 0.0
        }
        
        # Extrair métricas por rodada
        if hasattr(history, 'metrics_distributed'):
            for round_num, metrics in enumerate(history.metrics_distributed.items(), 1):
                round_metrics = {
                    "round": round_num,
                    "accuracy": metrics.get("accuracy", 0),
                }
                
                if "auc_macro" in metrics:
                    round_metrics["auc_macro"] = metrics["auc_macro"]
                
                processed["rounds"].append(round_metrics)
                
                # Verificar convergência
                metric_value = round_metrics.get(target_metric, 0)
                if metric_value >= target_threshold and processed["convergence_round"] is None:
                    processed["convergence_round"] = round_num
                
                # Rastrear melhor métrica
                if metric_value > processed["best_metric"]:
                    processed["best_metric"] = metric_value
                    processed["best_round"] = round_num
        
        return processed

class FederatedExperimentRunner:
    """Executor de experimentos federados."""
    
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
        Executa um único experimento federado.
        
        Args:
            federation_data: Dados da federação (do FederatedDataManager)
            model_name: Nome do modelo a usar
            
        Returns:
            Resultados do experimento
        """
        # Configurar simulador
        simulator = FederatedLearningSimulator(
            model_name=model_name,
            dataset_info=federation_data['dataset_info'],
            config=self.config['fl_training'],
            device=self.device
        )
        
        # Executar simulação
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
        Executa múltiplos experimentos.
        
        Args:
            federations: Lista de federações para experimentar
            experiment_name: Nome do experimento
            
        Returns:
            Lista de resultados
        """
        results = []
        
        for i, federation in enumerate(federations):
            print(f"\n--- Experimento {i+1}/{len(federations)} ---")
            
            result = self.run_single_experiment(federation)
            
            # Adicionar metadados
            result['experiment_name'] = experiment_name
            result['federation_id'] = federation['federation_id']
            result['partition_metadata'] = federation['partition_metadata']
            
            results.append(result)
            
            # Log progresso
            final_metric = result['final_metrics'].get(
                federation['dataset_info']['target_metric'], 0
            )
            print(f"  Métrica final: {final_metric:.4f}")
            
            if result.get('convergence_round'):
                print(f"  Convergência na rodada: {result['convergence_round']}")
        
        return results