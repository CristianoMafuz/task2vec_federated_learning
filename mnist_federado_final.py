import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
import random
import task_similarity
import copy
import seaborn as sns

from collections import defaultdict, OrderedDict
from typing import List, Tuple

from flwr.common import Context, Metrics
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

from task2vec import Task2Vec
from models import get_model




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 3
ROUNDS = 10  # Reduzido para teste
NUM_CLIENTS = 10
SEED = 42


# CNN para MNIST (1 canal)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Funções auxiliares para parâmetros (baseadas na documentação oficial)
def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    net.train()
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(testloader.dataset) > 0:
        loss /= len(testloader.dataset)
        accuracy = correct / total
    else:
        loss, accuracy = 0.0, 0.0
    return loss, accuracy

def partition_data(dataset, mode="iid", noise_ratio=0.0, test_id=0):
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    class_indices = {i: [] for i in range(10)}
    
    for idx, (_, label) in enumerate(dataset):
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_indices[label].append(idx)

    # Embaralhar ANTES de particionar (determinismo)
    for indices in class_indices.values():
        random.shuffle(indices)

    client_indices = [[] for _ in range(NUM_CLIENTS)]

    if mode == "iid":
        all_indices = [i for indices in class_indices.values() for i in indices]
        random.shuffle(all_indices)
        split = np.array_split(all_indices, NUM_CLIENTS)
        for i in range(NUM_CLIENTS):
            client_indices[i] = list(split[i])

    elif mode == "non-iid":
        for client_id in range(NUM_CLIENTS):
            cls = client_id % 10
            client_indices[client_id] = class_indices[cls]

    elif mode.startswith("non-iid-noise"):
        ratio = noise_ratio
        for client_id in range(NUM_CLIENTS):
            cls = client_id % 10
            main_class_data = class_indices[cls].copy()  # Cópia para não modificar original
            
            # Calcular quantidades
            num_main = int(len(main_class_data) * (1 - ratio))
            num_noise_per_class = int(len(main_class_data) * ratio / 9)  # Dividir ruído entre 9 classes
            
            # Selecionar dados da classe principal
            selected_main = main_class_data[:num_main]
            
            # Selecionar dados de ruído de outras classes
            selected_noise = []
            for other_cls in range(10):
                if other_cls != cls:
                    other_class_data = class_indices[other_cls].copy()
                    # Pegar uma quantidade proporcional de cada classe
                    noise_from_this_class = min(num_noise_per_class, len(other_class_data))
                    selected_noise.extend(other_class_data[:noise_from_this_class])
            
            # Embaralhar o ruído
            random.shuffle(selected_noise)
            
            # Combinar dados principais e ruído
            client_indices[client_id] = selected_main + selected_noise
            
            random.shuffle(client_indices[client_id])

    # Debug: mostrar distribuição dos dados
    print("\n--- Distribuição de dados por cliente ---")
    for i, indices in enumerate(client_indices):
        class_counts = defaultdict(int)
        for idx in indices:
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            class_counts[label] += 1
        print(f"Cliente {i}: {dict(class_counts)} (total: {len(indices)})")
    
    return client_indices

def compute_task_similarity(client_datasets, probe_network, stage="before", client_models=None, test_id=0):
    print(f"\n=== Computando Task2Vec - {stage.upper()} ===")
    
    # GARANTIR DETERMINISMO GLOBAL
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    task_vectors = []
    client_names = [f"Cliente {i+1}" for i in range(len(client_datasets))]
    
    # DEBUG: Amostragem representativa
    print("=== DEBUG: Verificando distribuições dos clientes ===")
    for i, dataset in enumerate(client_datasets):
        class_counts = defaultdict(int)
        
        # Pegar amostras aleatórias
        total_dataset_size = len(dataset)
        sample_size = min(1000, total_dataset_size)
        
        # Gerar índices aleatórios para amostragem
        sample_indices = random.sample(range(total_dataset_size), sample_size)
        
        for idx in sample_indices:
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            class_counts[label] += 1
        
        # Mostrar distribuição percentual
        percentages = {cls: (count/sample_size)*100 for cls, count in class_counts.items()}
        dominant_class = max(class_counts, key=class_counts.get)
        dominant_pct = percentages[dominant_class]
        
        print(f"Cliente {i+1}: classe dominante={dominant_class} ({dominant_pct:.1f}%), "
              f"amostra={sample_size}/{total_dataset_size}")
        print(f"  Top 3 classes: {sorted(percentages.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    for i, dataset in enumerate(client_datasets):
        print(f"\nProcessando Cliente {i+1}...")
        
        # Criar dataset com transform adequado para ResNet
        class TransformedDataset(Dataset):
            def __init__(self, original_dataset, client_id):
                self.dataset = original_dataset
                self.client_id = client_id
                
                # DEBUG: Verificar algumas amostras
                print(f"  Dataset transformado criado para Cliente {client_id+1}")
                if len(original_dataset) > 0:
                    sample_img, sample_label = original_dataset[0]
                    print(f"  Amostra original: shape={sample_img.shape}, label={sample_label}")
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                img, label = self.dataset[idx]
                
                # DEBUG: Verificar transformação
                original_shape = img.shape
                
                # img já é um tensor do transform original
                if img.dim() == 3 and img.size(0) == 1:  # [1, H, W]
                    img = img.repeat(3, 1, 1)  # [3, H, W]
                elif img.dim() == 2:  # [H, W]
                    img = img.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
                
                # Redimensionar para 224x224
                img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
                img = img.squeeze(0)  # Remove batch dimension
                
                # DEBUG: Mostrar transformação apenas para a primeira amostra
                if idx == 0:
                    print(f"  Transformação: {original_shape} → {img.shape}")
                
                return img, label
        
        transformed_dataset = TransformedDataset(dataset, i)
        current_probe_network = copy.deepcopy(probe_network)

        # DEBUG: Verificar se a rede probe está sendo criada corretamente
        print(f"  Criando Task2Vec para Cliente {i+1}...")
        
        try:
            # Extrair Task2Vec embedding
            task_embedding = Task2Vec(current_probe_network, max_samples=1000, skip_layers=6)
            embedding = task_embedding.embed(transformed_dataset)
            
            # DEBUG: Verificar o embedding
            print(f"  Embedding extraído: shape={embedding.hessian.shape}")
            print(f"  Estatísticas do embedding: mean={np.mean(embedding.hessian):.6f}, "
                  f"std={np.std(embedding.hessian):.6f}")
            print(f"  Primeiros 5 valores: {embedding.hessian[:5]}")
            
            task_vectors.append(embedding)
            
        except Exception as e:
            print(f"  ERRO ao processar Cliente {i+1}: {e}")
            # Criar um embedding dummy para não quebrar
            dummy_embedding = type('obj', (object,), {'hessian': np.random.randn(1000)})
            task_vectors.append(dummy_embedding)
    
    # DEBUG: Verificar se todos os embeddings são válidos
    print(f"\n=== DEBUG: Verificando embeddings finais ===")
    for i, embedding in enumerate(task_vectors):
        if hasattr(embedding, 'hessian'):
            print(f"Cliente {i+1}: embedding shape={embedding.hessian.shape}, "
                  f"mean={np.mean(embedding.hessian):.6f}")
        else:
            print(f"Cliente {i+1}: ERRO - embedding inválido")
    
    # Plot da matriz de similaridade
    task_similarity.plot_distance_matrix(
        task_vectors,
        client_names,
        distance="cosine",
        test_id=test_id
    )

    # Calcular e mostrar matriz de similaridade
    n_clients = len(task_vectors)
    similarity_matrix = np.zeros((n_clients, n_clients))

    print(f"\n=== DEBUG: Matriz de Similaridade {stage.upper()} ===")
    for i in range(n_clients):
        for j in range(n_clients):
            if hasattr(task_vectors[i], 'hessian') and hasattr(task_vectors[j], 'hessian'):
                similarity_matrix[i, j] = np.corrcoef(task_vectors[i].hessian, task_vectors[j].hessian)[0, 1]
            else:
                similarity_matrix[i, j] = 0.0  # Valor padrão para embeddings inválidos
                
            print(f"Cliente {i+1} vs Cliente {j+1}: {similarity_matrix[i, j]:.4f}")
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                annot_kws={"size": 8},
                xticklabels=[f"C{i+1}" for i in range(n_clients)],
                yticklabels=[f"C{i+1}" for i in range(n_clients)])
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.title(f"Matriz de Similaridade entre Clientes - {stage.upper()}")
    plt.xlabel("Cliente")
    plt.ylabel("Cliente")
    plt.tight_layout()
    plt.savefig(f'matriz_{stage}_{test_id}.png')
    plt.close()
    
    return task_vectors


# Cliente Flower baseado na documentação oficial
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=EPOCHS)
        return get_parameters(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate accuracy metrics."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def run_experiment(mode, probe_network, noise_ratio=0.0, test_id=0):
    print(f"\n=== Executando experimento: {mode} (noise={noise_ratio}) ===")
    
    # Transform para CNN (1 canal)
    transform_cnn = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Carregar datasets
    train_dataset = datasets.MNIST(root=".", train=True, download=True, transform=transform_cnn)
    test_dataset = datasets.MNIST(root=".", train=False, download=True, transform=transform_cnn)
    
    # Particionar dados
    client_indices = partition_data(train_dataset, mode, noise_ratio, test_id)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Criar datasets de clientes
    client_datasets = []
    
    for i in range(NUM_CLIENTS):
        subset = Subset(train_dataset, client_indices[i])
        client_datasets.append(subset)
    
    # Medir similaridade antes do treinamento federado
    task_vectors_before = compute_task_similarity(client_datasets, probe_network, "antes", test_id=test_id)
    
    print("\n=== Iniciando Treinamento Federado ===")
    
    def client_fn(context: Context) -> Client:
        """Create a Flower client representing a single organization."""
        # Load model
        net = CNN().to(DEVICE)
        
        # Read the partition-id to fetch data partition associated to this node
        partition_id = context.node_config["partition-id"]
        trainloader = DataLoader(client_datasets[partition_id], batch_size=BATCH_SIZE, shuffle=True)
        valloader = DataLoader(client_datasets[partition_id], batch_size=BATCH_SIZE)  # Usando mesmo dataset para simplificar
      
        # Create a single Flower client representing a single organization
        return FlowerClient(net, trainloader, valloader).to_client()

    def server_fn(context: Context) -> ServerAppComponents:
        """Construct components that set the ServerApp behaviour."""
        # Create FedAvg strategy
        strategy = FedAvg(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
            min_fit_clients=NUM_CLIENTS,  # Never sample less than NUM_CLIENTS clients for training
            min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
            min_available_clients=NUM_CLIENTS,  # Wait until all clients are available
            evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate custom metrics
        )
      
        # Configure the server for ROUNDS rounds of training
        config = ServerConfig(num_rounds=ROUNDS)
        return ServerAppComponents(strategy=strategy, config=config)

    # Create ClientApp and ServerApp
    client_app = ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)
    
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    
    # Run simulation
    history = run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )
    
    return None, None

if __name__ == "__main__":
    # Definir seeds uma vez
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    histories = {}
    task_vectors_all = {}
    
    experiments = [
        ("IID", "iid", 0.0, 1),
        ("Non-IID", "non-iid", 0.0, 2),
        ("Noise-5%", "non-iid-noise", 0.05, 3),
        ("Noise-25%", "non-iid-noise", 0.25, 4)
    ]

    # Criar a rede probe
    print("Criando a rede probe para Task2Vec...")
    probe_network_task2vec = get_model('resnet34', pretrained=True, num_classes=10).to(DEVICE)
    
    for name, mode, noise, test_id in experiments:
        print(f"\n{'='*50}")
        print(f"EXPERIMENTO: {name}")
        print(f"{'='*50}")
        # run_experiment(mode, noise)
        hist, vectors_before = run_experiment(mode, probe_network_task2vec, noise, test_id)
        # hist, vectors_before, vectors_after = run_experiment(mode, noise)
        histories[name] = hist
        task_vectors_all[name] = {'before': vectors_before} # Atualizado para o novo formato
        # task_vectors_all[name] = {
        #     'before': vectors_before,
        #     'after': vectors_after
        # }
    
    
    print("\n" + "="*50)
    print("EXPERIMENTOS CONCLUÍDOS!")
    print("="*50)
