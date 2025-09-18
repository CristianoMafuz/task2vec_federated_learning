# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from collections import defaultdict
import torch
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = defaultdict(int)
        self.avg = defaultdict(float)
        self.sum = defaultdict(int)
        self.count = defaultdict(int)

    def update(self, n=1, **val):
        for k in val:
            self.val[k] = val[k]
            self.sum[k] += val[k] * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]


def set_batchnorm_mode(model, train=True):
    """Allows to set batch_norm layer mode to train or eval, independendtly on the mode of the model."""
    def _set_batchnorm_mode(module):
        if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
            if train:
                module.train()
            else:
                module.eval()

    model.apply(_set_batchnorm_mode)


def get_error(output, target):
    pred = output.argmax(dim=1)
    correct = pred.eq(target).float().sum()
    return float((1. - correct / output.size(0)) * 100.)


def adjust_learning_rate(optimizer, epoch, optimizer_cfg):
    lr = optimizer_cfg.lr * (0.1 ** np.less(optimizer_cfg.schedule, epoch).sum())
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_device(model: torch.nn.Module):
    return next(model.parameters()).device





import torch
import numpy as np
import random
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import hashlib
from datetime import datetime

def set_reproducibility(seed: int):
    """Define seeds para reprodutibilidade total."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# def get_device():
#     """Retorna o dispositivo disponível."""
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_experiment_id(config: Dict[str, Any]) -> str:
    """Cria um ID único para o experimento baseado nos parâmetros."""
    config_str = json.dumps(config, sort_keys=True)
    hash_obj = hashlib.md5(config_str.encode())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{hash_obj.hexdigest()[:8]}"

def save_results(results: pd.DataFrame, output_dir: str, experiment_id: str):
    """Salva os resultados em CSV e Parquet."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_path / f"results_{experiment_id}.csv"
    parquet_path = output_path / f"results_{experiment_id}.parquet"
    
    results.to_csv(csv_path, index=False)
    results.to_parquet(parquet_path, index=False)
    
    print(f"Resultados salvos em:\n  - {csv_path}\n  - {parquet_path}")

def calculate_entropy(labels: np.ndarray) -> float:
    """Calcula a entropia de Shannon de uma distribuição de rótulos."""
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy

def log_experiment_metadata(config: Dict[str, Any], output_dir: str, experiment_id: str):
    """Registra metadados do experimento."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "environment": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    }
    
    metadata_path = output_path / f"metadata_{experiment_id}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadados salvos em: {metadata_path}")

def print_progress(current: int, total: int, prefix: str = "", suffix: str = ""):
    """Imprime uma barra de progresso simples."""
    bar_length = 50
    progress = current / total
    arrow = '=' * int(round(progress * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    print(f'\r{prefix} [{arrow}{spaces}] {int(progress * 100)}% {suffix}', end='')
    if current == total:
        print()

class ExperimentLogger:
    """Logger para rastrear métricas durante os experimentos."""
    
    def __init__(self, output_dir: str, experiment_id: str):
        self.output_dir = Path(output_dir)
        self.experiment_id = experiment_id
        self.logs = []
        
    def log(self, metrics: Dict[str, Any]):
        """Adiciona métricas ao log."""
        metrics['timestamp'] = datetime.now().isoformat()
        self.logs.append(metrics)
        
    def save(self):
        """Salva os logs em um arquivo JSON."""
        log_path = self.output_dir / f"logs_{self.experiment_id}.json"
        with open(log_path, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"Logs salvos em: {log_path}")