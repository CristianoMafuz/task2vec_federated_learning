#!/usr/bin/env python3
"""
Script para análise e visualização dos resultados dos experimentos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ResultsAnalyzer:
    """Analisador de resultados dos experimentos."""
    
    def __init__(self, results_path: str):
        """
        Args:
            results_path: Caminho para o arquivo de resultados (CSV ou Parquet)
        """
        self.results_path = Path(results_path)
        
        # Carregar dados
        if self.results_path.suffix == '.parquet':
            self.df = pd.read_parquet(self.results_path)
        else:
            self.df = pd.read_csv(self.results_path)
        
        self.df['dataset'] = self.df['dataset'].astype(str) 
        # Criar diretório para salvar figuras
        self.output_dir = self.results_path.parent / "analysis_alt"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Dados carregados: {len(self.df)} experimentos")
        print(f"Datasets: {self.df['dataset'].unique().tolist()}")
        print(f"Alphas: {sorted(self.df['alpha'].unique())}")
        print(f"Num clientes: {sorted(self.df['num_clients'].unique())}")
        print(f"Figuras serão salvas em: {self.output_dir}")
    
    def generate_full_report(self):
        """Gera relatório completo com todas as análises."""
        print("\n" + "="*80)
        print("INICIANDO ANÁLISE COMPLETA DOS RESULTADOS")
        print("="*80)
        
        # 1. Estatísticas descritivas
        print("\n1. Gerando estatísticas descritivas...")
        self.descriptive_statistics()
        
        # 2. Análise de correlação
        print("\n2. Analisando correlações...")
        self.correlation_analysis()
        
        # 3. Visualizações principais
        print("\n3. Gerando visualizações principais...")
        self.plot_readiness_vs_performance()
        self.plot_alpha_analysis()
        self.plot_client_size_analysis()
        self.plot_component_analysis()
        
        # 4. Análise por dataset
        print("\n4. Análise por dataset...")
        self.dataset_specific_analysis()
        
        # 5. Análise de convergência
        print("\n5. Análise de convergência...")
        self.convergence_analysis()
        
        # 6. Relatório de predição
        print("\n6. Avaliação de capacidade preditiva...")
        self.prediction_analysis()
        
        print(f"\n{'='*80}")
        print("ANÁLISE CONCLUÍDA!")
        print(f"Todas as figuras foram salvas em: {self.output_dir}")
        print(f"{'='*80}")
    
    def descriptive_statistics(self):
        """Gera estatísticas descritivas dos dados."""
        stats = {
            'readiness': self.df['readiness'].describe(),
            'final_performance': self.df['final_performance'].describe(),
            'convergence_round': self.df[self.df['convergence_round'] > 0]['convergence_round'].describe(),
        }
        
        # Salvar estatísticas
        stats_file = self.output_dir / "descriptive_statistics.txt"
        with open(stats_file, 'w') as f:
            f.write("ESTATÍSTICAS DESCRITIVAS\n")
            f.write("="*50 + "\n\n")
            
            for metric, desc in stats.items():
                f.write(f"{metric.upper()}:\n")
                f.write(str(desc) + "\n\n")
            
            # Taxa de sucesso por dataset
            f.write("TAXA DE SUCESSO POR DATASET:\n")
            for dataset in self.df['dataset'].unique():
                success_rate = self.df[self.df['dataset'] == dataset]['target_achieved'].mean()
                f.write(f"  {dataset}: {success_rate:.2%}\n")
        
        print(f"  Estatísticas salvas em: {stats_file}")
    
    def correlation_analysis(self):
        """Analisa correlações entre readiness e performance."""
        correlations = {}
        
        # Correlação geral
        spearman_corr, spearman_p = spearmanr(self.df['readiness'], self.df['final_performance'])
        pearson_corr, pearson_p = pearsonr(self.df['readiness'], self.df['final_performance'])
        
        correlations['geral'] = {
            'spearman': (spearman_corr, spearman_p),
            'pearson': (pearson_corr, pearson_p)
        }
        
        # Correlação por dataset
        for dataset in self.df['dataset'].unique():
            dataset_df = self.df[self.df['dataset'] == dataset]
            if len(dataset_df) > 3:
                spearman_corr, spearman_p = spearmanr(dataset_df['readiness'], dataset_df['final_performance'])
                pearson_corr, pearson_p = pearsonr(dataset_df['readiness'], dataset_df['final_performance'])
                correlations[dataset] = {
                    'spearman': (spearman_corr, spearman_p),
                    'pearson': (pearson_corr, pearson_p)
                }
        
        # Salvar correlações
        corr_file = self.output_dir / "correlations.txt"
        with open(corr_file, 'w') as f:
            f.write("ANÁLISE DE CORRELAÇÕES\n")
            f.write("="*50 + "\n\n")
            
            for name, corrs in correlations.items():
                f.write(f"{name.upper()}:\n")
                f.write(f"  Spearman: {corrs['spearman'][0]:.4f} (p={corrs['spearman'][1]:.4f})\n")
                f.write(f"  Pearson: {corrs['pearson'][0]:.4f} (p={corrs['pearson'][1]:.4f})\n\n")
        
        print(f"  Correlações salvas em: {corr_file}")
        return correlations
    
    def plot_readiness_vs_performance(self):
        """Plota gráfico de dispersão readiness vs performance."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Readiness vs Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Gráfico geral
        ax = axes[0, 0]
        scatter = ax.scatter(self.df['readiness'], self.df['final_performance'], 
                           c=self.df['alpha'], cmap='viridis', alpha=0.6, s=30)
        ax.set_xlabel('Readiness Index')
        ax.set_ylabel('Final Performance')
        ax.set_title('Geral (colorido por α)')
        plt.colorbar(scatter, ax=ax, label='Alpha')
        
        # Linha de tendência
        z = np.polyfit(self.df['readiness'], self.df['final_performance'], 1)
        p = np.poly1d(z)
        ax.plot(self.df['readiness'], p(self.df['readiness']), "r--", alpha=0.8, linewidth=2)
        
        # 2. Por dataset
        ax = axes[0, 1]
        for dataset in self.df['dataset'].unique():
            dataset_df = self.df[self.df['dataset'] == dataset]
            ax.scatter(dataset_df['readiness'], dataset_df['final_performance'], 
                      label=dataset, alpha=0.7, s=30)
        ax.set_xlabel('Readiness Index')
        ax.set_ylabel('Final Performance')
        ax.set_title('Por Dataset')
        ax.legend()
        
        # 3. Box plots por quartis de readiness
        ax = axes[1, 0]
        self.df['readiness_quartile'] = pd.qcut(self.df['readiness'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        sns.boxplot(data=self.df, x='readiness_quartile', y='final_performance', ax=ax)
        ax.set_title('Performance por Quartil de Readiness')
        ax.set_xlabel('Quartil de Readiness')
        
        # 4. Heatmap de correlação por alpha
        ax = axes[1, 1]
        corr_by_alpha = []
        alpha_values = []
        for alpha in sorted(self.df['alpha'].unique()):
            alpha_df = self.df[self.df['alpha'] == alpha]
            if len(alpha_df) > 3:
                corr, _ = spearmanr(alpha_df['readiness'], alpha_df['final_performance'])
                corr_by_alpha.append(corr)
                alpha_values.append(alpha)
        
        ax.bar(range(len(alpha_values)), corr_by_alpha, color='skyblue')
        ax.set_xticks(range(len(alpha_values)))
        ax.set_xticklabels([f'α={a}' for a in alpha_values], rotation=45)
        ax.set_ylabel('Correlação (Spearman)')
        ax.set_title('Correlação por Valor de Alpha')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_path = self.output_dir / "readiness_vs_performance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Gráfico salvo: {save_path}")
    
    def plot_alpha_analysis(self):
        """Análise específica do impacto do parâmetro alpha."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análise do Parâmetro Alpha (Heterogeneidade)', fontsize=16, fontweight='bold')
        
        # 1. Readiness por alpha
        ax = axes[0, 0]
        sns.boxplot(data=self.df, x='alpha', y='readiness', ax=ax)
        ax.set_title('Distribuição de Readiness por Alpha')
        ax.set_xlabel('Alpha (Heterogeneidade)')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. Performance por alpha
        ax = axes[0, 1]
        sns.boxplot(data=self.df, x='alpha', y='final_performance', ax=ax)
        ax.set_title('Distribuição de Performance por Alpha')
        ax.set_xlabel('Alpha (Heterogeneidade)')
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Taxa de sucesso por alpha
        ax = axes[1, 0]
        success_rates = self.df.groupby('alpha')['target_achieved'].mean()
        ax.bar(range(len(success_rates)), success_rates.values, color='lightcoral')
        ax.set_xticks(range(len(success_rates)))
        ax.set_xticklabels([f'α={a}' for a in success_rates.index], rotation=45)
        ax.set_ylabel('Taxa de Sucesso')
        ax.set_title('Taxa de Sucesso por Alpha')
        ax.set_ylim(0, 1)
        
        # 4. Readiness vs Performance por alpha (heatmap)
        ax = axes[1, 1]
        pivot_data = self.df.groupby(['alpha', pd.qcut(self.df['readiness'], 5)])['final_performance'].mean().unstack()
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax)
        ax.set_title('Performance Média (Readiness vs Alpha)')
        ax.set_xlabel('Quintis de Readiness')
        ax.set_ylabel('Alpha')
        
        plt.tight_layout()
        save_path = self.output_dir / "alpha_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Gráfico salvo: {save_path}")
    
    def plot_client_size_analysis(self):
        """Análise do impacto do número de clientes."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análise do Número de Clientes', fontsize=16, fontweight='bold')
        
        # 1. Readiness por número de clientes
        ax = axes[0, 0]
        sns.boxplot(data=self.df, x='num_clients', y='readiness', ax=ax)
        ax.set_title('Distribuição de Readiness por Nº de Clientes')
        ax.set_xlabel('Número de Clientes')
        
        # 2. Performance por número de clientes
        ax = axes[0, 1]
        sns.boxplot(data=self.df, x='num_clients', y='final_performance', ax=ax)
        ax.set_title('Distribuição de Performance por Nº de Clientes')
        ax.set_xlabel('Número de Clientes')
        
        # 3. Tempo de convergência
        ax = axes[1, 0]
        conv_data = self.df[self.df['convergence_round'] > 0]
        if not conv_data.empty:
            sns.boxplot(data=conv_data, x='num_clients', y='convergence_round', ax=ax)
            ax.set_title('Tempo de Convergência por Nº de Clientes')
            ax.set_xlabel('Número de Clientes')
            ax.set_ylabel('Rodada de Convergência')
        else:
            ax.text(0.5, 0.5, 'Dados de convergência\nnão disponíveis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Tempo de Convergência - Sem Dados')
        
        # 4. Correlação readiness-performance por tamanho
        ax = axes[1, 1]
        correlations = []
        client_sizes = []
        for size in sorted(self.df['num_clients'].unique()):
            size_df = self.df[self.df['num_clients'] == size]
            if len(size_df) > 3:
                corr, _ = spearmanr(size_df['readiness'], size_df['final_performance'])
                correlations.append(corr)
                client_sizes.append(size)
        
        ax.plot(client_sizes, correlations, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Número de Clientes')
        ax.set_ylabel('Correlação (Spearman)')
        ax.set_title('Correlação Readiness-Performance vs Nº Clientes')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / "client_size_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Gráfico salvo: {save_path}")
    
    def plot_component_analysis(self):
        """Análise dos componentes do índice de readiness."""
        components = ['cohesion', 'dispersion', 'coverage', 'avg_entropy', 'density']
        available_components = [c for c in components if c in self.df.columns]
        
        if not available_components:
            print("  Componentes do readiness não encontrados nos dados")
            return
        
        n_components = len(available_components)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Análise dos Componentes do Readiness Index', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        # Correlação de cada componente com performance
        for i, component in enumerate(available_components):
            ax = axes[i]
            ax.scatter(self.df[component], self.df['final_performance'], alpha=0.6, s=30)
            
            # Linha de tendência
            z = np.polyfit(self.df[component], self.df['final_performance'], 1)
            p = np.poly1d(z)
            ax.plot(self.df[component], p(self.df[component]), "r--", alpha=0.8, linewidth=2)
            
            # Correlação
            corr, p_val = spearmanr(self.df[component], self.df['final_performance'])
            ax.set_title(f'{component.title()}\n(ρ = {corr:.3f}, p = {p_val:.3f})')
            ax.set_xlabel(component.replace('_', ' ').title())
            ax.set_ylabel('Final Performance')
        
        # Matriz de correlação dos componentes
        if len(available_components) > 1:
            ax = axes[len(available_components)]
            corr_matrix = self.df[available_components + ['final_performance']].corr(method='spearman')
            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax)
            ax.set_title('Matriz de Correlação (Spearman)')
        
        # Remover subplots não utilizados
        for i in range(len(available_components) + 1, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        save_path = self.output_dir / "component_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Gráfico salvo: {save_path}")
    
    def dataset_specific_analysis(self):
        """Análise específica por dataset."""
        datasets = self.df['dataset'].unique()
        
        for dataset in datasets:
            dataset_df = self.df[self.df['dataset'] == dataset]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Análise Específica - {dataset.upper()}', fontsize=16, fontweight='bold')
            
            # 1. Readiness vs Performance
            ax = axes[0, 0]
            scatter = ax.scatter(dataset_df['readiness'], dataset_df['final_performance'], 
                               c=dataset_df['alpha'], cmap='viridis', alpha=0.7, s=40)
            ax.set_xlabel('Readiness Index')
            ax.set_ylabel('Final Performance')
            ax.set_title('Readiness vs Performance')
            plt.colorbar(scatter, ax=ax, label='Alpha')
            
            # Linha de tendência
            if len(dataset_df) > 1:
                z = np.polyfit(dataset_df['readiness'], dataset_df['final_performance'], 1)
                p = np.poly1d(z)
                ax.plot(dataset_df['readiness'], p(dataset_df['readiness']), "r--", alpha=0.8, linewidth=2)
            
            # 2. Distribuição por alpha
            ax = axes[0, 1]
            alpha_stats = dataset_df.groupby('alpha').agg({
                'readiness': 'mean',
                'final_performance': 'mean',
                'target_achieved': 'mean'
            })
            
            x_pos = range(len(alpha_stats))
            ax.bar([p - 0.2 for p in x_pos], alpha_stats['readiness'], 0.4, label='Readiness', alpha=0.7)
            ax2 = ax.twinx()
            ax2.bar([p + 0.2 for p in x_pos], alpha_stats['final_performance'], 0.4, 
                   label='Performance', color='orange', alpha=0.7)
            
            ax.set_xlabel('Alpha')
            ax.set_ylabel('Readiness Index', color='blue')
            ax2.set_ylabel('Performance', color='orange')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'α={a}' for a in alpha_stats.index])
            ax.set_title('Métricas por Alpha')
            
            # 3. Taxa de sucesso
            ax = axes[1, 0]
            success_by_config = dataset_df.groupby(['num_clients', 'alpha'])['target_achieved'].mean().unstack()
            sns.heatmap(success_by_config, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax)
            ax.set_title('Taxa de Sucesso (Clientes vs Alpha)')
            ax.set_xlabel('Alpha')
            ax.set_ylabel('Número de Clientes')
            
            # 4. Estatísticas resumo
            ax = axes[1, 1]
            ax.axis('off')
            
            # Calcular estatísticas
            corr, p_val = spearmanr(dataset_df['readiness'], dataset_df['final_performance'])
            success_rate = dataset_df['target_achieved'].mean()
            best_config = dataset_df.loc[dataset_df['final_performance'].idxmax()]
            
            stats_text = f"""
ESTATÍSTICAS - {dataset.upper()}

Correlação Readiness-Performance:
  Spearman: {corr:.4f} (p={p_val:.4f})

Taxa de Sucesso Geral: {success_rate:.2%}

Melhor Configuração:
  α = {best_config['alpha']}, {best_config['num_clients']} clientes
  Readiness: {best_config['readiness']:.4f}
  Performance: {best_config['final_performance']:.4f}

Amostras Totais: {len(dataset_df)}
Performance Média: {dataset_df['final_performance'].mean():.4f} ± {dataset_df['final_performance'].std():.4f}
Readiness Médio: {dataset_df['readiness'].mean():.4f} ± {dataset_df['readiness'].std():.4f}
            """
            ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            save_path = self.output_dir / f"dataset_analysis_{dataset.lower()}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Gráfico salvo: {save_path}")
    
    def convergence_analysis(self):
        """Análise de convergência dos experimentos."""
        conv_data = self.df[self.df['convergence_round'] > 0].copy()
        
        if conv_data.empty:
            print("  Dados de convergência não disponíveis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análise de Convergência', fontsize=16, fontweight='bold')
        
        # 1. Readiness vs Tempo de convergência
        ax = axes[0, 0]
        scatter = ax.scatter(conv_data['readiness'], conv_data['convergence_round'], 
                           c=conv_data['alpha'], cmap='viridis', alpha=0.7, s=40)
        ax.set_xlabel('Readiness Index')
        ax.set_ylabel('Rodada de Convergência')
        ax.set_title('Readiness vs Tempo de Convergência')
        plt.colorbar(scatter, ax=ax, label='Alpha')
        
        # 2. Distribuição do tempo de convergência
        ax = axes[0, 1]
        ax.hist(conv_data['convergence_round'], bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(conv_data['convergence_round'].mean(), color='red', linestyle='--', 
                  label=f'Média: {conv_data["convergence_round"].mean():.1f}')
        ax.set_xlabel('Rodada de Convergência')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição do Tempo de Convergência')
        ax.legend()
        
        # 3. Convergência por alpha
        ax = axes[1, 0]
        conv_by_alpha = conv_data.groupby('alpha')['convergence_round'].agg(['mean', 'std']).fillna(0)
        x_pos = range(len(conv_by_alpha))
        ax.bar(x_pos, conv_by_alpha['mean'], yerr=conv_by_alpha['std'], 
               capsize=5, alpha=0.7, color='skyblue')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'α={a}' for a in conv_by_alpha.index])
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Rodada de Convergência (Média ± Std)')
        ax.set_title('Tempo de Convergência por Alpha')
        
        # 4. Performance final vs tempo de convergência
        ax = axes[1, 1]
        ax.scatter(conv_data['convergence_round'], conv_data['final_performance'], 
                  alpha=0.7, s=40)
        
        # Linha de tendência
        if len(conv_data) > 1:
            z = np.polyfit(conv_data['convergence_round'], conv_data['final_performance'], 1)
            p = np.poly1d(z)
            ax.plot(conv_data['convergence_round'], p(conv_data['convergence_round']), 
                   "r--", alpha=0.8, linewidth=2)
        
        corr, p_val = spearmanr(conv_data['convergence_round'], conv_data['final_performance'])
        ax.set_xlabel('Rodada de Convergência')
        ax.set_ylabel('Performance Final')
        ax.set_title(f'Performance vs Tempo de Convergência\n(ρ = {corr:.3f}, p = {p_val:.3f})')
        
        plt.tight_layout()
        save_path = self.output_dir / "convergence_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Gráfico salvo: {save_path}")
    
    def prediction_analysis(self):
        """Avalia a capacidade preditiva do readiness index."""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        # Preparar features
        feature_cols = ['readiness', 'num_clients', 'alpha']
        
        components = ['cohesion', 'dispersion', 'coverage', 'avg_entropy', 'density']
        available_components = [c for c in components if c in self.df.columns]
        feature_cols.extend(available_components)
        
        X = self.df[feature_cols].fillna(0)
        y = self.df['final_performance']
        
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            results[name] = {'cv_r2_mean': cv_scores.mean(), 'cv_r2_std': cv_scores.std()}
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        feature_importance = dict(zip(feature_cols, rf_model.feature_importances_))
                
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Avaliação de Capacidade Preditiva', fontsize=16, fontweight='bold')
        
        # 1. Scores dos modelos
        ax = axes[0]
        model_names = list(results.keys())
        means = [results[name]['cv_r2_mean'] for name in model_names]
        stds = [results[name]['cv_r2_std'] for name in model_names]
        x_pos = range(len(model_names))
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=['lightblue', 'lightcoral'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names)
        ax.set_ylabel('R² Score (CV)')
        ax.set_title('Performance dos Modelos Preditivos')
        ax.set_ylim(0, 1)
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
            
        # 2. Importância das features
        ax = axes[1]
        N_TOP_FEATURES = 15
        
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        sorted_idx = np.argsort(importance)[::-1]
        
        top_features = [features[i] for i in sorted_idx][:N_TOP_FEATURES]
        top_importance = [importance[i] for i in sorted_idx][:N_TOP_FEATURES]
        
        top_features.reverse()
        top_importance.reverse()
        
        y_pos = range(len(top_features))
        ax.barh(y_pos, top_importance, alpha=0.7, color='lightgreen')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importância da Feature')
        ax.set_title(f'Top {min(N_TOP_FEATURES, len(features))} Features Mais Importantes') # Mais robusto

        fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.1, wspace=0.3)

        # SALVANDO A FIGURA:
        save_path = self.output_dir / "prediction_analysis.png"
        plt.savefig(save_path, dpi=300) 
        
        plt.close()
        print(f"  Gráfico salvo: {save_path}")
        
        pred_file = self.output_dir / "prediction_results.txt"
        with open(pred_file, 'w') as f:
            f.write("ANÁLISE DE CAPACIDADE PREDITIVA\n")
            f.write("="*50 + "\n\n")
            f.write("PERFORMANCE DOS MODELOS (Cross-Validation R²):\n")
            for name, result in results.items():
                f.write(f"  {name}: {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}\n")
            f.write(f"\nIMPORTÂNCIA DAS FEATURES (Random Forest) - Todas as {len(feature_importance)} features:\n")
            all_features_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, imp) in enumerate(all_features_sorted, 1):
                f.write(f"  {i:2d}. {feature}: {imp:.4f}\n")
        
        print(f"  Resultados de predição salvos em: {pred_file}")
    
    def task2vec_detailed_analysis(self):
        """Análise detalhada específica do Task2Vec e seus componentes."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Análise Detalhada Task2Vec - Índice de Readiness', fontsize=16, fontweight='bold')
        
        # 1. Distribuição do Readiness Index por dataset
        ax = axes[0, 0]
        for dataset in self.df['dataset'].unique():
            dataset_df = self.df[self.df['dataset'] == dataset]
            ax.hist(dataset_df['readiness'], alpha=0.6, label=dataset, bins=20)
        ax.set_xlabel('Readiness Index')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição do Readiness Index por Dataset')
        ax.legend()
        
        # 2. Readiness vs Alpha com intervalo de confiança
        ax = axes[0, 1]
        alpha_stats = self.df.groupby('alpha').agg({
            'readiness': ['mean', 'std'],
            'readiness_ci_lower': 'mean',
            'readiness_ci_upper': 'mean'
        }).round(4)
        
        alphas = alpha_stats.index
        means = alpha_stats[('readiness', 'mean')]
        ci_lower = alpha_stats[('readiness_ci_lower', 'mean')]
        ci_upper = alpha_stats[('readiness_ci_upper', 'mean')]
        
        ax.errorbar(alphas, means, 
                   yerr=[means - ci_lower, ci_upper - means], 
                   fmt='o-', capsize=5, linewidth=2, markersize=8)
        ax.set_xlabel('Alpha (Heterogeneidade)')
        ax.set_ylabel('Readiness Index')
        ax.set_title('Readiness Index vs Alpha\n(com intervalos de confiança)')
        ax.grid(True, alpha=0.3)
        
        # 3. Heatmap dos componentes Task2Vec
        ax = axes[1, 0]
        components = ['cohesion', 'dispersion', 'coverage', 'avg_entropy', 'density']
        available_components = [c for c in components if c in self.df.columns]
        
        if available_components:
            # Normalizar componentes para comparação
            component_data = self.df[available_components].copy()
            for comp in available_components:
                component_data[comp] = (component_data[comp] - component_data[comp].min()) / \
                                     (component_data[comp].max() - component_data[comp].min())
            
            # Agrupar por alpha e calcular média
            heatmap_data = component_data.groupby(self.df['alpha']).mean()
            
            sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax)
            ax.set_title('Componentes Task2Vec Normalizados por Alpha')
            ax.set_xlabel('Alpha')
            ax.set_ylabel('Componentes')
        else:
            ax.text(0.5, 0.5, 'Componentes Task2Vec\nnão disponíveis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Componentes Task2Vec - Dados Indisponíveis')
        
        # 4. Readiness vs Heterogeneidade da federação (entropia)
        ax = axes[1, 1]
        if 'federation_entropy_mean' in self.df.columns:
            scatter = ax.scatter(self.df['federation_entropy_mean'], self.df['readiness'], 
                               c=self.df['alpha'], cmap='viridis', alpha=0.7, s=40)
            ax.set_xlabel('Entropia Média da Federação')
            ax.set_ylabel('Readiness Index')
            ax.set_title('Readiness vs Heterogeneidade da Federação')
            plt.colorbar(scatter, ax=ax, label='Alpha')
            
            # Linha de tendência
            z = np.polyfit(self.df['federation_entropy_mean'], self.df['readiness'], 1)
            p = np.poly1d(z)
            ax.plot(self.df['federation_entropy_mean'], p(self.df['federation_entropy_mean']), 
                   "r--", alpha=0.8, linewidth=2)
            
            # Correlação
            corr, p_val = spearmanr(self.df['federation_entropy_mean'], self.df['readiness'])
            ax.text(0.05, 0.95, f'ρ = {corr:.3f}\np = {p_val:.3f}', 
                   transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
        else:
            ax.text(0.5, 0.5, 'Dados de entropia\nda federação não disponíveis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Readiness vs Entropia - Sem Dados')
        
        # 5. Variação do Readiness por tamanho da amostra
        ax = axes[2, 0]
        if 'total_samples' in self.df.columns:
            # Criar bins de tamanho de amostra
            self.df['sample_bins'] = pd.cut(self.df['total_samples'], 
                                          bins=5, labels=['Muito Baixo', 'Baixo', 'Médio', 'Alto', 'Muito Alto'])
            
            sns.boxplot(data=self.df, x='sample_bins', y='readiness', ax=ax)
            ax.set_title('Readiness por Tamanho Total de Amostras')
            ax.set_xlabel('Tamanho da Amostra')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'Dados de tamanho\nde amostra não disponíveis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Readiness vs Amostras - Sem Dados')
        
        # 6. Eficiência do Task2Vec (Readiness vs Performance por custo computacional)
        ax = axes[2, 1]
        # Proxy para "custo": número de clientes (mais clientes = mais embeddings para calcular)
        efficiency_score = self.df['final_performance'] / (self.df['num_clients'] * 0.01)  # Normalizar
        
        scatter = ax.scatter(self.df['readiness'], efficiency_score, 
                           c=self.df['num_clients'], cmap='plasma', alpha=0.7, s=40)
        ax.set_xlabel('Readiness Index')
        ax.set_ylabel('Eficiência (Performance/Custo)')
        ax.set_title('Eficiência Task2Vec: Readiness vs Performance/Custo')
        plt.colorbar(scatter, ax=ax, label='Número de Clientes')
        
        # Correlação
        corr, p_val = spearmanr(self.df['readiness'], efficiency_score)
        ax.text(0.05, 0.95, f'ρ = {corr:.3f}\np = {p_val:.3f}', 
               transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
        
        plt.tight_layout()
        save_path = self.output_dir / "task2vec_detailed_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Análise Task2Vec salva: {save_path}")
        
        # Relatório específico Task2Vec
        self._generate_task2vec_report()
    
    def _generate_task2vec_report(self):
        """Gera relatório específico sobre o desempenho do Task2Vec."""
        report_path = self.output_dir / "task2vec_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("RELATÓRIO DE ANÁLISE TASK2VEC\n")
            f.write("="*80 + "\n\n")
            
            # 1. Estatísticas do Readiness Index
            f.write("1. ESTATÍSTICAS DO READINESS INDEX:\n")
            f.write(f"   Média geral: {self.df['readiness'].mean():.4f} ± {self.df['readiness'].std():.4f}\n")
            f.write(f"   Mínimo: {self.df['readiness'].min():.4f}\n")
            f.write(f"   Máximo: {self.df['readiness'].max():.4f}\n")
            f.write(f"   Mediana: {self.df['readiness'].median():.4f}\n\n")
            
            # 2. Correlações principais
            f.write("2. CORRELAÇÕES PRINCIPAIS:\n")
            main_corr, main_p = spearmanr(self.df['readiness'], self.df['final_performance'])
            f.write(f"   Readiness vs Performance: {main_corr:.4f} (p={main_p:.4f})\n")
            
            if 'federation_entropy_mean' in self.df.columns:
                entropy_corr, entropy_p = spearmanr(self.df['readiness'], self.df['federation_entropy_mean'])
                f.write(f"   Readiness vs Entropia Fed.: {entropy_corr:.4f} (p={entropy_p:.4f})\n")
            
            alpha_corr, alpha_p = spearmanr(self.df['readiness'], self.df['alpha'])
            f.write(f"   Readiness vs Alpha: {alpha_corr:.4f} (p={alpha_p:.4f})\n\n")
            
            # 3. Análise por componentes
            f.write("3. ANÁLISE DOS COMPONENTES:\n")
            components = ['cohesion', 'dispersion', 'coverage', 'avg_entropy', 'density']
            for comp in components:
                if comp in self.df.columns:
                    comp_corr, comp_p = spearmanr(self.df[comp], self.df['final_performance'])
                    f.write(f"   {comp.title()} vs Performance: {comp_corr:.4f} (p={comp_p:.4f})\n")
            f.write("\n")
            
            # 4. Performance por variante de readiness
            f.write("4. PERFORMANCE POR VARIANTE DE READINESS:\n")
            if 'readiness_variant' in self.df.columns:
                for variant in self.df['readiness_variant'].unique():
                    variant_df = self.df[self.df['readiness_variant'] == variant]
                    success_rate = variant_df['target_achieved'].mean()
                    avg_performance = variant_df['final_performance'].mean()
                    f.write(f"   {variant}: {success_rate:.2%} sucesso, {avg_performance:.4f} performance média\n")
            else:
                f.write("   Dados de variante não disponíveis\n")
            f.write("\n")
            
            # 5. Intervalos de confiança
            f.write("5. ANÁLISE DOS INTERVALOS DE CONFIANÇA:\n")
            if 'readiness_ci_lower' in self.df.columns and 'readiness_ci_upper' in self.df.columns:
                self.df['ci_width'] = self.df['readiness_ci_upper'] - self.df['readiness_ci_lower']
                f.write(f"   Largura média do IC: {self.df['ci_width'].mean():.4f}\n")
                f.write(f"   Largura mínima do IC: {self.df['ci_width'].min():.4f}\n")
                f.write(f"   Largura máxima do IC: {self.df['ci_width'].max():.4f}\n")
                
                # Correlação entre largura do IC e performance
                ic_corr, ic_p = spearmanr(self.df['ci_width'], self.df['final_performance'])
                f.write(f"   Largura IC vs Performance: {ic_corr:.4f} (p={ic_p:.4f})\n")
            else:
                f.write("   Dados de intervalo de confiança não disponíveis\n")
            f.write("\n")
            
            # 6. Recomendações baseadas na análise
            f.write("6. CONCLUSÕES E RECOMENDAÇÕES:\n")
            f.write(f"   - Correlação Readiness-Performance: {'FORTE' if abs(main_corr) > 0.7 else 'MODERADA' if abs(main_corr) > 0.4 else 'FRACA'}\n")
            f.write(f"   - Significância estatística: {'SIM' if main_p < 0.05 else 'NÃO'}\n")
            
            best_alpha = self.df.groupby('alpha')['readiness'].mean().idxmax()
            f.write(f"   - Melhor alpha para Readiness: {best_alpha}\n")
            
            best_clients = self.df.groupby('num_clients')['readiness'].mean().idxmax()
            f.write(f"   - Melhor nº de clientes para Readiness: {best_clients}\n")
            
            if abs(main_corr) > 0.4 and main_p < 0.05:
                f.write("   - RECOMENDAÇÃO: Task2Vec demonstra capacidade preditiva útil\n")
            else:
                f.write("   - ATENÇÃO: Task2Vec pode precisar de ajustes para melhor predição\n")
        
        print(f"  Relatório Task2Vec salvo em: {report_path}")
    
    def generate_summary_table(self):
        """Gera tabela resumo dos principais resultados."""
        summary_data = []
        
        for dataset in self.df['dataset'].unique():
            dataset_df = self.df[self.df['dataset'] == dataset]
            
            # Correlação geral
            # corr, p_val = spearmanr(dataset_df['readiness'], dataset_df['final_performance'])
            
            # Por alpha
            for alpha in sorted(dataset_df['alpha'].unique()):
                alpha_df = dataset_df[dataset_df['alpha'] == alpha]
                
                if len(alpha_df) > 3:
                    alpha_corr, alpha_p = spearmanr(alpha_df['readiness'], alpha_df['final_performance'])
                    
                    summary_data.append({
                        'dataset': dataset,
                        'alpha': alpha,
                        'n_experiments': len(alpha_df),
                        'success_rate': alpha_df['target_achieved'].mean(),
                        'avg_readiness': alpha_df['readiness'].mean(),
                        'avg_performance': alpha_df['final_performance'].mean(),
                        'correlation': alpha_corr,
                        'p_value': alpha_p,
                        'avg_convergence': alpha_df[alpha_df['convergence_round'] > 0]['convergence_round'].mean()
                    })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Salvar tabela
        table_path = self.output_dir / "summary_table.csv"
        summary_df.to_csv(table_path, index=False)
        
        # Criar tabela formatada
        table_formatted_path = self.output_dir / "summary_table_formatted.txt"
        with open(table_formatted_path, 'w') as f:
            f.write("TABELA RESUMO DOS RESULTADOS\n")
            f.write("="*100 + "\n\n")
            
            f.write(f"{'Dataset':<12} {'Alpha':<8} {'N':<6} {'Success%':<10} {'Readiness':<12} "
                   f"{'Performance':<12} {'Correlation':<12} {'P-value':<10} {'Conv.Round':<10}\n")
            f.write("-"*100 + "\n")
            
            for _, row in summary_df.iterrows():
                f.write(f"{row['dataset']:<12} {row['alpha']:<8.1f} {row['n_experiments']:<6} "
                       f"{row['success_rate']:<10.2%} {row['avg_readiness']:<12.4f} "
                       f"{row['avg_performance']:<12.4f} {row['correlation']:<12.4f} "
                       f"{row['p_value']:<10.4f} {row['avg_convergence']:<10.1f}\n")
        
        print(f"  Tabela resumo salva em: {table_path}")
        print(f"  Tabela formatada salva em: {table_formatted_path}")
        
        return summary_df


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Analisar resultados dos experimentos Task2Vec + Federated Learning"
    )
    parser.add_argument(
        "results_path",
        type=str,
        help="Caminho para arquivo de resultados (CSV ou Parquet)"
    )
    parser.add_argument(
        "--analysis-type",
        choices=['full', 'correlation', 'visualization', 'prediction', 'summary'],
        default='full',
        help="Tipo de análise a executar"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Filtrar por dataset específico"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Filtrar por valor de alpha específico"
    )
    
    args = parser.parse_args()
    
    # Verificar se arquivo existe
    if not Path(args.results_path).exists():
        print(f"Erro: Arquivo '{args.results_path}' não encontrado!")
        return 1
    
    try:
        # Criar analisador
        analyzer = ResultsAnalyzer(args.results_path)
        
        # Aplicar filtros se especificados
        if args.dataset:
            analyzer.df = analyzer.df[analyzer.df['dataset'] == args.dataset]
            print(f"Filtrado para dataset: {args.dataset}")
        
        if args.alpha is not None:
            analyzer.df = analyzer.df[analyzer.df['alpha'] == args.alpha]
            print(f"Filtrado para alpha: {args.alpha}")
        
        # Executar análise
        if args.analysis_type == 'full':
            analyzer.generate_full_report()
            analyzer.generate_summary_table()
            
        elif args.analysis_type == 'correlation':
            analyzer.correlation_analysis()
            
        elif args.analysis_type == 'visualization':
            analyzer.plot_readiness_vs_performance()
            analyzer.plot_alpha_analysis()
            analyzer.plot_client_size_analysis()
            analyzer.plot_component_analysis()
            
        elif args.analysis_type == 'prediction':
            analyzer.prediction_analysis()
            
        elif args.analysis_type == 'summary':
            analyzer.generate_summary_table()
        
        return 0
        
    except Exception as e:
        print(f"Erro durante análise: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())