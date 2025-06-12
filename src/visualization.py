"""
Módulo para visualização de dados do e-commerce.

Este módulo contém funções para criar visualizações e gráficos
a partir dos dados do dataset de e-commerce da Olist.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def set_visualization_style():
    """
    Configura o estilo de visualização para os gráficos.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('viridis')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14


def plot_time_series(df, date_column, value_column, title, xlabel='Data', ylabel=None, 
                     figsize=(15, 8), save_path=None, color='darkblue'):
    """
    Cria um gráfico de série temporal.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados
    date_column : str
        Nome da coluna de data
    value_column : str
        Nome da coluna de valores
    title : str
        Título do gráfico
    xlabel : str, optional
        Rótulo do eixo x
    ylabel : str, optional
        Rótulo do eixo y
    figsize : tuple, optional
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar a figura
    color : str, optional
        Cor da linha
    """
    plt.figure(figsize=figsize)
    plt.plot(df[date_column], df[value_column], color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else value_column)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_bar_chart(df, x_column, y_column, title, xlabel=None, ylabel=None, 
                  figsize=(12, 8), save_path=None, palette='viridis', horizontal=False):
    """
    Cria um gráfico de barras.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados
    x_column : str
        Nome da coluna para o eixo x
    y_column : str
        Nome da coluna para o eixo y
    title : str
        Título do gráfico
    xlabel : str, optional
        Rótulo do eixo x
    ylabel : str, optional
        Rótulo do eixo y
    figsize : tuple, optional
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar a figura
    palette : str, optional
        Paleta de cores
    horizontal : bool, optional
        Se True, cria um gráfico de barras horizontal
    """
    plt.figure(figsize=figsize)
    
    if horizontal:
        ax = sns.barplot(y=x_column, x=y_column, data=df, palette=palette)
    else:
        ax = sns.barplot(x=x_column, y=y_column, data=df, palette=palette)
    
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_column)
    plt.ylabel(ylabel if ylabel else y_column)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_histogram(df, column, title, xlabel=None, ylabel='Frequência', 
                  bins=30, figsize=(12, 8), save_path=None, kde=True, color='darkblue'):
    """
    Cria um histograma.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados
    column : str
        Nome da coluna para o histograma
    title : str
        Título do gráfico
    xlabel : str, optional
        Rótulo do eixo x
    ylabel : str, optional
        Rótulo do eixo y
    bins : int, optional
        Número de bins
    figsize : tuple, optional
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar a figura
    kde : bool, optional
        Se True, adiciona uma estimativa de densidade kernel
    color : str, optional
        Cor do histograma
    """
    plt.figure(figsize=figsize)
    sns.histplot(df[column], kde=kde, bins=bins, color=color)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else column)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_scatter(df, x_column, y_column, title, xlabel=None, ylabel=None, 
                figsize=(12, 8), save_path=None, color='darkblue', alpha=0.5, add_trend=False):
    """
    Cria um gráfico de dispersão.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados
    x_column : str
        Nome da coluna para o eixo x
    y_column : str
        Nome da coluna para o eixo y
    title : str
        Título do gráfico
    xlabel : str, optional
        Rótulo do eixo x
    ylabel : str, optional
        Rótulo do eixo y
    figsize : tuple, optional
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar a figura
    color : str, optional
        Cor dos pontos
    alpha : float, optional
        Transparência dos pontos
    add_trend : bool, optional
        Se True, adiciona uma linha de tendência
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(x=x_column, y=y_column, data=df, alpha=alpha, color=color)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_column)
    plt.ylabel(ylabel if ylabel else y_column)
    
    if add_trend:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_column], df[y_column])
        x = np.array([df[x_column].min(), df[x_column].max()])
        y = intercept + slope * x
        plt.plot(x, y, 'r', label=f'y = {slope:.4f}x + {intercept:.4f}, R² = {r_value**2:.4f}')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_heatmap(df, title, figsize=(12, 10), save_path=None, cmap='viridis', annot=True):
    """
    Cria um mapa de calor de correlação.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados
    title : str
        Título do gráfico
    figsize : tuple, optional
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar a figura
    cmap : str, optional
        Mapa de cores
    annot : bool, optional
        Se True, adiciona anotações com os valores
    """
    plt.figure(figsize=figsize)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=annot, fmt='.2f', linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_pie_chart(df, column, title, figsize=(10, 10), save_path=None, colors=None, autopct='%1.1f%%'):
    """
    Cria um gráfico de pizza.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados
    column : str
        Nome da coluna para o gráfico de pizza
    title : str
        Título do gráfico
    figsize : tuple, optional
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar a figura
    colors : list, optional
        Lista de cores
    autopct : str, optional
        Formato para exibir percentuais
    """
    plt.figure(figsize=figsize)
    value_counts = df[column].value_counts()
    plt.pie(value_counts, labels=value_counts.index, autopct=autopct, colors=colors, shadow=True)
    plt.title(title)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_box_plot(df, x_column, y_column, title, xlabel=None, ylabel=None, 
                 figsize=(12, 8), save_path=None, palette='viridis'):
    """
    Cria um box plot.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados
    x_column : str
        Nome da coluna para o eixo x
    y_column : str
        Nome da coluna para o eixo y
    title : str
        Título do gráfico
    xlabel : str, optional
        Rótulo do eixo x
    ylabel : str, optional
        Rótulo do eixo y
    figsize : tuple, optional
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar a figura
    palette : str, optional
        Paleta de cores
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x=x_column, y=y_column, data=df, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_column)
    plt.ylabel(ylabel if ylabel else y_column)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_count_plot(df, column, title, xlabel=None, ylabel='Contagem', 
                   figsize=(12, 8), save_path=None, palette='viridis', horizontal=False):
    """
    Cria um gráfico de contagem.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados
    column : str
        Nome da coluna para o gráfico de contagem
    title : str
        Título do gráfico
    xlabel : str, optional
        Rótulo do eixo x
    ylabel : str, optional
        Rótulo do eixo y
    figsize : tuple, optional
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar a figura
    palette : str, optional
        Paleta de cores
    horizontal : bool, optional
        Se True, cria um gráfico de contagem horizontal
    """
    plt.figure(figsize=figsize)
    
    if horizontal:
        ax = sns.countplot(y=column, data=df, palette=palette)
    else:
        ax = sns.countplot(x=column, data=df, palette=palette)
    
    plt.title(title)
    plt.xlabel(xlabel if xlabel else column)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_subplots(nrows, ncols, figsize=(15, 10)):
    """
    Cria uma grade de subplots.
    
    Parameters:
    -----------
    nrows : int
        Número de linhas
    ncols : int
        Número de colunas
    figsize : tuple, optional
        Tamanho da figura
        
    Returns:
    --------
    tuple
        (fig, axes) - figura e array de eixos
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    return fig, axes


def save_figure(fig, save_path, dpi=300):
    """
    Salva uma figura.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figura para salvar
    save_path : str
        Caminho para salvar a figura
    dpi : int, optional
        Resolução da figura
    """
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Figura salva em {save_path}")


if __name__ == "__main__":
    # Exemplo de uso
    set_visualization_style()
    
    # Criar dados de exemplo
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    values = np.random.normal(100, 15, len(dates)) + np.sin(np.arange(len(dates)) * 0.1) * 30
    df = pd.DataFrame({'date': dates, 'sales': values})
    
    # Criar visualização
    plot_time_series(df, 'date', 'sales', 'Vendas Diárias - Exemplo', 
                    ylabel='Vendas (R$)', save_path='../reports/figures/example_time_series.png')
    
    print("Exemplo de visualização criado com sucesso!")
