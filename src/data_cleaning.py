"""
Módulo para limpeza e preparação de dados do e-commerce.

Este módulo contém funções para limpar, transformar e preparar os dados
do dataset de e-commerce da Olist para análise.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_datasets(base_path='../data/raw/'):
    """
    Carrega os datasets do e-commerce da Olist.
    
    Parameters:
    -----------
    base_path : str
        Caminho base para os arquivos de dados
        
    Returns:
    --------
    dict
        Dicionário contendo os dataframes carregados
    """
    try:
        datasets = {
            'orders': pd.read_csv(f'{base_path}olist_orders_dataset.csv'),
            'order_items': pd.read_csv(f'{base_path}olist_order_items_dataset.csv'),
            'products': pd.read_csv(f'{base_path}olist_products_dataset.csv'),
            'customers': pd.read_csv(f'{base_path}olist_customers_dataset.csv'),
            'sellers': pd.read_csv(f'{base_path}olist_sellers_dataset.csv')
        }
        
        print("Datasets carregados com sucesso!")
        for name, df in datasets.items():
            print(f"{name}: {df.shape[0]} linhas, {df.shape[1]} colunas")
            
        return datasets
    
    except FileNotFoundError as e:
        print(f"Erro ao carregar datasets: {e}")
        return None


def convert_dates(df, date_columns):
    """
    Converte colunas de texto para datetime.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contendo as colunas de data
    date_columns : list
        Lista de nomes das colunas a serem convertidas
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame com as colunas convertidas
    """
    df_copy = df.copy()
    
    for col in date_columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    
    return df_copy


def extract_date_features(df, date_column):
    """
    Extrai características de data (ano, mês, dia, dia da semana) de uma coluna datetime.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contendo a coluna de data
    date_column : str
        Nome da coluna de data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame com as novas colunas extraídas
    """
    df_copy = df.copy()
    
    if date_column in df_copy.columns and pd.api.types.is_datetime64_dtype(df_copy[date_column]):
        df_copy[f'{date_column}_year'] = df_copy[date_column].dt.year
        df_copy[f'{date_column}_month'] = df_copy[date_column].dt.month
        df_copy[f'{date_column}_day'] = df_copy[date_column].dt.day
        df_copy[f'{date_column}_dayofweek'] = df_copy[date_column].dt.dayofweek
        df_copy[f'{date_column}_quarter'] = df_copy[date_column].dt.quarter
        df_copy[f'{date_column}_is_weekend'] = df_copy[date_column].dt.dayofweek.isin([5, 6]).astype(int)
    else:
        print(f"Erro: {date_column} não é uma coluna datetime válida")
    
    return df_copy


def handle_missing_values(df, strategy='drop'):
    """
    Trata valores ausentes no DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com valores ausentes
    strategy : str
        Estratégia para lidar com valores ausentes ('drop', 'fill_mean', 'fill_median', 'fill_zero')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame com valores ausentes tratados
    """
    df_copy = df.copy()
    
    if strategy == 'drop':
        df_copy = df_copy.dropna()
    elif strategy == 'fill_mean':
        for col in df_copy.select_dtypes(include=[np.number]).columns:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    elif strategy == 'fill_median':
        for col in df_copy.select_dtypes(include=[np.number]).columns:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    elif strategy == 'fill_zero':
        df_copy = df_copy.fillna(0)
    else:
        print(f"Estratégia '{strategy}' não reconhecida. Usando 'drop' como padrão.")
        df_copy = df_copy.dropna()
    
    return df_copy


def merge_order_data(orders, order_items, products=None, customers=None):
    """
    Combina dados de pedidos, itens, produtos e clientes.
    
    Parameters:
    -----------
    orders : pandas.DataFrame
        DataFrame de pedidos
    order_items : pandas.DataFrame
        DataFrame de itens de pedido
    products : pandas.DataFrame, optional
        DataFrame de produtos
    customers : pandas.DataFrame, optional
        DataFrame de clientes
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame combinado
    """
    # Merge pedidos e itens
    merged_data = pd.merge(orders, order_items, on='order_id', how='inner')
    
    # Adicionar produtos se disponível
    if products is not None:
        merged_data = pd.merge(merged_data, products, on='product_id', how='left')
    
    # Adicionar clientes se disponível
    if customers is not None:
        merged_data = pd.merge(merged_data, customers, on='customer_id', how='left')
    
    return merged_data


def calculate_delivery_time(orders):
    """
    Calcula o tempo de entrega em dias para cada pedido.
    
    Parameters:
    -----------
    orders : pandas.DataFrame
        DataFrame de pedidos com colunas de data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame com colunas adicionais de tempo de entrega
    """
    df = orders.copy()
    
    # Verificar se as colunas necessárias existem e são do tipo datetime
    date_cols = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in date_cols:
        if col not in df.columns:
            print(f"Coluna {col} não encontrada no DataFrame")
            return df
        if not pd.api.types.is_datetime64_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calcular tempo de entrega real (em dias)
    df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.total_seconds() / (24 * 3600)
    
    # Calcular tempo de entrega estimado (em dias)
    df['estimated_delivery_time_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.total_seconds() / (24 * 3600)
    
    # Calcular diferença entre tempo real e estimado (negativo = entregue antes do prazo)
    df['delivery_time_difference'] = df['delivery_time_days'] - df['estimated_delivery_time_days']
    
    # Marcar se entrega foi feita dentro do prazo
    df['delivered_on_time'] = df['delivery_time_difference'] <= 0
    
    return df


def prepare_data_for_analysis(base_path='../data/raw/', save_processed=True):
    """
    Função principal que executa todo o pipeline de preparação de dados.
    
    Parameters:
    -----------
    base_path : str
        Caminho base para os arquivos de dados
    save_processed : bool
        Se True, salva os dados processados em arquivos
        
    Returns:
    --------
    dict
        Dicionário contendo os dataframes processados
    """
    # Carregar datasets
    datasets = load_datasets(base_path)
    if not datasets:
        return None
    
    # Converter datas
    date_columns = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                   'order_delivered_customer_date', 'order_estimated_delivery_date']
    datasets['orders'] = convert_dates(datasets['orders'], date_columns)
    
    # Extrair características de data
    datasets['orders'] = extract_date_features(datasets['orders'], 'order_purchase_timestamp')
    
    # Tratar valores ausentes
    for key in datasets:
        datasets[key] = handle_missing_values(datasets[key], strategy='fill_median')
    
    # Calcular tempo de entrega
    datasets['orders'] = calculate_delivery_time(datasets['orders'])
    
    # Combinar dados
    merged_data = merge_order_data(
        datasets['orders'], 
        datasets['order_items'], 
        datasets['products'], 
        datasets['customers']
    )
    datasets['merged_data'] = merged_data
    
    # Salvar dados processados
    if save_processed:
        import os
        processed_path = '../data/processed/'
        os.makedirs(processed_path, exist_ok=True)
        
        for key, df in datasets.items():
            df.to_csv(f'{processed_path}{key}_processed.csv', index=False)
        
        print(f"Dados processados salvos em {processed_path}")
    
    return datasets


if __name__ == "__main__":
    # Executar o pipeline de preparação de dados
    processed_data = prepare_data_for_analysis()
    
    if processed_data:
        print("Pipeline de preparação de dados executado com sucesso!")
        for key, df in processed_data.items():
            print(f"{key}: {df.shape[0]} linhas, {df.shape[1]} colunas")
