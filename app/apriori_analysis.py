import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from datetime import datetime

def preprocess_data(data):
    """
    Preprocess the transaction data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw transaction data
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data
    """
    # Convert date to datetime with correct format
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', dayfirst=True)
    
    # Clean item descriptions
    data['itemDescription'] = data['itemDescription'].str.strip().str.lower()
    
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Remove rows with missing values
    data = data.dropna()
    
    return data

def analyze_transactions(data):
    """
    Analyze transaction patterns
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Transaction data
        
    Returns:
    --------
    dict
        Dictionary containing various transaction statistics
    """
    stats = {
        'total_transactions': data['Member_number'].nunique(),
        'total_items': data['itemDescription'].nunique(),
        'avg_items_per_transaction': data.groupby('Member_number')['itemDescription'].count().mean(),
        'most_common_items': data['itemDescription'].value_counts().head(10).to_dict(),
        'transaction_by_date': data.groupby(data['Date'].dt.date)['Member_number'].nunique().to_dict()
    }
    return stats

def run_apriori(data, min_support, min_confidence):
    """
    Run Apriori algorithm on the transaction data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Transaction data where each row represents a transaction
    min_support : float
        Minimum support threshold
    min_confidence : float
        Minimum confidence threshold
        
    Returns:
    --------
    tuple
        (frequent_itemsets, rules, binary_data, stats)
    """
    # Preprocess data
    data = preprocess_data(data)
    
    # Get transaction statistics
    stats = analyze_transactions(data)
    
    # Create binary matrix with boolean type
    binary_data = pd.crosstab(data['Member_number'], data['itemDescription'])
    binary_data = binary_data.astype(bool)  # Convert to boolean type
    
    # Generate frequent itemsets
    frequent_itemsets = apriori(binary_data, 
                              min_support=min_support,
                              use_colnames=True)
    
    # Generate association rules with additional metrics
    rules = association_rules(frequent_itemsets, 
                            metric="confidence",
                            min_threshold=min_confidence)
    
    # Add additional metrics if rules exist
    if not rules.empty:
        rules['lift'] = rules['lift'].round(3)
        rules['confidence'] = rules['confidence'].round(3)
        rules['support'] = rules['support'].round(3)
        
        # Sort rules by lift
        rules = rules.sort_values('lift', ascending=False)
    
    return frequent_itemsets, rules, binary_data, stats 