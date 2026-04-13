import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency

def impute_missing_values(dataset, column_name, strategy):
    imputer = SimpleImputer(strategy=strategy)
    dataset[[column_name]] = imputer.fit_transform(dataset[[column_name]])
    return dataset

def replace_outliers_with_nan(dataset):
    exclude = ['final_result', 'final_coursework_score']
    cnumeric_columns = dataset.select_dtypes(include=['number']).columns.tolist()

    for col in cnumeric_columns:
        if col not in exclude:
            q1 = dataset[col].quantile(0.25)
            q3 = dataset[col].quantile(0.75)
            middle = q3 - q1
            lower_bound = q1 - 1.5 * middle
            upper_bound = q3 + 1.5 * middle

            replacing_small_values = dataset[col] < lower_bound
            replacing_large_values = dataset[col] > upper_bound
            dataset.loc[replacing_small_values | replacing_large_values, col] = np.nan

    return dataset

def check_redundant_attributes(dataset):
    exclude = ['final_result', 'final_coursework_score']
    redundant_attributes =[]

    # get numeric columns
    numeric_colums = []
    aux = dataset.select_dtypes(include=['number']).columns.tolist()
    for coloana in aux:
        if coloana not in exclude:
            numeric_colums.append(coloana)

    correlation_matrix = dataset[numeric_colums].corr().abs()
    # check above the main diagonal for redundant attributes
    for i in range(len(numeric_colums)):
        for j in range(i + 1, len(numeric_colums)):
            col1 = numeric_colums[i]
            col2 = numeric_colums[j]
            val = correlation_matrix.loc[col1, col2]
            
            if val >= 0.8:
                print(f"{col1} - {col2}: {val:.4f} correlation score")
                redundant_attributes.append(col2)

    print("\n")
    return redundant_attributes

def check_null_values(dataset):
    result = []

    column_values = dataset.isna().sum()
    extreme_null_values = column_values[column_values > 5500]
    null_values = column_values[column_values > 0]
    
    for col in extreme_null_values.index:
        result.append(col)

    for col in null_values.index:
        print(f"{col}: {null_values[col]} null values")
    
    print("\n")
    return result

def impute_columns(dataset):
    null_columns = dataset.columns[dataset.isna().any()].tolist()
    
    for col in null_columns:
        if 'clicks_' in col:
            dataset = impute_missing_values(dataset, col, 'constant') # substitute with 0
        elif is_numeric_dtype(dataset[col]):
            dataset = impute_missing_values(dataset, col, 'median') # substitute with median
        else:
            dataset = impute_missing_values(dataset, col, 'most_frequent') # substitute with most frequent string

    return dataset

def standardize_data(dataset):
    exclude = ['final_result', 'final_coursework_score']
    numeric_columns = dataset.select_dtypes(include=['number']).columns.tolist()
    
    scaling_columns = []
    for col in numeric_columns:
        if col not in exclude:
            scaling_columns.append(col)

    scaler = StandardScaler()
    dataset[scaling_columns] = scaler.fit_transform(dataset[scaling_columns])

    return dataset