import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def replace_outliers_with_nan(train_dataset, test_dataset):
    exclude = ['final_result', 'final_coursework_score']
    numeric_columns = train_dataset.select_dtypes(include=['number']).columns.tolist()

    for col in numeric_columns:
        if col not in exclude:
            q1 = train_dataset[col].quantile(0.05)
            q3 = train_dataset[col].quantile(0.95)
            middle = q3 - q1
            lower_bound = q1 - 1.5 * middle
            upper_bound = q3 + 1.5 * middle

            replacing_small_values = train_dataset[col] < lower_bound
            replacing_large_values = train_dataset[col] > upper_bound
            train_dataset.loc[replacing_small_values | replacing_large_values, col] = np.nan

            if col in test_dataset.columns:
                test_small = test_dataset[col] < lower_bound
                test_large = test_dataset[col] > upper_bound
                test_dataset.loc[test_small | test_large, col] = np.nan

    return train_dataset, test_dataset

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
            
            if val >= 0.9:
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
        
    return result

def impute_columns(train_dataset, test_dataset):
    null_columns = train_dataset.columns[train_dataset.isna().any()].tolist()
    
    for col in null_columns:
        if  is_numeric_dtype(train_dataset[col]):
            imputer = SimpleImputer(strategy='mean')
        else:
            imputer = SimpleImputer(strategy='most_frequent')

        train_dataset[[col]] = imputer.fit_transform(train_dataset[[col]])

        if col in test_dataset.columns:
            test_dataset[[col]] = imputer.transform(test_dataset[[col]])

    return train_dataset, test_dataset

def standardize_data(train_dataset, test_dataset):
    exclude = ['final_result', 'final_coursework_score']
    numeric_columns = train_dataset.select_dtypes(include=['number']).columns.tolist()
    
    scaling_columns = []
    for col in numeric_columns:
        if col not in exclude:
            scaling_columns.append(col)

    scaler = StandardScaler()
    train_dataset[scaling_columns] = scaler.fit_transform(train_dataset[scaling_columns])
    test_dataset[scaling_columns] = scaler.transform(test_dataset[scaling_columns])

    return train_dataset, test_dataset

def merge_total_activity(dataset):
    click_columns = [col for col in dataset.columns if 'clicks_' in col and col != 'clicks_freq_init']
    
    dataset['total_activity'] = dataset[click_columns].sum(axis=1)
    dataset = dataset.drop(columns=click_columns)
    
    return dataset

def convert_categorical_to_numeric(dataset):
    clicks_map = {'missing_value': 0, 'low': 1, 'mid': 2, 'high': 3}
    age_map = {'0-35': 1, '35-55': 2, '55<=': 3}
    imd_map = {
        '0-10%': 1, '10-20': 2, '20-30%': 3, '30-40%': 4, 
        '40-50%': 5, '50-60%': 6, '60-70%': 7, '70-80%': 8, 
        '80-90%': 9, '90-100%': 10}
    education_map = {
        'No Formal quals': 0, 'Lower Than A Level': 1,
        'A Level or Equivalent': 2, 'HE Qualification': 3,
        'Post Graduate Qualification': 4}
    
    if 'clicks_freq_init' in dataset.columns:
        dataset['clicks_freq_init'] = dataset['clicks_freq_init'].map(clicks_map)
    
    if 'age_band' in dataset.columns:
        dataset['age_band'] = dataset['age_band'].map(age_map)
        
    if 'imd_band' in dataset.columns:
        dataset['imd_band'] = dataset['imd_band'].map(imd_map)

    if 'highest_education' in dataset.columns:
        dataset['highest_education'] = dataset['highest_education'].map(education_map)

    return dataset