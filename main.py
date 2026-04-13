import pandas as pd
from DataAnalysis import (
    analyze_numeric_attributes, 
    analyze_categorical_attributes, 
    analyze_class_balance,
    analyze_all_correlations
)
from DataProcessing import (
    check_redundant_attributes,
    check_null_values,
    impute_columns
)

def analyze_dataset(dataset):
    # analyze columns
    numeric_features = ['studied_credits', 'mean_score_early']
    categorical_features = ['disability', 'highest_education']
    
    analyze_numeric_attributes(dataset, numeric_features)
    analyze_categorical_attributes(dataset, categorical_features)
    
    class_target = 'final_result'    
    analyze_class_balance(dataset, class_target)
    
    reg_target = 'final_coursework_score'
    analyze_all_correlations(dataset, 'studied_credits', 'highest_education', class_target, reg_target)

def process_dataset(dataset):
    # remove redundant and mostly null columns
    redundant_attributes = check_redundant_attributes(dataset)
    dataset = dataset.drop(columns=redundant_attributes)

    null_columns = check_null_values(dataset)
    dataset = dataset.drop(columns=null_columns)

    # replace outliers with NaN then impute those values
    dataset = replace_outliers_with_nan(dataset)
    dataset = impute_columns(dataset)
    
    return dataset

def main():
    train_dataset = pd.read_csv('CA_OUALD_train.csv', skipinitialspace=True)
    
    analyze_dataset(train_dataset)
    train_dataset = process_dataset(train_dataset)

if __name__ == "__main__":
    main()