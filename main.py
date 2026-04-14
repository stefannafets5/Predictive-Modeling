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
    impute_columns,
    replace_outliers_with_nan,
    standardize_data,
    merge_total_activity
)
from Clasify import run_model

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

def process_dataset(train_dataset, test_dataset):
    # remove redundant and mostly null columns
    redundant_attributes = check_redundant_attributes(train_dataset)
    train_dataset = train_dataset.drop(columns=redundant_attributes)
    test_dataset = test_dataset.drop(columns=redundant_attributes)

    null_columns = check_null_values(train_dataset)
    train_dataset = train_dataset.drop(columns=null_columns)
    test_dataset = test_dataset.drop(columns=null_columns)

    # replace outliers with NaN then impute those values
    train_dataset = replace_outliers_with_nan(train_dataset)
    train_dataset = impute_columns(train_dataset)

    test_dataset = replace_outliers_with_nan(test_dataset)
    test_dataset = impute_columns(test_dataset)

    train_dataset = merge_total_activity(train_dataset)
    test_dataset = merge_total_activity(test_dataset)

    train_dataset, test_dataset = standardize_data(train_dataset, test_dataset)

    return train_dataset, test_dataset

def main():
    train_dataset = pd.read_csv('CA_OUALD_train.csv', skipinitialspace=True)
    test_dataset = pd.read_csv('CA_OUALD_test.csv', skipinitialspace=True)
    
    analyze_dataset(train_dataset)
    train_dataset, test_dataset = process_dataset(train_dataset, test_dataset)
    run_model(train_dataset, test_dataset)

if __name__ == "__main__":
    main()