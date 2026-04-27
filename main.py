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
    merge_total_activity,
    convert_categorical_to_numeric
)
from Clasify import run_model
from Regression import run_regression

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
    # remove redundant and useless attributes
    redundant_attributes = check_redundant_attributes(train_dataset)
    train_dataset = train_dataset.drop(columns=redundant_attributes)
    test_dataset = test_dataset.drop(columns=redundant_attributes)
    train_dataset = train_dataset.drop(columns=['region', 'gender'], errors='ignore')
    test_dataset = test_dataset.drop(columns=['region', 'gender'], errors='ignore')

    # convert categorical variables to numeric in oreder to minimize the final number of columns
    train_dataset = convert_categorical_to_numeric(train_dataset)
    test_dataset = convert_categorical_to_numeric(test_dataset)

    # merge click_columns to make less data and copmpute faster
    train_dataset = merge_total_activity(train_dataset)
    test_dataset = merge_total_activity(test_dataset)

    # replace outliers with NaN
    train_dataset, test_dataset = replace_outliers_with_nan(train_dataset, test_dataset)

    # impute NaN values from dataset and outlier transformation
    train_dataset, test_dataset = impute_columns(train_dataset, test_dataset)

    train_dataset, test_dataset = standardize_data(train_dataset, test_dataset)

    return train_dataset, test_dataset

def main():
    train_dataset = pd.read_csv('CA_OUALD_train.csv', skipinitialspace=True)
    test_dataset = pd.read_csv('CA_OUALD_test.csv', skipinitialspace=True)
    
    analyze_dataset(train_dataset)
    train_dataset, test_dataset = process_dataset(train_dataset, test_dataset)
    run_model(train_dataset, test_dataset)
    run_regression(train_dataset, test_dataset)

if __name__ == "__main__":
    main()