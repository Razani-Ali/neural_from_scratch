import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


#############################################################################################################################

def impute_data(
    features: np.ndarray, 
    targets: np.ndarray = None, 
    feature_method: str = 'mean', 
    class_method: str = 'mode', 
    specified_value_class: str = 'miss', 
    missing_list: list = ['?', '_', '', '-']):
    
    # Convert features and targets to DataFrames for easier manipulation
    features_df = pd.DataFrame(features)
    if targets is not None:
        targets_df = pd.DataFrame(targets)

    # Replace specified missing values with NaN
    features_df.replace(missing_list, np.NaN, inplace=True)
    if targets is not None:
        targets_df.replace(missing_list, np.NaN, inplace=True)

    # Impute missing features based on the specified method
    if feature_method == 'zero':
        # Fill missing values with 0
        features_df = features_df.fillna(0.0)
    elif feature_method == 'drop':
        # Drop rows with any missing values in features
        non_nan_indices = features_df.dropna().index
        features_df = features_df.loc[non_nan_indices]
        if targets is not None:
            targets_df = targets_df.loc[non_nan_indices]
    elif feature_method == 'mean':
        # Fill missing values with mean of each column
        mean_columns = np.nanmean(features_df, axis=0)
        for col in features_df.columns:
            features_df[col] = features_df[col].fillna(mean_columns[col])
    elif feature_method == 'median':
        # Fill missing values with median of each column
        median_columns = np.nanmedian(features_df, axis=0)
        for col in features_df.columns:
            features_df[col] = features_df[col].fillna(median_columns[col])
    elif feature_method == 'mode':
        # Fill missing values with mode of each column
        mode_columns = features_df.mode().iloc[0]
        features_df = features_df.fillna(mode_columns)
    elif feature_method == 'ffill':
        # Forward fill missing values
        features_df = features_df.fillna(method='ffill')
        non_nan_indices = features_df.dropna().index
        features_df = features_df.loc[non_nan_indices]
        if targets is not None:
            targets_df = targets_df.loc[non_nan_indices]
    elif feature_method == 'bfill':
        # Backward fill missing values
        features_df = features_df.fillna(method='bfill')
        non_nan_indices = features_df.dropna().index
        features_df = features_df.loc[non_nan_indices]
        if targets is not None:
            targets_df = targets_df.loc[non_nan_indices]
    elif feature_method == 'interpolate':
        # Interpolate missing values
        features_df = features_df.interpolate()

    if targets is not None:
        # Handle target imputation based on the specified method
        if class_method == 'specified':
            # Fill missing values with a specified value
            targets_df = targets_df.fillna(specified_value_class)
        elif class_method == 'drop':
            # Drop rows with any missing values in targets
            non_nan_indices = targets_df.dropna().index
            targets_df = targets_df.loc[non_nan_indices]
            features_df = features_df.loc[non_nan_indices]
        elif class_method == 'mode':
            # Fill missing values with mode of each column
            mode_columns = targets_df.mode().iloc[0]
            targets_df = targets_df.fillna(mode_columns)
        elif class_method == 'ffill':
            # Forward fill missing values
            targets_df = targets_df.fillna(method='ffill')
            non_nan_indices = targets_df.dropna().index
            targets_df = targets_df.loc[non_nan_indices]
            features_df = features_df.loc[non_nan_indices]
        elif class_method == 'bfill':
            # Backward fill missing values
            targets_df = targets_df.fillna(method='bfill')
            non_nan_indices = targets_df.dropna().index
            targets_df = targets_df.loc[non_nan_indices]
            features_df = features_df.loc[non_nan_indices]
        elif class_method == 'KNN':
            # Use K-Nearest Neighbors imputation
            knn_imputer = KNNImputer()
            if not targets_df.empty:
                for col in targets_df.columns:
                    targets_df[col] = knn_imputer.fit_transform(targets_df[[col]])
        elif class_method == 'mice':
            # Use Multiple Imputation by Chained Equations (MICE)
            mice_imputer = IterativeImputer()
            if not targets_df.empty:
                for col in targets_df.columns:
                    targets_df[col] = mice_imputer.fit_transform(targets_df[[col]])
        elif class_method in ['mean', 'median', 'zero', 'interpolate']:
            if class_method == 'zero':
                # Fill missing values with 0
                targets_df = targets_df.fillna(0)
            elif class_method == 'mean':
                # Fill missing values with mean of each column
                mean_columns = np.nanmean(targets_df, axis=0)
                for col in targets_df.columns:
                    targets_df[col] = targets_df[col].fillna(mean_columns[col])
            elif class_method == 'median':
                # Fill missing values with median of each column
                median_columns = np.nanmedian(targets_df, axis=0)
                for col in targets_df.columns:
                    targets_df[col] = targets_df[col].fillna(median_columns[col])
            elif class_method == 'interpolate':
                # Interpolate missing values
                targets_df = targets_df.interpolate()

    # Convert DataFrames back to numpy arrays
    features = features_df.to_numpy()
    if targets is not None:
        targets = targets_df.to_numpy()
        
    # Return imputed data    
    if targets is not None:
        return features, targets
    else:
        return features

#############################################################################################################################

def supervised_impute(
    features: np.ndarray, 
    targets: np.ndarray, 
    feature_method: str = 'mean', 
    class_method: str = 'mode', 
    specified_value_class: str = 'miss', 
    missing_list: list = ['?', '_', '', '-']):
    
    # Unique classes in the target array
    unique_classes = np.unique(targets)
    
    imputed_features_list = []
    imputed_targets_list = []
    
    # Loop over each unique class in the target data
    for target_class in unique_classes:
        # Separate the data for the current class
        class_indices = np.where(targets == target_class)[0]
        class_features = features[class_indices]
        class_targets = targets[class_indices]
        
        # Impute the data for the current class
        imputed_class_features, imputed_class_targets = impute_data(
            features=class_features, 
            targets=class_targets, 
            feature_method=feature_method, 
            class_method=class_method, 
            specified_value_class=specified_value_class, 
            missing_list=missing_list
        )
        
        # Store the imputed data
        imputed_features_list.append(imputed_class_features)
        imputed_targets_list.append(imputed_class_targets.reshape(-1))  # Ensure correct dimension for targets
    
    # Concatenate the imputed data for all classes
    imputed_features = np.vstack(imputed_features_list)
    imputed_targets = np.hstack(imputed_targets_list)
    
    return imputed_features, imputed_targets
