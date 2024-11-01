import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Loads data from a CSV file, encodes the target variable, and returns features and target.

    Parameters:
    - file_path (str): Path to the dataset CSV file.

    Returns:
    - X (DataFrame): Features without irrelevant columns like 'id'.
    - y (Series): Binary target variable (0 = Benign, 1 = Malignant).
    """
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Encode target variable: 0 for Benign, 1 for Malignant
    df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})
    
    # Drop the 'id' column as it is not a feature
    X = df.drop(columns=["id", "diagnosis"], errors="ignore")
    y = df["diagnosis"]
    
    return X, y

def preprocess_data(X, y, test_size=0.2):
    """
    Splits the data into training and testing sets, then scales the features.

    Parameters:
    - X (DataFrame): Features.
    - y (Series): Target variable.
    - test_size (float): Proportion of the dataset to include in the test split.

    Returns:
    - X_train (ndarray): Scaled training features.
    - X_test (ndarray): Scaled testing features.
    - y_train (Series): Training target variable.
    - y_test (Series): Testing target variable.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
