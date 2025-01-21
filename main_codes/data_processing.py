# Import Necessary Library
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder #, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


def data_splitting(X, y, random_state = 100, test_size=0.3):
    # Split the Training Data
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = test_size, 
                                                        random_state = random_state) #100
    return X_train, X_test, y_train, y_test

# Data Pre-processing
def pre_process_data(numerical_features, categorical_features, X, y, columns_to_use, random_state = 10, test_size = 0.3):
    # Pipeline for Pre-processing
    preprocessor_pipeline = ColumnTransformer(
                                                transformers = [
                                                                ('numerical', SimpleImputer(strategy = 'mean'), numerical_features),
                                                                ('categorical', OrdinalEncoder(), categorical_features)
                                                                ]
                                            )
    
    # Split Data
    X_train, X_test, y_train, y_test = data_splitting(X, y, random_state = random_state, test_size = test_size)
    # Apply the Transformation
    # Train Data
    X_train_transformed = pd.DataFrame(preprocessor_pipeline.fit_transform(X_train), columns = columns_to_use)
    # Test Data
    X_test_transformed = pd.DataFrame(preprocessor_pipeline.transform(X_test), columns = columns_to_use)
    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor_pipeline

# Data Pre-processing without splitting
def pre_process_data_no_split(numerical_features, categorical_features, X, y, columns_to_use):
    # Pipeline for Pre-processing
    preprocessor_pipeline = ColumnTransformer([
                                                    ('numerical', Pipeline([
                                                        ('imputer', SimpleImputer(strategy='mean')),
                                                        ('scaler', MinMaxScaler())
                                                    ]), numerical_features),
                                                    ('categorical', OrdinalEncoder(), categorical_features)
                                                ])
    
    # Apply the Transformation
    # Train Data
    X_train_transformed = pd.DataFrame(preprocessor_pipeline.fit_transform(X), columns = columns_to_use)
    y_transformed = pd.DataFrame(MinMaxScaler().fit_transform(y.values.reshape(-1, 1)), columns=['y'])
    return X_train_transformed, y_transformed, preprocessor_pipeline