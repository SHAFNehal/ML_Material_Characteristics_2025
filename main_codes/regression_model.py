# Import necessary Libraries
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV #, RandomizedSearchCV
# THe machine learning models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor


from tqdm import tqdm

# Define the Regression Models
# Initialize all regressors
regressors = {
                'Linear_Regression'     : LinearRegression(),
                'Random_Forest'         : RandomForestRegressor(n_estimators=100, 
                                                                random_state=420),
                'Decision_Tree'         : DecisionTreeRegressor(),
                'Ridge'                 : Ridge(),
                'Lasso'                 : Lasso(),
                'Gradient_Boosting'     : GradientBoostingRegressor(),
                'AdaBoost'              : AdaBoostRegressor(),
                'Bagging'               : BaggingRegressor(),
            }

# We need hyper-parameter tuner for all of the models
param_grids = {
    'Linear_Regression': {'fit_intercept':[True]},
    
    'Random_Forest': {'n_estimators': [100, 200, 300],
                      'max_depth': [10, 20, 30],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4]},


    'Decision_Tree': {'max_depth': [None, 10, 20, 30],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4]},

    'Ridge': {'alpha': [0.1, 1, 10, 100]},

    'Lasso': {'alpha': [0.1, 1, 10, 100]},

    'Gradient_Boosting': {'n_estimators': [100, 200, 300],
                          'learning_rate': [0.01, 0.1, 0.2],
                          'max_depth': [3, 5, 10]},

    'AdaBoost': {'n_estimators': [50, 100, 200]},

    'Bagging': {'n_estimators': [10, 50, 100],
                'max_samples': [0.5, 1.0],
                'max_features': [0.5, 1.0]},

            }

# Function to train and evaluate models
def train_evaluate_models(models, 
                          param_grids,
                          X_train, 
                          y_train, 
                          X_test, 
                          y_test):
    # Store the results in a list
    results = []
    model_trained = {}
    # loop over the models list
    for name, model in tqdm(models.items(), colour= 'red'):
        # print("\nWorking on: " + name + "\n")
        # We will use grid search for hyper parameter tuning
        tuner = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error')
        tuner.fit(X_train, y_train)
        # Compute the MSE scores
        best_model = tuner.best_estimator_
        pred = best_model.predict(X_test)
        # Store the results
        mse = mean_squared_error(y_test, pred)
        results.append([name, mse, tuner.best_params_ if name in param_grids else 'N/A'])
        # Store the trained Models
        model_trained[name] = best_model


    # Store the final results as pandas dataframe
    final_results = pd.DataFrame(results, columns=['Model', 'MSE', 'R2', 'Best Parameters'])
    
    return final_results, model_trained