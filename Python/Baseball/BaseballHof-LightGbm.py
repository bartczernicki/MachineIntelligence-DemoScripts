from flaml import AutoML
from flaml.data import load_openml_dataset
from sklearn.datasets import load_iris
import pandas as pd

baseballPlayers = pd.read_csv('MLBBaseballBattersFullTraining.csv')

# Drop un-needed features
baseballPlayers.drop('FullPlayerName', axis=1, inplace=True)
baseballPlayers.drop('LastYearPlayed', axis=1, inplace=True)
baseballPlayers.drop('ID', axis=1, inplace=True)

# Convert bool to 1/0
# print(baseballPlayers['OnHallOfFameBallot'].dtypes)
y_train_OnHallOfFameBallot = baseballPlayers.OnHallOfFameBallot.eq(True).mul(1)
y_train_InductedToHallOfFame = baseballPlayers.InductedToHallOfFame.eq(True).mul(1)
baseballPlayers.drop('OnHallOfFameBallot', axis=1, inplace=True)
baseballPlayers.drop('InductedToHallOfFame', axis=1, inplace=True)

# Print Results
print(baseballPlayers.head(n=10).to_string(index=False))
print(y_train_InductedToHallOfFame.head(n=10).to_string(index=False))

# # Initialize an AutoML instance
automl = AutoML()
# Specify automl goal and constraint
automl_settings = {
    "time_budget": 100,  # total running time in seconds
    "metric": 'accuracy',  # primary metrics for regression can be chosen from: ['mae','mse','r2']
    "estimator_list": ['lgbm', 'catboost', 'rf', 'extra_tree'],  # list of ML learners; we tune lightgbm in this example
    "task": 'classification',  # task type  
    "log_file_name": 'baseball_experiment.log',  # flaml log file
    "log_type": 'all',
    "seed": 200,    # random seed
    "n_jobs": 1,
    "eval_method": 'cv',
    "n_splits": 4
}


# # # Train with labeled input data
automl.fit(X_train=baseballPlayers, y_train=y_train_InductedToHallOfFame, **automl_settings)


# # # Predict
# # print(automl.predict_proba(X_train))
# # # Print the best model
print(automl.model.estimator)