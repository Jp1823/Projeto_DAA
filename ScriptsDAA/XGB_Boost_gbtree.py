import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb
from xgboost import XGBClassifier
import time

# Load datasets
hipp_train = pd.read_csv('train_radiomics_hipocamp.csv', na_filter=False)
hipp_test = pd.read_csv('test_radiomics_hipocamp.csv', na_filter=False)

# Remove columns with single values and irrelevant columns
colunas_remover = ['Image', 'diagnostics_Image-original_Hash', 'diagnostics_Mask-original_Hash',
                   'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMassIndex',
                   'diagnostics_Mask-original_CenterOfMass', 'Mask']
hipp_train_c = hipp_train.drop(columns=colunas_remover, errors='ignore').loc[:, hipp_train.nunique() > 1]
hipp_test_c = hipp_test.drop(columns=colunas_remover, errors='ignore').loc[:, hipp_test.nunique() > 1]

# Map the 'Transition' column to numeric values
mapping = {
    'CN-CN': 0,  # Normal state
    'CN-MCI': 1,  # Intermediate state
    'MCI-MCI': 2,  # Intermediate state
    'MCI-AD': 3,  # Dementia
    'AD-AD': 4    # Dementia
}
hipp_train_c['Transition'] = hipp_train_c['Transition'].map(mapping)

# Drop rows with null values in the 'Transition' column
hipp_train_c.dropna(subset=['Transition'], inplace=True)

# Prepare data for training
X_train = hipp_train_c.drop(['Transition', 'ID'], axis=1, errors='ignore')
y_train = hipp_train_c['Transition']

# Split the training data into train and validation sets
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=2022)

# Define the parameter grid for RandomizedSearchCV
# param_grid = {
#     'n_estimators': [50, 100, 200, 300],        # Number of boosting rounds
#     'max_depth': [3, 4, 5, 6],                 # Tree depth
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],   # Learning rate
#     'min_child_weight': [1, 3, 5],             # Minimum child weight
#     'subsample': [0.6, 0.8, 1.0],              # Subsample ratio of training samples
#     'colsample_bytree': [0.6, 0.8, 1.0],       # Subsample ratio of columns
#     'gamma': [0, 0.1, 0.3, 0.5],               # Minimum loss reduction
#     'reg_alpha': [0, 1, 10],                   # L1 regularization
#     'reg_lambda': [1, 10, 100]                 # L2 regularization
# }

# Initialize the XGBoost Classifier with `xgb_gbtree`
param_grid = {
    'booster': ['gbtree'],                   # Only use gbtree booster
    'colsample_bytree': [0.6, 0.8, 1.0],    # Column sampling
    'gamma': [0.0, 0.1, 0.2, 0.3, 0.5],     # Minimum loss reduction
    'learning_rate': [0.01, 0.05, 0.1, 0.2], # Learning rate
    'max_delta_step': [0, 1, 2],            # Max delta step for imbalanced data
    'max_depth': [3, 4, 5, 6],              # Depth of trees
    'min_child_weight': [1, 3, 5],          # Minimum sum of weights for child nodes
    'n_estimators': [50, 100, 150, 200],    # Number of boosting rounds
    'reg_alpha': [0.0, 0.01, 0.1, 1.0],     # L1 regularization
    'reg_lambda': [1.0, 5.0, 10.0, 20.0],   # L2 regularization
    'subsample': [0.6, 0.8, 1.0]            # Row sampling
}

xgb_clf = XGBClassifier(objective='multi:softmax', num_class=5, random_state=42)

# Setup RandomizedSearchCV
# random_search = RandomizedSearchCV(
#     estimator=xgb_clf,
#     param_distributions=param_grid,
#     n_iter=50,                   # Number of random combinations to try
#     scoring='accuracy',          # Use accuracy for scoring
#     cv=3,                        # 3-fold cross-validation
#     verbose=2,                   # Display progress
#     random_state=42,             # For reproducibility
#     n_jobs=-1                    # Use all available cores
# )

random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_grid,
    n_iter=50,               # Number of random combinations to try
    scoring='accuracy',      # Use accuracy for scoring
    cv=3,                    # 3-fold cross-validation
    verbose=2,               # Display progress
    random_state=42,         # For reproducibility
    n_jobs=-1                # Use all available cores
)


# Perform RandomizedSearchCV
print("Running RandomizedSearchCV...")
random_search.fit(X_train_split, y_train_split)

# Display the best parameters
print("Best parameters found: ", random_search.best_params_)

# Get the best model
best_xgb_model = random_search.best_estimator_

# Make predictions on the validation set
predictions = best_xgb_model.predict(X_test_split)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test_split, predictions, target_names=list(mapping.keys())))

# Make predictions on the final test set
X_test = hipp_test_c.drop(columns=['ID'], errors='ignore')
predictions_mapped = pd.Series(best_xgb_model.predict(X_test)).map({v: k for k, v in mapping.items()})

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'RowId': range(1, len(predictions_mapped) + 1),
    'Result': predictions_mapped
})

# Verify the number of predictions and save the submission
if len(submission_df) < 100:
    print(f"Warning: The test set contains less than 100 entries. Submission will have only {len(submission_df)} predictions.")
else:
    print(f"Total number of predictions: {len(submission_df)}")

submission_df.to_csv('submissionXGB_gbtree.csv', index=False)
print(f"Submission saved successfully with {len(submission_df)} predictions.")
