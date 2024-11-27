import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb
import time
from xgboost import XGBClassifier


# Carregar datasets
hipp_train = pd.read_csv('train_radiomics_hipocamp.csv', na_filter=False)
hipp_test = pd.read_csv('test_radiomics_hipocamp.csv', na_filter=False)

# Remover colunas com apenas um valor e colunas irrelevantes
colunas_remover = ['Image', 'diagnostics_Image-original_Hash', 'diagnostics_Mask-original_Hash',
                   'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMassIndex',
                   'diagnostics_Mask-original_CenterOfMass', 'Mask']
hipp_train_c = hipp_train.drop(columns=colunas_remover, errors='ignore').loc[:, hipp_train.nunique() > 1]
hipp_test_c = hipp_test.drop(columns=colunas_remover, errors='ignore').loc[:, hipp_test.nunique() > 1]

# Mapear a coluna 'Transition' para valores numéricos
mapping = {
    'CN-CN': 0,  # Estado Normal
    'CN-MCI': 1,  # Estado Intermediário
    'MCI-MCI': 2,  # Estado Intermediário
    'MCI-AD': 3,  # Demência
    'AD-AD': 4    # Demência
}
hipp_train_c['Transition'] = hipp_train_c['Transition'].map(mapping)

# Remover qualquer entrada com valores nulos na coluna 'Transition' após o mapeamento
hipp_train_c.dropna(subset=['Transition'], inplace=True)

# Preparar dados para treinamento
X_train = hipp_train_c.drop(['Transition', 'ID'], axis=1, errors='ignore')
y_train = hipp_train_c['Transition']

# define data_dmatrix
data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)

# Dividir o conjunto de treinamento em treino e teste
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=2022)

# Definir o espaço de parâmetros para RandomizedSearchCV
# declare parameters
param_grid = {
    'n_estimators': [50, 100, 200, 300],        # Number of boosting rounds
    'max_depth': [3, 4, 5, 6],                 # Tree depth
    'learning_rate': [0.01, 0.05, 0.1, 0.2],   # Learning rate
    'min_child_weight': [1, 3, 5],             # Minimum child weight
    'subsample': [0.6, 0.8, 1.0],              # Subsample ratio of training samples
    'colsample_bytree': [0.6, 0.8, 1.0],       # Subsample ratio of columns
    'gamma': [0, 0.1, 0.3, 0.5],               # Minimum loss reduction
    'alpha': [0, 1, 10]                        # L1 regularization
}

# Initialize the XGBoost Classifier
xgb_clf = XGBClassifier(objective='multi:softmax', num_class=5, random_state=42)

# Setup RandomizedSearchCV, cross-validation and scoring
random_search = RandomizedSearchCV(
    estimator=xgb_clf, 
    param_distributions=param_grid, 
    n_iter=50,                 # Number of random combinations to try
    scoring='accuracy',        # Use accuracy for scoring
    cv=3,                      # 3-fold cross-validation
    verbose=2,                 # Display progress
    random_state=42,           # For reproducibility
    n_jobs=-1                  # Use all available cores
)

# Perform RandomizedSearchCV
print("Running RandomizedSearchCV...")
random_search.fit(X_train, y_train)

# Display the best parameters
print("Best parameters found: ", random_search.best_params_)

# Get the best model
best_xgb_model = random_search.best_estimator_

# Make predictions on the validation set
predictions = best_xgb_model.predict(X_test_split)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test_split, predictions, target_names=list(mapping.keys())))

# Fazer previsões no conjunto de teste final com o melhor modelo
X_test = hipp_test_c.drop(columns=['ID'], errors='ignore')
predictions_mapped = pd.Series(best_xgb_model.predict(X_test)).map({v: k for k, v in mapping.items()})

# Criar o DataFrame de submissão
submission_df = pd.DataFrame({
    'RowId': range(1, len(predictions_mapped) + 1),
    'Result': predictions_mapped
})

# Verificação de número de previsões e salvar submissão
if len(submission_df) < 100:
    print(f"Aviso: O conjunto de teste contém menos de 100 entradas. Submissão terá apenas {len(submission_df)} previsões.")
else:
    print(f"Número total de previsões: {len(submission_df)}")

submission_df.to_csv('submissionXGBRand.csv', index=False)
print(f"Submissão salva com sucesso com exatamente {len(submission_df)} previsões.")