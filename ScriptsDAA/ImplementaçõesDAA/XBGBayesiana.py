import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Carregar datasets
hipp_train = pd.read_csv('/mnt/c/Users/João/Documents/DAA/Projeto/train_radiomics_hipocamp.csv', na_filter=False)
hipp_test = pd.read_csv('/mnt/c/Users/João/Documents/DAA/Projeto/test_radiomics_hipocamp.csv', na_filter=False)

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

# Dividir o conjunto de treinamento em treino e teste
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=2022)

# Definir o modelo LGBMClassifier
lgbm_model = LGBMClassifier(random_state=2022)

# Define o espaço de busca com BayesSearchCV
param_space = {
    'learning_rate': Real(1e-3, 0.5, prior='log-uniform'),
    'num_leaves': Integer(31, 127),
    'min_child_samples': Integer(5, 100),
    'subsample': Real(0.1, 1.0),
    'colsample_bytree': Real(0.1, 1.0),
    'lambda_l1': Real(1e-5, 10.0, prior='log-uniform'),  
    'lambda_l2': Real(1e-5, 10.0, prior='log-uniform'),  
    'max_depth': Integer(3, 15),
    'min_data_in_leaf': Integer(20, 100),
    'max_bin': Integer(100, 500),
    'min_split_gain': Real(0.0, 0.1)
}

opt = BayesSearchCV(
    estimator=lgbm_model,
    search_spaces=param_space,
    n_iter=50,  
    scoring='accuracy',  
    cv=5,  
    random_state=2022,
    verbose=2,
    n_jobs=-1
)

# Treina o modelo
opt.fit(X_train_split, y_train_split)

# Mostra os melhores parâmetros
print("Melhores parâmetros encontrados:", opt.best_params_)

# Avalia no conjunto de teste
best_model = opt.best_estimator_
predictions = best_model.predict(X_test_split)
accuracy = accuracy_score(y_test_split, predictions)
print("Acurácia no conjunto de teste:", accuracy)

# Exibir o classification report
print(classification_report(y_test_split, predictions, target_names=list(mapping.keys())))

# Fazer previsões no conjunto de teste final com o melhor modelo
X_test = hipp_test_c.drop(columns=['ID'], errors='ignore')
predictions_mapped = pd.Series(best_model.predict(X_test)).map({v: k for k, v in mapping.items()})

# Criar o DataFrame de submissão
submission_df = pd.DataFrame({
    'RowId': range(1, len(predictions_mapped) + 1),
    'Result': predictions_mapped
})

# Salvar a submissão
submission_df.to_csv('submissionXGBBayesiana2.csv', index=False)
print(f"Submissão salva com sucesso com exatamente {len(submission_df)} previsões.")
