import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import time

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

# Definir os bins e os labels corretamente para a coluna 'Age'
bins = [55, 65, 70, 75, 78, 81, 84, 86, 88, np.inf]
labels = ['55-65', '65-70', '70-75', '75-78', '78-81', '81-84', '84-86', '86-88', '88+']

# Aplicar o binning na coluna 'Age' para o conjunto de treino e teste
hipp_train_c['Age_Bin'] = pd.cut(hipp_train_c['Age'], bins=bins, labels=labels, right=False)
hipp_test_c['Age_Bin'] = pd.cut(hipp_test_c['Age'], bins=bins, labels=labels, right=False)

# Criar as colunas binárias para cada faixa etária no conjunto de treino e teste
for label in labels:
    hipp_train_c[label] = (hipp_train_c['Age_Bin'] == label).astype(int)
    hipp_test_c[label] = (hipp_test_c['Age_Bin'] == label).astype(int)

# Remover a coluna 'Age' e 'Age_Bin' do conjunto de treino e teste
hipp_train_c.drop(columns=['Age', 'Age_Bin'], inplace=True)
hipp_test_c.drop(columns=['Age', 'Age_Bin'], inplace=True)

# Preparar dados para treinamento
X_train = hipp_train_c.drop(['Transition', 'ID'], axis=1)
y_train = hipp_train_c['Transition']

# Dividir o conjunto de treinamento em treino e teste
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=2022)

# Definir o grid de hiperparâmetros para o XGBClassifier
param_grid = {
    'n_estimators': [97],  
    'max_depth': [3],               
    'learning_rate': [0.2],   
    'subsample': [0.6],         
    'colsample_bytree': [0.6],  
    'gamma': [0.1],
    'min_child_weight': [3, 5],
    'reg_alpha': [0.01, 0.1],
    'reg_lambda': [10, 100],
    'max_delta_step': [1, 10],
    'booster': ['gbtree', 'dart']
}

# Configurar o modelo XGBClassifier e o GridSearchCV
xgb_model = XGBClassifier(random_state=2022, use_label_encoder=False, eval_metric='mlogloss')
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

# Executar a busca
start_time = time.time()
grid_search.fit(X_train_split, y_train_split)
elapsed_time = time.time() - start_time

print(f"Tempo para encontrar o melhor modelo: {elapsed_time:.3f} segundos")
print("Melhores parâmetros encontrados:", grid_search.best_params_)
print("Melhor acurácia:", grid_search.best_score_)

# Usar o melhor modelo encontrado para fazer previsões
best_xgb_model = grid_search.best_estimator_
predictions = best_xgb_model.predict(X_test_split)

# Exibir o classification report
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
    print("Aviso: O conjunto de teste contém menos de 100 entradas. Submissão terá apenas", len(submission_df), "previsões.")
else:
    print("Número total de previsões:", len(submission_df))

submission_df.to_csv('XGBNewFeature.csv', index=False)
print("Submissão salva com sucesso com exatamente", len(submission_df), "previsões.")