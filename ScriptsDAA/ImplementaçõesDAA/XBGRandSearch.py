import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# Remover qualquer entrada com valores nulos na coluna 'Transition' após o mapeamento
hipp_train_c.dropna(subset=['Transition'], inplace=True)

# Preparar dados para treinamento
X_train = hipp_train_c.drop(['Transition', 'ID'], axis=1, errors='ignore')
y_train = hipp_train_c['Transition']

# Dividir o conjunto de treinamento em treino e teste
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=2022)

# Definir o espaço de parâmetros para RandomizedSearchCV
param_distributions = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 10, 100],
    'scale_pos_weight': [1, 2, 5],
    'tree_method': ['hist']
}

# Instanciar o modelo XGBClassifier
xgb_model = XGBClassifier(random_state=2022, eval_metric='mlogloss')

# Configurar o RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=50,  # Define quantas combinações aleatórias testar
    cv=3,  # Número de folds para validação cruzada
    scoring='accuracy',
    verbose=2,
    n_jobs=-1,  # Usa todos os núcleos disponíveis para acelerar o processo
    random_state=2022
)

# Executar a busca com medição de tempo
start_time = time.time()
random_search.fit(X_train_split, y_train_split)
elapsed_time = time.time() - start_time

# Exibir os melhores parâmetros e resultados
print(f"Tempo para encontrar o melhor modelo: {elapsed_time:.2f} segundos")
print("Melhores parâmetros encontrados:", random_search.best_params_)
print(f"Melhor acurácia durante a validação cruzada: {random_search.best_score_:.4f}")

# Usar o melhor modelo encontrado para fazer previsões
best_xgb_model = random_search.best_estimator_
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
    print(f"Aviso: O conjunto de teste contém menos de 100 entradas. Submissão terá apenas {len(submission_df)} previsões.")
else:
    print(f"Número total de previsões: {len(submission_df)}")

submission_df.to_csv('submissionXGBRand.csv', index=False)
print(f"Submissão salva com sucesso com exatamente {len(submission_df)} previsões.")