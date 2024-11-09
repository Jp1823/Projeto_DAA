import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
import time
import shap

# Carregar datasets
hipp_train = pd.read_csv('/mnt/c/Users/João/Documents/DAA/Projeto/train_radiomics_hipocamp.csv', na_filter=False)
hipp_test = pd.read_csv('/mnt/c/Users/João/Documents/DAA/Projeto/test_radiomics_hipocamp.csv', na_filter=False)

# Exibir informações básicas e verificar valores nulos
print("Informações do dataset de treino:")
hipp_train.info()
print("\nInformações do dataset de teste:")
hipp_test.info()

# Remover colunas com apenas um valor e colunas irrelevantes
colunas_remover = ['Image', 'diagnostics_Image-original_Hash', 'diagnostics_Mask-original_Hash',
                   'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMassIndex',
                   'diagnostics_Mask-original_CenterOfMass', 'Mask']
hipp_train_c = hipp_train.drop(columns=colunas_remover, errors='ignore').loc[:, hipp_train.nunique() > 1]
hipp_test_c = hipp_test.drop(columns=colunas_remover, errors='ignore').loc[:, hipp_test.nunique() > 1]

# Mapear a coluna 'Transition' para valores numéricos
mapping = {
    'CN-CN': 1,  # Estado Normal
    'CN-MCI': 2,  # Estado Intermediário
    'MCI-MCI': 3,  # Estado Intermediário
    'MCI-AD': 4,  # Demência
    'AD-AD': 5    # Demência
}
hipp_train_c['Transition'] = hipp_train_c['Transition'].map(mapping)

# Preparar dados para treinamento
X_train = hipp_train_c.drop(['Transition', 'ID'], axis=1)
y_train = hipp_train_c['Transition']

# Dividir o conjunto de treinamento em treino e teste
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=2022)

# Treinar o modelo
pg_model = RandomForestClassifier(bootstrap=True, max_depth=10, random_state=2022)
pg_model.fit(X_train_split, y_train_split)

# Fazer previsões no conjunto de teste
predictions = pg_model.predict(X_test_split)

# Exibir o classification report
print(classification_report(y_test_split, predictions, target_names=list(mapping.keys())))

#Medir o MDI
start_time = time.time()

mdi_importances = pd.Series(pg_model.feature_importances_, index=X_test_split.columns)

elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
print("Feature importances using MDI:\n", mdi_importances )
mdi_importances.to_csv("mdi_importances.csv", index=True)
#Implementação do permutation importance

result = permutation_importance(
    pg_model, X_test_split, y_test_split, n_repeats=10, random_state=42, n_jobs=2
)

# Ordenar os índices de importâncias em ordem crescente
sorted_importances_idx = result.importances_mean.argsort()

# Criar um DataFrame com as importâncias, ordenado
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=X_test_split.columns[sorted_importances_idx],
)

# Salvar o DataFrame de importâncias em um arquivo CSV
importances.to_csv("permutation_importances.csv", index=False)

#Implementação do SHAP
shap.initjs()
explainer = shap.TreeExplainer(pg_model)
shap_values = explainer(X_test_split)
shap.plots.heatmap(shap_values[:,:,1])
shap_df = pd.DataFrame(shap_values.values[:, :, 1], columns=X_test_split.columns)

# Guardar o Shap no csv
shap_df.to_csv("shap_importances.csv", index=False)

# Fazer previsões no conjunto de teste final
X_test = hipp_test_c.drop(columns=['ID'], errors='ignore')

# Verificar valores nulos em X_test
print("Valores nulos em X_test:")
print(X_test.isnull().sum())

# Dicionário inverso para mapear de volta para os rótulos originais
inverse_mapping = {1: 'CN-CN', 2: 'CN-MCI', 3: 'MCI-MCI', 4: 'MCI-AD', 5: 'AD-AD'}
predictions_mapped = pd.Series(pg_model.predict(X_test)).map(inverse_mapping)

# Criar o DataFrame de submissão preservando o ID original do conjunto de teste
submission_df = pd.DataFrame({
    'RowId': range(1, len(predictions_mapped) + 1),  # Criar IDs de 0 até o número de previsões
    'Result': predictions_mapped
})

# Garantindo que estamos pegando até 100 previsões
if len(submission_df) < 100:
    print("Aviso: O conjunto de teste contém menos de 100 entradas. Submissão terá apenas", len(submission_df), "previsões.")
else:
    print("Número total de previsões:", len(submission_df))

# Salvar as previsões
submission_df.to_csv('submission.csv', index=False)
print("Submissão salva com sucesso com exatamente", len(submission_df), "previsões.")