import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix



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
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.3, random_state=2022)

# Ajustar as classes para começar do índice 0
y_train_split -= 1
y_test_split -= 1

# Treinar o modelo
xgb_model = XGBClassifier(max_depth=1, objective='multi:softprob', num_class=5)
xgb_model.fit(X_train_split, y_train_split)

# Avaliar o modelo
xgb_score = xgb_model.score(X_test_split, y_test_split)
print("Accuracy: %.2f%%" % (xgb_score * 100))

# Fazer previsões no conjunto de teste
xgb_predictions = xgb_model.predict(X_test_split)

# Exibir o classification report
print(classification_report(y_test_split, xgb_predictions))

# Gerar a matriz de confusão
conf_matrix = confusion_matrix(y_test_split, xgb_predictions)

# Exibir a matriz de confusão
conf_matrix_df = pd.DataFrame(conf_matrix, index=mapping.values(), columns=mapping.values())
print("Matriz de Confusão:")
print(conf_matrix_df)


# Fazer previsões no conjunto de teste final
X_test = hipp_test_c.drop(columns=['ID'], errors='ignore')

# Verificar valores nulos em X_test
print("Valores nulos em X_test:")
print(X_test.isnull().sum())

# Dicionário inverso para mapear de volta para os rótulos originais
inverse_mapping = {1: 'CN-CN', 2: 'CN-MCI', 3: 'MCI-MCI', 4: 'MCI-AD', 5: 'AD-AD'}
predictions_mapped = pd.Series(xgb_model.predict(X_test) + 1).map(inverse_mapping)

# Criar o DataFrame de submissão preservando o ID original do conjunto de teste
submission_df = pd.DataFrame({
    'RowId': range(1, len(predictions_mapped) + 1),  # IDs de 1 até o número de previsões
    'Result': predictions_mapped
})

# Garantindo que estamos pegando até 100 previsões
if len(submission_df) < 100:
    print("Aviso: O conjunto de teste contém menos de 100 entradas. Submissão terá apenas", len(submission_df), "previsões.")
else:
    print("Número total de previsões:", len(submission_df))

# Salvar as previsões
submission_df.to_csv('submissionXGB.csv', index=False)
print("Submissão salva com sucesso com exatamente", len(submission_df), "previsões.")
