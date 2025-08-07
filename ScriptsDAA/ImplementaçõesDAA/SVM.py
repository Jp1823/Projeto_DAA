import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Carregar datasets
hipp_train = pd.read_csv('train_radiomics_hipocamp.csv', na_filter=False)
hipp_test = pd.read_csv('test_radiomics_hipocamp.csv', na_filter=False)
hipp_control = pd.read_csv('train_radiomics_occipital_CONTROL.csv', na_filter=False)

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

# Aplicar PCA para reduzir dimensionalidade
pca = PCA(n_components=50)  # Reduzir para 50 componentes principais
X_train_pca = pca.fit_transform(X_train_split)
X_test_pca = pca.transform(X_test_split)

# Treinar o modelo SVM nos atributos reduzidos
best_svm_rbf = SVC(C=50, kernel='rbf', probability=True)
best_svm_rbf.fit(X_train_pca, y_train_split)

# Fazer previsões no conjunto de teste
predictions_svm = best_svm_rbf.predict(X_test_pca)
print("Relatório de Classificação do SVM:")
print(classification_report(y_test_split, predictions_svm, target_names=list(mapping.keys()), zero_division=1))
print("Accuracy do SVM:", accuracy_score(y_test_split, predictions_svm))

# Fazer previsões no conjunto de teste final
X_test_final = hipp_test_c.drop(columns=['ID'], errors='ignore')
X_test_final_pca = pca.transform(X_test_final)  # Transformar o conjunto de teste final com PCA
test_predictions_svm = best_svm_rbf.predict(X_test_final_pca)

# Submissão
predictions_mapped = pd.Series(test_predictions_svm).map({v: k for k, v in mapping.items()})
submission_df = pd.DataFrame({
    'RowId': range(1, len(predictions_mapped) + 1),
    'Result': predictions_mapped
})
submission_df.to_csv('SVM-Submission-PCA.csv', index=False)
print("\nSubmissão salva com sucesso com", len(submission_df), "previsões.")
