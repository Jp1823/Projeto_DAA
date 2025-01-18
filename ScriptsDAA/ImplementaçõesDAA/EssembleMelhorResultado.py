import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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

# Configurando os modelos com os melhores parâmetros
best_svm_rbf = SVC(C=50, kernel='rbf', probability=True).fit(X_train_split, y_train_split)
best_xgb_model = XGBClassifier(booster='gblinear', colsample_bytree=0.6, gamma=0.1, learning_rate=0.2,
                                max_delta_step=1, max_depth=3, min_child_weight=3, n_estimators=97,
                                reg_alpha=0.01, reg_lambda=10, subsample=0.6, random_state=2022).fit(X_train_split, y_train_split)
xgb_model = XGBClassifier(booster='dart', colsample_bytree=0.6, gamma=0.1, learning_rate=0.2,
                          max_delta_step=1, max_depth=3, min_child_weight=5, n_estimators=97,
                          reg_alpha=0.01, reg_lambda=10, subsample=0.6, eval_metric='mlogloss').fit(X_train_split, y_train_split)
rf_model = RandomForestClassifier(n_estimators=75, max_depth=None, min_samples_split=10,
                                   min_samples_leaf=4, random_state=2022).fit(X_train_split, y_train_split)

# Criar dicionário com todos os modelos já treinados
models = {
    'svm_rbf': best_svm_rbf,
    'xgb_G': best_xgb_model,
    'xgb_R': xgb_model,
    'rf': rf_model
}

# Mapear cada classe para o melhor modelo baseado nas métricas
class_model_mapping = {
    0: 'xgb_R',  # Escolha do melhor modelo para a classe 0
    1: 'xgb_R',  # Escolha do melhor modelo para a classe 1
    2: 'xgb_R',  # Escolha do melhor modelo para a classe 2
    3: 'svm_rbf',  # Escolha do melhor modelo para a classe 3
    4: 'svm_rbf'   # Escolha do melhor modelo para a classe 4
}

# Obter previsões e probabilidades de todos os modelos no conjunto de teste
predictions = {}
probabilities = {}
for name, model in models.items():
    try:
        predictions[name] = model.predict(X_test_split)
        probabilities[name] = model.predict_proba(X_test_split)
    except Exception as e:
        print(f"Erro ao gerar previsões para o modelo {name}: {e}")

# Lista para armazenar as previsões finais
final_predictions = []

# Para cada amostra no conjunto de teste
for i in range(len(X_test_split)):
    class_probs = {}
    for class_label, model_name in class_model_mapping.items():
        prob = probabilities.get(model_name, [[0] * len(class_model_mapping)])[i][class_label]
        class_probs[class_label] = prob
    best_class = max(class_probs.items(), key=lambda x: x[1])[0]
    final_predictions.append(best_class)

# Avaliar o modelo
final_predictions = np.array(final_predictions)
print("Relatório de Classificação do Smart Ensemble:")
print(classification_report(y_test_split, final_predictions, target_names=list(mapping.keys())))
print("Accuracy do Smart Ensemble:", accuracy_score(y_test_split, final_predictions))

# Fazer previsões no conjunto de teste final
X_test_final = hipp_test_c.drop(columns=['ID'], errors='ignore')
test_probabilities = {}
for name, model in models.items():
    try:
        test_probabilities[name] = model.predict_proba(X_test_final)
    except Exception as e:
        print(f"Erro ao gerar probabilidades para o modelo {name} no conjunto final: {e}")

final_test_predictions = []
for i in range(len(X_test_final)):
    class_probs = {}
    for class_label, model_name in class_model_mapping.items():
        prob = test_probabilities.get(model_name, [[0] * len(class_model_mapping)])[i][class_label]
        class_probs[class_label] = prob
    best_class = max(class_probs.items(), key=lambda x: x[1])[0]
    final_test_predictions.append(best_class)

# Submissão
predictions_mapped = pd.Series(final_test_predictions).map({v: k for k, v in mapping.items()})
submission_df = pd.DataFrame({
    'RowId': range(1, len(predictions_mapped) + 1),
    'Result': predictions_mapped
})
submission_df.to_csv('Smart-Ensemble-Sequential.csv', index=False)
print("\nSubmissão salva com sucesso com", len(submission_df), "previsões.")