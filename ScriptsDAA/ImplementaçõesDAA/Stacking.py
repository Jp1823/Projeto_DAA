from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split, StratifiedKFold
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

# Função para gerar as features de stacking
def get_stacking_features(models, X, y):
    """
    Gera as features de stacking usando validação cruzada.
    
    Parameters:
    - models: dicionário de modelos base.
    - X: DataFrame com as features de entrada.
    - y: Série com os rótulos de saída.
    
    Returns:
    - stacking_features: array numpy com as previsões dos modelos base.
    """
    n_classes = len(np.unique(y))
    stacking_features = np.zeros((X.shape[0], len(models) * n_classes))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
    
    for i, (name, model) in enumerate(models.items()):
        for train_idx, val_idx in skf.split(X, y):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            val_predictions = model.predict_proba(X.iloc[val_idx])
            stacking_features[val_idx, i * n_classes:(i + 1) * n_classes] = val_predictions
    
    return stacking_features

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

# Dividir o conjunto de treinamento em treino e validação
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

# Definir os modelos
models = {
    'SVM': best_svm_rbf,
    'XGB1': best_xgb_model,
    'XGB2': xgb_model,
    'RandomForest': rf_model
}

# Função para gerar as features para o meta-modelo
def get_stacking_features(models, X, y):
    stacking_features = np.zeros((X.shape[0], len(models) * len(np.unique(y))))
    for i, (name, model) in enumerate(models.items()):
        stacking_features[:, i * len(np.unique(y)):(i + 1) * len(np.unique(y))] = cross_val_predict(
            model, X, y, cv=5, method='predict_proba'
        )
    return stacking_features

# Gerar previsões usando validação cruzada para o meta-modelo
stacking_features_train = get_stacking_features(models, X_train_split, y_train_split)

# Treinar o meta-modelo
meta_model = LogisticRegression(random_state=2022)
meta_model.fit(stacking_features_train, y_train_split)

# Gerar previsões para o conjunto de teste de validação
stacking_features_test = np.zeros((X_test_split.shape[0], len(models) * len(np.unique(y_train_split))))
for i, (name, model) in enumerate(models.items()):
    stacking_features_test[:, i * len(np.unique(y_train_split)):(i + 1) * len(np.unique(y_train_split))] = model.predict_proba(X_test_split)

final_predictions = meta_model.predict(stacking_features_test)

# Avaliar o desempenho
print("Relatório de Classificação do Stacking Ensemble:")
print(classification_report(y_test_split, final_predictions, target_names=list(mapping.keys())))
print("Accuracy do Stacking Ensemble:", accuracy_score(y_test_split, final_predictions))

# Previsões finais para o conjunto de teste
X_test_final = hipp_test_c.drop(['ID'], axis=1)  # Apenas removendo 'ID', pois 'Transition' não existe
stacking_features_final_test = np.zeros((X_test_final.shape[0], len(models) * len(np.unique(y_train_split))))
for i, (name, model) in enumerate(models.items()):
    stacking_features_final_test[:, i * len(np.unique(y_train_split)):(i + 1) * len(np.unique(y_train_split))] = model.predict_proba(X_test_final)

final_test_predictions = meta_model.predict(stacking_features_final_test)

# Submissão
predictions_mapped = pd.Series(final_test_predictions).map({v: k for k, v in mapping.items()})
submission_df = pd.DataFrame({
    'RowId': range(1, len(predictions_mapped) + 1),
    'Result': predictions_mapped
})
submission_df.to_csv('Stacking-Ensemble-Submissions.csv', index=False)
print("\nSubmissão salva com sucesso com", len(submission_df), "previsões.")