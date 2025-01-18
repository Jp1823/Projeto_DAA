import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
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

# Configurando o modelo SVM com os melhores parâmetros
svm_model = SVC(C=50, kernel='rbf')

#Treinar com o modelo
svm_model.fit(X_train_split, y_train_split)

# Configurar o modelo XGBoost com os melhores parâmetros
best_xgb_model = XGBClassifier(
    booster='gblinear',
    colsample_bytree=0.6,
    gamma=0.1,
    learning_rate=0.2,
    max_delta_step=1,
    max_depth=3,
    min_child_weight=3,
    n_estimators=97,
    reg_alpha=0.01,
    reg_lambda=10,
    subsample=0.6,
    random_state=2022
)

xgb_model = XGBClassifier(
    booster='dart', 
    colsample_bytree=0.6,
    gamma=0.1,
    learning_rate=0.2,
    max_delta_step=1,
    max_depth=3,
    min_child_weight=5,
    n_estimators=97,  
    reg_alpha=0.01,
    reg_lambda=10,  
    subsample=0.6,
    eval_metric='mlogloss'
)

# Treinar os modelos XGBoost
best_xgb_model.fit(X_train_split, y_train_split)
xgb_model.fit(X_train_split, y_train_split)

best_params = {
    'n_estimators': 75,
    'max_depth': None,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
}

# Inicializando o modelo Random Forest com os melhores parâmetros
rf_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=2022
)

# Treinando o modelo com os melhores parâmetros
rf_model.fit(X_train_split, y_train_split)

# Fazer previsões para cada modelo
svm_rbf_preds = svm_model.predict(X_test_split)
xgb_g_preds = best_xgb_model.predict(X_test_split)
xgb_r_preds = xgb_model.predict(X_test_split)
rf_preds = rf_model.predict(X_test_split)

# Exibir previsões
svm_rbf_preds = svm_model.predict(X_test_split)
xgb_g_preds = best_xgb_model.predict(X_test_split)
xgb_r_preds = xgb_model.predict(X_test_split)
rf_preds = rf_model.predict(X_test_split)

# Relatórios de Classificação e Accuracy
print("Relatório de Classificação do modelo SVM (RBF):")
print(classification_report(y_test_split, svm_rbf_preds, target_names=list(mapping.keys())))
print("Accuracy do modelo SVM (RBF):", accuracy_score(y_test_split, svm_rbf_preds))

print("\nRelatório de Classificação do modelo XGB (G):")
print(classification_report(y_test_split, xgb_g_preds, target_names=list(mapping.keys())))
print("Accuracy do modelo XGB (G):", accuracy_score(y_test_split, xgb_g_preds))

print("\nRelatório de Classificação do modelo XGB (R):")
print(classification_report(y_test_split, xgb_r_preds, target_names=list(mapping.keys())))
print("Accuracy do modelo XGB (R):", accuracy_score(y_test_split, xgb_r_preds))

print("\nRelatório de Classificação do modelo Random Forest:")
print(classification_report(y_test_split, rf_preds, target_names=list(mapping.keys())))
print("Accuracy do modelo Random Forest:", accuracy_score(y_test_split, rf_preds))

# Definir os modelos ajustados
ensemble_model = VotingClassifier(
    estimators=[
        ('svm_rbf', svm_model),
        ('xgb_G', best_xgb_model),
        ('xgb_R', xgb_model),
        ('rf', rf_model)
    ],
    voting='hard',  # Votação por maioria de votos
    weights=[1, 2, 3, 2]  # Pesos para priorizar XGB (R)
)

# Treinar o modelo ensemble
ensemble_model.fit(X_train_split, y_train_split)

# Previsões do modelo ensemble
ensemble_preds = ensemble_model.predict(X_test_split)

# Avaliação do modelo ensemble
print("Relatório de Classificação do modelo Ensemble:")
print(classification_report(y_test_split, ensemble_preds, target_names=list(mapping.keys())))
print("Accuracy do modelo Ensemble:", accuracy_score(y_test_split, ensemble_preds))

# Fazer previsões no conjunto de teste final com o ensemble_model
X_test_final = hipp_test_c.drop(columns=['ID'], errors='ignore')
predictions_final = ensemble_model.predict(X_test_final)
predictions_mapped = pd.Series(predictions_final).map({v: k for k, v in mapping.items()})

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

submission_df.to_csv('SVM-Essemble.csv', index=False)
print("Submissão salva com sucesso com exatamente", len(submission_df), "previsões.")