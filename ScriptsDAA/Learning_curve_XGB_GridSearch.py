import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

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
inverse_mapping = {v: k for k, v in mapping.items()}  # Create reverse mapping
hipp_train_c['Transition'] = hipp_train_c['Transition'].map(mapping)

# Preparar dados para treinamento
X_train = hipp_train_c.drop(['Transition', 'ID'], axis=1)
y_train = hipp_train_c['Transition']

# Dividir o conjunto de treinamento em treino e teste
test_size = max(100 / len(X_train), 0.1)  # Ensure test size is sufficient
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train, y_train, test_size=test_size, random_state=2022
)

# Definir e treinar o modelo
xgb_model = XGBClassifier(
    booster='dart',
    colsample_bytree=0.6,
    gamma=0.1,
    learning_rate=0.2,
    max_delta_step=1,
    max_depth=3,
    min_child_weight=3,
    n_estimators=97,
    reg_alpha=0.1,
    reg_lambda=100,
    subsample=0.6,
    random_state=2022
)
xgb_model.fit(X_train_split, y_train_split)

# Fazer previsões no conjunto de teste dividido
predictions = xgb_model.predict(X_test_split)
predictions_mapped = pd.Series(predictions).map(inverse_mapping)

# Exibir métricas
cm = confusion_matrix(y_test_split, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(mapping.keys()))
disp.plot(cmap=plt.cm.Blues)
plt.show()

print("Classification Report:")
print(classification_report(y_test_split, predictions, target_names=list(mapping.keys())))

# Fazer previsões no conjunto de teste final e mapear para rótulos
X_test = hipp_test_c.drop(columns=['ID'], errors='ignore')
final_predictions = xgb_model.predict(X_test)
final_predictions_mapped = pd.Series(final_predictions).map(inverse_mapping)

# Criar o DataFrame de submissão
submission_df = pd.DataFrame({
    'RowId': range(1, len(final_predictions_mapped) + 1),
    'Result': final_predictions_mapped
})

# Verificar número de previsões e salvar submissão
if len(submission_df) < 100:
    print("Aviso: O conjunto de teste contém menos de 100 entradas. Submissão terá apenas", len(submission_df), "previsões.")
else:
    print("Número total de previsões:", len(submission_df))

submission_df.to_csv('submission.csv', index=False)
print("Submissão salva com sucesso com exatamente", len(submission_df), "previsões.")
