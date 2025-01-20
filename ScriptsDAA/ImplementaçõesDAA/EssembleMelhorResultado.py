import itertools
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, pair_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import seaborn as sns

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


def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print(f"Results for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    print(classification_report(y_test, predictions, zero_division=1))
    
    return predictions

# Evaluate each model
for name, model in models.items():
    print(f"Evaluating {name}...")
    evaluate_model(model, X_test_split, y_test_split, name)
    
    
# Ensemble Evaluation (already implemented in your code)
print("Evaluating Smart Ensemble...")
print("Relatório de Classificação do Smart Ensemble:")
print(classification_report(y_test_split, final_predictions, target_names=list(mapping.keys())))
print("Accuracy do Smart Ensemble:", accuracy_score(y_test_split, final_predictions))

def plot_last_100_matrices(models, X_test_final, final_predictions, mapping, n_last=100):
    # Get the last n predictions
    last_n = min(n_last, len(final_predictions))  # In case we have fewer than 100 predictions
    last_predictions = final_predictions[-last_n:]
    last_X = X_test_final.iloc[-last_n:]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    axes = axes.ravel()
    
    # Create list of labels
    labels = list(mapping.keys())
    
    # Get predictions from each model for the last n samples
    for idx, (name, model) in enumerate(models.items()):
        model_predictions = model.predict(last_X)
        
        # Create confusion matrix between this model and fusion
        cm = confusion_matrix(last_predictions, model_predictions)
        
        # Plot manually to avoid tick issues
        im = axes[idx].imshow(cm, interpolation='nearest', cmap=plt.cm.YlOrRd)
        
        # Add colorbar
        fig.colorbar(im, ax=axes[idx])
        
        # Set title and labels
        axes[idx].set_title(f'Confusion Matrix - {name.upper()}\nvs Fusion Model', pad=20, fontsize=12)
        axes[idx].set_xlabel('Model Prediction', fontsize=10)
        axes[idx].set_ylabel('Fusion Model Prediction', fontsize=10)
        
        # Set ticks and labels
        tick_marks = np.arange(len(labels))
        axes[idx].set_xticks(tick_marks)
        axes[idx].set_yticks(tick_marks)
        axes[idx].set_xticklabels(labels, rotation=45, ha='right')
        axes[idx].set_yticklabels(labels)
        
        # Add text annotations in the cells
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            axes[idx].text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        # Add agreement percentage
        agreement = np.mean(last_predictions == model_predictions) * 100
        axes[idx].text(0.05, -0.15, f'Agreement with fusion: {agreement:.1f}%', 
                      transform=axes[idx].transAxes, fontsize=10)
    
    # Create distribution plot for fusion model predictions
    fusion_predictions = pd.Series(last_predictions).map({v: k for k, v in mapping.items()})
    sns.countplot(x=fusion_predictions, ax=axes[-2])
    axes[-2].set_title(f'Distribution of Fusion Model Predictions\n(Last {last_n} samples)', pad=20, fontsize=12)
    axes[-2].set_xticklabels(axes[-2].get_xticklabels(), rotation=45, ha='right', fontsize=10)
    
    # Remove the last subplot
    fig.delaxes(axes[-1])
    
    plt.suptitle(f'Model Comparisons for Last {last_n} Predictions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print(f"\nDistribution of Last {last_n} Predictions (Fusion Model):")
    print("-" * 50)
    value_counts = fusion_predictions.value_counts()
    for class_name, count in value_counts.items():
        percentage = (count / last_n) * 100
        print(f"{class_name}: {count} predictions ({percentage:.1f}%)")
    
    # Print agreement percentages
    print("\nModel Agreement with Fusion Model:")
    print("-" * 50)
    for name, model in models.items():
        model_predictions = model.predict(last_X)
        agreement = np.mean(last_predictions == model_predictions) * 100
        print(f"{name}: {agreement:.1f}% agreement")

# Call the function
plot_last_100_matrices(models, X_test_final, final_test_predictions, mapping)
