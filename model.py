from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

# Définir une liste d'hyperparamètres à tester
hyperparameters = [
    {'out_channels': 16, 'kernel_size': 3, 'pool_size': 2, 'stride': 1},
    {'out_channels': 32, 'kernel_size': 5, 'pool_size': 2, 'stride': 1},
    {'out_channels': 64, 'kernel_size': 3, 'pool_size': 3, 'stride': 1}
]

for params in hyperparameters:
    # Initialiser le modèle avec les hyperparamètres actuels
    model = MultiTaskModel(in_channels, params['out_channels'], params['kernel_size'], params['pool_size'], params['stride'], fc_input_size, phrase_classes)
    
    # Entraînement du modèle
    start_time = time.time()
    # Code d'entraînement du modèle
    training_time = time.time() - start_time
    
    # Évaluation sur les données de test
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')
    
    # Affichage des résultats
    print(f"Hyperparameters: {params}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Training Time: {training_time} seconds\n")
