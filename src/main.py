import numpy as np
import config
from data_loader import load_data
from graph_builder import run_ball_tree, compute_correlation_matrix, find_threshold, rbo_matrix, jaccardK, jaccardK_median, jaccardKMax
from utils import create_folds, compute_confidence_interval
from gcn_model import GCNClassifier
import random
import torch

CORRELATION_FUNCTIONS= {'rbo': rbo_matrix, 'jaccard_max': jaccardKMax, 'jaccard_median': jaccardK_median, 'jaccardk': jaccardK}

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    DATASET_TO_USE= 'flowers'
    FEATURES_EXTRACTOR= 'resnet'
    CORRELATION_FUNCTION_TO_USE= 'rbo'
    USE_AUTOMATIC_THRESHOLD= True
    MANUAL_THRESHOLD= 0.4

    print(f'\n--- Iniciando Experimento para {DATASET_TO_USE.upper()} com features {FEATURES_EXTRACTOR.upper()} ---\n')

    features, labels= load_data(DATASET_TO_USE, FEATURES_EXTRACTOR)

    try:
        selected_correlation_func= CORRELATION_FUNCTIONS[CORRELATION_FUNCTION_TO_USE]
    except KeyError:
        print(f"Erro: Função de correlação '{CORRELATION_FUNCTION_TO_USE}' não reconhecida.")
        print(f"Opções válidas são: {list(CORRELATION_FUNCTIONS.keys())}")
        return


    ranked_lists= run_ball_tree(features, k=config.L)
    correlation_matrix= compute_correlation_matrix(ranked_lists, top_k=config.top_k, correlation_func=selected_correlation_func)

    if USE_AUTOMATIC_THRESHOLD:
        best_threshold, _= find_threshold(correlation_matrix=correlation_matrix, ranked_lists=ranked_lists, k_graph=config.k_graph, L= config.L)
        threshold_to_use= best_threshold
    else:
        threshold_to_use= MANUAL_THRESHOLD
        print(f"\nUsando limiar manual: {threshold_to_use}")
    
    folds= create_folds(features, labels, n_folds= config.N_folds)
    clf= GCNClassifier('gcn-net', ranked_lists, len(labels), number_neighbors=config.k_graph, learning_rate = config.learning_rate)
    fold_acuracies= []

    for fold_idx, (test_idx, train_idx) in enumerate(folds):
        repetition_accuracies= []
        for _ in range(config.N_repetitions):
            clf.prepare(test_index=test_idx, train_index=train_idx, features=features, labels=labels, matrix=correlation_matrix, limiar=threshold_to_use, correlation='default')
            _, predictions= clf.train_and_predict()
            test_labels= [labels[i] for i in test_idx]
            acc= sum(1 for i, p in enumerate(predictions) if test_labels[i] == p) / len(predictions)
            repetition_accuracies.append(acc)
        
        mean_acc_for_fold= np.mean(repetition_accuracies)
        fold_acuracies.append(mean_acc_for_fold)

    print(clf.edge_index)

    final_mean, final_std= compute_confidence_interval(fold_acuracies)
    print("\n--- Resultados Finais ---")
    print(f"Dataset: {DATASET_TO_USE}, Features: {FEATURES_EXTRACTOR}, Limiar: {threshold_to_use:.4f}")
    print(f"Acurácia Final: {final_mean * 100:.2f}% ± {final_std * 100:.2f}%")

if __name__ == '__main__':
    main()
