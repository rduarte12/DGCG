import numpy as np
from sklearn.neighbors import BallTree
import config


def run_ball_tree(features, k):
    if not isinstance(features, np.ndarray):
        raise ValueError('As features devem estar em formato numpy')
    if features.ndim != 2:
        raise ValueError('As features devem ser um vetor 2D (n_samples, n_features)')
    
    tree= BallTree(features)
    _, ranked_lists= tree.query(features, k=k)
    return ranked_lists

def rbo_matrix(rks1, rks2, top_k):
    stored = set()
    r= 0.9
    acum_inter = 0
    score = 0
    img1_leftover = set()
    img2_leftover = set()

    for k in range(top_k):
        img1_elm = rks1[k]
        img2_elm = rks2[k]

        if img1_elm not in stored and img1_elm == img2_elm:
            acum_inter += 1
            stored.add(img1_elm)
        else:
            if img1_elm not in stored:
                if img1_elm in img2_leftover:
                    acum_inter += 1
                    stored.add(img1_elm)
                    img2_leftover.remove(img1_elm)
                else:
                    img1_leftover.add(img1_elm)
            if img2_elm not in stored:
                if img2_elm in img1_leftover:
                    acum_inter += 1
                    stored.add(img2_elm)
                    img1_leftover.remove(img2_elm)
                else:
                    img2_leftover.add(img2_elm)

        score += (r**k) * (acum_inter / (k + 1))

    scrN = (1 - r) * score
    return scrN

def jaccardK_median(rks1, rks2, top_k):
    scores = []
    stored = set()
    stored_img1 = set()
    stored_img2 = set()
    acum_inter = 0
    img1_leftover = set()
    img2_leftover = set()

    for k in range(top_k):
        img1_elm = rks1[k]
        img2_elm = rks2[k]

        if img1_elm not in stored and img1_elm == img2_elm:
            acum_inter += 1
            stored.add(img1_elm)
            stored_img1.add(img1_elm)
            stored_img2.add(img2_elm)
        else:
            if img1_elm not in stored:
                if img1_elm in img2_leftover:
                    acum_inter += 1
                    stored.add(img1_elm)
                    img2_leftover.remove(img1_elm)
                    stored_img1.add(img1_elm)
                else:
                    img1_leftover.add(img1_elm)
                    stored_img1.add(img1_elm)
            if img2_elm not in stored:
                if img2_elm in img1_leftover:
                    acum_inter += 1
                    stored.add(img2_elm)
                    img1_leftover.remove(img2_elm)
                    stored_img2.add(img2_elm)
                else:
                    img2_leftover.add(img2_elm)
                    stored_img2.add(img2_elm)

        denominador = len(stored_img1) + len(stored_img2) - acum_inter
        if denominador > 0:
            score = acum_inter / denominador
            scores.append(score)

    score= np.median(scores)
    return score

def jaccardK(rks1, rks2, top_k):
    score = 0

    x_leftover = set()
    y_leftover = set()
    stored = set()  
    stored_x = set()
    stored_y = set()
    cur_inter = 0
    for i in range(top_k):
        x_elm = rks1[i]
        y_elm = rks2[i]
        if x_elm not in stored and x_elm == y_elm:
            cur_inter += 1
            stored.add(x_elm)
            stored_x.add(x_elm)
            stored_y.add(y_elm)
        else:
            if x_elm not in stored:
                if x_elm in y_leftover:
            
                    cur_inter += 1
                    stored.add(x_elm)
                    stored_x.add(x_elm)
                    y_leftover.remove(x_elm)
                else:
                    x_leftover.add(x_elm)
                    stored_x.add(x_elm)
            if y_elm not in stored:
                if y_elm in x_leftover:
        
                    cur_inter += 1
                    stored.add(y_elm)
                    stored_y.add(y_elm)
                    x_leftover.remove(y_elm)
                else:
                    y_leftover.add(y_elm)
                    stored_y.add(y_elm)

        score += cur_inter / (len(stored_x)+len(stored_y)-cur_inter)

    return score / top_k

def jaccardKMax(rks1, rks2, top_k):
    stored = set()
    stored_img1 = set()
    stored_img2 = set()
    acum_inter = 0
    max_score=0 
    score = 0
    img1_leftover = set()
    img2_leftover = set()

    for k in range(top_k):
        img1_elm = rks1[k]
        img2_elm = rks2[k]

        if img1_elm not in stored and img1_elm == img2_elm:
            acum_inter += 1
            stored.add(img1_elm)
            stored_img1.add(img1_elm)
            stored_img2.add(img2_elm)
        else:
            if img1_elm not in stored:
                if img1_elm in img2_leftover:
                    acum_inter += 1
                    stored.add(img1_elm)
                    img2_leftover.remove(img1_elm)
                    stored_img1.add(img1_elm)
                else:
                    img1_leftover.add(img1_elm)
                    stored_img1.add(img1_elm)
            if img2_elm not in stored:
                if img2_elm in img1_leftover:
                    acum_inter += 1
                    stored.add(img2_elm)
                    img1_leftover.remove(img2_elm)
                    stored_img2.add(img2_elm)
                else:
                    img2_leftover.add(img2_elm)
                    stored_img2.add(img2_elm)

        denominador = len(stored_img1) + len(stored_img2) - acum_inter
        if denominador > 0:
            score = acum_inter / denominador
            if score > max_score:
                max_score= score
        
    return max_score

def compute_correlation_matrix(ranked_lists, top_k, correlation_func=rbo_matrix):
    n_samples= len(ranked_lists)
    correlation_matrix= np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(config.L):
            neighbor_j= ranked_lists[i][j]
            score= correlation_func(ranked_lists[i], ranked_lists[neighbor_j], top_k)
            correlation_matrix[i, neighbor_j]= score
    
    return correlation_matrix

def compute_coef_for_threshold(correlation_matrix, ranked_lists, threshold, k_graph):
    n_samples= correlation_matrix.shape[0]
    num_edges= 0

    for i in range(n_samples):
        for j in range(k_graph):
            neighbor_idx= ranked_lists[i][j]
            
            if correlation_matrix[i, neighbor_idx] > threshold:
                num_edges += 1

    total_possible_edges= n_samples * 200
    
    return num_edges/total_possible_edges

def find_threshold(correlation_matrix, ranked_lists, k_graph, L, initial_threshold=0.5, target_density=(0.03, 0.04), max_iter= 10):
    print(f"\n---Iniciando busca automática por limiar ideal ---")

    low, high= 0.0, 1.0
    best_threshold= initial_threshold

    for i in range(1, max_iter+1):
        coef= compute_coef_for_threshold(correlation_matrix, ranked_lists, best_threshold, k_graph)
        print(f'Iteração {i}: Limiar= {best_threshold:.4f}, Densidade= {coef:.4f}')

        if target_density[0] <= coef <= target_density[1]:
            print(f"\nEncontrado Limiar ideal dentro do intervalo alvo em {i} iterações.")
            return best_threshold, coef
        
        if coef < target_density[0]:
            high= best_threshold
            best_threshold= (low + high)/2
        else:
            low= best_threshold
            best_threshold= (low + high)/2

    print("\nBusca finalizada. Não foi possível convergir para o intervalo exato.")
    print(f"\nRetornando o limiar mais próximo encontrado: {best_threshold:.4f}")
    return best_threshold, coef
