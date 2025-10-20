from typing import Tuple, List, Dict

import numpy as np


def adaboost_train(
    data_train: np.ndarray, label_train: np.ndarray, T: int
) -> Tuple[List[Dict], List[float]]:
    num_samples, num_features = data_train.shape
    classes, class_map = np.unique(label_train, return_inverse=True)
    num_classes = len(classes)

    D = np.ones(num_samples) / num_samples  # 初始化樣本權重

    stumps = []  # 儲存每個決策樹樁的參數
    alphas = []  # 儲存每個決策樁的權重

    for m in range(T):
        best_stump = None
        best_error = float('inf')

        # 遍歷每個特徵以尋找最佳分割
        for feature_i in range(num_features):
            # 1. 根據當前特徵值對樣本索引進行排序
            sorted_indices = np.argsort(data_train[:, feature_i])

            # 根據排序後的索引重新排列權重和標籤
            sorted_weights = D[sorted_indices]
            sorted_labels_int = class_map[sorted_indices]

            # 2. 初始化左右分區的類別權重總和
            # weights_left[k] = 左分區中類別 k 的權重總和
            weights_left = np.zeros(num_classes)
            # 初始時，所有點都在右分區
            weights_right = np.array(
                [
                    np.sum(sorted_weights[sorted_labels_int == k])
                    for k in range(num_classes)
                ]
            )

            # 3. 線性掃描所有可能的分割點
            for i in range(num_samples - 1):
                # 將第 i 個樣本從右分區移動到左分區
                current_label_idx = sorted_labels_int[i]
                current_weight = sorted_weights[i]
                weights_left[current_label_idx] += current_weight
                weights_right[current_label_idx] -= current_weight

                # 如果相鄰的特徵值相同，則它們之間不能作為分割點，跳過
                val_i = data_train[sorted_indices[i], feature_i]
                val_i_plus_1 = data_train[sorted_indices[i + 1], feature_i]
                if val_i == val_i_plus_1:
                    continue

                # 閾值設定在兩個不同值的樣本中間
                threshold = (val_i + val_i_plus_1) / 2

                # 4. 找到左右分區的最佳預測類別（加權眾數）
                c_left_idx = np.argmax(weights_left)
                c_right_idx = np.argmax(weights_right)

                # 5. 計算加權錯誤率
                # 錯誤率 = (左分區總權重 - 左分區最佳類別的權重) + (右分區總權重 - 右分區最佳類別的權重)
                error = (np.sum(weights_left) - weights_left[c_left_idx]) + (
                    np.sum(weights_right) - weights_right[c_right_idx]
                )

                # 6. 更新最佳決策樁
                # 在此優化下，'lt' 和 'gt' 兩種不等式會得到相同的最小錯誤率。
                # 我們統一使用 'lt' 形式儲存，其中 class1 代表小於等於閾值的預測，class2 代表大於的預測。
                if error < best_error:
                    best_error = error
                    best_stump = {
                        'feature_index': feature_i,
                        'threshold': threshold,
                        'inequality': 'lt',  # 'lt' is sufficient
                        'class1': classes[c_left_idx],
                        'class2': classes[c_right_idx],
                    }

        stump = best_stump
        epsilon = best_error

        # 如果錯誤率太高，提前停止
        if epsilon >= 1 - (1 / num_classes):
            print(f'訓練在第 {m + 1} 輪停止，錯誤率過高: {epsilon:.4f}')
            break

        # 計算弱分類器的權重 alpha
        beta = epsilon / (1 - epsilon)
        alpha = np.log(1 / beta)

        # 為了更新樣本權重 D，需要生成當前最佳弱分類器的預測結果
        feature_values = data_train[:, stump['feature_index']]
        predictions = np.empty(num_samples, dtype=object)
        predictions[feature_values <= stump['threshold']] = stump['class1']
        predictions[feature_values > stump['threshold']] = stump['class2']

        # 更新樣本權重 D
        matches = (predictions == label_train).astype(float)
        D *= np.power(beta, matches)
        D /= np.sum(D)  # 歸一化

        stumps.append(stump)
        alphas.append(alpha)

        if best_error < 1e-9:  # 接近於 0
            print(f'訓練在第 {m + 1} 輪達到完美分類。')
            break

    return stumps, alphas


def softmax(x: np.ndarray) -> np.ndarray:
    """計算 softmax"""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def adaboost_predict_proba(X: np.ndarray, stumps: list, alphas: list) -> np.ndarray:
    """
    使用 Adaboost 進行預測，並回傳每個類別的機率。
    """
    num_samples = X.shape[0]
    classes = np.unique([s['class1'] for s in stumps] + [s['class2'] for s in stumps])
    # 確保 classes 的順序是固定的，例如 [0, 1, 2]
    classes = sorted(list(classes))
    class_map = {c: i for i, c in enumerate(classes)}
    
    # 初始化每個樣本對每個類別的信心分數
    scores = np.zeros((num_samples, len(classes)))

    for alpha, stump in zip(alphas, stumps):
        feature_index = stump['feature_index']
        threshold = stump['threshold']
        
        # 根據決策樁進行預測
        preds_c1_mask = X[:, feature_index] <= threshold
        preds_c2_mask = ~preds_c1_mask

        # 找到 c1 和 c2 在 scores 陣列中的索引
        c1_idx = class_map[stump['class1']]
        c2_idx = class_map[stump['class2']]

        # 將 alpha 加到對應類別的分數上
        scores[preds_c1_mask, c1_idx] += alpha
        scores[preds_c2_mask, c2_idx] += alpha

    # 使用 softmax 將分數轉換為機率
    probabilities = softmax(scores)
    
    return probabilities, classes


def adaboost_predict(X: np.ndarray, stumps: list, alphas: list) -> np.ndarray:
    probabilities, classes = adaboost_predict_proba(X, stumps, alphas)
    # 找出每個樣本機率最高的類別索引
    max_indices = np.argmax(probabilities, axis=1)
    # 將索引轉換回原始類別標籤
    final_preds = np.array(classes)[max_indices]
    return final_preds
