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


def adaboost_predict(X: np.ndarray, stumps: list, alphas: list) -> np.ndarray:
    num_samples = X.shape[0]
    classes = np.unique([s['class1'] for s in stumps] + [s['class2'] for s in stumps])
    class_votes = {c: np.zeros(num_samples) for c in classes}

    for alpha, stump in zip(alphas, stumps):
        feature_index = stump['feature_index']
        threshold = stump['threshold']
        inequality = stump['inequality']
        c1 = stump['class1']
        c2 = stump['class2']

        preds = np.empty(num_samples, dtype=object)
        if inequality == 'lt':
            preds[X[:, feature_index] <= threshold] = c1
            preds[X[:, feature_index] > threshold] = c2
        else:  # gt
            preds[X[:, feature_index] > threshold] = c1
            preds[X[:, feature_index] <= threshold] = c2

        for c in classes:
            class_votes[c] += alpha * (preds == c).astype(float)

    # 找出每個樣本得票最高的類別
    final_preds = np.array(
        [max(class_votes, key=lambda c: class_votes[c][i]) for i in range(num_samples)]
    )

    return final_preds
