from typing import Tuple

import cv2
import numpy as np
from collections import Counter

def difference(array: np.ndarray):
    x = []
    for i in range(len(array) - 1):
        x.append(array[i + 1] - array[i])

    return np.array(x)


def find_peek(array: np.ndarray):
    peek = difference(difference(array))
    peek_pos = np.argmax(peek) + 2
    return peek_pos


def k_means(points: np.ndarray):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    flags = cv2.KMEANS_RANDOM_CENTERS
    length = []
    max_k = min(10, points.shape[0])
    for k in range(2, max_k + 1):
        avg = 0
        for i in range(5):
            compactness, _, _ = cv2.kmeans(
                points, k, None, criteria, 10, flags)
            avg += compactness
        avg /= 5
        length.append(avg)

    peek_pos = find_peek(length)
    k = peek_pos + 2
    return k, cv2.kmeans(points, k, None, criteria, 10, flags)[1]  # labels


def get_group_center(points1: np.ndarray, points2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    k, labels = k_means(points1)
    labels = labels.flatten()
    labels_dict =  Counter(labels)
    print("聚类结果：",k ,labels_dict)

    selected_centers1 = []
    selected_centers2 = []
    for i in range(k):
        center1 = np.mean(points1[labels == i], axis=0)
        center2 = np.mean(points2[labels == i], axis=0)
        selected_centers1.append(center1)
        selected_centers2.append(center2)

    selected_centers1, selected_centers2 = np.array(
        selected_centers1), np.array(selected_centers2)


    return points1, points2


def main():
    x = np.array([[1, 1], [4, 2], [2, 2], [8, 80],[1, 1], [4, 2], [2, 2], [8, 80],[1, 1], [4, 2], [2, 2], [8, 80],[1, 1], [4, 2], [2, 2], [8, 80]], dtype=np.float32)
    y = np.array([[1, 1], [4, 2], [2, 2], [8, 80],[1, 1], [4, 2], [2, 2], [8, 80],[1, 1], [4, 2], [2, 2], [8, 80],[1, 1], [4, 2], [2, 2], [8, 80]], dtype=np.float32)
    print(get_group_center(x, y))
    pass


if __name__ == "__main__":
    main()