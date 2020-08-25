from typing import Tuple
from math import ceil


def get_feature_sizes(img_size: Tuple[int, int],
                      min_level: int,
                      max_level: int):
    feature_size_list = [img_size]
    for _ in range(max_level):
        feature_size_list.append((
            ceil(feature_size_list[-1][0] / 2),
            ceil(feature_size_list[-1][1] / 2),
        ))

    return feature_size_list[min_level: max_level+1]


if __name__ == '__main__':
    print(get_feature_sizes((500, 500), 3, 7))
