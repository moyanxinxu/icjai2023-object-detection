from math import sqrt

import torch


class PriorBox:
    def __init__(self, image_size=300):
        self.image_size = image_size
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.steps = [8, 16, 32, 64, 100, 300]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    def __call__(self):
        boxes = []
        for index in range(len(self.feature_maps)):
            feature_map_size = self.feature_maps[index]
            scale_factor = self.image_size / self.steps[index]

            normalized_min_size = self.min_sizes[index] / self.image_size
            normalized_max_size = self.max_sizes[index] / self.image_size

            min_size = normalized_min_size
            max_size = sqrt(normalized_min_size * normalized_max_size)

            for row in range(feature_map_size):
                for col in range(feature_map_size):
                    normalized_box_center_x = (col + 0.5) / scale_factor
                    normalized_box_center_y = (row + 0.5) / scale_factor

                    boxes.append([normalized_box_center_x, normalized_box_center_y, min_size, min_size])
                    boxes.append([normalized_box_center_x, normalized_box_center_y, max_size, max_size])

                    for item in self.aspect_ratios[index]:
<<<<<<< HEAD:net/utils/priorbox.py
                        boxes.append([normalized_box_center_x, normalized_box_center_y, min_size * sqrt(item), min_size / sqrt(item)])
                        boxes.append([normalized_box_center_x, normalized_box_center_y, min_size / sqrt(item), min_size * sqrt(item)])
=======
                        boxes.append([normalized_box_center_x, normalized_box_center_y, min_size * sqrt(item),
                                      min_size / sqrt(item)])
                        boxes.append([normalized_box_center_x, normalized_box_center_y, min_size / sqrt(item),
                                      min_size * sqrt(item)])
>>>>>>> parent of 357af3c (refactor):utils/priorbox.py
        boxes = torch.Tensor(boxes).clamp(min=0, max=1)
        return boxes


if __name__ == '__main__':
    priors = PriorBox(300)()
    print(priors.shape)