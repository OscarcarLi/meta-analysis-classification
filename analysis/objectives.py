import numpy as np
import torch


def var_reduction_disc(batch_x, batch_y):
    all_classes = np.unique(batch_y)
    class_centres = {}
    for y in all_classes:
        class_centres[y] = torch.mean(batch_x[batch_y==y, :], dim=0)
    obj = 0.
    cnt = 1
    for y in all_classes:
        for y_hat in all_classes:
            if y_hat != y:
                disc_direction = class_centres[y] - class_centres[y_hat]
                class_features = batch_x[batch_y==y, :]
                projection = class_features @  disc_direction
                obj += torch.var(projection)
                cnt += 1
    return obj / cnt
