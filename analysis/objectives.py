import numpy as np
import torch


def var_reduction_disc(batch_x, batch_y):
    
    all_classes = np.unique(batch_y.detach().cpu().numpy())
    class_centres = {}
    for y in all_classes:
        class_centres[y] = torch.mean(batch_x[batch_y==y, :], dim=0)
    obj1 = 0.
    cnt1 = 1
    for y in all_classes:
        for y_hat in all_classes:
            if y_hat != y:
                disc_direction = (class_centres[y] - class_centres[y_hat]).detach()
                disc_direction = disc_direction / torch.norm(disc_direction, p=2)
                class_features = batch_x[batch_y==y]
                projection = class_features @  disc_direction
                if len(projection) > 1:
                    obj1 += max(0., torch.var(projection) - 0.01)
                    cnt1 += 1
    obj2 = 0.
    cnt2 = 0
    for i in range(len(all_classes)):
        for j in range(i+1, len(all_classes)):
            obj2 += max(0., 5.0 - ((class_centres[all_classes[i]] - class_centres[all_classes[j]])**2).sum())
            cnt2 += 1 
    return obj1 / cnt1 + obj2 / cnt2




def var_reduction_disc_perp(batch_x, batch_y):
    
    all_classes = np.unique(batch_y.detach().cpu().numpy())
    class_centres = {}
    for y in all_classes:
        class_centres[y] = torch.mean(batch_x[batch_y==y, :], dim=0)
    obj1 = 0.
    cnt1 = 1
    for y in all_classes:
        for y_hat in all_classes:
            if y_hat != y:
                disc_direction = (class_centres[y] - class_centres[y_hat]).detach()
                disc_direction = disc_direction / torch.norm(disc_direction, p=2)
                class_features = batch_x[batch_y==y]
                projection = class_features @  (
                    torch.eye(disc_direction.shape[0], device=disc_direction.device) -\
                    disc_direction @ disc_direction.t())
                if len(projection) > 1:
                    var_perp = sum(((projection - projection.mean(dim=0, keepdim=True))**2).mean(dim=0))
                    obj1 += max(0., var_perp - 0.01)
                    cnt1 += 1
    obj2 = 0.
    cnt2 = 0
    for i in range(len(all_classes)):
        for j in range(i+1, len(all_classes)):
            obj2 += max(0., 5.0 - ((class_centres[all_classes[i]] - class_centres[all_classes[j]])**2).sum())
            cnt2 += 1 
    return obj1 / cnt1 + obj2 / cnt2




def var_reduction(batch_x, batch_y):
    
    all_classes = np.unique(batch_y.detach().cpu().numpy())
    # print("all_classes", all_classes)
    class_centres = {}
    for y in all_classes:
        class_centres[y] = torch.mean(batch_x[batch_y==y], dim=0)
        # print(class_centres[y].shape)
    obj1 = 0.
    cnt1 = 0
    for y in all_classes:
        class_features = batch_x[batch_y==y]
        obj1 += max(0., sum(((class_features - class_centres[y].detach())**2).mean(dim=0)) - 0.01) 
        cnt1 += 1
        # print("var:", y, sum(((class_features - class_centres[y])**2).mean(dim=0)).item())
    obj2 = 0.
    cnt2 = 0
    for i in range(len(all_classes)):
        for j in range(i+1, len(all_classes)):
            obj2 += max(0., 5.0 - ((class_centres[all_classes[i]] - class_centres[all_classes[j]])**2).sum())
            cnt2 += 1 
            # print("sep:", all_classes[i], all_classes[j], ((class_centres[i] - class_centres[j])**2).sum().item())
    return obj1 / cnt1 +  obj2 / cnt2    
