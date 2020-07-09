import numpy as np
import torch


def var_reduction_disc(batch_x, batch_y, model):
    
    all_classes = np.unique(batch_y.detach().cpu().numpy())
    # take only top sqrt(d) classes
    n_classes = len(all_classes)
    all_classes = all_classes[:int(np.ceil(np.sqrt(n_classes)))]
    class_centres = {}
    for y in all_classes:
        class_centres[y] = torch.mean(batch_x[batch_y==y, :], dim=0)
        # print("centre", y, class_centres[y].shape, batch_x.shape, batch_y.shape, batch_x[batch_y==y, :].shape)
    obj1 = 0.
    cnt1 = 0
    for y in all_classes:
        for y_hat in all_classes:
            if y_hat != y:
                with torch.no_grad():
                    # w = model.module.fc.L.weight_v[y] / (torch.norm(model.module.fc.L.weight_v[y])+0.00001)
                    disc_direction = (class_centres[y] - class_centres[y_hat])
                    # disc_direction = torch.randn(disc_direction.shape, device=disc_direction.device)
                    # w = w.unsqueeze(1)
                    disc_direction = disc_direction.unsqueeze(1)
                    # print("shapes", w.shape, disc_direction.shape)
                    # disc_direction = torch.mm(torch.eye(w.shape[0], device=w.device) - w@w.t(), disc_direction).squeeze()
                    disc_direction = disc_direction.div(torch.norm(disc_direction+0.00001, p=2)).detach()
                class_features = batch_x[batch_y==y]
                # print("class_features", class_features.shape, "disc_direction", disc_direction.shape)
                projection = class_features @  disc_direction
                if len(projection) > 1:
                    # print(y, y_hat, projection)
                    obj1 += torch.var(projection)
                    cnt1 += 1
    # obj2 = 0.
    # cnt2 = 0
    # for i in range(len(all_classes)):
    #     for j in range(i+1, len(all_classes)):
    #         obj2 += max(0., 10.0 - ((class_centres[all_classes[i]] - class_centres[all_classes[j]])**2).sum())
    #         cnt2 += 1 
    # print("aux loss: ", obj1 / cnt1, obj1, cnt1)
    # + obj2 / cnt2
    return (obj1 / cnt1)  if cnt1 > 0 else torch.tensor(0., device=batch_x.device)




def var_reduction_ortho(batch_x, batch_y, fc):
    
    # classifier wts
    W = fc.L.weight
    K = fc.K
    outdim = fc.outdim

    # take only top sqrt(d) classes
    all_classes = np.unique(batch_y.detach().cpu().numpy())
    n_classes = len(all_classes)
    all_classes = all_classes[:int(np.ceil(np.sqrt(n_classes)))]
    n_classes = len(all_classes)

    # project onto hyper spehere
    batch_x = batch_x.div(torch.norm(batch_x, dim=1, keepdim=True)+0.00001)
    
    # reduce variance null space of class specific linear projection
    obj1 = 0.
    obj2 = 0.
    cnt = 0
    for y in all_classes:
        class_features = batch_x[batch_y==y]
        w = W.reshape(outdim, K, -1)[y, :, :] 
        
        # retain orthonormality
        assert w.shape[0] <= w.shape[1] 
        obj2 += torch.sum((w @ w.t() - torch.eye(w.shape[0], device=w.device))**2)

        w = w / (torch.norm(w, dim=1, keepdim=True)+0.00001)
        ortho_proj = torch.mm((torch.eye(w.shape[1], device=w.device) - w.t()@w), class_features.t())
        obj1 += torch.sum(torch.mean(ortho_proj ** 2, 0))
        cnt += 1

    # rfc inter-class var
    assert W.shape[0] <= W.shape[1] 
    inter_var = 0.
    intra_var = 0.
    for i in range(n_classes):
        f_i = batch_x[batch_y==all_classes[i]]
        for j in range(i+1, n_classes):
            f_j = batch_x[batch_y==all_classes[j]]
            inter_var += 1.0 - torch.mean(f_i @ f_j.t())
        intra_var += 1.0 - torch.mean(f_i @ f_i.t())

    obj3 = intra_var * len(all_classes) / (inter_var + 0.00001)


    # print("obj1:", 100. * obj1 / cnt1)
    # print("obj2:", obj2)
    return 100. * (obj1 / cnt) + (obj2 / cnt) + obj3



def rfc_and_pc(batch_x, batch_y, fc):
    
    # classifier wts
    W = fc.running_pc
    # outdim*K, indim
    K = fc.K
    outdim = fc.outdim

    # take only top sqrt(d) classes
    all_classes = np.unique(batch_y.detach().cpu().numpy())
    n_classes = len(all_classes)
    all_classes = all_classes[:int(np.ceil(np.sqrt(n_classes)))]
    n_classes = len(all_classes)
    
    # project onto hyper spehere
    batch_x = batch_x.div(torch.norm(batch_x, dim=1, keepdim=True)+0.00001)
    
    # reduce variance null space of class specific linear projection
    obj1 = 0.
    cnt1 = 0
    for y in all_classes:
        class_features = batch_x[batch_y==y]
        w = W.reshape(outdim, K, -1)[y, :, :] 
        w = w / (torch.norm(w, dim=1, keepdim=True)+0.00001)
        ortho_proj = torch.mm((torch.eye(w.shape[1], device=w.device) - w.t()@w), class_features.t())
        obj1 += torch.sum(torch.mean(ortho_proj ** 2, 0))
        cnt1 += 1

    # rfc inter-class var
    assert W.shape[0] <= W.shape[1] 
    inter_var = 0.
    intra_var = 0.
    for i in range(n_classes):
        f_i = batch_x[batch_y==all_classes[i]]
        for j in range(i+1, n_classes):
            f_j = batch_x[batch_y==all_classes[j]]
            inter_var += 1.0 - torch.mean(f_i @ f_j.t())
        intra_var += 1.0 - torch.mean(f_i @ f_i.t())

    obj2 = intra_var * len(all_classes) / (inter_var + 0.00001)
    
    # update fc
    fc.update_pc(batch_x, batch_y)

    return 100. * (obj1 / cnt1) + obj2



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
