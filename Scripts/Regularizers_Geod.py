import torch

def comp_di(X_d):
    di = torch.empty((0, 1999), requires_grad=True).cuda()
    for i in range(X_d.shape[0]):
        cd = X_d[i]
        cd = torch.reshape(cd, (1, -1))
        
        x_d = torch.cat([X_d[:i, :], X_d[i+1:, :]], dim=0)
        d = x_d - cd
        d = torch.norm(d, dim=1, p=2).reshape((1, -1)).cuda()
        di = torch.cat((di, d), dim=0)    
    return di

def formulation1(X_geod, X_d2, p):
    loss = torch.tensor(0.0, requires_grad=True).cuda()
    for i in range(X_d2.shape[0]):
        cd2 = X_d2[i]
        cd2 = torch.reshape(cd2, (1, -1))
        
        x_d2 = torch.cat((X_d2[:i, :], X_d2[i+1:, :]), dim=0)
        
        d2 = x_d2 - cd2
        d2 = torch.norm(d2, dim=1, p=p)
#         d1, d2 = d1 / torch.mean(d1), d2 / torch.mean(d2)
        
        d1 = torch.cat((X_geod[i][:i, :], X_geod[i][:i+1, :]), dim=0)
        
        r = d1 / d2
        
        r[torch.isnan(r)] = 0
        r[torch.isinf(r)] = 0
        loss = loss + torch.var(r)
    return loss


def formulation2(X_geod, X_d2, p):
    R = torch.empty((0), requires_grad=True).cuda()
    for i in range(X_d2.shape[0]):
        cd2 = X_d2[i]
        cd2 = torch.reshape(cd2, (1, -1))
        
        x_d2 = torch.cat((X_d2[:i, :], X_d2[i+1:, :]), dim=0)
        
        d2 = x_d2 - cd2
#         d1, d2 = torch.linalg.norm(d1, dim=1, ord=2), torch.linalg.norm(d2, dim=1, ord=2)
        d2 = torch.norm(d2, dim=1, p=p)
        
        d1 = torch.cat((X_geod[i][:i], X_geod[i][i+1:]), dim=0)
        r = d1 / d2
        
        r[torch.isnan(r)] = 0
        r[torch.isinf(r)] = 0

        R = torch.cat((R, r), dim=0)
        
    return torch.var(R)

def sammons_stress(X_geod, X_d2, p):
    """
    X_d1: points in original distribution (need to update)
    X_d2: points in latent distribution
    """
    error = torch.tensor(0.0).cuda()
    denom = torch.tensor(0.0).cuda()
    for j in range(1, X_d2.shape[0]):
        x_d2ij = X_d2[:j, :]
        x_d2j = X_d2[j]
        dist_2j = x_d2ij - x_d2j
        dist_2j = torch.norm(dist_2j, p=p, dim=1)
        
        dist_1j = X_geod[j][:j]
        
        diff = dist_1j - dist_2j
    
        diff = diff ** 2
        
        diff = torch.div(diff, dist_1j)
        
        diff = diff[~torch.isinf(diff)]
        diff = diff[~torch.isnan(diff)]
        
        error = error + torch.sum(diff)
        denom = denom + torch.sum(dist_1j)
        
    return torch.div(error, denom)
#-------------change-------------
def formulation1log(X_geod_batch,x_latent,p):
    X_geod_batch_indices = torch.triu_indices(X_geod_batch.size()[0],X_geod_batch.size()[0],offset=1)
    X_geod_batch = X_geod_batch[X_geod_batch_indices[0],X_geod_batch_indices[1]]
    #print("After:",X_geod_batch)
    d_input = X_geod_batch
    #print(d_input)
    mean_d_input = d_input.mean()
    # std dev of d_input
    std_d_input = d_input.std(unbiased = False)
    # d_input_standardized = (d_input - mean_d_input)/std_d_input
    d_input_standardized = torch.div(torch.sub(d_input,mean_d_input),std_d_input)
    d_input_standardized = d_input_standardized.to("cuda:0")
    
    # Similarly for the Batch of Encoded Data Points
    d_latent = torch.cdist(x_latent,x_latent, p)
    d_latent_indices = torch.triu_indices(x_latent.size()[0],x_latent.size()[0],offset=1)
    d_latent = d_latent[d_latent_indices[0],d_latent_indices[1]]
    #print("d latent",d_latent)
    mean_d_latent = d_latent.mean()
    std_d_latent = d_latent.std(unbiased = False)
    d_latent_standardized = torch.div(torch.sub(d_latent,mean_d_latent),std_d_latent)
    reg_loss = torch.sub(d_input_standardized ,d_latent_standardized).abs().mean()
    
    return reg_loss


def formulation2log(X_geod, X_d2, p):
    R = torch.empty((0), requires_grad=True).cuda()
    for i in range(X_d2.shape[0]):
        cd2 = X_d2[i]
        cd2 = torch.reshape(cd2, (1, -1))
        
        x_d2 = torch.cat((X_d2[:i, :], X_d2[i+1:, :]), dim=0)
        
        d2 = x_d2 - cd2
#         d1, d2 = torch.linalg.norm(d1, dim=1, ord=2), torch.linalg.norm(d2, dim=1, ord=2)
        d2 = torch.norm(d2, dim=1, p=p)
        
        d1 = torch.cat((X_geod[i][:i], X_geod[i][i+1:]), dim=0)
        
        r = d2 / d1
        
        r[torch.isnan(r)] = 1
        
        r = torch.log(r)
        
        r[torch.isinf(r)] = 1

        R = torch.cat((R, r), dim=0)
        
    return torch.var(R)

def noreg(X_geod, X_d2, p):
    return torch.tensor(0.0).cuda()

def get_regularizer(reg):
    if reg == "reg1":
        return formulation1
    elif reg == "reg2":
        return formulation2
    elif reg == "reg1log":
        return formulation1log
    elif reg == "reg2log":
        return formulation2log
    elif reg == "sammon":
        return sammons_stress
    elif reg == "noreg":
        return noreg
    else:
        raise Exception("Invalid regularizer")
