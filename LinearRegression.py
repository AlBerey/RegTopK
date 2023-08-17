# This is the Code for Linear Regression Experiment
# Consistent with what presented in the paper and Section 2 of the Supplementary file


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as Fn
import copy 

import matplotlib.pyplot as plt
import pandas
import SynthDataSet as sds
import DistLearnLinearReg as fl

# Select the device
device = torch.device("mps" if torch.has_mps else "cpu")

### Hyper-parameters
Num_clients = 20
lr = 0.01
batch_size_min = 4000
round_length = [1 for _ in range(Num_clients)]  
# round_length = [np.random.randint(1,50) for _ in range(Num_clients)] 
num_rounds = 2500

### Dataset Parameters
Dataset_size = Num_clients*500
dim_x = 100
dist_cls = 'uniform'

# set batch size of each client somewhere between <batch_size_min> and the maximum possible batch size (i.e., deterministic SGD)
batch_size = [np.random.randint(batch_size_min,Dataset_size) for _ in range(Num_clients)]

### Sparsification Parameters

# Reference case, no sparsification, i.e., s = 1
sprs_ref = 1
num_ref = int(dim_x*sprs_ref)

# Sparsified cases
sprs_clients = .4
num_k = int(dim_x*sprs_clients)

# No sparsification at server
sprs_server = 1
num_server = int(dim_x*sprs_server)

# RegTopK Parameter
Mu = 4
### Dataset Generation
dataset = sds.SynthDataSet(Dataset_size, Num_clients, model_size=dim_x, dist_cls=dist_cls, mean_power_client=5, purterb_var=.5)
dataset.Generate(device)

## Generate Test Dataset
dataset.TrainToTest(ratio = 1, device=device)

# Define Global Models
model_gl = nn.Linear(dim_x, 1, bias=False).to(device)
alg = torch.optim.SGD(model_gl.parameters(), lr=lr)
Spr_er_server = torch.zeros(dim_x).to(device)

# topK
model_gl_topK = copy.deepcopy(model_gl).to(device)
alg_topK = torch.optim.SGD(model_gl_topK.parameters(), lr=lr)
Spr_er_topK = [torch.zeros(dim_x).to(device) for _ in range(Num_clients)]
Spr_er_topK_server = torch.zeros(dim_x).to(device)

# RegTopK
model_gl_regK = copy.deepcopy(model_gl).to(device)
alg_regK = torch.optim.SGD(model_gl_regK.parameters(), lr=lr)
Spr_er_regK = [torch.zeros(dim_x).to(device) for _ in range(Num_clients)]
Spr_er_regK_server = torch.zeros(dim_x).to(device)
sel_regK = [torch.zeros(dim_x).to(device) for _ in range(Num_clients)]
Delta = [1e8*torch.ones(dim_x).to(device) for _ in range(Num_clients)]
deviation = 1e5

# Reference model with global minimum
model_true = nn.Linear(dim_x, 1, bias=False).to(device)
model_true.weight.data = dataset.x_star

# Initialization
# counter of local SGD steps
cnt = [0 for _ in range(Num_clients)]
# flag of validation step
flg = 0

# MSE lists
MSE_fl = []
MSE_topK = []
MSE_regK = []
MSE_true = []

# Difference to optimal solution
dif_FL = []
dif_topK = []
dif_regK = []

# Training
for rnd in range(num_rounds):

    W_loc = []
    W_loc_topK = []
    W_loc_regK = []
    grd_regK = []

    for k in range(Num_clients):
        
        model_loc = fl.model_copier(model_gl).to(device)
        alg_loc = torch.optim.Adam(model_loc.parameters(), lr=lr)

        model_loc_topK = fl.model_copier(model_gl_topK).to(device)
        alg_loc_topK = torch.optim.SGD(model_loc_topK.parameters(), lr=lr)

        model_loc_regK = fl.model_copier(model_gl_regK).to(device)
        alg_loc_regK = torch.optim.SGD(model_loc_regK.parameters(), lr=lr)


        for _ in range(round_length[k]):

            # Collect a batch
            cnt[k], X_train, y_train = fl.next_batch(cnt[k], dataset.train_dataset[k], batch_size[k])
            
            # model evaluation
            alg_loc.zero_grad()
            y_hat = model_loc(X_train)
            y_hat = y_hat.reshape(y_train.shape)
            loss = Fn.mse_loss(y_hat,y_train)
            loss.backward()
            alg_loc.step()

            # model evaluation topK
            alg_loc_topK.zero_grad()
            y_hat_topK = model_loc_topK(X_train)
            y_hat_topK = y_hat_topK.reshape(y_train.shape)
            loss_topK = Fn.mse_loss(y_hat_topK,y_train)
            loss_topK.backward()
            alg_loc_topK.step()

            # model evaluation regK
            alg_loc_regK.zero_grad()
            y_hat_regK = model_loc_regK(X_train)
            y_hat_regK = y_hat_regK.reshape(y_train.shape)
            loss_regK = Fn.mse_loss(y_hat_regK,y_train)
            loss_regK.backward()
            alg_loc_regK.step()
        
        # Save and Sparsify

        # FL full
        W_k = model_loc.weight.grad.detach()
        W_loc.append(W_k)

        # FL topK 
        W_k_topK = model_loc_topK.weight.grad.detach()
        W_k_sprs_topK, not_used, Spr_er_topK[k] = fl.topK(W_k_topK, Spr_er_topK[k], num_k, device=device)
        
        W_loc_topK.append(W_k_sprs_topK)

        # FL reg-topK 
        W_k_regK = model_loc_regK.weight.grad.detach()
        Params_k = [Delta[k], Mu]
        W_k_spr_regK, grd_k_regK, Spr_er_regK[k], sel_regK[k] = fl.RegTopK(W_k_regK, Spr_er_regK[k], Params_k, num_k, device=device)

        grd_regK.append(grd_k_regK)
        W_loc_regK.append(W_k_spr_regK)

    
    # Model aggregation and downlink sparsification

    # Full FL
    W_aggr = sum(W_loc)/Num_clients
    W_aggr_spr, not_used, Spr_er_server = fl.topK(W_aggr, Spr_er_server, num_ref, device=device)

    alg.zero_grad()
    model_gl.weight.grad = W_aggr_spr.detach()
    alg.step()

    dif_rnd = torch.norm(model_gl.weight.data.flatten() - dataset.x_star.flatten(),p=2)**2
    dif_FL.append(dif_rnd.to('cpu').item())

    # topK
    W_aggr_topK = sum(W_loc_topK)/Num_clients
    W_aggr_topK_spr, not_used, Spr_er_topK_server = fl.topK(W_aggr_topK, Spr_er_topK_server, num_server, device=device)

    alg_topK.zero_grad()
    model_gl_topK.weight.grad = W_aggr_topK_spr.detach()
    alg_topK.step()

    dif_topK_rnd = torch.norm(model_gl_topK.weight.data.flatten() - dataset.x_star.flatten(),p=2)**2
    dif_topK.append(dif_topK_rnd.to('cpu').item())

    # regK
    W_aggr_regK = sum(W_loc_regK)/Num_clients
    W_aggr_regK_spr, grd_regK_mean, Spr_er_regK_server = fl.topK(W_aggr_regK, Spr_er_regK_server, num_server, device=device)


    alg_regK.zero_grad()
    model_gl_regK.weight.grad = W_aggr_regK_spr.detach()
    x_regK = copy.deepcopy(model_gl_regK.weight.data.detach())
    alg_regK.step()
    model_gl_regK, deviation = fl.termination(model_gl_regK, x_regK, deviation)

    

    dif_regK_rnd = torch.norm(model_gl_regK.weight.data.flatten() - dataset.x_star.flatten(),p=2)**2
    dif_regK.append(dif_regK_rnd.to('cpu').item())

    # update Delta
    Delta = fl.Delta_updater(grd_aggr=grd_regK_mean, grd_clients=grd_regK, selection_clients=sel_regK)
    

    # Validate every 20 iterations
    if (rnd+1) % 20 == 0:
        print(f'round {rnd+1}:')
        flg += 1
        with torch.no_grad():
            mse = 0
            mse_topK = 0
            mse_regK = 0
            mse_true = 0

            for x_test, y_test in dataset.test_set_true:
            
                y_pred = model_gl(x_test)
                mse += (y_pred-y_test)**2

                y_pred_topK = model_gl_topK(x_test)
                mse_topK += (y_pred_topK-y_test)**2

                y_pred_regK = model_gl_regK(x_test)
                mse_regK += (y_pred_regK-y_test)**2

                y_pred_true = model_true(x_test)
                mse_true += (y_pred_true-y_test)**2

            MSE_fl.append( np.log10(mse.item() / dataset.test_size_true) )
            MSE_topK.append(np.log10(mse_topK.item() / dataset.test_size_true))
            MSE_regK.append(np.log10(mse_regK.item() / dataset.test_size_true))
            MSE_true.append(np.log10(mse_true.item() / dataset.test_size_true))

# counter for total iterations
rounds = [i for i in range(num_rounds)]
# counter for validation steps
valid_steps = [i+1 for i in range(flg)]

# Plot distance from optimum
plt.figure('log |x_t - x_star|^2')
plt.plot(rounds, np.log10(dif_FL), label='Dist SGD')
plt.plot(rounds, np.log10(dif_topK), label='TopK')
plt.plot(rounds, np.log10(dif_regK), label='RegTopK')
plt.legend()
plt.show()


# Save Data
Data = [f"{i}    {np.log10(dif_FL[i])}" for i in range(num_rounds)]
Data.append("Dist SGD Ends")
for i in range(num_rounds):
    Data.append(f"{i}    {np.log10(dif_topK[i])}")
Data.append("TopK Sparsification Ends")
for i in range(num_rounds):
    Data.append(f"{i}    {np.log10(dif_regK[i])}")
Data.append("RegTopK Sparsification Ends")

# Save into a LaTex table
df = pandas.DataFrame(Data)
df.style.hide(axis="index").to_latex(f'./Diff_to_Optima_Mu_{Mu}_Sparsity_{sprs_clients}.tex',hrules=True)