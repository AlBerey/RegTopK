# Module for generating synthetic dataset for Linear Regression problem
# Details on its generation is given in Section "Numerical Validation" in the main paper and also Section 2 of Supplementary file

import numpy as np
import torch
import random

class SynthDataSet():
    '''
    Generate synthetic dataset: y_i = a_i * x + e_i
    Size: total size of the dataset (including all clients)
    Num_clients: number of clients in the network
    dist_cls: <'uniform'> = all clients get same number of datapoints
              <'random_size'> = share of each client is generated randomly
    mean_power_client: mean for each client (entries of x) is generated Gaussian with mean zero and vatiance <mean_power_client>
    purterb_var: variance of the noise added to each data-point (e_i)
    '''
    def __init__(self, Size, Num_clients, model_size, dist_cls = 'uniform', mean_power_client = 5, purterb_var = 0.5):
        self.dataset_size = Size
        self.num_clients = Num_clients
        self.model_size = model_size
        self.distribution_class = dist_cls


        if dist_cls == 'uniform':
            self.sharing_profile = np.ones(Num_clients, dtype=np.float64)/Num_clients 
        elif dist_cls == 'random_size':
            self.sharing_profile = np.random.uniform(1,8, Num_clients)
            self.sharing_profile /= np.sum(self.sharing_profile)
        else:
            raise ValueError('Invalid distribution class')
        
        client_size = np.floor(self.sharing_profile[:-1]*Size).astype(int)
        self.client_size = list(client_size)
        self.client_size.append(Size - np.sum(client_size))


        self.alpha = torch.tensor(mean_power_client)
        self.model_mean = torch.sqrt(self.alpha)*torch.randn(Num_clients)
        self.epsilon_var = torch.tensor(purterb_var)
        self.generated = False
        
        
    # method to generate the dataset
    def Generate(self, device = 'cpu'):
        '''
        device: the device to be used by torch, e.g., 'cpu', 'cuda' or 'mps'
        '''
        self.model_mean = self.model_mean.to(device)
        self.epsilon_var = self.epsilon_var.to(device)

        u_0 = torch.randn(self.model_size,1).to(device)
        
        X_set = []
        dist_set = []
        Gram = []
        z = []
        for k in range(self.num_clients):
            x_k = self.model_mean[k] * torch.ones(self.model_size,1).to(device) + torch.randn(self.model_size,1).to(device)
            X_set.append(x_k)

            A_k = torch.randn(self.client_size[k], self.model_size).to(device)
            y_k = torch.mm(A_k, x_k).reshape([self.client_size[k]]) + torch.sqrt(self.epsilon_var) * torch.randn(self.client_size[k]).to(device)
            dist_set.append([A_k, y_k])
            # Save gram matrices for calculating the global optima 
            Gram_k = torch.mm(A_k.T,A_k)
            z_k = torch.mm(A_k.T,y_k.reshape(-1,1))
            Gram.append(Gram_k)
            z.append(z_k)
        
        # Calculate the global optima
        G_sum = sum(Gram)
        z_sum = sum(z)

        x_star = torch.mm(torch.inverse(G_sum), z_sum)
        self.x_star = x_star.reshape(self.model_size)

        
        self.X_true = X_set
        self.generated = True
        self.train_dataset = dist_set
        
    # method to reshape the training set into a test dataset
    def TrainToTest(self, ratio = 1, device ='cpu'):
        '''
        ratio: fraction of train dataset to be included
        device: the device to be used by torch, e.g., 'cpu', 'cuda' or 'mps'
        '''
        if self.generated:
            test_set_true = []
            for k in range(self.num_clients):
                A_k = self.train_dataset[k][0].to(device)
                y_k = self.train_dataset[k][1].to(device)
                for i in range(self.client_size[k]):
                    test_set_true.append([A_k[i], y_k[i]])
            
            random.shuffle(test_set_true)
            self.test_size_true = int(ratio*self.dataset_size)
            self.test_set_true = test_set_true[:self.test_size_true] 
        else:
            raise ValueError('The Dataset has not been yet generated! Use the method .Generate()')
    
    # method to generate a test dataset independent of train dataset
    def indepTestSet(self, ratio = 1, device ='cpu'):
        '''
        This test dataset is independent of train dataset; however, it has the same feature vectors, i.e., same x's
        ratio: size of test dataset / size of train dataset
        device: the device to be used by torch, e.g., 'cpu', 'cuda' or 'mps'
        '''
        if self.generated:
            self.epsilon_var = self.epsilon_var.to(device)
            test_set_indep = []
            for k in range(self.num_clients):
                A_k_indp = torch.randn(self.client_size[k], self.model_size).to(device)
                x_k = self.X_true[k].to(device)
                y_k_indp = torch.mm(A_k_indp, x_k).reshape([self.client_size[k]]) + torch.sqrt(self.epsilon_var) * torch.randn(self.client_size[k]).to(device)
                for i in range(self.client_size[k]):
                    test_set_indep.append([A_k_indp[i], y_k_indp[i]])
            
            random.shuffle(test_set_indep)
            self.test_size_idep = int(ratio*self.dataset_size)
            self.test_set_indep = test_set_indep[:self.test_size_idep]

        else:
            raise ValueError('The Dataset has not been yet generated! Use the method .Generate()')
