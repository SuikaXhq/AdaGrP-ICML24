from methods.base import FMTLMethod, Client, Server
import numpy as np
from tqdm import tqdm
import logging
import time

class IFCAClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subgroup_label = None
    
    def update(self):
        # self.update_cluster_identity()
        self.theta = self.alpha_list[self.subgroup_label, :]
        return super().update()
        
    def update_cluster_identity(self):
        '''
        residual = y_i - X_i beta - Z_i theta_i
        F_i = residual.T W residual
        ---------
        residual: (k, n) ndarray, residual[i, :] = residual for theta of cluster i
        '''
        residual = self.y - self.X @ self.beta - (self.Z @ self.alpha_list.T).T
        K = residual.shape[0]
        F_i_list = np.zeros(K)
        for k in range(residual.shape[0]):
            F_i_list[k] = residual[k, :] @ self.get_W() @ residual[k, :]
        self.subgroup_label = np.argmin(F_i_list)
    
    def upload(self):
        return self.beta, self.theta, self.subgroup_label
        
    def download(self, key: str, **content):
        if key == 'params':
            self.beta = content['beta']
            self.alpha_list = content['alpha_list']
        else:
            raise NotImplementedError

class IFCAServer(Server):
    def __init__(self, M, p, q, k, client_list=None):
        super().__init__(M, p, q, client_list=client_list)
        self.k = k # number of cluster
        self.beta = np.random.normal(size=(self.p,))
        self.alpha_list = np.random.normal(size=(self.k, self.q))
        self.beta_list = np.tile(self.beta, (self.M, 1))
        self.theta_list = np.zeros((self.M, self.q))
        self.subgroup_labels = np.arange(M, dtype=int)
        
    def receive(self, prompt, **kwparams):
        if prompt == 'update':
            self.participate_idx = kwparams['participate_idx']
            self.beta_list[self.participate_idx, :] = kwparams['beta_list']
            self.theta_list[self.participate_idx, :] = kwparams['theta_list']
            self.subgroup_labels[self.participate_idx] = kwparams['subgroup_labels']
        else:
            raise NotImplementedError
    
    def broadcast(self, prompt: str):
        if prompt == 'params':
            for client in self.client_list:
                client.download(prompt, beta=self.beta, alpha_list=self.alpha_list)
        else:
            raise NotImplementedError
        
    def aggregate(self):
        assert self.beta_list is not None
        assert self.theta_list is not None
        self._update_client_sample_size()
        self.beta = np.average(self.beta_list[self.participate_idx, :], axis=0, weights=self.N[self.participate_idx])
        for s in range(self.k):
            subgroup_s_indices = np.where(self.subgroup_labels[self.participate_idx] == s)[0]
            if len(subgroup_s_indices) > 0:
                self.alpha_list[s, :] = np.average(self.theta_list[self.participate_idx, :][subgroup_s_indices, :], axis=0,
                                                   weights=self.N[self.participate_idx][subgroup_s_indices])
        return self.beta, self.alpha_list

class IFCAMethod(FMTLMethod):
    def __init__(self,
        p, q, k,
        n_clients: int = 0,
        max_round: int = 100,
        max_step: int = 100,
        lr: float = 1e-4,
        sigma_u2 = None,
        sigma_e2 = None,
        args = None,
        setting = None
    ) -> None:
        self.args = args
        self.data = None
        self.k = k # number of cluster
        self.p = p
        self.q = q
        self.M = n_clients
        self.R = max_round
        self.T = max_step
        self.lr = lr
        self.sigma_u2 = sigma_u2
        self.sigma_e2 = sigma_e2
        
        self.server = IFCAServer(self.M, self.p, self.q, self.k)
        self.clients = [IFCAClient(lr=self.lr, max_step=self.T, early_stop=args.early_stop) for _ in range(self.M)]
        self.server.add_clients(self.clients)
        self.participate_idx = np.arange(self.M)
        
    def fit(self):
        estimate_full = []
        
        for r in tqdm(range(self.R), desc='Communication round'):
            beta_list = []
            theta_list = []
            subgroup_labels = []
            local_steps = np.zeros((self.M,), dtype=int)
            aggregate_time = 0
            if self.updater is not None:
                self.updater(round=r)
            self.server.broadcast('params')
            for i, client in enumerate(self.clients):
                if i in self.participate_idx:
                    start_time = time.perf_counter()
                    client.update_cluster_identity()
                    end_time = time.perf_counter()
                    aggregate_time += end_time - start_time
                    local_steps[i] = client.update()
                    beta_i, theta_i, subgroup_labels_i = client.upload()
                    beta_list.append(beta_i)
                    theta_list.append(theta_i)
                    subgroup_labels.append(subgroup_labels_i)
            aggregate_time /= self.M
            self.server.receive('update', beta_list=beta_list, theta_list=theta_list, subgroup_labels=subgroup_labels, participate_idx=self.participate_idx)
            
            start_time = time.perf_counter()
            self.server.aggregate()
            end_time = time.perf_counter()
            aggregate_time += end_time - start_time
            
            estimate = self.server.get_estimate()
            estimate['average_steps'] = local_steps.mean()
            estimate['aggregate_time'] = aggregate_time
            estimate_full.append(estimate)
            
        logging.info('IFCA estimate finished.')
        return estimate_full