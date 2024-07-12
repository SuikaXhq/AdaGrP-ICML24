from methods.base import FMTLMethod, Client, Server
import numpy as np
from tqdm import tqdm
import logging

class FeSEMClient(Client):
    def __init__(self, server=None, lr=1e-3, max_step=1000, early_stop=False):
        super().__init__(server, lr=lr, max_step=max_step, early_stop=early_stop)
    
    def upload(self):
        return self.beta, self.theta
        
    def download(self, key: str, **content):
        if key == 'params':
            self.beta = content['beta']
            self.theta = content['theta']
        else:
            raise NotImplementedError

class FeSEMServer(Server):
    def __init__(self, M, p, q, k, client_list=None):
        super().__init__(M, p, q, client_list=client_list)
        self.k = k # number of cluster
        self.beta = np.random.normal(size=(self.p,))
        self.alpha_list = np.random.normal(size=(self.k, self.q))
        self.beta_list = np.tile(self.beta, (self.M, 1))
        self.theta_list = np.zeros((self.M, self.q))
        self.subgroup_labels = np.arange(M, dtype=int)
        self.distances = np.zeros((self.M, self.k))
        
    def receive(self, prompt, **kwparams):
        if prompt == 'update':
            self.participate_idx = kwparams['participate_idx']
            self.beta_list[self.participate_idx, :] = kwparams['beta_list']
            self.theta_list[self.participate_idx, :] = kwparams['theta_list']
        elif prompt == 'initialize':
            self.participate_idx = kwparams['participate_idx']
        else:
            raise NotImplementedError
    
    def broadcast(self, prompt: str):
        if prompt == 'params':
            for i, client in enumerate(self.client_list):
                client.download(prompt, beta=self.beta, theta=self.alpha_list[self.subgroup_labels[i], :])
        else:
            raise NotImplementedError
    
    def calc_distance(self, participate_idx):
        distances = np.linalg.norm(self.theta_list[participate_idx, None, :] - self.alpha_list[None, :, :], axis=-1)
        assert distances.shape == (len(participate_idx), self.k)
        return distances
    
    def update_clusters(self):
        self.distances[self.participate_idx, :] = self.calc_distance(self.participate_idx)
        self.subgroup_labels[self.participate_idx] = np.argmin(self.distances[self.participate_idx, :], axis=-1)
        
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

class FeSEMMethod(FMTLMethod):
    def __init__(self,
        p, q, k,
        n_clients: int = 0,
        max_round: int = 100,
        max_step: int = 100,
        lr: float = 1e-4,
        sigma_u2 = None,
        sigma_e2 = None,
        args = None,
        setting = None,
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
        
        self.server = FeSEMServer(self.M, self.p, self.q, self.k)
        self.clients = [FeSEMClient(lr=self.lr, max_step=self.T, early_stop=args.early_stop) for _ in range(self.M)]
        self.server.add_clients(self.clients)
        self.participate_idx = np.arange(self.M)
        
    def fit(self):
        estimate_full = []
        self.server.receive('initialize', participate_idx=self.participate_idx)
        self.server.update_clusters()
        
        for r in tqdm(range(self.R), desc='Communication round'):
            beta_list = []
            theta_list = []
            if self.updater is not None:
                self.updater(round=r)
            self.server.broadcast('params')
            for i, client in enumerate(self.clients):
                if i in self.participate_idx:
                    client.update()
                    beta_i, theta_i = client.upload()
                    beta_list.append(beta_i)
                    theta_list.append(theta_i)
            self.server.receive('update', beta_list=beta_list, theta_list=theta_list, participate_idx=self.participate_idx)
            self.server.update_clusters()
            self.server.aggregate()
            estimate_full.append(self.server.get_estimate())
            
        logging.info('FeSEM estimate finished.')
        return estimate_full