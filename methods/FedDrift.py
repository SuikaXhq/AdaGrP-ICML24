from methods.base import FMTLMethod, Client, Server
import numpy as np
from tqdm import tqdm
import logging
import time

class FedDriftClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subgroup_label = None
        self.min_loss = 0
    
    def update(self):
        self.theta = self.alpha_list[self.subgroup_label, :]
        return super().update()
    
    def calculate_model_losses(self):
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
        return F_i_list
        
    def update_cluster_identity(self, identity=None):
        if identity is not None:
            self.subgroup_label = identity
            return
        else:
            loss_list = self.calculate_model_losses()
            self.subgroup_label = np.argmin(loss_list)
            return self.subgroup_label, loss_list.min()
    
    def upload(self):
        return self.beta, self.theta, self.subgroup_label
        
    def download(self, key: str, **content):
        if key == 'params':
            self.beta = content['beta']
            self.alpha_list = content['alpha_list']
        else:
            raise NotImplementedError

class FedDriftServer(Server):
    def __init__(self, M, p, q, delta, client_list=None):
        super().__init__(M, p, q, client_list=client_list)
        self.k = 1
        self.delta = delta
        self.beta = np.random.normal(size=(self.p,))
        self.alpha_list = np.random.normal(size=(self.k, self.q))
        self.beta_list = np.tile(self.beta, (self.M, 1))
        self.theta_list = np.zeros((self.M, self.q))
        self.subgroup_labels = np.zeros(M, dtype=int)
        self.loss_clients_last = np.zeros(self.M)
        
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
    
    def get_estimate(self):
        current_subgroups = np.unique(self.subgroup_labels)
        return {
            'beta': self.beta.copy(),
            'alpha': self.alpha_list[current_subgroups].copy(),
            'subgroup_labels': [np.where(current_subgroups == label)[0][0] for label in self.subgroup_labels]
        }

    def update_concepts(self):
        self.broadcast('params')
        for i, client in enumerate(self.client_list):
            self.subgroup_labels[i], client_min_loss = client.update_cluster_identity()
            if client_min_loss > self.loss_clients_last[i] + self.delta:
                # add new cluster
                self.k += 1
                self.alpha_list = np.concatenate((self.alpha_list, self.theta_list[[i]]), axis=0)
                client.update_cluster_identity(self.k-1)
                self.subgroup_labels[i] = self.k - 1
            self.loss_clients_last[i] = client_min_loss
            
        # collect losses
        losses = np.zeros((self.k, self.k))
        self.broadcast('params')
        client_losses = np.zeros((self.M, self.k))
        for i, client in enumerate(self.client_list):
            client_losses[i] = client.calculate_model_losses()
        subgroup_member_numbers = [np.sum(self.subgroup_labels == s) for s in range(self.k)]
        for i in range(self.k):
            for j in range(self.k):
                losses[i, j] = np.sum(client_losses[:, i][self.subgroup_labels == j]) / subgroup_member_numbers[j] if subgroup_member_numbers[j] > 0 else 0
        
        # update cluster distances
        distances = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(i, self.k):
                if i == j:
                    distances[i, j] = np.inf
                else:
                    distances[i, j] = np.amax([
                        losses[i, j] - losses[i, i],
                        losses[j, i] - losses[j, j],
                        0
                    ])
                    distances[j, i] = distances[i, j]
        
        # merge
        while distances.min() < self.delta:
            i, j = np.where(distances == distances.min())
            i = i[0]
            j = j[0]
            new_model_weight = [
                np.sum(self.subgroup_labels == i),
                np.sum(self.subgroup_labels == j)
            ]
            if new_model_weight[0] == 0:
                new_model_weight[0] += 1
            if new_model_weight[1] == 0:
                new_model_weight[1] += 1
            new_model = np.average(self.alpha_list[[i, j]], axis=0, weights=new_model_weight)
            subgroup_label_map = np.arange(self.k)
            subgroup_label_map[j] = i
            subgroup_label_map[subgroup_label_map > j] -= 1
            self.subgroup_labels = subgroup_label_map[self.subgroup_labels]
            for c, client in enumerate(self.client_list):
                client.update_cluster_identity(self.subgroup_labels[c])
            self.alpha_list = np.delete(self.alpha_list, j, axis=0)
            self.alpha_list[i] = new_model
            distances[:, i] = np.amax(distances[:, [i, j]], axis=1)
            distances[i, :] = distances[:, i]
            distances = np.delete(distances, j, axis=1)
            distances = np.delete(distances, j, axis=0)
            self.k -= 1

class FedDriftMethod(FMTLMethod):
    def __init__(self,
        p, q,
        n_clients: int = 0,
        max_round: int = 100,
        time_step_interval: int = 10,
        delta: float = 0.04,
        max_step: int = 100,
        lr: float = 1e-4,
        sigma_u2 = None,
        sigma_e2 = None,
        args = None,
        setting = None
    ) -> None:
        self.args = args
        self.data = None
        self.time_step_interval = time_step_interval
        self.p = p
        self.q = q
        self.M = n_clients
        self.R = max_round
        self.T = max_step
        self.lr = lr
        self.delta = delta
        self.sigma_u2 = sigma_u2
        self.sigma_e2 = sigma_e2
        
        self.server = FedDriftServer(self.M, self.p, self.q, self.delta)
        self.clients = [FedDriftClient(lr=self.lr, max_step=self.T, early_stop=args.early_stop) for _ in range(self.M)]
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
            if r % self.time_step_interval == 0:
                if self.updater is not None:
                    self.updater(round=r)
                start_time = time.perf_counter()
                self.server.update_concepts()
                end_time = time.perf_counter()
                aggregate_time += end_time - start_time
            self.server.broadcast('params')
            for i, client in enumerate(self.clients):
                if i in self.participate_idx:
                    local_steps[i] = client.update()
                    beta_i, theta_i, subgroup_labels_i = client.upload()
                    beta_list.append(beta_i)
                    theta_list.append(theta_i)
                    subgroup_labels.append(subgroup_labels_i)
            # aggregate_time /= self.M
            self.server.receive('update', beta_list=beta_list, theta_list=theta_list, subgroup_labels=subgroup_labels, participate_idx=self.participate_idx)
            
            start_time = time.perf_counter()
            self.server.aggregate()
            end_time = time.perf_counter()
            aggregate_time += end_time - start_time
            
            estimate = self.server.get_estimate()
            estimate['average_steps'] = local_steps.mean()
            estimate['aggregate_time'] = aggregate_time
            estimate_full.append(estimate)
            
        logging.info('FedDrift estimate finished.')
        return estimate_full