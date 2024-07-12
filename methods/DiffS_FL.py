from methods.base import FMTLMethod, Client, Server
import numpy as np
from scipy.stats import chi2
from tqdm import tqdm
import logging

class BaseDiffSClient(Client):
    def __init__(self, server = None, lr=1e-3, max_step=1000):
        super().__init__(server, lr=lr, max_step=max_step)
        self.sigma = None
        self.GTWG_inv = None
    
    def load_data(self, data: dict, sigma_u2=None, sigma_e2=None) -> None:
        self.sigma = None
        self.GTWG_inv = None
        return super().load_data(data, sigma_u2, sigma_e2)

    def get_GTWG_inv(self):
        if self.GTWG_inv is None:
            self.GTWG_inv = np.linalg.pinv(self.get_GTWG())
        return self.GTWG_inv
    
    def get_sigma(self):
        if self.sigma is None:
            self.sigma = np.linalg.pinv(self.get_GTWG())[self.p:, self.p:]
        return self.sigma

    def upload(self):
        return self.beta, self.theta

    def download(self, key: str, **content) -> bool:
        if key == 'domainGLS':
            self.domainGLS_list = content['domainGLS_list']
            self.sigma_batch = content['sigma_batch']
            return True
        elif key == 'params':
            self.beta = content['beta']
            self.theta = content['theta']
            return True
        else:
            raise NotImplementedError

    def domain_difference(self):
        self.std_domain_distance = np.zeros(len(self.sigma_batch))
        for j, sigma_j in enumerate(self.sigma_batch):
            theta_j = self.domainGLS_list[j]
            delta_theta = self.theta_domain - theta_j
            sigma_inv_delta_theta, _, _, _ = np.linalg.lstsq(self.sigma + sigma_j, delta_theta, rcond=None)
            self.std_domain_distance[j] = delta_theta @ sigma_inv_delta_theta
        return self.std_domain_distance
    
    def domain_GLS(self):
        '''
        Sigma_i = ( G.T W G )^-1 [-q:, -q:]
                  --------------
                   self.GTWG_inv
        theta_domain = (G.T W G)^-1 G.T W y
                       ------------ -------
                      self.GTWG_inv self.GTWy
        '''
        
        self.sigma = self.get_GTWG_inv()[self.p:, self.p:]
        self.theta_domain = (self.get_GTWG_inv() @ self.get_GTWy())[self.p:]
        return self.theta_domain, self.sigma

class BaseDiffSServer(Server):
    def __init__(self, M: int, p: int, q: int, nu: float = 1e-2):
        super().__init__(M, p, q)
        self.nu = nu
        self.beta_list = None
        self.beta = np.random.normal(size=(self.p,))
        self.theta_list = None
        self.alpha_list = np.random.normal(size=(self.M, self.q))
        self.sigma_batch = None
        self.domain_distances = np.zeros((M, M))
        self.subgroup_labels = np.arange(M, dtype=int)
            
    def receive(self, prompt, **kwparams):
        if prompt == 'update':
            self.beta_list = kwparams['beta_list']
            self.theta_list = kwparams['theta_list']
        elif prompt == 'domain distances':
            self.domain_distances = kwparams['domain_distances']
        elif prompt == 'domainGLS':
            self.domainGLS_list = kwparams['domainGLS_list']
            self.sigma_batch = kwparams['sigma_batch']
        else:
            raise NotImplementedError

    def aggregate(self) -> tuple:
        assert self.beta_list is not None
        assert self.theta_list is not None
        self._update_client_sample_size()
        self.beta = np.average(self.beta_list, axis=0, weights=self.N)
        S = max(self.subgroup_labels) + 1
        self.alpha_list = np.zeros((S, self.q))
        for k in range(S):
            subgroup_k_indices = np.where(self.subgroup_labels == k)[0]
            self.alpha_list[k, :] = np.average(self.theta_list[subgroup_k_indices, :], axis=0, weights=self.N[subgroup_k_indices])
        return self.beta, self.alpha_list

    def calc_learnability_structure(self) -> np.ndarray:
        subgroup_labels = np.arange(self.M, dtype=int)
        delta_matrix = self.domain_distances.copy()
        delta_matrix[np.diag_indices_from(delta_matrix)] = np.inf
        threshold = chi2.ppf(1-self.nu, self.q)
        subgroups = [{i} for i in range(self.M)]
        for _ in range(self.M):
            if np.amin(delta_matrix) > threshold:
                break
            delta_over_threshold = delta_matrix <= threshold
            link_count = np.sum(delta_over_threshold, axis=1)
            link_count[link_count==0] = np.amax(link_count) + 1 # int infinity
            u_index = np.argmin(link_count)
            v_index = np.argmin(delta_matrix[u_index, :])

            # 合并u,v
            subgroups[u_index] |= subgroups[v_index]
            del subgroups[v_index]
            delta_matrix[u_index, :] = np.amax(delta_matrix[(u_index, v_index), :], axis=0)
            delta_matrix[:, u_index] = delta_matrix[u_index, :]
            delta_matrix = np.delete(delta_matrix, v_index, axis=0)
            delta_matrix = np.delete(delta_matrix, v_index, axis=1)
        for i, subgroup in enumerate(subgroups):
            subgroup_labels[list(subgroup)] = i
        self.subgroup_labels = subgroup_labels
        return subgroup_labels

    def broadcast(self, key: str):
        if key == 'domainGLS':
            for client in self.client_list:
                client.download(key, domainGLS_list=self.domainGLS_list, sigma_batch=self.sigma_batch)
        elif key == 'params':
            assert self.alpha_list is not None
            for i, label in enumerate(self.subgroup_labels):
                self.client_list[i].download(key, beta=self.beta, theta=self.alpha_list[label, :])
        else:
            raise NotImplementedError

class DiffSFLMethod(FMTLMethod):
    def __init__(self,
        p, q,
        n_clients: int = 0,
        max_round: int = 100,
        max_step: int = 100,
        lr: float = 1e-4,
        nu: float = 1e-3,
        sigma_u2 = None,
        sigma_e2 = None
    ):
        '''
        params:
            n_clients - Client数量
            max_round - 最大通信轮数
            max_step - 最大Local更新步数
            lr - 初始学习率
            nu - Learnability Recovery Hyper-Parameter
        '''
        self.data = None
        self.p = p
        self.q = q
        self.M = n_clients
        self.R = max_round
        self.T = max_step
        self.lr = lr
        self.nu = nu
        self.sigma_u2 = sigma_u2
        self.sigma_e2 = sigma_e2

        self.server = BaseDiffSServer(self.M, self.p, self.q, self.nu)
        self.clients = [BaseDiffSClient(lr=self.lr, max_step=self.T) for _ in range(self.M)]
        self.server.add_clients(self.clients)


    
    def fit(self):
        beta_list = np.zeros([self.M, self.p])
        theta_list = np.zeros([self.M, self.q])
        domainGLS_list = np.zeros([self.M, self.q])
        delta_matrix = np.zeros([self.M, self.M])
        sigma_batch = np.zeros((self.M, self.q, self.q))
        estimate_full = []

        for i, client in enumerate(self.clients):
            domainGLS_list[i, :], sigma_batch[i, ...] = client.domain_GLS()
        self.server.receive('domainGLS', domainGLS_list=domainGLS_list, sigma_batch=sigma_batch)
        self.server.broadcast('domainGLS')

        for i, client in enumerate(self.clients):
            delta_matrix[i, :] = client.domain_difference()
        self.server.receive('domain distances', domain_distances=delta_matrix)
        self.server.calc_learnability_structure()

        for r in tqdm(range(self.R), desc='Communication round'):
            self.server.broadcast('params')
            for i, client in enumerate(self.clients):
                client.update()
                beta_list[i], theta_list[i] = client.upload()
            self.server.receive('update', beta_list=beta_list, theta_list=theta_list)
            self.server.aggregate()
            estimate_full.append(self.server.get_estimate())
            
        logging.info('Final Estimate: {} subgroups.'.format(max(self.server.subgroup_labels) + 1))
        return estimate_full
            