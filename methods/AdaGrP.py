import time
from methods.base import FMTLMethod, Client, Server
import numpy as np
from tqdm import tqdm
from scipy.stats import chi2
import logging
from sklearn.metrics import normalized_mutual_info_score as nmi
from bisect import bisect


class AdaGrPClient(Client):
    # def get_local_loss_trace(self):
    #     output_lists = [
    #         ['step', 'loss'],
    #     ]
    #     for i, loss in enumerate(self.loss_trace):
    #         output_lists.append([i, loss])
    #     return output_lists

    def upload(self):
        return self.beta, self.theta

    def download(self, key: str, **content) -> bool:
        if key == 'params':
            self.beta = content['beta']
            self.theta = content['theta']
            return True
        else:
            raise NotImplementedError

class AdaGrPServer(Server):
    def __init__(self, M, p, q, nu=None):
        super().__init__(M, p, q)
        self.M = M
        self.nu = nu
        self.beta = np.random.normal(size=(self.p,))
        self.beta_list = np.tile(self.beta, (self.M, 1))
        self.alpha_list = np.random.normal(size=(self.M, self.q))
        self.theta_list = self.alpha_list
        self.GTWG_batch = None
        self.sigma_batch = np.tile(np.eye(self.q), (self.M, 1, 1))
        self.domain_distances = np.zeros((M, M))
        self.subgroup_labels = np.arange(M, dtype=int)
        self.last_subgroup_labels = self.subgroup_labels
        
    def receive(self, prompt, **kwparams):
        if prompt == 'update':
            self.participate_idx = kwparams['participate_idx']
            self.beta_list[self.participate_idx, :] = kwparams['beta_list']
            self.theta_list[self.participate_idx, :] = kwparams['theta_list']
        elif prompt == 'GTWG':
            self.GTWG_batch = kwparams['GTWG_batch']
        else:
            raise NotImplementedError
    
    def aggregate(self) -> tuple:
        assert self.beta_list is not None
        assert self.theta_list is not None
        self._update_client_sample_size()
        self.beta = np.average(self.beta_list[self.participate_idx, :], axis=0, weights=self.N[self.participate_idx])
        S = max(self.subgroup_labels) + 1
        self.alpha_list = np.zeros((S, self.q))
        for k in range(S):
            subgroup_k_indices = np.where(self.subgroup_labels[self.participate_idx] == k)[0]
            if len(subgroup_k_indices) != 0:
                self.alpha_list[k, :] = np.average(self.theta_list[self.participate_idx, :][subgroup_k_indices, :], axis=0, 
                                                   weights=self.N[self.participate_idx][subgroup_k_indices])
        return self.beta, self.alpha_list

    def calc_sigma(self) -> np.ndarray:
        assert self.GTWG_batch is not None
        self.sigma_batch[self.participate_idx, ...] = np.linalg.pinv(self.GTWG_batch)[..., self.p:, self.p:]
        return self.sigma_batch
    
    def _calc_domain_distance(self, i, j) -> float:
        assert self.theta_list is not None
        assert self.sigma_batch is not None
        delta_theta = self.theta_list[i] - self.theta_list[j]
        sigma_inv_delta_theta, _, _, _ = np.linalg.lstsq(self.sigma_batch[i, ...] + self.sigma_batch[j, ...], delta_theta, rcond=None)
        return delta_theta @ sigma_inv_delta_theta
        
    def update_domain_distances(self) -> np.ndarray:
        for i in range(self.M):
            for j in range(i+1, self.M):
                self.domain_distances[i, j] = self._calc_domain_distance(i, j)
                self.domain_distances[j, i] = self.domain_distances[i, j]
        return self.domain_distances
        
    def update_learnability_structure(self) -> np.ndarray:
        self.last_subgroup_labels = self.subgroup_labels
        subgroup_labels = np.arange(self.M, dtype=int)
        delta_matrix = self.domain_distances.copy()
        # threshold = chi2.ppf(1 - self.nu, self.q)
        threshold = self.threshold_decision(chi2.ppf(1 - self.nu, self.q) if self.nu else None)
        delta_matrix[np.diag_indices_from(delta_matrix)] = np.inf
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
    
    def threshold_decision(self, initial_threshold=None):
        if initial_threshold is None:
            initial_threshold = chi2.ppf(1 - 1e-4, self.q)
        distances = []
        for i in range(self.M):
            for j in range(i+1, self.M):
                distances.append(self.domain_distances[i, j])
        distances = np.sort(distances)

        def _density_estimate(x):
            kN = int(np.sqrt(self.M*(self.M-1)/2))+1 # 近邻点数量
            left_ptr = bisect(distances, x) - 1
            right_ptr = left_ptr + 1
            k_nearest_ptr = -1
            for _ in range(kN):
                if right_ptr >= len(distances):
                    k_nearest_ptr = left_ptr
                    left_ptr -= 1
                elif left_ptr < 0 or x - distances[left_ptr] >= distances[right_ptr] - x:
                    k_nearest_ptr = right_ptr
                    right_ptr += 1
                else:
                    k_nearest_ptr = left_ptr
                    left_ptr -= 1
            return kN / len(distances) / (distances[k_nearest_ptr] - x) / 2
        
        threshold_corrected = initial_threshold
        current_ptr = bisect(distances, threshold_corrected)
        if current_ptr == len(distances) - 1:
            threshold_corrected = distances[current_ptr] + 1e-2
        while (threshold_corrected >= chi2.ppf(0.9, self.q)):
            chi_prop = current_ptr / len(distances)
            if _density_estimate(threshold_corrected) < 2 * chi_prop * chi2.pdf(threshold_corrected, self.q):
                break
            else:
                current_ptr -= 1
                threshold_corrected = (distances[current_ptr] + distances[current_ptr-1]) / 2
        
        return threshold_corrected
    
    def is_learnability_structure_updated(self) -> bool:
        return nmi(self.subgroup_labels, self.last_subgroup_labels) < 1

    def broadcast(self, key: str):
        if key == 'params':
            assert self.alpha_list is not None
            for i, label in enumerate(self.subgroup_labels):
                self.client_list[i].download(key, beta=self.beta, theta=self.alpha_list[label, :])
        elif key == 'lr_reset':
            for client in self.client_list:
                client.reset_lr()
        else:
            raise NotImplementedError
    

class AdaGrPMethod(FMTLMethod):
    def __init__(self,
        p, q,
        n_clients: int = 0,
        max_round: int = 100,
        max_step: int = 100,
        lr: float = 1e-3,
        nu = None,
        sigma_u2 = None,
        sigma_e2 = None,
        ignore_first_n_rounds=0,
        args = None,
        setting = None
    ):
        '''
        params:
            n_clients - Client数量
            max_round - 最大通信轮数
            max_step - 最大Local更新步数
            lr - 初始学习率
            nu - Learnability Recovery Hyper-Parameter
        '''
        self.args = args
        self.setting = setting
        self.save_folder = 'M{:d}_n{:d}_S{:d}_p{:d}_q{:d}'.format(setting['M'], setting['n'], setting['S'], setting['p'], setting['q']) if setting is not None else None
        self.data = None
        self.p = p
        self.q = q
        self.nu = nu
        self.M = n_clients
        self.R = max_round
        self.T = max_step
        self.lr = lr
        self.sigma_u2 = sigma_u2
        self.sigma_e2 = sigma_e2
        self.ignore_n_rounds = ignore_first_n_rounds

        self.server = AdaGrPServer(self.M, self.p, self.q, self.nu)
        self.clients = [AdaGrPClient(lr=self.lr, max_step=self.T, early_stop=args.early_stop) for _ in range(self.M)]
        self.server.add_clients(self.clients)
        self.participate_idx = np.arange(self.M)

    
    
    def fit(self):
        estimate_full = []
        
        for r in tqdm(range(self.R), desc='Communication round'):
            if self.updater is not None:
                self.updater(round=r)
            beta_list = []
            theta_list = []
            GTWG_batch = []
            local_steps = np.zeros((self.M,), dtype=int)
            self.server.broadcast('params')
            # if self.server.is_learnability_structure_updated(): self.server.broadcast('lr_reset')
            for i, client in enumerate(self.clients):
                if i in self.participate_idx:
                    local_steps[i] = client.update()
                    # save_any(self.args, client.get_local_loss_trace(), f'{self.save_folder}/loss_trace_C{i}_R{r}.csv')
                    beta_i, theta_i = client.upload()
                    beta_list.append(beta_i)
                    theta_list.append(theta_i)
                    GTWG_batch.append(client.get_GTWG())
            GTWG_batch = np.asarray(GTWG_batch)
            self.server.receive('update', beta_list=beta_list, theta_list=theta_list, participate_idx=self.participate_idx)
            self.server.receive('GTWG', GTWG_batch=GTWG_batch)
            
            start_time = time.perf_counter()
            self.server.calc_sigma()
            if r >= self.ignore_n_rounds:
                self.server.update_domain_distances()
                self.server.update_learnability_structure()
            self.server.aggregate()
            end_time = time.perf_counter()
            
            estimate = self.server.get_estimate()
            estimate['average_steps'] = local_steps.mean()
            estimate['local_steps'] = local_steps
            estimate['aggregate_time'] = end_time - start_time
            estimate_full.append(estimate)
            
        logging.info('Final Estimate: {} subgroups.'.format(max(self.server.subgroup_labels) + 1))
        return estimate_full
            