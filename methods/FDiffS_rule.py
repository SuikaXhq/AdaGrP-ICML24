import time
from methods.base import FMTLMethod, Client, Server
import numpy as np
from tqdm import tqdm
from scipy.stats import chi2
import logging
# from multiprocessing import Pool
from utils.helper import save_any
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class FDiffSRuleClient(Client):
    # def __init__(self, server=None, lr=1e-3, max_step=1000, early_stop=False):
    #     super().__init__(server, lr=lr, max_step=max_step, early_stop=early_stop)
    
    # def update(self):
    #     self.loss_trace = []
    #     for i in range(self.max_step):
    #         self.step()
    #         # self.loss_trace.append(self.loss())
    #         # self.loss_valid_trace.append(self.loss_valid())
    #         # if self.is_converged():
    #         #     break
    #     return i
    
    # def step(self):
    #     params, grad = self.grad()
    #     self.update_count += 1
    #     assert self.update_count != 0
    #     # params_update = params - self.lr / self.update_count * grad
    #     params_update = params - self.lr * grad
    #     self.beta, self.theta = params_update[:self.p], params_update[self.p:]
    
    # def reset_lr(self):
    #     # reset learning rate
    #     self.update_count = 0
    
    # def is_converged(self):
    #     # min_loss = np.inf
    #     # stall_count = 0
    #     # for loss in self.loss_trace:
    #     #     if loss < min_loss:
    #     #         min_loss = loss
    #     #         stall_count = 0
    #     #     else:
    #     #         stall_count += 1
    #     #     if stall_count >= self.converge_judge:
    #     #         return True
    #     # return False
    #     loss_valid_current = self.loss_valid_trace[-1]
    #     if loss_valid_current < self.min_loss_valid:
    #         self.min_loss_valid = loss_valid_current
    #         self.min_loss_valid_idx = len(self.loss_valid_trace) - 1
    #         return False
    #     else:
    #         return len(self.loss_valid_trace) - self.min_loss_valid_idx >= self.converge_judge
    
    def get_local_loss_trace(self):
        output_lists = [
            ['step', 'loss'],
        ]
        for i, loss in enumerate(self.loss_trace):
            output_lists.append([i, loss])
        return output_lists

    def upload(self):
        return self.beta, self.theta

    def download(self, key: str, **content) -> bool:
        if key == 'params':
            self.beta = content['beta']
            self.theta = content['theta']
            return True
        else:
            raise NotImplementedError

class FDiffSRuleServer(Server):
    def __init__(self, M, p, q, nu=1e-2):
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
        # threshold = chi2.ppf(1-self.nu, self.q)
        threshold = self.threshold_decision()
        # logging.info(f'Delta matrix: {np.sum(delta_matrix<threshold)} zero(s), {np.sum(delta_matrix>threshold)} one(s).')
        # logging.info(f'Delta matrix: {delta_matrix}')
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
    
    def threshold_decision(self):
        distances = []
        for i in range(self.M):
            for j in range(i+1, self.M):
                distances.append(self.domain_distances[i, j])
        distances = np.sort(distances)
        # max gap method
        # distance_prev = distances[0]
        # max_gap = 0
        # max_gap_idx = 0
        # for i, distance in enumerate(distances):
        #     gap = distance - distance_prev
        #     if gap > max_gap:
        #         max_gap = gap
        #         max_gap_idx = i
        #     distance_prev = distance
        # return distances[max_gap_idx] - max_gap / 2
    
        kN = int(np.sqrt(self.M*(self.M-1)/2))+1 # 近邻点数量
        # kN = int((M*(M-1)/2)*0.5) + 1
        density = []
        xs = []
        for k in range(len(distances)-kN+1):
            density.append(
                kN / len(distances) / (distances[k+kN-1] - distances[k])
            )
            xs.append(
                (distances[k+kN-1] + distances[k]) / 2
            )
        xs = np.asarray(xs)
        xs_4th_order = PolynomialFeatures(degree=5).fit_transform(xs.reshape([-1, 1]))
        regression_model = LinearRegression().fit(xs_4th_order, density)
        poly = np.poly1d([
            5 * regression_model.coef_[5],
            4 * regression_model.coef_[4],
            3 * regression_model.coef_[3],
            2 * regression_model.coef_[2],
            regression_model.coef_[1],
        ])
        threshold_new = np.min(poly.roots[poly.roots>0])
        return threshold_new
    
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
    

class FDiffSRuleMethod(FMTLMethod):
    def __init__(self,
        p, q,
        n_clients: int = 0,
        max_round: int = 100,
        max_step: int = 100,
        lr: float = 1e-3,
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
        self.M = n_clients
        self.R = max_round
        self.T = max_step
        self.lr = lr
        self.sigma_u2 = sigma_u2
        self.sigma_e2 = sigma_e2
        self.ignore_n_rounds = ignore_first_n_rounds

        self.server = FDiffSRuleServer(self.M, self.p, self.q)
        self.clients = [FDiffSRuleClient(lr=self.lr, max_step=self.T, early_stop=args.early_stop) for _ in range(self.M)]
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
            estimate['aggregate_time'] = end_time - start_time
            estimate_full.append(estimate)
            
        logging.info('Final Estimate: {} subgroups.'.format(max(self.server.subgroup_labels) + 1))
        return estimate_full
            