from abc import ABC, abstractmethod
import numpy as np
from utils.metrics import error

class Client(ABC):
    def __init__(self, server = None, lr = 1e-3, max_step = 1000, early_stop = False):
        self.server = server
        self.lr = lr
        self.update_count = 0
        self.max_step = max_step
        self.G = None
        self.W = None
        self.W_valid = None
        self.GTWG = None
        self.GTWy = None
        self.early_stop = early_stop
        if self.early_stop:
            self.loss_valid_list = np.zeros(self.max_step)
            self.min_loss_valid_idx = 0
            self.converge_judge = 50
            self.beta_best = None
            self.theta_best = None
    
    def set_server(self, server):
        self.server = server
    
    @abstractmethod
    def upload(self): pass
    
    def get_G(self):
        if self.G is None:
            self.G = np.concatenate((self.X, self.Z), axis=1)
        return self.G
    
    def get_W(self):
        if self.W is None:
            self.W = np.linalg.pinv(self.sigma_e2 * np.eye(self.n) + self.sigma_u2 * self.Z @ self.Z.T)
        return self.W
    
    def get_W_valid(self):
        if self.W_valid is None:
            n_valid = self.X_valid.shape[0]
            self.W_valid = np.linalg.pinv(self.sigma_e2 * np.eye(n_valid) + self.sigma_u2 * self.Z_valid @ self.Z_valid.T)
        return self.W_valid
    
    def get_GTWG(self):
        if self.GTWG is None:
            self.GTWG = self.get_G().T @ self.get_W() @ self.get_G()
        return self.GTWG
    
    def get_GTWy(self):
        if self.GTWy is None:
            self.GTWy = self.get_G().T @ self.get_W() @ self.y
        return self.GTWy
    
    def update(self):
        for i in range(self.max_step):
            self.step()
            if self.early_stop:
                self.loss_valid_list[i] = self.loss_valid()
                if i == 0 or (self.loss_valid_list[self.min_loss_valid_idx] - self.loss_valid_list[i] ) / self.loss_valid_list[self.min_loss_valid_idx] > 0:
                    self.min_loss_valid_idx = i
                    self.beta_best = self.beta
                    self.theta_best = self.theta
                else:
                    if i - self.min_loss_valid_idx >= self.converge_judge:
                        # self.min_loss_valid_idx = 0
                        break
        if self.early_stop:
            self.beta = self.beta_best
            self.theta = self.theta_best
            return self.min_loss_valid_idx+1
        else:
            return i+1
            
    def step(self):
        params, grad = self.grad()
        self.update_count += 1
        assert self.update_count != 0
        params_update = params - self.lr * grad
        self.beta, self.theta = params_update[:self.p], params_update[self.p:]
    
    def grad(self):
        # grad = Gi.T Wi Gi (beta, theta) - Gi.T Wi yi
        #        ----------                 ----------
        #        self.GTWG                  self.GTWy
        params = np.concatenate((self.beta, self.theta))
        return params, self.get_GTWG() @ params - self.get_GTWy()
    
    def loss(self):
        assert self.X is not None
        assert self.Z is not None
        assert self.y is not None
        # loss = (yi - Xi beta - Zi theta).T Wi (yi - Xi beta - Zi theta)
        residual = self.y - self.X @ self.beta - self.Z @ self.theta
        return residual @ self.get_W() @ residual
    
    def loss_valid(self):
        assert self.X_valid is not None
        assert self.Z_valid is not None
        assert self.y_valid is not None
        # # loss = (yi - Xi beta - Zi theta).T Wi (yi - Xi beta - Zi theta)
        # residual = self.y_valid - self.X_valid @ self.beta - self.Z_valid @ self.theta
        # return residual @ self.get_W_valid() @ residual
        return error(
            self.y_valid,
            self.X_valid @ self.beta + self.Z_valid @ self.theta,
            reduction='avg'
        )

    def load_data(self, data: dict, sigma_u2=None, sigma_e2=None) -> None:
        '''
        params:
            data - dict:
                {
                    'X': (n*p) ndarray,
                    'Z': (n*q) ndarray, 
                    'y': (n) ndarray,
                    'train_idx': (0.7*n) ndarray, 训练样本索引
                    'valid_idx': (0.1*n) ndarray, 验证样本索引
                }
        '''
        # self.data = data
        self.G = None
        self.W = None
        self.GTWG = None
        self.GTWy = None
        self.train_idx = data['train_idx']
        self.valid_idx = data['valid_idx']
        self.X = data['X'][self.train_idx]
        self.Z = data['Z'][self.train_idx]
        self.y = data['y'][self.train_idx]
        self.X_valid = data['X'][self.valid_idx]
        self.Z_valid = data['Z'][self.valid_idx]
        self.y_valid = data['y'][self.valid_idx]
        self.p = self.X.shape[1]
        self.q = self.Z.shape[1]
        self.n = self.y.shape[0]
        if sigma_u2 is not None: self.sigma_u2 = sigma_u2
        if sigma_e2 is not None: self.sigma_e2 = sigma_e2

class Server(ABC):
    def __init__(self, M: int, p: int, q: int, client_list = None):
        self.client_list = client_list if client_list is not None else []
        self.beta = None
        self.alpha_list = None
        self.subgroup_labels = None
        self.M = M
        self.N = None
        self.p = p
        self.q = q

    @abstractmethod
    def aggregate(self): pass

    @abstractmethod
    def broadcast(self): pass
    
    def get_estimate(self):
        return {
            'beta': self.beta.copy(),
            'alpha': self.alpha_list.copy(),
            'subgroup_labels': self.subgroup_labels.copy(),
        }
    
    def add_client(self, client: Client) -> None:
        self.client_list.append(client)
    
    def add_clients(self, clients: list) -> None:
        self.client_list.extend(clients)
    
    def _update_client_sample_size(self) -> None:
        self.N = np.asarray([client.n for client in self.client_list])

class FMTLMethod(ABC):
    def __init__(self,
        sigma_u2 = None,
        sigma_e2 = None
    ):
        self.sigma_u2 = sigma_u2
        self.sigma_e2 = sigma_e2
        self.server = None
        self.updater = None
        self.clients = []
        
        
    def load_data(self, data, client_idx=None, sigma_u2=None, sigma_e2=None):
        '''
        params:
            data - list(dict), 共M个元素, 每个元素是一个Client数据: 
                {
                    'X': (n*p) ndarray,
                    'Z': (n*q) ndarray, 
                    'y': (n) ndarray,
                    'train_idx': (0.7*n) ndarray, 训练样本索引
                    'valid_idx': (0.1*n) ndarray, 验证样本索引
                }
        '''
        self.data = data
        self.M = len(data)
        self.data_train = [{
            'X': data[i]['X'][data[i]['train_idx']],
            'Z': data[i]['Z'][data[i]['train_idx']],
            'y': data[i]['y'][data[i]['train_idx']],
        } for i in range(self.M)]
        self.data_valid = [{
            'X': data[i]['X'][data[i]['valid_idx']],
            'Z': data[i]['Z'][data[i]['valid_idx']],
            'y': data[i]['y'][data[i]['valid_idx']],
        } for i in range(self.M)]
        
        self.p = data[0]['X'].shape[1]
        self.q = data[0]['Z'].shape[1]
        if sigma_u2: self.sigma_u2 = sigma_u2
        if sigma_e2: self.sigma_e2 = sigma_e2
        for i, client in enumerate(self.clients):
            if client_idx is None or i in client_idx:
                client.load_data(data[i], self.sigma_u2, self.sigma_e2)
                
        self.n = np.asarray(
            [client.n for client in self.clients]
        )
    
    @abstractmethod
    def fit(self) -> dict: pass
    # estimate - list(dict), with R (rounds) elements, each element is a dict:
    # {
    #     'beta': (p) ndarray, estimated beta.
    #     'alpha': (S^, q) ndarray, estimated alpha.
    #     'subgroup_labels': (M) ndarray, estimated learnability structure.
    # }

    def load_updater(self, updater):
        self.updater = updater