from abc import ABC, abstractmethod
from methods.base import FMTLMethod
import numpy as np
import logging

class FMTLMethodUpdater(ABC):
    def __init__(self, method: FMTLMethod = None) -> None:
        self.r = 0
        self.method = method
    
    def __call__(self, round):
        self.r = round
        
    def relate_method(self, method):
        self.method = method
    
class AddClientUpdater(FMTLMethodUpdater):
    def __init__(self, method: FMTLMethod, args, trigger_round=0) -> None:
        super().__init__(method)
        # self.client_to_add = clients
        # self.data_to_add = data
        self.add_rate = args.add_rate
        self.trigger_round = trigger_round
        
    # def add_client(self):
    #     self.method.server.add_clients(self.client_to_add)
    #     self.method.load_data(self.data_to_add, client_idx=range(self.method.M, self.method.M+len(self.client_to_add)))
        
    def __call__(self, round):
        super().__call__(round)
        if self.r < self.trigger_round:
            self.method.participate_idx = np.arange(int(self.add_rate*self.method.M), self.method.M)
        else:
            self.method.participate_idx = np.arange(self.method.M)
            
class ChangeDataUpdater(FMTLMethodUpdater):
    def __init__(self, method: FMTLMethod, data: list, trigger_round=0) -> None:
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
        super().__init__(method)
        self.trigger_round = trigger_round
        self.data = data[1:] + data[:1] # shift by 1
        
    def change_data(self):
        self.method.load_data(self.data)
        
    def __call__(self, round):
        super().__call__(round)
        if self.r >= self.trigger_round:
            self.change_data()
            logging.info('Data changed at round {}.'.format(self.r))
            
class ChangeSigmaUpdater(FMTLMethodUpdater):
    def __init__(self, method: FMTLMethod, data: list, trigger_rounds: list) -> None:
        '''
        params:
            data - data_per_step[];
                data_per_step - dict: 
                {
                    'beta': (p) ndarray, global parameter
                    'alpha': (S, q) ndarray, heterogeneous parameters
                    'subgroup_labels': (M) ndarray, int, subgroup labels (label in [0, S-1] as index of alpha)
                    'datasets': list(dict), 共M个元素, 每个元素是一个Client数据:
                    {
                        'X': (n[i], p) ndarray, global features
                        'Z': (n[i], q) ndarray, heterogeneous features
                        'y': (n[i]) ndarray, responses
                        'u': (q) ndarray, random effects
                        'train_idx': (0.7*n[i]) ndarray, int, training indices
                        'valid_idx': (0.1*n[i]) ndarray, int, validation indices
                        'test_idx': (0.2*n[i]) ndarray, int, test indices
                    }
                    'sigma_u2': float, sigma of random effects
                    'sigma_e2': float, sigma of noise
                }
        '''
        super().__init__(method)
        self.trigger_rounds = trigger_rounds
        self.data = data
        self.step = -1
        
    def get_data(self):
        return self.data[self.step]
    
    def __call__(self, round):
        super().__call__(round)
        if self.step+1 < len(self.trigger_rounds) and self.r == self.trigger_rounds[self.step+1]:
            self.step += 1
            data_step = self.get_data()
            if self.method:
                self.method.load_data(data_step['datasets'], sigma_u2=data_step['sigma_u2'], sigma_e2=data_step['sigma_e2'])
                logging.info('Data changed at round {}.'.format(self.r))
        return self.get_data()
            
class NOAADataUpdater(FMTLMethodUpdater):
    def __init__(self, method: FMTLMethod, data: list, trigger_interval=10, steps=5) -> None:
        super().__init__(method)
        self.trigger_interval = trigger_interval
        self.steps = steps
        self.step_current = -1
        self.r_last_changed = 0
        self.data = [{} for _ in range(len(data))]
        for i, dataset in enumerate(data):
            self.data[i]['X'] = dataset['X'].reshape([self.steps, 1500 // self.steps, -1])
            self.data[i]['Z'] = dataset['Z'].reshape([self.steps, 1500 // self.steps, -1])
            self.data[i]['y'] = dataset['y'].reshape([self.steps, 1500 // self.steps])
            self.data[i]['train_idx'] = dataset['train_idx'].reshape([self.steps, -1])[0]
            self.data[i]['valid_idx'] = dataset['valid_idx'].reshape([self.steps, -1])[0]
            self.data[i]['test_idx'] = dataset['test_idx'].reshape([self.steps, -1])[0]
        
    def change_data(self):
        data_to_update = [{} for _ in range(len(self.data))]
        for i, data_i in enumerate(data_to_update):
            data_i['X'] = self.data[i]['X'][self.step_current]
            data_i['Z'] = self.data[i]['Z'][self.step_current]
            data_i['y'] = self.data[i]['y'][self.step_current]
            data_i['train_idx'] = self.data[i]['train_idx']
            data_i['valid_idx'] = self.data[i]['valid_idx']
            data_i['test_idx'] = self.data[i]['test_idx']
        logging.info('Data changed at round {}. Step {}.'.format(self.r, self.step_current))
        if self.method is not None: self.method.load_data(data_to_update)
        return data_to_update
    
    def __call__(self, round):
        super().__call__(round)
        if (self.r - self.r_last_changed) % self.trigger_interval == 0:
            self.step_current += 1
            self.r_last_changed = self.r
            self.data_latest = self.change_data()
        return self.data_latest


def load_updater(args, method, data):
    if args.type == 'normal':
        return None
    elif args.type == 'add_client':
        assert args.dataset == 'synthetic'
        return AddClientUpdater(method, args, trigger_round=args.trigger_round)
    elif args.type == 'change_data':
        assert args.dataset == 'synthetic'
        return ChangeDataUpdater(method, data, trigger_round=args.trigger_round)
    elif args.type == 'change_sigma':
        assert args.dataset == 'synthetic'
        return ChangeSigmaUpdater(method, data, trigger_rounds=args.trigger_round_list)
    elif args.type == 'noaa':
        assert args.dataset == 'noaa'
        return NOAADataUpdater(method, data, trigger_interval=args.trigger_interval)
    else:
        raise NotImplementedError