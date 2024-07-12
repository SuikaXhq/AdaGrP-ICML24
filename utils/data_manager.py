import os
import numpy as np
import logging
from tqdm import tqdm
import pickle
import scipy.io as sio

def load_data(args, M=None, n=None, S=None, p=None, q=None) -> dict: # type: ignore
    if args.dataset == 'synthetic':
        # checking arguments
        assert M is not None
        assert n is not None
        assert S is not None
        assert p is not None
        assert q is not None
        
        # loading data
        if args.type == 'change_sigma':
            data_path = os.path.join('data', args.dataset, 'sigma', 'data_M{:d}_n{:d}_S{:d}_p{:d}_q{:d}.pkl'.format(M, n, S, p, q))
        else:
            data_path = os.path.join('data', args.dataset, 'data_M{:d}_n{:d}_S{:d}_p{:d}_q{:d}.pkl'.format(M, n, S, p, q))
        data = pickle.load(open(data_path, 'rb'))
        logging.info('Data loaded from %s.' % data_path)
        return data
    else:
        if args.dataset == 'noaa':
            data_path = os.path.join('data', 'noaa', 'climate_adagrp.pkl')
            data = pickle.load(open(data_path, 'rb'))
            logging.info('Data loaded from %s.' % data_path)
            return data
        else:
            raise NotImplementedError

# def generate_data_by_existing_dataset(data, M_to_generate, n_to_generate):
#     beta = data['beta']
#     alpha = data['alpha']
#     p = beta.shape[0]
#     q = alpha.shape[1]
#     sigma_u2 = data['sigma_u2']
#     sigma_e2 = data['sigma_e2']
#     M = len(data['datasets'])
#     S = len(alpha)
#     datasets = []
#     data_generated = {
#         'beta': beta,
#         'alpha': alpha,
#         'subgroup_labels': np.asarray([[i] * M_to_generate for i in range(S)]),
#     }
#     U = np.random.multivariate_normal(np.zeros(q), np.eye(q) * sigma_u2, M_to_generate * S)
#     sigma_data = 0.3*np.ones(p+q) + 0.7*np.eye(p+q)
#     for i in range(M * M_to_generate):
#         dataset = {}
#         G = np.random.multivariate_normal(np.zeros(p+q), sigma_data, n_to_generate)
#         X = G[:, :p]
#         Z = G[:, p:]
#         u = U[i, :]
#         y = X @ beta + Z @ (alpha[subgroup_labels[i], :] + u) + np.random.normal(0, np.sqrt(sigma_e2), n)
#         dataset['X'] = X
#         dataset['Z'] = Z
#         dataset['y'] = y
#         dataset['u'] = u

def generate_data_with_different_sigma(M, n, S, p, q, sigma_u2: list, sigma_e2: list, replicate_num=5, save=True):
    '''
    Parameters:
        M - Client数量
        n - 每个Client的样本数量
        S - Subgroup数量
        p - X的维度
        q - Z的维度
        sigma_u2: list - u的方差
        sigma_e2: list - epsilon的方差
        replicate_num - 重复次数
    '''
    assert M > 0
    assert n > 0
    assert S > 0
    assert S <= M
    assert p > 0
    assert q > 0
    assert len(sigma_u2) == len(sigma_e2)
    assert replicate_num > 0
    
    T = len(sigma_u2)
    data_full = []
    '''
        data_full - list: data[];
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
    for _ in tqdm(range(replicate_num), desc='Generating data replicates'):
        data = []
        for t in range(T):
            data.append(generate_data(M, n, S, p, q, sigma_u2[t], sigma_e2[t], 1, False)[0])
        data_full.append(data)
    
    if save:
        # save datasets
        if not os.path.exists('data/synthetic/sigma'):
            os.makedirs('data/synthetic/sigma')
        data_path = 'data/synthetic/sigma/data_M{}_n{}_S{}_p{}_q{}.pkl'.format(M, n, S, p, q)
        pickle.dump(
            data_full,
            open(data_path, 'wb')
        )
        logging.info('Data has successfully saved to {}.'.format(data_path))
            

def generate_data(M, n, S, p, q, sigma_u2=1, sigma_e2=2, replicate_num=100, save=True):
    '''
    Parameters:
        M - Client数量
        n - 每个Client的样本数量
        S - Subgroup数量
        p - X的维度
        q - Z的维度
        sigma_u2 - u的方差
        sigma_e2 - epsilon的方差
        replicate_num - 重复次数
    '''
    assert M > 0
    assert n > 0
    assert S > 0
    assert S <= M
    assert p > 0
    assert q > 0
    assert replicate_num > 0
    
    logging.info('Start generating data with arguments: M={}, n={}, S={}, p={}, q={}, sigma_u2={}, sigma_e2={}, replicate_num={}'.format(M, n, S, p, q, sigma_u2, sigma_e2, replicate_num))
    data_full = []
    
    for _ in tqdm(range(replicate_num), desc='Generating data replicates'):
        data = {}
        '''
        data_full - list: data[];
        data - dict: 
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
        # generate effects
        beta = np.random.normal(0, 1, p) * 4 - 2
        data['beta'] = beta

        alpha_0 = (np.arange(q) / (q-1) - 1/2) * np.sqrt(q) * 4 * sigma_u2
        if S % 2 == 1:
            alpha_0 += 1
        alpha = np.zeros([S, q])
        alpha[0, :] = alpha_0
        for s in range(1, S):
            alpha[s, :] = np.roll(alpha[s-1, :], -1)
        # randomly rotate alpha
        random_matrix = np.random.uniform(-1, 1, [q, q])
        rotate_matrix, _ = np.linalg.qr(random_matrix)
        alpha = alpha @ rotate_matrix
        
        data['alpha'] = alpha
        
        # random effects
        U = np.random.multivariate_normal(np.zeros(q), np.eye(q) * sigma_u2, M)
        
        # generate subgroup labels
        subgroup_labels = np.zeros(M, dtype=int)
        base_members = np.random.choice(M, S, replace=False)
        subgroup_labels[base_members] = np.arange(S)
        
        rest_members = np.setdiff1d(np.arange(M, dtype=int), base_members, assume_unique=True)
        subgroup_labels[rest_members] = np.random.choice(S, M-S, replace=True)
        
        data['subgroup_labels'] = subgroup_labels
        
        # generate datasets
        datasets = []
        sigma_data = 0.3*np.ones(p+q) + 0.7*np.eye(p+q)
        for i in range(M):
            dataset = {}
            G = np.random.multivariate_normal(np.zeros(p+q), sigma_data, n)
            X = G[:, :p]
            Z = G[:, p:]
            u = U[i, :]
            y = X @ beta + Z @ (alpha[subgroup_labels[i], :] + u) + np.random.normal(0, np.sqrt(sigma_e2), n)
            dataset['X'] = X
            dataset['Z'] = Z
            dataset['y'] = y
            dataset['u'] = u
            
            # train-valid-test split
            train_idx = np.random.choice(n, int(0.7*n), replace=False)
            valid_idx = np.random.choice(
                np.setdiff1d(np.arange(n), train_idx, assume_unique=True),
                int(0.1*n), replace=False
            )
            test_idx = np.setdiff1d(
                np.arange(n), 
                np.concatenate([train_idx, valid_idx]), assume_unique=True
            )
            dataset['train_idx'] = train_idx
            dataset['valid_idx'] = valid_idx
            dataset['test_idx'] = test_idx
            
            datasets.append(dataset)
        
        data['datasets'] = datasets
        data['sigma_u2'] = sigma_u2
        data['sigma_e2'] = sigma_e2
        data_full.append(data)

    if save:
        # save datasets
        if not os.path.exists('data/synthetic'):
            os.makedirs('data/synthetic')
        data_path = 'data/synthetic/data_M{}_n{}_S{}_p{}_q{}.pkl'.format(M, n, S, p, q)
        pickle.dump(
            data_full,
            open(data_path, 'wb')
        )
        logging.info('Data has successfully saved to {}.'.format(data_path))
    
    return data_full

class SyntheticDataSettingIterator():
    def __init__(self, args) -> None:
        '''
        生成模拟数据参数迭代器
        '''
        assert args.dataset == 'synthetic'
        
        self.sigma_u2 = args.sigma_u2
        self.sigma_e2 = args.sigma_e2
        self.M = args.M
        self.n = args.n
        self.S = args.S
        self.p_equals_q = args.p_equals_q
        self.p = args.p
        if not args.p_equals_q:
            self.q = args.q
        self.replicate_num = args.replicate_num
        self.keys = ('M', 'n', 'S', 'p', 'q')
    
    def __iter__(self):
        M = self.M[0]
        n = self.n[0]
        S = self.S[0]
        if self.p_equals_q:
            p = q = self.p[0]
        else:
            p = self.p[0]
            q = self.q[0]
        
        for M_ in self.M:
            yield dict(zip(self.keys, (M_, n, S, p, q)))
        if len(self.n) > 1:
            for n_ in self.n[1:]:
                yield dict(zip(self.keys, (M, n_, S, p, q)))
        if len(self.S) > 1:
            for S_ in self.S[1:]:
                yield dict(zip(self.keys, (M, n, S_, p, q)))
        if self.p_equals_q:
            if len(self.p) > 1:
                for dim_ in self.p[1:]:
                    yield dict(zip(self.keys, (M, n, S, dim_, dim_)))
        else:
            if len(self.p) > 1:
                for p_ in self.p[1:]:
                    yield dict(zip(self.keys, (M, n, S, p_, q)))
            if len(self.q) > 1:
                for q_ in self.q[1:]:
                    yield dict(zip(self.keys, (M, n, S, p, q_)))
        

def create_synthetic_data(args):
    '''
    生成模拟数据
    '''
    assert args.generate
    assert args.dataset == 'synthetic'
    
    logging.info('Start generating synthetic data.')
    for kwparams in SyntheticDataSettingIterator(args):
        if args.type == 'change_sigma':
            generate_data_with_different_sigma(**kwparams, sigma_u2=args.sigma_u2_list, sigma_e2=args.sigma_e2_list, replicate_num=args.replicate_num)
        else:
            generate_data(**kwparams, sigma_u2=args.sigma_u2, sigma_e2=args.sigma_u2, replicate_num=args.replicate_num)
    logging.info('All data have generated. Exiting...')

def _convert_data_to_mat(data, path):
    sio.savemat(path, data)
    logging.info('Data has successfully saved to {}.'.format(path))
    
def convert_data_to_mat(args):
    '''
    将pkl格式的数据转换为mat格式
    '''
    if args.dataset == 'synthetic':
        for kwparams in SyntheticDataSettingIterator(args):
            data = load_data(args, **kwparams)
            path = 'data/synthetic/data_M{}_n{}_S{}_p{}_q{}.mat'.format(kwparams['M'], kwparams['n'], kwparams['S'], kwparams['p'], kwparams['q'])
            _convert_data_to_mat(data, path)
    else:
        pass
    # data_mat = {}
    # for key, value in data.items():
    #     if key == 'datasets':
    #         datasets = []
    #         for dataset in value:
    #             dataset_mat = {}
    #             for key, value in dataset.items():
    #                 if key in ['X', 'Z']:
    #                     dataset_mat[key] = value.astype(np.float64)
    #                 else:
    #                     dataset_mat[key] = value
    #             datasets.append(dataset_mat)
    #         data_mat['datasets'] = datasets
    #     else:
    #         data_mat[key] = value
    # data_path = 'data/synthetic/data_M{}_n{}_S{}_p{}_q{}.mat'.format(args.M[0], args.n[0], args.S[0], args.p[0], args.q[0])
    # scipy.io.savemat(data_path, data_mat)
    # logging.info('Data has successfully converted as {}.'.format(data_path))