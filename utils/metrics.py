import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import mean_squared_error as mse

def rmse(x: np.ndarray, y: np.ndarray, reduction='sum') -> np.ndarray:
    assert x.shape == y.shape
    r = np.sqrt(np.mean((x - y)**2, axis=-1))
    if reduction == 'sum':
        return r.sum()
    elif reduction == 'avg':
        return r.mean()
    elif reduction == 'all':
        return r
    else:
        raise NotImplementedError

# def error(x: np.ndarray, y: np.ndarray, reduction='sum') -> np.ndarray:
#     assert x.shape == y.shape
#     if reduction == 'sum':
#         r = mse(x, y, multioutput='raw_values')
#         return r.sum()
#     elif reduction == 'avg':
#         return mse(x, y)
#     elif reduction == 'all':
#         return mse(x, y, multioutput='raw_values')
#     else:
#         raise NotImplementedError
    
def error(x: np.ndarray, y: np.ndarray, reduction='sum') -> np.ndarray:
    assert x.shape == y.shape
    r = (x-y)**2
    if reduction == 'sum':
        return r.sum()
    elif reduction == 'avg':
        return r.mean()
    elif reduction == 'all':
        return r
    else:
        raise NotImplementedError
    
def evaluate(data, estimate, exp_type, add_rate=0) -> dict:
    '''
    Parameters:
    data - dict,
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
    estimate - dict,
        {
            'beta': (p) ndarray, estimated beta.
            'alpha': (S^, q) ndarray, estimated alpha.
            'subgroup_labels': (M) ndarray, estimated learnability structure.
        }
    exp_type - 'normal' or 'add_client'
        normal: normal experiment
        add_client: ignore first `add_rate`% clients
        change_data: shift dataset by 1 (data_new[i] = data[i+1], data_new[-1] = data[0])
    '''
    data = data.copy()
    if 'subgroup_labels' not in estimate: estimate['subgroup_labels'] = estimate['subgroups'] # compatibility
    M = len(data['datasets'])
    if exp_type == 'change_data':
        data['datasets'] = data['datasets'][1:] + data['datasets'][:1]
        data['subgroup_labels'] = np.concatenate((data['subgroup_labels'][1:], data['subgroup_labels'][:1]))
    elif exp_type == 'add_client':
        assert add_rate >= 0
        data['datasets'] = data['datasets'][int(add_rate*M):]
        data['subgroup_labels'] = data['subgroup_labels'][int(add_rate*M):]
        estimate['subgroup_labels'] = estimate['subgroup_labels'][int(add_rate*M):]
    rmse_beta = None
    rmse_theta = None
    nmi_score = None
    if 'beta' in data:
        rmse_beta = rmse(data['beta'], estimate['beta'])
    if 'subgroup_labels' in data:
        nmi_score = nmi(data['subgroup_labels'], estimate['subgroup_labels'])
    if 'alpha' in data:
        rmse_theta = rmse(
            data['alpha'][data['subgroup_labels'], ...],
            estimate['alpha'][estimate['subgroup_labels'], ...],
            reduction='avg'
        )
    
    # evaluate overall sample error
    datasets = data['datasets']
    error_full = np.zeros(len(datasets))
    for i, dataset in enumerate(datasets):
        idx = dataset['test_idx']
        X = dataset['X'][idx]
        Z = dataset['Z'][idx]
        y = dataset['y'][idx]
        theta_est = estimate['alpha'][estimate['subgroup_labels'][i], :]
        error_full[i] = error(
            y,
            X @ estimate['beta'] + Z @ theta_est,
            reduction='avg'
        )
    error_mean = np.mean(error_full)
    error_max = np.amax(error_full)
    
    return {
        'NMI': nmi_score,
        'RMSE_beta': rmse_beta,
        'RMSE_theta': rmse_theta,
        'S_estimate': estimate['alpha'].shape[0],
        'Error_mean': error_mean,
        'Error_max': error_max,
    }
    
def evaluate_per_round(data, estimate, args, updater=None):
    '''
    Parameters:
    data - the same as above
    estimate - list(dict), with R (rounds) elements, each element is a dict:
        {
            'beta': (p) ndarray, estimated beta.
            'alpha': (S^, q) ndarray, estimated alpha.
            'subgroup_labels': (M) ndarray, estimated learnability structure.
        }
    '''
    R = len(estimate)
    evaluation_full = []
    exp_type = 'normal'
    for r in range(R):
        if args.type == 'add_client' and r < args.trigger_round:
            exp_type = 'add_client'
        elif args.type == 'change_data' and r >= args.trigger_round:
            exp_type = 'change_data'
        elif args.type =='change_sigma':
            data = updater(r)
        elif args.type == 'noaa':
            data['datasets'] = updater(r)
        evaluation_full.append(evaluate(data, estimate[r], exp_type, args.add_rate))
            
    return evaluation_full