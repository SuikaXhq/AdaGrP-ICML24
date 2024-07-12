import argparse
import logging
import os
import pickle
import csv
import time

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run Federated Mutual Transfer Learning.")
    
    parser.add_argument('--test', action='store_true',
                        help='Test the algorithms.')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=[
                            'synthetic',
                            'noaa',
                        ],
                        help='Dataset Name. Raise error when not implemented.')
    parser.add_argument('--method', type=str, default='FDiffS',
                        choices=[
                            'FDiffS',
                            'DiffS_FL',
                            'IFCA',
                            'FeSEM',
                            'AdaGrP',
                            'FedDrift'
                        ],
                        help='Method Name. Raise error when not implemented.')
    parser.add_argument('--type', type=str, default='normal',
                        choices=[
                            'normal',
                            'add_client',
                            'change_data',
                            'change_sigma',
                            'noaa'
                        ],
                        help='''
                        Experiment types.
                        normal: Normal Experiment with no concept drift.
                        add_client: New clients arrive in the middle of training.
                        change_data: data are shifted by 1 with the order of clients in the middle of training. ([0:M-1] -> [1:M-1, 0])
                        change_sigma: sigma are changed during training.
                        noaa: NOAA experiment. 25 years as one step.
                        ''')
    parser.add_argument('--early_stop', action='store_true',
                        help='Whether to use early stop.')
    parser.add_argument('--trigger_round', type=int, default=15,
                        help='Trigger round number for add_client and change_data types.')
    parser.add_argument('--trigger_round_list', nargs='+', type=int,
                        default=[0, 10, 20],
                        help='Trigger round list for change_sigma.')
    parser.add_argument('--trigger_interval', type=int, default=10,
                        help='Trigger interval (in round) for noaa exp.')
    parser.add_argument('--time_step_interval', type=int, default=10,
                        help='Time step interval (in round) for FedDrift exp.')
    parser.add_argument('--ignore_first_n_rounds', type=int, default=0,
                        help='do not trigger recovery in the first n rounds.')
    parser.add_argument('--add_rate', type=float, default=0.2,
                        help='Excluding rate of clients when conducting add_client experiments.')
    parser.add_argument('--seed', type=int, default=123457,
                        help='Random seed.')
    parser.add_argument('--M', nargs='+', type=int,
                        default=[50, 100, 150, 200, 250, 300],
                        help='Number of clients.')
    parser.add_argument('--n', nargs='+', type=int,
                        default=[200, 400, 600, 800, 1000],
                        help='Sample size of each client.')
    parser.add_argument('--S', nargs='+', type=int,
                        default=[3, 5, 7, 11],
                        help='Number of subgroups.')
    parser.add_argument('--p', nargs='+', type=int,
                        default=[10, 20, 30, 40, 50],
                        help='Dimension of global features.')
    parser.add_argument('--q', nargs='+', type=int,
                        default=[10, 20, 30, 40, 50],
                        help='Dimension of heterogeneous features.')
    parser.add_argument('--p_equals_q', action='store_true',
                        help='State that p equals q. The q argument will be ignored.')
    parser.add_argument('--replicate_num', type=int, default=5,
                        help='Number of replicates.')
    parser.add_argument('--max_round', type=int, default=30,
                        help='Maximum number of communication rounds.')
    parser.add_argument('--max_step', type=int, default=10000,
                        help='Maximum number of local iterations in each round.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--nu', type=float, default=1e-2,
                        help='Learnability Recovery Hyper-Parameter for DiffS.')
    parser.add_argument('--nu_initial', type=float, default=1e-3,
                        help='Initial Learnability Recovery Threshold for AdaGrP.')
    parser.add_argument('--k', type=int, default=0,
                        help='Number of clusters for IFCA. 0 means auto, -1 means use real clusters (synthetic only).')
    parser.add_argument('--delta', type=float, default=0.04,
                        help='Delta hyper-parameter for FedDrift.')
    
    # Synthetic Data
    sub_parsers = parser.add_subparsers(description='Sub-commands for reproducing.')
    generate_parser = sub_parsers.add_parser('synthetic', help='Conducting synthetic experiments with optional arguments.')
    generate_parser.set_defaults(dataset='synthetic')
    generate_parser.add_argument('--generate', action='store_true',
                                help='Generate synthetic data.')
    generate_parser.add_argument('--sigma_u2', type=float, default=0.5,
                                help='Variance of random effect.')
    generate_parser.add_argument('--sigma_e2', type=float, default=1,
                                help='Variance of noise.')
    generate_parser.add_argument('--sigma_u2_list', nargs='+', type=float,
                                default=[0.5, 1, 0.25],
                                help='List of sigma u2. Must have the same length with sigma_e2_list.')
    generate_parser.add_argument('--sigma_e2_list', nargs='+', type=float,
                                default=[1, 2, 0.5],
                                help='List of sigma e2. Must have the same length with sigma_u2_list.')

    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    
    if args.dataset == 'synthetic' and args.generate:
        args.save_dir = os.path.join('data', args.dataset, 'generate_log')
    else:
        if args.method == 'FDiffS' or args.method == 'DiffS_FL':
            method_subdir = 'R{}_T{}_lr{}_nu{}'.format(args.max_round, args.max_step, args.lr, args.nu)
        elif args.method == 'IFCA' or args.method == 'FeSEM':
            method_subdir = 'R{}_T{}_lr{}_k{}'.format(args.max_round, args.max_step, args.lr, args.k if args.k != -1 else 'real')
        elif args.method == 'FDiffS_rule' or args.method == 'AdaGrP':
            method_subdir = 'R{}_T{}_lr{}'.format(args.max_round, args.max_step, args.lr)
        elif args.method == 'FedDrift':
            method_subdir = 'R{}_T{}_lr{}_delta{}'.format(args.max_round, args.max_step, args.lr, args.delta)
        else:
            raise NotImplementedError
        args.save_dir = os.path.join('results', args.dataset, args.type, args.method, method_subdir)

    return args

def create_log_id(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return log_count

def logging_config(folder, name,
                   level=logging.DEBUG,
                   console_level=logging.DEBUG):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".log")
    print("All logs will be saved to %s" %logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    logconsole = logging.StreamHandler()
    logconsole.setLevel(console_level)
    logconsole.setFormatter(formatter)
    logging.root.addHandler(logconsole)

def save_estimate(args, estimate, M=None, n=None, S=None, p=None, q=None):
    if M is not None:
        assert args.dataset == 'synthetic'
        save_dir = os.path.join(args.save_dir, 'M{:d}_n{:d}_S{:d}_p{:d}_q{:d}'.format(M, n, S, p, q))
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pickle.dump(estimate, open(os.path.join(save_dir, 'estimate.pkl'), 'wb'))
    logging.info('Estimate saved to %s.' % os.path.join(save_dir, 'estimate.pkl'))

def load_estimate(args, M=None, n=None, S=None, p=None, q=None):
    '''
    estimate: list(dict) with R (communication round) elements:
        {
            'beta': (p) ndarray, estimated beta,
            'alpha': (S, q) ndarray, estimated alpha list
            'subgroup_labels': (M) ndarray(int), estimated subgroup labels
        }
    '''
    if M is not None:
        assert args.dataset == 'synthetic'
        load_dir = os.path.join(args.save_dir, 'M{:d}_n{:d}_S{:d}_p{:d}_q{:d}'.format(M, n, S, p, q))
    else:
        load_dir = args.save_dir
    estimate = pickle.load(open(os.path.join(load_dir, 'estimate.pkl'), 'rb'))
    logging.info('Estimate loaded from %s.' % os.path.join(load_dir, 'estimate.pkl'))
    return estimate

def save_results(args, results, save_name='results.csv'):
    if args.dataset == 'synthetic':
        with open(os.path.join(args.save_dir, save_name), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([
                'M',
                'n',
                'S',
                'p',
                'q',
                'S_estimate',
                'NMI',
                'NMI_std',
                'RMSE_beta',
                'RMSE_beta_std',
                'RMSE_theta',
                'RMSE_theta_std',
                'Error_mean',
                'Error_std',
                'Error_max'
            ])
            for setting_list, result in results.items():
                writer.writerow([
                    *setting_list,
                    result['S_estimate'],
                    result['NMI'],
                    result['NMI_std'],
                    result['RMSE_beta'],
                    result['RMSE_beta_std'],
                    result['RMSE_theta'],
                    result['RMSE_theta_std'],
                    result['Error_mean'],
                    result['Error_std'],
                    result['Error_max']
                ])
    else:
        with open(os.path.join(args.save_dir, 'results.csv'), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([
                'M',
                'n',
                'p',
                'q',
                'S_estimate',
                'Error_mean',
                'Error_max'
            ])
            writer.writerow([
                args.M[0],
                args.N[0],
                args.p[0],
                args.p[0] if args.p_equals_q else args.q[0],
                results['S_estimate'],
                results['Error_mean'],
                results['Error_max']
            ])
                
def save_evaluation(args, evaluation, M=None, n=None, S=None, p=None, q=None, replicate=None):
    if M is not None:
        assert args.dataset == 'synthetic'
        save_dir = os.path.join(args.save_dir, 'M{:d}_n{:d}_S{:d}_p{:d}_q{:d}'.format(M, n, S, p, q))
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    R = len(evaluation)
        
    file_path = os.path.join(save_dir, 'evaluation_full{}.csv'.format('_rep{}'.format(replicate) if replicate is not None else ''))
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        if args.dataset == 'synthetic':
            writer.writerow([
                'Round',
                'S_estimate',
                'NMI',
                'RMSE_beta',
                'RMSE_theta',
                'Error_mean',
                'Error_max'
            ])
            for r in range(R):
                evaluation_r = evaluation[r]
                writer.writerow([
                    r,
                    evaluation_r['S_estimate'],
                    evaluation_r['NMI'],
                    evaluation_r['RMSE_beta'],
                    evaluation_r['RMSE_theta'],
                    evaluation_r['Error_mean'],
                    evaluation_r['Error_max'],
                ])
        else:
            writer.writerow([
                'Round',
                'S_estimate',
                'Error_mean',
                'Error_max'
            ])
            for r in range(R):
                evaluation_r = evaluation[r]
                writer.writerow([
                    r,
                    evaluation_r['S_estimate'],
                    evaluation_r['Error_mean'],
                    evaluation_r['Error_max'],
                ])
            
    logging.info('Estimate saved to %s.' % file_path)
    
def save_any(args, output_lists, file_name):
    file_path = os.path.join(args.save_dir, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for line in output_lists:
            writer.writerow(line)
    logging.info('File saved to %s.' % file_path)
    
def timelog(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logging.info('Time elapsed: {:.2f}s'.format(end - start))
        return result
    return wrapper
        