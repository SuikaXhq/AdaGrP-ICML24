from utils import helper, data_manager
from utils.data_manager import SyntheticDataSettingIterator
from methods import load_method
from utils.metrics import evaluate_per_round
import logging
from tqdm import tqdm
import numpy as np
from utils.experiment_environment import load_updater

def test(args):
    def save_final_results(results, round=-1):
        total_results = {
            'S_estimate': np.mean([results[i][round]['S_estimate'] for i in range(args.replicate_num)]),
            'NMI': np.mean([results[i][round]['NMI'] for i in range(args.replicate_num)]),
            'NMI_std': np.std([results[i][round]['NMI'] for i in range(args.replicate_num)]),
            'RMSE_beta': np.mean([results[i][round]['RMSE_beta'] for i in range(args.replicate_num)]),
            'RMSE_beta_std': np.std([results[i][round]['RMSE_beta'] for i in range(args.replicate_num)]),
            'RMSE_theta': np.mean([results[i][round]['RMSE_theta'] for i in range(args.replicate_num)]),
            'RMSE_theta_std': np.std([results[i][round]['RMSE_theta'] for i in range(args.replicate_num)]),
            'Error_mean': np.mean([results[i][round]['Error_mean'] for i in range(args.replicate_num)]),
            'Error_max': np.mean([results[i][round]['Error_max'] for i in range(args.replicate_num)]),
            'Error_std': np.std([results[i][round]['Error_mean'] for i in range(args.replicate_num)]),
        }
        
        return total_results
        # logging.info('Test results:')
        # logging.info('S_estimate: {}'.format(total_results['S_estimate']))
        # logging.info('NMI: {}'.format(total_results['NMI']))
        # logging.info('RMSE_beta: {}'.format(total_results['RMSE_beta']))
        # logging.info('RMSE_theta: {}'.format(total_results['RMSE_theta']))
        # logging.info('Error_mean: {}'.format(total_results['Error_mean']))
        # synthetic_results[setting_list] = total_results
        
        # if round == -1:
        #     save_name = 'results.csv'
        # else:
        #     save_name = f'results_r{round}.csv'
        # helper.save_results(args, synthetic_results, save_name=save_name)
        
    if args.dataset == 'synthetic':
        logging.info('Testing with synthetic datasets.')
        synthetic_settings = SyntheticDataSettingIterator(args)
        synthetic_results = [{} for _ in range(args.max_round)]
        for setting in synthetic_settings:
            logging.info('Synthetic setting:')
            for key, value in setting.items():
                logging.info('{}: {}'.format(key, value))
            setting_list = (setting['M'], setting['n'], setting['S'], setting['p'], setting['q'])
            estimate = helper.load_estimate(args, **setting)
            data_full = data_manager.load_data(args, **setting)
            logging.info('Test with {} replicates...'.format(args.replicate_num))
            results = []
            for i in tqdm(range(args.replicate_num), desc='Testing on replicates'):
                data = data_full[i]
                est = estimate[i]
                if args.type == 'change_sigma':
                    updater = load_updater(args, None, data)
                else:
                    updater = None
                result = evaluate_per_round(data, est, args, updater)
                results.append(result)
                helper.save_evaluation(args, result, *setting_list, i)
            for t in range(len(results[0])):
                synthetic_results[t][setting_list] = save_final_results(results, round=t)
        if args.type == 'change_sigma':
            for step_round in args.trigger_round_list:
                step_round_end = step_round + args.time_step_interval - 1
                save_name = f'results_r{step_round_end}.csv'
                helper.save_results(args, synthetic_results[step_round_end], save_name=save_name)
        else:
            helper.save_results(args, synthetic_results[-1])
            
    else:
        logging.info('Testing with dataset {}.'.format(args.dataset))
        data = data_manager.load_data(args)
        estimate = helper.load_estimate(args)
        updater = load_updater(args, None, data['datasets'])
        result = evaluate_per_round(data, estimate, args, updater)
        helper.save_evaluation(args, result)

def train(args, M=None, n=None, S=None, p=None, q=None):
    if args.dataset == 'synthetic':
        logging.info('Training with synthetic datasets.')
        if M is not None:
            assert n is not None
            assert S is not None
            assert p is not None
            assert q is not None
            logging.info('Synthetic setting:')
            logging.info('M: {}'.format(M))
            logging.info('n: {}'.format(n))
            logging.info('S: {}'.format(S))
            logging.info('p: {}'.format(p))
            logging.info('q: {}'.format(q))
            data_full = data_manager.load_data(args, M=M, n=n, S=S, p=p, q=q)
            setting = {
                'M': M,
                'n': n,
                'S': S,
                'p': p,
                'q': q,
            }
            logging.info('Train with {} replicates...'.format(args.replicate_num))
            estimate_full = []
            for i in tqdm(range(args.replicate_num), desc='Training on replicates'):
                if args.type == 'change_sigma':
                    data_step0 = data_full[i][0]
                    method = load_method(args, synthetic_setting=setting, sigma_u2=data_step0['sigma_u2'], sigma_e2=data_step0['sigma_e2'], setting=setting)
                    data_to_load = data_step0['datasets']
                    method.load_data(data_to_load)
                    data = data_full[i]
                else:
                    method = load_method(args, synthetic_setting=setting, sigma_u2=data_full[i]['sigma_u2'], sigma_e2=data_full[i]['sigma_e2'], setting=setting)
                    data = data_full[i]['datasets']
                    method.load_data(data)
                updater = load_updater(args, method, data)
                method.load_updater(updater)
                estimate = method.fit()
                estimate_full.append(estimate)
            logging.info('Learning finished. Saving results...')
            helper.save_estimate(args, estimate_full, **setting)
            return estimate_full
        else:
            synthetic_settings = SyntheticDataSettingIterator(args)
            estimate_full = []
            for setting in synthetic_settings:
                train(args, **setting)
            logging.info('All settings learned. Testing...')
            test(args)
                
    else:
        logging.info('Training with dataset {}.'.format(args.dataset))
        data = data_manager.load_data(args)
        method = load_method(args, sigma_u2=data['sigma_u2'], sigma_e2=data['sigma_e2'])
        updater = load_updater(args, method, data['datasets'])
        method.load_data(data['datasets'])
        method.load_updater(updater)
        estimate = method.fit()
        logging.info('Learning finished. Saving results...')
        helper.save_estimate(args, estimate)


def main():
    args = helper.parse_args()

    log_save_id = helper.create_log_id(args.save_dir)
    helper.logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id))
    logging.info('成功读取参数。')
    logging.info(args)

    np.random.seed(args.seed)
    
    if args.dataset == 'synthetic' and args.generate:
        data_manager.create_synthetic_data(args)
        return
        
    if args.test:
        test(args)
    else:
        train(args)

if __name__ == '__main__':
    main()