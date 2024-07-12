from .base import FMTLMethod
from .FDiffS import FDiffSMethod
from .DiffS_FL import DiffSFLMethod
from .FeSEM import FeSEMMethod
from .IFCA import IFCAMethod
from .FDiffS_rule import FDiffSRuleMethod
from .AdaGrP import AdaGrPMethod
from .FedDrift import FedDriftMethod

def load_method(args, synthetic_setting=None, **kwargs) -> FMTLMethod:
    max_round = args.max_round
    max_step = args.max_step
    lr = args.lr
    nu = args.nu
    
    if args.dataset == 'synthetic':
        assert synthetic_setting is not None
        n_clients = synthetic_setting['M']
        p = synthetic_setting['p']
        q = synthetic_setting['q']
    else:
        n_clients = args.M[0]
        p = args.p[0]
        q = args.q[0]
    
    if args.method == 'FDiffS':
        return FDiffSMethod(
            p=p,
            q=q,
            n_clients=n_clients,
            max_round=max_round,
            max_step=max_step,
            lr=lr,
            nu=nu,
            ignore_first_n_rounds=args.ignore_first_n_rounds,
            args=args,
            **kwargs
        )
    elif args.method == 'FDiffS_rule':
        return FDiffSRuleMethod(
            p=p,
            q=q,
            n_clients=n_clients,
            max_round=max_round,
            max_step=max_step,
            lr=lr,
            args=args,
            **kwargs
        )
    elif args.method == 'AdaGrP':
        return AdaGrPMethod(
            p=p,
            q=q,
            nu=args.nu_initial,
            n_clients=n_clients,
            max_round=max_round,
            max_step=max_step,
            lr=lr,
            args=args,
            **kwargs
        )
    elif args.method == 'DiffS_FL':
        return DiffSFLMethod(
            p=p,
            q=q,
            n_clients=n_clients,
            max_round=max_round,
            max_step=max_step,
            lr=lr,
            nu=nu,
            args=args,
            **kwargs
        )
    elif args.method == 'IFCA':
        if args.k == -1:
            assert synthetic_setting is not None
            k = synthetic_setting['S']
        else:
            k = args.k
        return IFCAMethod(
            p=p,
            q=q,
            n_clients=n_clients,
            max_round=max_round,
            max_step=max_step,
            lr=lr,
            k=k,
            args=args,
            **kwargs
        )
    elif args.method == 'FedDrift':
        return FedDriftMethod(
            p=p,
            q=q,
            delta=args.delta,
            n_clients=n_clients,
            max_round=max_round,
            time_step_interval=args.time_step_interval,
            max_step=max_step,
            lr=lr,
            args=args,
            **kwargs
        )
    elif args.method == 'FeSEM':
        if args.k == -1:
            assert synthetic_setting is not None
            k = synthetic_setting['S']
        else:
            k = args.k
        return FeSEMMethod(
            p=p,
            q=q,
            n_clients=n_clients,
            max_round=max_round,
            max_step=max_step,
            lr=lr,
            k=k,
            args=args,
            **kwargs
        )
    else:
        raise NotImplementedError