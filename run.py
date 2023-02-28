"""Load model, data and corresponding configs. Trigger training."""
import os
import pathlib
import sys
import pdb
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
from torch.profiler import profile, record_function, ProfilerActivity

from baselines.sklearn_tune import run_sklearn_hypertuning
from npt.column_encoding_dataset import ColumnEncodingDataset
from npt.configs import build_parser
from npt.distribution import distributed_train_wrapper
from npt.train import Trainer
from npt.utils.model_init_utils import init_model_opt_scaler_from_dataset
from npt.utils.viz_att_maps import viz_att_maps
from npt.utils.dataUtil import getCudaFreeMem
from npt.datasets.imputation import formWinDicts

def main(args):
    """Load model, data, configs, start training."""
    args, wandb_args = setup_args(args)
    run_cv(args=args, wandb_args=wandb_args)


def setup_args(args):
    print('Configuring arguments...')

    if args.exp_azure_sweep:
        print('Removing old logs.')
        os.system('rm -r wandb')

    if args.np_seed == -1:
        args.np_seed = np.random.randint(0, 1000)
    if args.torch_seed == -1:
        args.torch_seed = np.random.randint(0, 1000)
    if args.exp_name is None:
        args.exp_name = f'{wandb.util.generate_id()}'
    if (args.exp_group is None) and (args.exp_n_runs > 1):
        # Assuming you want to do CV, group runs together.
        args.exp_group = f'{wandb.util.generate_id()}'
        print(f"Doing k-FOLD CV. Assigning group name {args.exp_group}.")

    if args.exp_azure_sweep:
        print("Azure sweep run!")
        # Our configs may run oom. That's okay.
        os.environ['WANDB_AGENT_DISABLE_FLAPPING'] = 'true'

    if not isinstance(args.model_augmentation_bert_mask_prob, dict):
        print('Reading dict for model_augmentation_bert_mask_prob.')
        # Well, this is ugly. But I blame it on argparse.
        # There is just no good way to parse dicts as arguments.
        # Good thing, I don't care about code security.
        exec(
            f'args.model_augmentation_bert_mask_prob = '
            f'{args.model_augmentation_bert_mask_prob}')

    if not isinstance(args.model_label_bert_mask_prob, dict):
        print('Reading dict for model_augmentation_bert_mask_prob.')
        exec(
            f'args.model_label_bert_mask_prob = '
            f'{args.model_label_bert_mask_prob}')

    if not args.model_bert_augmentation:
        for value in args.model_augmentation_bert_mask_prob.values():
            assert value == 0
        for value in args.model_label_bert_mask_prob.values():
            assert value == 1

    if (args.model_class == 'sklearn-baselines' and
        args.sklearn_model == 'TabNet' and not args.data_force_reload):
        raise ValueError('For TabNet, user must specify data_force_reload '
                         'to encode data in a TabNet-compatible manner.')

    pathlib.Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)

    # Set seeds
    np.random.seed(args.np_seed)
    torch.manual_seed(args.np_seed)

    # Resolve CUDA device(s)
    if args.exp_use_cuda and torch.cuda.is_available():
        if args.exp_device is not None:
            print(f'Running model with CUDA on device {args.exp_device}.')
            exp_device = args.exp_device
        else:
            print(f'Running model with CUDA')
            exp_device = 'cuda:0'    
            torch.cuda.manual_seed(args.np_seed)
            # torch.backends.cudnn.benchmark = True
            # torch.backends.cudnn.enabled = True
    else:
        print('Running model on CPU.')
        exp_device = 'cpu'

    args.exp_device = exp_device

    wandb_args = dict(
        project=args.project,
        entity=args.entity,
        dir=args.wandb_dir,
        reinit=True,
        name=args.exp_descr,
        group=args.exp_group)

    return args, wandb_args


def run_cv(args, wandb_args):

    if args.mp_distributed:
        wandb_run = None
        c = args
    else:
        wandb_run = wandb.init(**wandb_args)
        args.cv_index = 0
        wandb.config.update(args, allow_val_change=True)
        c = wandb.config

    if c.model_class in ['NPT', 'MC_Dropout']:
        #RR add try except block for catching and profiling memory
        try:
            if c.debug_memory_profile:
                with profile(activities=[ProfilerActivity.CPU],profile_memory=True, record_shapes=True) as prof:
                        run_cv_splits(wandb_args, args, c, wandb_run)
            else:
                run_cv_splits(wandb_args, args, c, wandb_run)
        except BaseException as err:
            if c.debug_memory_profile: print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=100000))
            print(f"Unexpected {err=}, {type(err)=}")
            raise
    elif c.model_class == 'sklearn-baselines':
        if c.data_set=='imputation':
            run_sklearn_hypertuning(
                ColumnEncodingDataset(c, dataset_iter=c.num_win, subWin_size=c.subWin_size), wandb_args, args, c, wandb_run)
        else:
            run_sklearn_hypertuning(
                ColumnEncodingDataset(c), wandb_args, args, c, wandb_run)


def run_cv_splits(wandb_args, args, c, wandb_run):

    if c.data_set=='imputation':
        datasets=[]
        if (c.subWin_size==100) and (not c.multiple_win):
            for i in [c.num_win]:
                dataset = ColumnEncodingDataset(c,dataset_iter=i, subWin_size=c.subWin_size)
                dataset.load_next_cv_split()
                datasets.append(dataset)
        else:
            formWinDicts(c.num_win, c.subWin_size)
            for i in range(c.num_win):
                dataset = ColumnEncodingDataset(c,dataset_iter=i, subWin_size=c.subWin_size)
                dataset.load_next_cv_split()
                datasets.append(dataset)
        
    else:
        dataset = ColumnEncodingDataset(c)

    #######################################################################
    # Distributed Setting
    if c.mp_distributed:
        torch.manual_seed(c.torch_seed)

        # Fix from
        # https://github.com/facebookresearch/maskrcnn-benchmark/issues/103
        # torch.multiprocessing.set_sharing_strategy('file_system')

        dataset.load_next_cv_split()
        dataset.dataset_gen = None
        args = {'dataset': dataset, 'c': c, 'wandb_args': wandb_args}
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(
            distributed_train_wrapper, nprocs=c.mp_gpus, args=(args,),
            join=True)
        mp.set_start_method('fork')
        return

    starting_cv_index = 0
    total_n_cv_splits = min(dataset.n_cv_splits, c.exp_n_runs)

    # Since we're doing CV by default, model init is in a loop.
    for cv_index in range(starting_cv_index, total_n_cv_splits):
        print(f'CV Index: {cv_index}')

        print(f'Train-test Split {cv_index + 1}/{dataset.n_cv_splits}')

        if c.exp_n_runs < dataset.n_cv_splits:
            print(
                f'c.exp_n_runs = {c.exp_n_runs}. '
                f'Stopping at {c.exp_n_runs} splits.')

        # New wandb logger for each run
        if cv_index > 0:
            wandb_args['name'] = f'{wandb.util.generate_id()}'
            args.exp_name = wandb_args['name']
            args.cv_index = cv_index
            wandb_run = wandb.init(**wandb_args)
            wandb.config.update(args, allow_val_change=True)

        #######################################################################
        # Load New CV Split
        if not c.data_set=='imputation':
            dataset.load_next_cv_split()

        if c.viz_att_maps:
            print('Attempting to visualize attention maps.')
            return viz_att_maps(c, dataset, wandb_run, cv_index, total_n_cv_splits)

        if c.model_class == 'DKL':
            print(f'Running DKL on dataset {c.data_set}.')
            from baselines.models.dkl_run import main
            return main(c, dataset)
        
        if c.model_class == 'MC_Dropout':
            print(f'Running MLP on dataset {c.data_set}.')
            from baselines.models.mc_dropout_run import main
            return main(c, dataset)

        #######################################################################
        # Initialise Model
        # RR: before initializing the model , pass the dataset along with c for 
        # sub-window only
        if c.data_set=='imputation':
            if c.showGPUUtil: print(f"GPU free, reserve, allocated before init model:{getCudaFreeMem()}")
            model, optimizer, scaler, paramsCount = init_model_opt_scaler_from_dataset(
            dataset=datasets[0], c=c, device=c.exp_device) # RR init the model with the
            # first dataset, as we only need D, N for initializing the model
            if c.showGPUUtil: print(f"GPU free, reserve, allocated after init model:{getCudaFreeMem()}")
        else:
            model, optimizer, scaler, paramsCount = init_model_opt_scaler_from_dataset(
            dataset=dataset, c=c, device=c.exp_device)
        if wandb is not None: wandb.run.summary.update({"count of model parameters": paramsCount})
        # if not c.exp_azure_sweep:
        #     wandb.watch(model, log="all", log_freq=10)

        #######################################################################
        # Run training
        if c.data_set=='imputation':
            trainer = Trainer(
            model=model, optimizer=optimizer, scaler=scaler,
            c=c, wandb_run=wandb_run, cv_index=cv_index, dataset=datasets)

        else:
            trainer = Trainer(
                model=model, optimizer=optimizer, scaler=scaler,
                c=c, wandb_run=wandb_run, cv_index=cv_index, dataset=dataset)
        trainer.train_and_eval()

        wandb_run.finish()


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    main(args)
