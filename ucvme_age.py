

import sys
import math
import os

# Add workspace directory to Python path for module imports
workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

import time
import shutil
import datetime
import pandas as pd
import itertools

import click
import yaml
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm
import subprocess

import models
import datasets
import utils
import utils_pixelwise

from torch.distributions.normal import Normal

# Set to True to save per-epoch prediction CSVs (train_pred_*, val_predmcd0_*, z_val_epch*_prd.csv).
# Disabled by default to avoid huge files (especially for pixel-wise tasks).
SAVE_PREDICTION_CSVS = False


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@click.command("ucvme")
@click.option("--config", type=click.Path(exists=True, file_okay=True), default=None,
              help="Path to config YAML file (optional, overrides defaults)")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--pretrained/--random", default=None)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=None)
@click.option("--test_only/--run_all", default=None)

@click.option("--num_epochs", type=int, default=None)   
@click.option("--lr", type=float, default=None)
@click.option("--weight_decay", type=float, default=None)
@click.option("--lr_step_period", type=int, default=None)
@click.option("--num_workers", type=int, default=None)
@click.option("--batch_size", type=int, default=None)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=None)

@click.option("--reduced_set/--full_set", default=None)
@click.option("--rd_label", type=int, default=None) 
@click.option("--rd_unlabel", type=int, default=None)
@click.option("--ssl_mult", type=int, default=None)
@click.option("--w_ulb", type=float, default=None)

@click.option("--pad_param", type=int, default=None)


@click.option("--y_mean", type=float, default=None)
@click.option("--y_std", type=float, default=None)

@click.option("--samp_fq", type=int, default=None)
@click.option("--samp_ssl", type=int, default=None)

@click.option("--drp_p", type=float, default=None)
@click.option("--model", type=click.Choice(['resnet50', 'efficientnetb0', 'unet'], case_sensitive=False), 
              default=None, help='Model architecture: resnet50, efficientnetb0, or unet')
def run(
    config=None,
    data_dir=None,
    output=None,
    pretrained=None,
    weights=None,
    run_test=None,
    test_only=None,

    num_epochs=None,
    lr=None,
    weight_decay=None,
    lr_step_period=None,
    num_workers=None,
    batch_size=None,
    device=None,
    seed=None,

    reduced_set=None,
    rd_label=None,
    rd_unlabel=None,

    ssl_mult=None,
    w_ulb=None,

    pad_param=None,
    y_mean=None,
    y_std=None,
    samp_fq=None,
    samp_ssl=None,

    drp_p=None,
    model=None
):
    # Load config file if provided
    if config:
        cfg = load_config(config)
        print(f"Loaded config from: {config}")
        
        # Extract values from config, but allow CLI args to override
        model_name = model or cfg['model']['name']
        pretrained = pretrained if pretrained is not None else bool(cfg['model']['pretrained'])
        drp_p = drp_p if drp_p is not None else float(cfg['model']['drp_p'])
        
        data_dir = data_dir or cfg['data']['data_dir']
        reduced_set = reduced_set if reduced_set is not None else bool(cfg['data']['reduced_set'])
        pad_param = pad_param if pad_param is not None else int(cfg['data']['pad_param'])
        
        # Dataset selection
        dataset_name = cfg['data'].get('dataset_name', 'utkface')
        target_column = cfg['data'].get('target_column', 'age')
        image_dir = cfg['data'].get('image_dir', None)  # For UTKFace: "UTKFace", for So2Sat: "So2Sat_POP_Part2"
        file_list_name = cfg['data'].get('file_list_name', 'FileList.csv')
        
        # Handle percentage-based or absolute number-based label/unlabel split
        label_percentage = cfg['data'].get('label_percentage', None)
        unlabel_percentage = cfg['data'].get('unlabel_percentage', None)
        
        # Get absolute numbers if not provided via CLI
        if rd_label is None:
            rd_label = cfg['data'].get('rd_label', None)
        if rd_unlabel is None:
            rd_unlabel = cfg['data'].get('rd_unlabel', None)
        
        # Calculate from percentages if provided (takes precedence over absolute numbers)
        if label_percentage is not None or unlabel_percentage is not None:
            # Need to load data to get total train samples
            data = pd.read_csv(os.path.join(data_dir, "FileList.csv"))
            data["SPLIT"].map(lambda x: x.upper())
            total_train_samples = len(data[data['SPLIT'] == 'TRAIN'])
            
            if label_percentage is not None:
                rd_label = int(total_train_samples * label_percentage)
                print(f"Calculated labeled samples ({label_percentage*100:.1f}%): {rd_label}")
            
            if unlabel_percentage is not None:
                rd_unlabel = int(total_train_samples * unlabel_percentage)
                print(f"Calculated unlabeled samples ({unlabel_percentage*100:.1f}%): {rd_unlabel}")
            
            # If only one percentage is provided, calculate the other
            if label_percentage is not None and unlabel_percentage is None:
                rd_unlabel = total_train_samples - rd_label
                print(f"Calculated unlabeled samples (remaining): {rd_unlabel}")
            elif unlabel_percentage is not None and label_percentage is None:
                rd_label = total_train_samples - rd_unlabel
                print(f"Calculated labeled samples (remaining): {rd_label}")
            
            print(f"Total training samples: {total_train_samples}")
            print(f"Using {rd_label} labeled + {rd_unlabel} unlabeled = {rd_label + rd_unlabel} total")
        
        num_epochs = num_epochs if num_epochs is not None else int(cfg['training']['num_epochs'])
        lr = lr if lr is not None else float(cfg['training']['lr'])
        weight_decay = weight_decay if weight_decay is not None else float(cfg['training']['weight_decay'])
        lr_step_period = lr_step_period if lr_step_period is not None else int(cfg['training']['lr_step_period'])
        batch_size = batch_size if batch_size is not None else int(cfg['training']['batch_size'])
        num_workers = num_workers if num_workers is not None else int(cfg['training']['num_workers'])
        seed = seed if seed is not None else int(cfg['training']['seed'])
        
        ssl_mult = ssl_mult if ssl_mult is not None else int(cfg['ssl']['ssl_mult'])
        w_ulb = w_ulb if w_ulb is not None else float(cfg['ssl']['w_ulb'])
        samp_fq = samp_fq if samp_fq is not None else int(cfg['ssl']['samp_fq'])
        samp_ssl = samp_ssl if samp_ssl is not None else int(cfg['ssl']['samp_ssl'])
        
        y_mean = y_mean if y_mean is not None else float(cfg['target']['y_mean'])
        y_std = y_std if y_std is not None else float(cfg['target']['y_std'])
        
        # Output can come from CLI or config (CLI takes precedence)
        if output is None:
            output = cfg['misc'].get('output', None)
        device = device if device is not None else cfg['misc']['device']
        run_test = run_test if run_test is not None else cfg['misc']['run_test']
        test_only = test_only if test_only is not None else cfg['misc']['test_only']
    else:
        # Use defaults if config not provided
        model_name = model or 'resnet50'
        pretrained = pretrained if pretrained is not None else True
        drp_p = drp_p if drp_p is not None else 0.05
        data_dir = data_dir or "DATA_DIR"
        reduced_set = reduced_set if reduced_set is not None else True
        rd_label = rd_label if rd_label is not None else 1000
        rd_unlabel = rd_unlabel if rd_unlabel is not None else 9518
        pad_param = pad_param if pad_param is not None else 5
        dataset_name = 'utkface'  # Default
        target_column = 'age'  # Default
        image_dir = None
        file_list_name = 'FileList.csv'
        num_epochs = num_epochs if num_epochs is not None else 30
        lr = lr if lr is not None else 0.0001
        weight_decay = weight_decay if weight_decay is not None else 1e-3
        lr_step_period = lr_step_period if lr_step_period is not None else 10
        num_workers = num_workers if num_workers is not None else 4
        batch_size = batch_size if batch_size is not None else 32
        seed = seed if seed is not None else 0
        ssl_mult = ssl_mult if ssl_mult is not None else -1
        w_ulb = w_ulb if w_ulb is not None else 10
        samp_fq = samp_fq if samp_fq is not None else 5
        samp_ssl = samp_ssl if samp_ssl is not None else 5
        y_mean = y_mean if y_mean is not None else 35
        y_std = y_std if y_std is not None else 11
        run_test = run_test if run_test is not None else True
        test_only = test_only if test_only is not None else False

    command_args = sys.argv[:]

    print("Run with options:")
    for carg_itr in command_args:
        print(carg_itr)
    
    print(f"Using model: {model_name}")

    if reduced_set:
        # Ensure we have both values (fallback to defaults if not set)
        if rd_label is None:
            rd_label = 1000
        if rd_unlabel is None:
            rd_unlabel = 9518
        
        if not os.path.isfile(os.path.join(data_dir, "FileList_ssl_{}_{}.csv".format(rd_label, rd_unlabel))):
            print("Generating new file list for ssl dataset")
            np.random.seed(0)
            
            data = pd.read_csv(os.path.join(data_dir, "FileList.csv"))
            data["SPLIT"].map(lambda x: x.upper())

            file_name_list = np.array(data[data['SPLIT']== 'TRAIN']['FileName'])
            np.random.shuffle(file_name_list)

            label_list = file_name_list[:rd_label]
            # Take unlabeled samples after labeled ones, up to rd_unlabel
            end_idx = min(rd_label + rd_unlabel, len(file_name_list))
            unlabel_list = file_name_list[rd_label:end_idx]

            data['SSL_SPLIT'] = "UNLABELED"
            data.loc[data['FileName'].isin(label_list), 'SSL_SPLIT'] = "LABELED"

            data.to_csv(os.path.join(data_dir, "FileList_ssl_{}_{}.csv".format(rd_label, rd_unlabel)),index = False)

    ssl_mult_choice = ssl_mult

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    def worker_init_fn(worker_id):                            
        # print("worker id is", torch.utils.data.get_worker_info().id)
        # https://discuss.pytorch.org/t/in-what-order-do-dataloader-workers-do-their-job/88288/2
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Set default output directory
    if output is None:
        assert 1==2, "need output option"

    os.makedirs(output, exist_ok = True)
    bkup_tmstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "gpu":
        device = torch.device("cuda")
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        assert 1==2, "wrong parameter for device"


    # Model selection
    if model_name.lower() == 'resnet50':
        model_fn = models.resnet50_unc
    elif model_name.lower() == 'efficientnetb0':
        model_fn = models.efficientnetb0_unc
    elif model_name.lower() == 'unet':
        model_fn = models.unet_unc
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'resnet50', 'efficientnetb0', or 'unet'")
    
    model = model_fn(pretrained=pretrained, drp_p=drp_p)
    model = torch.nn.DataParallel(model)

    model_1 = model_fn(pretrained=pretrained, drp_p=drp_p)
    model_1 = torch.nn.DataParallel(model_1)

    model.to(device)
    model_1.to(device)


    if weights is not None:
        checkpoint = torch.load(weights)
        if checkpoint.get('state_dict'):
            model.load_state_dict(checkpoint['state_dict'])
        elif checkpoint.get('state_dict_0'):
            model.load_state_dict(checkpoint['state_dict_0'])
        else:
            assert 1==2, "state dict not found"


    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    optim_1 = torch.optim.Adam(model_1.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler_1 = torch.optim.lr_scheduler.StepLR(optim_1, lr_step_period)

    # Get dataset class from registry
    dataset_class = datasets.get_dataset(dataset_name)
    print(f"Using dataset: {dataset_name} ({dataset_class.__name__})")
    
    # Prepare dataset kwargs
    dataset_kwargs_base = {
        "target_type": [target_column],
        "mean": 0.,  # Will be calculated
        "std": 1.    # Will be calculated
    }
    
    # Add dataset-specific parameters
    if image_dir is not None:
        dataset_kwargs_base["image_dir"] = image_dir
    if file_list_name is not None:
        dataset_kwargs_base["file_list_name"] = file_list_name
    
    # Add normalize_mean and normalize_std for custom datasets (e.g., so2sat_pop_custom)
    if dataset_name == 'so2sat_pop_custom':
        dataset_kwargs_base["normalize_mean"] = y_mean
        dataset_kwargs_base["normalize_std"] = y_std
    
    # Calculate mean and std
    mean, std = utils.get_mean_and_std(dataset_class(root=data_dir, split="train", **dataset_kwargs_base))
    print("mean std", mean, std)
    
    # Update kwargs with calculated mean/std
    kwargs = {
        "target_type": [target_column],
        "mean": mean,
        "std": std
    }
    if image_dir is not None:
        kwargs["image_dir"] = image_dir
    if file_list_name is not None:
        kwargs["file_list_name"] = file_list_name
    
    # Add normalize_mean and normalize_std for custom datasets (e.g., so2sat_pop_custom)
    if dataset_name == 'so2sat_pop_custom':
        kwargs["normalize_mean"] = y_mean
        kwargs["normalize_std"] = y_std
    
    # Check if preload is enabled in config
    preload_enabled = cfg.get('data', {}).get('preload', False) if config else False
    if preload_enabled:
        print("Preloading enabled: images will be loaded into RAM at startup")
        kwargs["preload"] = True

    # Set up datasets and dataloaders
    dataset = {}
    dataset_trainsub = {}
    if reduced_set:
        # SSL mode: split into labeled and unlabeled
        dataset_trainsub['lb'] = dataset_class(root=data_dir, split="train", **kwargs, pad=pad_param, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), ssl_type = 1, ssl_mult = ssl_mult_choice)
        dataset_trainsub['unlb_0'] = dataset_class(root=data_dir, split="train", **kwargs, pad=pad_param, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), ssl_type = 2)
        dataset['train'] = dataset_trainsub
    else:
        # Non-SSL mode: use full training dataset
        # For compatibility with training loop, create both datasets (they'll be the same)
        # The SSL loss will still be computed but acts as consistency regularization between the two models
        dataset_trainsub['lb'] = dataset_class(root=data_dir, split="train", **kwargs, pad=pad_param)
        dataset_trainsub['unlb_0'] = dataset_class(root=data_dir, split="train", **kwargs, pad=pad_param)
        dataset['train'] = dataset_trainsub
    # Validation should not use SSL postfix - it uses the original FileList.csv
    kwargs_val = kwargs.copy()
    kwargs_val.pop('ssl_postfix', None)  # Remove ssl_postfix for validation
    dataset["val"] = dataset_class(root=data_dir, split="val", **kwargs_val)

    with open(os.path.join(output, "log.csv"), "a") as f:

        f.write("Run timestamp: {}\n".format(bkup_tmstmp))

        epoch_resume = 0
        bestLoss = float("inf")
        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'], strict = False)
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])

            model_1.load_state_dict(checkpoint['state_dict_1'], strict = False)
            optim_1.load_state_dict(checkpoint['opt_dict_1'])
            scheduler_1.load_state_dict(checkpoint['scheduler_dict_1'])

            np_rndstate_chkpt = checkpoint['np_rndstate']
            trch_rndstate_chkpt = checkpoint['trch_rndstate']

            np.random.set_state(np_rndstate_chkpt)
            torch.set_rng_state(trch_rndstate_chkpt)

            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")
            f.write("# train row: epoch,phase,loss,r2_0,r2_1,time_sec,n_samples,mem_allocated,mem_reserved,batch_size,loss_reg_0,cps\n")
            f.write("# val row:   epoch,phase,loss,r2,mae,rmse,time_sec,n_samples,mem_allocated,mem_reserved,batch_size,0,0\n")

        if test_only:
            num_epochs = 0

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:

                start_time = time.time()

                if device.type == "cuda":
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.reset_peak_memory_stats(i)

                
                ds = dataset[phase]
                if phase == "train":
                    dataloader_lb = torch.utils.data.DataLoader(
                        ds['lb'], batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"), worker_init_fn=worker_init_fn)
                    dataloader_unlb_0 = torch.utils.data.DataLoader(
                        ds['unlb_0'], batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"), worker_init_fn=worker_init_fn)
                    


                    loss_tr, loss_reg_0, loss_reg_1, cps, cps_l, cps_s, yhat_0, yhat_1, y, mean_0_ls, mean_1_ls, var_0_ls, var_1_ls = run_epoch(model, 
                                                                                                                                model_1, 
                                                                                                                                dataloader_lb, 
                                                                                                                                dataloader_unlb_0, 
                                                                                                                                phase == "train", 
                                                                                                                                optim, 
                                                                                                                                optim_1, 
                                                                                                                                device, 
                                                                                                                                w_ulb = w_ulb, 
                                                                                                                                y_mean = y_mean, 
                                                                                                                                y_std = y_std, 
                                                                                                                                samp_fq = samp_fq, 
                                                                                                                                samp_ssl = samp_ssl)

                    r2_value_0 = sklearn.metrics.r2_score(y, yhat_0)
                    r2_value_1 = sklearn.metrics.r2_score(y, yhat_1)

                    f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                phase,
                                                                loss_tr,
                                                                r2_value_0,
                                                                r2_value_1,
                                                                time.time() - start_time,
                                                                y.size,
                                                                sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                batch_size,
                                                                loss_reg_0,
                                                                cps))
                    f.flush()
                
                    if SAVE_PREDICTION_CSVS:
                        with open(os.path.join(output, "train_pred_{}.csv".format(epoch)), "w") as f_trnpred:
                            for clmn in range(mean_0_ls.shape[1]):
                                f_trnpred.write("m_0_{},".format(clmn))
                            for clmn in range(mean_1_ls.shape[1]):
                                f_trnpred.write("m_1_{},".format(clmn))
                            for clmn in range(var_0_ls.shape[1]):
                                f_trnpred.write("v_0_{},".format(clmn))
                            for clmn in range(var_1_ls.shape[1]):
                                f_trnpred.write("v_1_{},".format(clmn))
                            f_trnpred.write("\n".format(clmn))
                            
                            for rw in range(mean_0_ls.shape[0]):
                                for clmn in range(mean_0_ls.shape[1]):
                                    f_trnpred.write("{},".format(mean_0_ls[rw, clmn]))
                                for clmn in range(mean_1_ls.shape[1]):
                                    f_trnpred.write("{},".format(mean_1_ls[rw, clmn]))
                                for clmn in range(var_0_ls.shape[1]):
                                    f_trnpred.write("{},".format(var_0_ls[rw, clmn]))
                                for clmn in range(var_1_ls.shape[1]):
                                    f_trnpred.write("{},".format(var_1_ls[rw, clmn]))
                                f_trnpred.write("\n".format(clmn))

                
                else:
    
                    ds = dataset[phase]
                    dataloader = torch.utils.data.DataLoader(
                        ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))                        
                    
                    loss_valit, yhat, y, var_hat, var_e, var_a, mean_0_ls, var_0_ls = run_epoch_val(model = model, model_1 = model_1, dataloader = dataloader, train = False, optim = None, device = device, block_size=None, y_mean = y_mean, y_std = y_std, samp_fq = samp_fq)

                    r2_value = sklearn.metrics.r2_score(y, yhat)
                    mae_value = sklearn.metrics.mean_absolute_error(y, yhat)
                    rmse_value = sklearn.metrics.mean_squared_error(y, yhat) ** 0.5
                    loss = loss_valit
                    
                    print(f"Epoch {epoch} - {phase}: R2={r2_value:.4f}, MAE={mae_value:.2f}, RMSE={rmse_value:.2f}", flush=True)

                    if SAVE_PREDICTION_CSVS:
                        with open(os.path.join(output, "z_{}_epch{}_prd.csv".format(phase, epoch)), "a") as pred_out:
                            pred_out.write("yhat,y,var_hat, var_e, var_a\n")
                            for pred_itr in range(y.shape[0]):
                                pred_out.write("{},{},{},{},{}\n".format(yhat[pred_itr],
                                y[pred_itr], 
                                var_hat[pred_itr], 
                                var_e[pred_itr], 
                                var_a[pred_itr]))
                            pred_out.flush()
                        
                        with open(os.path.join(output, "val_predmcd0_{}.csv".format(epoch)), "w") as f_trnpred:
                            for clmn in range(mean_0_ls.shape[1]):
                                f_trnpred.write("m_0_{},".format(clmn))
                            for clmn in range(var_0_ls.shape[1]):
                                f_trnpred.write("v_0_{},".format(clmn))
                            f_trnpred.write("\n".format(clmn))
                            
                            for rw in range(mean_0_ls.shape[0]):
                                for clmn in range(mean_0_ls.shape[1]):
                                    f_trnpred.write("{},".format(mean_0_ls[rw, clmn]))
                                for clmn in range(var_0_ls.shape[1]):
                                    f_trnpred.write("{},".format(var_0_ls[rw, clmn]))
                                f_trnpred.write("\n".format(clmn))

                    f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}".format(epoch,
                                                                phase,
                                                                loss,
                                                                r2_value,
                                                                mae_value,
                                                                rmse_value,
                                                                time.time() - start_time,
                                                                y.size,
                                                                sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                batch_size,
                                                                0,
                                                                0))
            
                    

                    f.write("\n")
                    f.flush()


            
            scheduler.step()
            scheduler_1.step()

            best_model_loss = loss_valit

            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'state_dict_1': model_1.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                "best_model_loss": best_model_loss,
                'r2': r2_value,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
                'opt_dict_1': optim_1.state_dict(),
                'scheduler_dict_1': scheduler_1.state_dict(),
                'np_rndstate': np.random.get_state(),
                'trch_rndstate': torch.get_rng_state()
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            
            if best_model_loss < bestLoss:
                print("saved best because {} < {}".format(best_model_loss, bestLoss))
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = best_model_loss


        # Load best weights for evaluation (after training or for test_only)
        best_pt_path = os.path.join(output, "best.pt")
        if num_epochs != 0 or (test_only and os.path.isfile(best_pt_path)):
            checkpoint = torch.load(best_pt_path)
            model.load_state_dict(checkpoint['state_dict'], strict = False)
            model_1.load_state_dict(checkpoint['state_dict_1'], strict = False)
            f.write("Best validation loss {} from epoch {}, R2 {}\n".format(checkpoint["best_model_loss"], checkpoint["epoch"], checkpoint["r2"]))
            f.flush()

        if run_test:

            split_list = ["test", "val"]
            
            # Test/val splits should not use SSL postfix - use original FileList.csv
            kwargs_test = kwargs.copy()
            kwargs_test.pop('ssl_postfix', None)  # Remove ssl_postfix for test/val
            
            for split in split_list: 

                dataloader = torch.utils.data.DataLoader(
                    dataset_class(root=data_dir, split=split, **kwargs_test),
                    batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), worker_init_fn=worker_init_fn)
                total_loss, yhat, y, _, _, _, _, _ = run_epoch_val(model = model, model_1 = model_1, dataloader = dataloader, train = False, optim = None, device = device, block_size=None, y_mean = y_mean, y_std = y_std, samp_fq = samp_fq)

                # Datasets return raw targets; run_epoch_val returns y=raw (concatenated), yhat=denormalized. Use y as-is for metrics.
                y_orig = y
                f.write("{} - {} (one clip) R2:   {:.3f}\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, sklearn.metrics.r2_score(y_orig, yhat)))
                f.write("{} - {} (one clip) MAE:  {:.2f}\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, sklearn.metrics.mean_absolute_error(y_orig, yhat)))
                f.write("{} - {} (one clip) RMSE: {:.2f}\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, sklearn.metrics.mean_squared_error(y_orig, yhat)**0.5))
                f.flush()







def run_epoch(model, 
            model_1, 
            dataloader_lb, 
            dataloader_unlb_0, 
            train, 
            optim, 
            optim_1, 
            device, 
            block_size=None, 
            run_dir = None, 
            test_val = None, 
            w_ulb = 10,  
            y_mean = 35, 
            y_std = 11, 
            samp_fq = 5, 
            samp_ssl = 5):
    
    model.train(train)
    model_1.train(train)

    total = 0  
    total_reg = 0 
    total_reg_1 = 0

    total_cps = 0
    total_cps_0 = 0
    total_cps_1 = 0


    n = 0 


    yhat_0 = []
    yhat_1 = []
    y = []

    mean2s_0_stack_ls = []
    mean2s_1_stack_ls = []
    var1s_0_stack_ls = []
    var1s_1_stack_ls = []

    start_frame_record = []
    vidpath_record = []

    torch.set_grad_enabled(train)

    total_itr_num = len(dataloader_lb)

    # Create iterators - avoid recreating iterators from exhausted DataLoaders with workers
    # as this causes deadlock. Since we iterate exactly total_itr_num times (len of labeled dataloader)
    # and unlabeled has more samples, neither should exhaust.
    dataloader_lb_itr = iter(dataloader_lb)
    dataloader_unlb_0_itr = iter(dataloader_unlb_0)

    for train_iter in range(total_itr_num):
        (X_ulb_0, outcome_ulb) = next(dataloader_unlb_0_itr)

        X_ulb_0 = X_ulb_0.to(device)

        all_output_unlb_0_pred_0, var_unlb_0_pred_0 = model(X_ulb_0)
        all_output_unlb_1_pred_0, var_unlb_1_pred_0 = model_1(X_ulb_0)
        
        mean1s_0 = []
        mean2s_0 = []
        var1s_0 = []

        mean1s_1 = []
        mean2s_1 = []
        var1s_1 = []

        X_ulb_in = X_ulb_0

        with torch.no_grad():
            for samp_ssl_itr in range(samp_ssl):
                mean1_raw_0, var1_raw_0 = model(X_ulb_in)
                
                # Handle pixel-wise vs image-level outputs
                if utils_pixelwise.is_pixelwise_output(mean1_raw_0):
                    # Pixel-wise: squeeze channel dim but keep spatial (B, H, W)
                    if mean1_raw_0.dim() == 4 and mean1_raw_0.size(1) == 1:
                        mean1_0 = mean1_raw_0.squeeze(1)
                        var1_0 = var1_raw_0.squeeze(1)
                    else:
                        mean1_0 = mean1_raw_0
                        var1_0 = var1_raw_0
                else:
                    # Image-level: flatten to 1D
                    mean1_0 = mean1_raw_0.view(-1)
                    var1_0 = var1_raw_0.view(-1)

                mean1s_0.append(mean1_0** 2)
                mean2s_0.append(mean1_0)
                var1s_0.append(var1_0)

                mean1_raw_1, var1_raw_1 = model_1(X_ulb_in)
                
                # Handle pixel-wise vs image-level outputs
                if utils_pixelwise.is_pixelwise_output(mean1_raw_1):
                    # Pixel-wise: squeeze channel dim but keep spatial (B, H, W)
                    if mean1_raw_1.dim() == 4 and mean1_raw_1.size(1) == 1:
                        mean1_1 = mean1_raw_1.squeeze(1)
                        var1_1 = var1_raw_1.squeeze(1)
                    else:
                        mean1_1 = mean1_raw_1
                        var1_1 = var1_raw_1
                else:
                    # Image-level: flatten to 1D
                    mean1_1 = mean1_raw_1.view(-1)
                    var1_1 = var1_raw_1.view(-1)

                mean1s_1.append(mean1_1** 2)
                mean2s_1.append(mean1_1)
                var1s_1.append(var1_1)


        mean2s_0_stack = torch.stack(mean2s_0, dim=1).to("cpu").detach().numpy()
        mean2s_0_stack_ls.append(mean2s_0_stack)
        var1s_0_stack = torch.stack(var1s_0, dim=1).to("cpu").detach().numpy()
        var1s_0_stack_ls.append(var1s_0_stack)

        mean1s_0_ = torch.stack(mean1s_0, dim=0).mean(dim=0)
        mean2s_0_ = torch.stack(mean2s_0, dim=0).mean(dim=0)
        var1s_0_ = torch.stack(var1s_0, dim=0).mean(dim=0)

        mean2s_1_stack = torch.stack(mean2s_1, dim=1).to("cpu").detach().numpy()
        mean2s_1_stack_ls.append(mean2s_1_stack)
        var1s_1_stack = torch.stack(var1s_1, dim=1).to("cpu").detach().numpy()
        var1s_1_stack_ls.append(var1s_1_stack)

        mean1s_1_ = torch.stack(mean1s_1, dim=0).mean(dim=0)
        mean2s_1_ = torch.stack(mean2s_1, dim=0).mean(dim=0)
        var1s_1_ = torch.stack(var1s_1, dim=0).mean(dim=0)


        all_output_unlb_0_pslb = mean2s_0_
        all_output_unlb_1_pslb = mean2s_1_

        avg_mean01 = (all_output_unlb_0_pslb + all_output_unlb_1_pslb)/2
        avg_var01 = (var1s_0_ + var1s_1_)/2

        # Handle pixel-wise vs image-level for consistency loss
        if utils_pixelwise.is_pixelwise_output(all_output_unlb_0_pred_0):
            # Pixel-wise: squeeze channel dim but keep spatial (B, H, W)
            if all_output_unlb_0_pred_0.dim() == 4 and all_output_unlb_0_pred_0.size(1) == 1:
                pred_0 = all_output_unlb_0_pred_0.squeeze(1)
                pred_1 = all_output_unlb_1_pred_0.squeeze(1)
                var_pred_0 = var_unlb_0_pred_0.squeeze(1)
                var_pred_1 = var_unlb_1_pred_0.squeeze(1)
            else:
                pred_0 = all_output_unlb_0_pred_0
                pred_1 = all_output_unlb_1_pred_0
                var_pred_0 = var_unlb_0_pred_0
                var_pred_1 = var_unlb_1_pred_0
        else:
            # Image-level: flatten to 1D
            pred_0 = all_output_unlb_0_pred_0.view(-1)
            pred_1 = all_output_unlb_1_pred_0.view(-1)
            var_pred_0 = var_unlb_0_pred_0.view(-1)
            var_pred_1 = var_unlb_1_pred_0.view(-1)

        loss_mse_cps_0 = ((pred_0 - avg_mean01)**2)
        loss_mse_cps_1 = ((pred_1 - avg_mean01)**2)

        loss_cmb_cps_0 = 0.5 * (torch.mul(torch.exp(-avg_var01), loss_mse_cps_0) + avg_var01 )
        loss_cmb_cps_1 = 0.5 * (torch.mul(torch.exp(-avg_var01), loss_mse_cps_1) + avg_var01 )

        loss_reg_cps0 = loss_cmb_cps_0.mean()
        loss_reg_cps1 = loss_cmb_cps_1.mean()

        
        var_loss_ulb_0 = ((var_pred_0 - avg_var01)**2).mean()
        var_loss_ulb_1 = ((var_pred_1 - avg_var01)**2).mean()


        loss_reg_cps = (loss_reg_cps0 + loss_reg_cps1) + (var_loss_ulb_0 + var_loss_ulb_1)

        (X, outcome ) = next(dataloader_lb_itr)


        y.append(outcome.detach().cpu().numpy())
        X = X.to(device)

        outcome = outcome.to(device)


        all_output = model(X)
        all_output_1 = model_1(X)                    
        

        mean_raw, var_raw = all_output
        mean_1_raw, var_1_raw = all_output_1
        
        # Check if output is pixel-wise or image-level
        is_pixelwise = utils_pixelwise.is_pixelwise_output(mean_raw)
        
        if is_pixelwise:
            # Pixel-wise regression (e.g., UNET)
            # Ensure outcome has same shape as prediction
            if outcome.dim() == 1:
                # Reshape outcome to match spatial dimensions (shouldn't happen with proper dataset)
                raise ValueError("Outcome should be spatial for pixel-wise regression")
            
            # Normalize target
            outcome_norm = (outcome - y_mean) / y_std
            
            # Squeeze channel dimension if present
            if mean_raw.dim() == 4 and mean_raw.size(1) == 1:
                mean = mean_raw.squeeze(1)
                var = var_raw.squeeze(1)
            else:
                mean = mean_raw
                var = var_raw
            
            if mean_1_raw.dim() == 4 and mean_1_raw.size(1) == 1:
                mean_1 = mean_1_raw.squeeze(1)
                var_1 = var_1_raw.squeeze(1)
            else:
                mean_1 = mean_1_raw
                var_1 = var_1_raw
            
            # Compute pixel-wise loss
            loss_mse = (mean - outcome_norm) ** 2
            loss1 = torch.mul(torch.exp(-(var + var_1) / 2), loss_mse)
            loss2 = (var + var_1) / 2
            loss = .5 * (loss1 + loss2)
            loss_reg_0 = loss.mean()
            
            loss_mse_1 = (mean_1 - outcome_norm) ** 2
            loss1_1 = torch.mul(torch.exp(-(var + var_1) / 2), loss_mse_1)
            loss2_1 = (var + var_1) / 2
            loss_1 = .5 * (loss1_1 + loss2_1)
            loss_reg_1 = loss_1.mean()
            
            # Store predictions (denormalized, flattened for metrics)
            yhat_0.append((mean.detach().cpu().numpy() * y_std + y_mean).flatten())
            yhat_1.append((mean_1.detach().cpu().numpy() * y_std + y_mean).flatten())
        else:
            # Image-level regression (e.g., ResNet, EfficientNet)
            mean = mean_raw.view(-1)
            var = var_raw.view(-1)
            mean_1 = mean_1_raw.view(-1)
            var_1 = var_1_raw.view(-1)
            
            loss_mse = (mean - (outcome - y_mean) / y_std) ** 2
            loss1 = torch.mul(torch.exp(-(var + var_1) / 2), loss_mse)
            loss2 = (var + var_1) / 2
            loss = .5 * (loss1 + loss2)
            loss_reg_0 = loss.mean()
            yhat_0.append(all_output[0].view(-1).to("cpu").detach().numpy() * y_std + y_mean)
            
            loss_mse_1 = (mean_1 - (outcome - y_mean) / y_std) ** 2
            loss1_1 = torch.mul(torch.exp(-(var + var_1) / 2), loss_mse_1)
            loss2_1 = (var + var_1) / 2
            loss_1 = .5 * (loss1_1 + loss2_1)
            loss_reg_1 = loss_1.mean()
            yhat_1.append(all_output_1[0].view(-1).to("cpu").detach().numpy() * y_std + y_mean)


        loss_reg = (loss_reg_0 + loss_reg_1)

        loss = loss_reg + w_ulb * loss_reg_cps + ((var_1 - var) ** 2).mean()

        
        if train:
            optim.zero_grad()
            optim_1.zero_grad()
            loss.backward()
            optim.step()
            optim_1.step()

        total += loss.item() * outcome.size(0)
        total_reg += loss_reg_0.item() * outcome.size(0)
        total_reg_1 += loss_reg_1.item() * outcome.size(0)

        total_cps += loss_reg_cps.item() * outcome.size(0)
        total_cps_0 += loss_reg_cps0.item() * outcome.size(0)
        total_cps_1 += loss_reg_cps1.item() * outcome.size(0)

        n += outcome.size(0)

        if train_iter % 10 == 0:
            print("phase {} itr {}/{}: ls {:.2f}({:.2f}) rg0 {:.4f} ({:.2f}) rg1 {:.4f} ({:.2f}) cps {:.4f} ({:.2f}) cps0 {:.4f} ({:.2f}) cps1 {:.4f} ({:.2f})".format(train,
                train_iter, total_itr_num, 
                total / n, loss.item(), 
                total_reg/n, loss_reg_0.item(), 
                total_reg_1/n, loss_reg_1.item(), 
                total_cps/n, loss_reg_cps.item(),
                total_cps_0/n, loss_reg_cps0.item(),
                total_cps_1/n, loss_reg_cps1.item()), flush = True)


    yhat_0 = np.concatenate(yhat_0)
    yhat_1 = np.concatenate(yhat_1)
    
    # Handle y concatenation (flatten if pixel-wise)
    y_list = []
    for y_item in y:
        if y_item.ndim > 1:
            y_list.append(y_item.flatten())
        else:
            y_list.append(y_item)
    y = np.concatenate(y_list)

    mean2s_0_stack_ls = np.concatenate(mean2s_0_stack_ls)
    mean2s_1_stack_ls = np.concatenate(mean2s_1_stack_ls)
    var1s_0_stack_ls = np.concatenate(var1s_0_stack_ls)
    var1s_1_stack_ls = np.concatenate(var1s_1_stack_ls)

    return total / n, total_reg / n, total_reg_1 / n, total_cps / n, total_cps_0 / n, total_cps_1 / n, yhat_0, yhat_1, y, mean2s_0_stack_ls, mean2s_1_stack_ls, var1s_0_stack_ls, var1s_1_stack_ls







def run_epoch_val(model, 
                model_1, 
                dataloader, 
                train, 
                optim, 
                device, 
                block_size=None, 
                y_mean = 35, 
                y_std = 11, 
                samp_fq = 5):


    model.train(False)
    model_1.train(False)

    total = 0 
    n = 0   

    yhat = []
    y = []

    var_hat = []
    var_e = []
    var_a = []

    mean2s_0_stack_ls = []
    var1s_0_stack_ls = []
    mean2s_0_stack_ls_m1 = []
    var1s_0_stack_ls_m1 = []

    mean2s_0_stack_ls_avg = []
    var1s_0_stack_ls_avg = []

    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (X, outcome) in dataloader:

                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)

                mean1s = []
                mean2s = []
                var1s = []


                mean1s_m1 = []
                mean2s_m1 = []
                var1s_m1 = []

                # Check if output is pixel-wise
                test_output = model(X)
                is_pixelwise = utils_pixelwise.is_pixelwise_output(test_output[0])
                
                for samp_itr in range(samp_fq):
                    all_ouput = model(X)
                    mean1_raw, var1_raw = all_ouput
                    
                    all_ouput_m1 = model_1(X)
                    mean1_raw_m1, var1_raw_m1 = all_ouput_m1
                    
                    if is_pixelwise:
                        # Pixel-wise: keep spatial dimensions, flatten for storage
                        if mean1_raw.dim() == 4 and mean1_raw.size(1) == 1:
                            mean1 = mean1_raw.squeeze(1).flatten(1)  # (B, H*W)
                            var1 = var1_raw.squeeze(1).flatten(1)
                        else:
                            mean1 = mean1_raw.flatten(1)
                            var1 = var1_raw.flatten(1)
                        
                        if mean1_raw_m1.dim() == 4 and mean1_raw_m1.size(1) == 1:
                            mean1_m1 = mean1_raw_m1.squeeze(1).flatten(1)
                            var1_m1 = var1_raw_m1.squeeze(1).flatten(1)
                        else:
                            mean1_m1 = mean1_raw_m1.flatten(1)
                            var1_m1 = var1_raw_m1.flatten(1)
                    else:
                        # Image-level: flatten to 1D
                        mean1 = mean1_raw.view(-1)
                        var1 = var1_raw.view(-1)
                        mean1_m1 = mean1_raw_m1.view(-1)
                        var1_m1 = var1_raw_m1.view(-1)

                    mean1s.append(mean1** 2)
                    mean2s.append(mean1)
                    var1s.append(torch.exp(var1))

                    mean1s_m1.append(mean1_m1** 2)
                    mean2s_m1.append(mean1_m1)
                    var1s_m1.append(torch.exp(var1_m1))


                mean2s_0_stack = torch.stack(mean2s, dim=1).to("cpu").detach().numpy()
                mean2s_0_stack_ls.append(mean2s_0_stack)
                var1s_0_stack = torch.stack(var1s, dim=1).to("cpu").detach().numpy()
                var1s_0_stack_ls.append(var1s_0_stack)

                mean1s_ = torch.stack(mean1s, dim=0).mean(dim=0)
                mean2s_ = torch.stack(mean2s, dim=0).mean(dim=0)
                var1s_ = torch.stack(var1s, dim=0).mean(dim=0)


                mean2s_0_stack_m1 = torch.stack(mean2s_m1, dim=1).to("cpu").detach().numpy()
                mean2s_0_stack_ls_m1.append(mean2s_0_stack_m1)
                var1s_0_stack_m1 = torch.stack(var1s_m1, dim=1).to("cpu").detach().numpy()
                var1s_0_stack_ls_m1.append(var1s_0_stack_m1)

                mean2s_0_stack_ls_avg.append((mean2s_0_stack + mean2s_0_stack_m1) / 2)
                var1s_0_stack_ls_avg.append((var1s_0_stack + var1s_0_stack_m1) / 2)


                mean1s_m1_ = torch.stack(mean1s_m1, dim=0).mean(dim=0)
                mean2s_m1_ = torch.stack(mean2s_m1, dim=0).mean(dim=0)
                var1s_m1_ = torch.stack(var1s_m1, dim=0).mean(dim=0)


                var2 = mean1s_ - mean2s_ ** 2
                var_ = var1s_ + var2
                var_norm = var_ / var_.max()         


                var2_m1 = mean1s_m1_ - mean2s_m1_ ** 2
                var_m1_ = var1s_m1_ + var2_m1
                var_m1_norm = var_m1_ / var_m1_.max()      



                # Denormalize predictions and store
                pred_mean = ((mean2s_ + mean2s_m1_) / 2).to("cpu").detach().numpy() * y_std + y_mean
                
                if is_pixelwise:
                    # Flatten pixel-wise predictions for metrics
                    yhat.append(pred_mean.flatten())
                else:
                    yhat.append(pred_mean)
                
                var_hat.append(((var_norm + var_m1_norm) / 2).to("cpu").detach().numpy())
                var_e.append(((var2 + var2_m1) / 2).to("cpu").detach().numpy())
                var_a.append(((var1s_ + var1s_m1_) / 2).to("cpu").detach().numpy())

                # Compute loss
                if is_pixelwise:
                    # Normalize outcome for pixel-wise
                    if outcome.dim() == 3:
                        outcome_norm = (outcome - y_mean) / y_std
                    else:
                        outcome_norm = (outcome - y_mean) / y_std
                    loss = torch.nn.functional.mse_loss((mean2s_ + mean2s_m1_) / 2, outcome_norm)
                else:
                    loss = torch.nn.functional.mse_loss((mean2s_ + mean2s_m1_) / 2, (outcome - y_mean) / y_std)

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item() * X.size(0)
                n += X.size(0)

                pbar.set_postfix_str("{:.2f} ({:.2f})".format(total / n, loss.item()))
                pbar.update()

    # Handle empty lists (e.g., if validation dataset is empty)
    if len(yhat) == 0:
        raise ValueError("Validation dataloader returned no batches. Check if validation split has data.")

    yhat = np.concatenate(yhat)
    var_hat = np.concatenate(var_hat)
    var_e = np.concatenate(var_e)
    var_a = np.concatenate(var_a)
    
    # Handle y concatenation (flatten if pixel-wise)
    y_list = []
    for y_item in y:
        if y_item.ndim > 1:
            y_list.append(y_item.flatten())
        else:
            y_list.append(y_item)
    y = np.concatenate(y_list)

    mean2s_0_stack_ls_avg = np.concatenate(mean2s_0_stack_ls_avg)
    var1s_0_stack_ls_avg = np.concatenate(var1s_0_stack_ls_avg)

    return total / n, yhat, y, var_hat, var_e, var_a, mean2s_0_stack_ls_avg, var1s_0_stack_ls_avg




if __name__ == '__main__':
    run()