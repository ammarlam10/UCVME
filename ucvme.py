"""
UCVME - Main training script supporting multiple datasets.
"""

import sys
import math
import os
import time
import datetime
import pandas as pd

import click
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm

import models
import datasets
import utils

from utils.config_loader import load_dataset_config


@click.command("ucvme")
@click.option("--dataset", type=str, required=True,
              help="Dataset name: so2sat_pop (more to be added)")
@click.option("--data_root", type=click.Path(exists=True, file_okay=False),
              default="/work/ammar/sslrp/data",
              help="Root directory containing datasets")
@click.option("--output", type=click.Path(file_okay=False), default=None, required=True)
@click.option("--pretrained/--random", default=True)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=True)
@click.option("--test_only/--run_all", default=False)

@click.option("--num_epochs", type=int, default=30)   
@click.option("--lr", type=float, default=0.0001)
@click.option("--weight_decay", type=float, default=1e-3)
@click.option("--lr_step_period", type=int, default=10)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=32)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)

@click.option("--reduced_set/--full_set", default=True)
@click.option("--rd_label", type=int, default=1000) 
@click.option("--rd_unlabel", type=int, default=9518)
@click.option("--ssl_mult", type=int, default=-1)
@click.option("--w_ulb", type=float, default=10)

@click.option("--pad_param", type=int, default=5)

@click.option("--y_mean", type=float, default=None)
@click.option("--y_std", type=float, default=None)

@click.option("--samp_fq", type=int, default=5)
@click.option("--samp_ssl", type=int, default=5)

@click.option("--drp_p", type=float, default=0.05)


def run(
    dataset,
    data_root="/work/ammar/sslrp/data",
    output=None,
    pretrained=True,
    weights=None,
    run_test=True,
    test_only=False,

    num_epochs=30,
    lr=0.0001,
    weight_decay=1e-3,
    lr_step_period=10,
    num_workers=4,
    batch_size=32,
    device=None,
    seed=0,

    reduced_set=True,
    rd_label=1000,
    rd_unlabel=9518,

    ssl_mult=-1,
    w_ulb=10,

    pad_param=5,
    y_mean=None,
    y_std=None,

    samp_fq=5,
    samp_ssl=5,

    drp_p=0.05
):
    """Main training function."""
    
    command_args = sys.argv[:]
    print("Run with options:")
    for carg_itr in command_args:
        print(carg_itr)

    # Load dataset configuration
    try:
        config = load_dataset_config(dataset)
        # Override data_root if provided
        original_data_root = config.get("data_root", "")
        if os.path.isabs(original_data_root):
            # If absolute path, use it directly
            config["data_root"] = original_data_root
        else:
            # If relative, join with provided data_root
            config["data_root"] = os.path.join(data_root, original_data_root.split("/")[-1])
        
        # Fallback: try dataset name as directory
        if not os.path.exists(config["data_root"]):
            fallback_path = os.path.join(data_root, dataset)
            if os.path.exists(fallback_path):
                config["data_root"] = fallback_path
            else:
                raise FileNotFoundError(
                    f"Data directory not found. Tried: {config['data_root']} and {fallback_path}"
                )
        
        print(f"Using data root: {config['data_root']}")
    except Exception as e:
        print(f"Error loading config for dataset '{dataset}': {e}")
        print(f"Available datasets: {datasets.list_available_datasets()}")
        return

    # Generate SSL splits if needed
    if reduced_set:
        ssl_file_list = os.path.join(
            config["data_root"], 
            "FileList_ssl_{}_{}.csv".format(rd_label, rd_unlabel)
        )
        
        if not os.path.isfile(ssl_file_list):
            print("Generating new file list for SSL dataset")
            _generate_ssl_splits(config, rd_label, rd_unlabel)

    ssl_mult_choice = ssl_mult

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Set default output directory
    if output is None:
        raise ValueError("--output option is required")

    os.makedirs(output, exist_ok=True)
    bkup_tmstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "gpu":
        device = torch.device("cuda")
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Invalid device parameter: {device}")

    # Initialize models
    model = models.resnet50_unc(pretrained=pretrained, drp_p=drp_p)
    model = torch.nn.DataParallel(model)

    model_1 = models.resnet50_unc(pretrained=pretrained, drp_p=drp_p)
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
            raise ValueError("State dict not found in checkpoint")

    # Initialize optimizers
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    optim_1 = torch.optim.Adam(model_1.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler_1 = torch.optim.lr_scheduler.StepLR(optim_1, lr_step_period)

    # Compute image statistics
    print("Computing image mean and std...")
    train_dataset_temp = datasets.get_dataset(
        dataset_name=dataset,
        config=config,
        split="train",
        ssl_type=0
    )
    mean, std = utils.get_mean_and_std(train_dataset_temp)
    print(f"Image mean: {mean}, std: {std}")

    # Get target statistics
    if y_mean is None or y_std is None:
        print("Computing target statistics...")
        y_mean, y_std = train_dataset_temp.get_target_stats()
        print(f"Target mean: {y_mean}, std: {y_std}")
    else:
        print(f"Using provided target stats: mean={y_mean}, std={y_std}")

    # Set up datasets and dataloaders
    dataset_dict = {}
    dataset_trainsub = {}
    
    if reduced_set:
        ssl_postfix = "_ssl_{}_{}".format(rd_label, rd_unlabel)
        
        dataset_trainsub['lb'] = datasets.get_dataset(
            dataset_name=dataset,
            config=config,
            split="train",
            ssl_type=1,
            ssl_postfix=ssl_postfix,
            ssl_mult=ssl_mult_choice,
            mean=mean,
            std=std,
            pad=pad_param
        )
        
        dataset_trainsub['unlb_0'] = datasets.get_dataset(
            dataset_name=dataset,
            config=config,
            split="train",
            ssl_type=2,
            ssl_postfix=ssl_postfix,
            mean=mean,
            std=std,
            pad=pad_param
        )
    else:
        raise NotImplementedError("Full set not yet implemented")

    dataset_dict['train'] = dataset_trainsub
    dataset_dict["val"] = datasets.get_dataset(
        dataset_name=dataset,
        config=config,
        split="val",
        ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel) if reduced_set else "",
        mean=mean,
        std=std
    )

    # Training loop
    with open(os.path.join(output, "log.csv"), "a") as f:
        f.write("Run timestamp: {}\n".format(bkup_tmstmp))

        epoch_resume = 0
        bestLoss = float("inf")
        
        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])

            model_1.load_state_dict(checkpoint['state_dict_1'], strict=False)
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

        if test_only:
            num_epochs = 0

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            
            for phase in ['train', 'val']:
                start_time = time.time()

                if device.type == "cuda":
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.reset_peak_memory_stats(i)

                ds = dataset_dict[phase]
                
                if phase == "train":
                    dataloader_lb = torch.utils.data.DataLoader(
                        ds['lb'], batch_size=batch_size, num_workers=num_workers,
                        shuffle=True, pin_memory=(device.type == "cuda"),
                        drop_last=True, worker_init_fn=worker_init_fn
                    )
                    dataloader_unlb_0 = torch.utils.data.DataLoader(
                        ds['unlb_0'], batch_size=batch_size, num_workers=num_workers,
                        shuffle=True, pin_memory=(device.type == "cuda"),
                        drop_last=True, worker_init_fn=worker_init_fn
                    )

                    loss_tr, loss_reg_0, loss_reg_1, cps, cps_l, cps_s, yhat_0, yhat_1, y, mean_0_ls, mean_1_ls, var_0_ls, var_1_ls = run_epoch(
                        model, model_1, dataloader_lb, dataloader_unlb_0,
                        phase == "train", optim, optim_1, device,
                        w_ulb=w_ulb, y_mean=y_mean, y_std=y_std,
                        samp_fq=samp_fq, samp_ssl=samp_ssl
                    )

                    r2_value_0 = sklearn.metrics.r2_score(y, yhat_0)
                    r2_value_1 = sklearn.metrics.r2_score(y, yhat_1)

                    f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        epoch, phase, loss_tr, r2_value_0, r2_value_1,
                        time.time() - start_time, y.size,
                        sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())),
                        sum(torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())),
                        batch_size, loss_reg_0, cps
                    ))
                    f.flush()

                else:
                    dataloader = torch.utils.data.DataLoader(
                        ds, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=(device.type == "cuda"),
                        drop_last=False
                    )

                    loss_valit, yhat, y, var_hat, var_e, var_a, mean_0_ls, var_0_ls = run_epoch_val(
                        model=model, model_1=model_1, dataloader=dataloader,
                        train=False, optim=None, device=device, block_size=None,
                        y_mean=y_mean, y_std=y_std, samp_fq=samp_fq
                    )

                    r2_value = sklearn.metrics.r2_score(y, yhat)
                    loss = loss_valit

                    f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        epoch, phase, loss, r2_value, time.time() - start_time,
                        y.size,
                        sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())),
                        sum(torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())),
                        batch_size, 0, 0
                    ))
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

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model_1.load_state_dict(checkpoint['state_dict_1'], strict=False)

            f.write("Best validation loss {} from epoch {}, R2 {}\n".format(
                checkpoint["best_model_loss"], checkpoint["epoch"], checkpoint["r2"]
            ))
            f.flush()

        if run_test:
            split_list = ["test", "val"]

            for split in split_list:
                dataloader = torch.utils.data.DataLoader(
                    datasets.get_dataset(
                        dataset_name=dataset,
                        config=config,
                        split=split,
                        ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel) if reduced_set else "",
                        mean=mean,
                        std=std
                    ),
                    batch_size=batch_size, num_workers=num_workers,
                    shuffle=False, pin_memory=(device.type == "cuda"),
                    worker_init_fn=worker_init_fn
                )
                total_loss, yhat, y, _, _, _, _, _ = run_epoch_val(
                    model=model, model_1=model_1, dataloader=dataloader,
                    train=False, optim=None, device=device, block_size=None,
                    y_mean=y_mean, y_std=y_std, samp_fq=samp_fq
                )

                f.write("{} - {} (one clip) R2:   {:.3f}\n".format(
                    datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split,
                    sklearn.metrics.r2_score(y, yhat)
                ))
                f.write("{} - {} (one clip) MAE:  {:.2f}\n".format(
                    datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split,
                    sklearn.metrics.mean_absolute_error(y, yhat)
                ))
                f.write("{} - {} (one clip) RMSE: {:.2f}\n".format(
                    datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split,
                    sklearn.metrics.mean_squared_error(y, yhat)**0.5
                ))
                f.flush()


def _generate_ssl_splits(config, rd_label, rd_unlabel):
    """Generate SSL splits for dataset."""
    data_root = config["data_root"]
    file_list_path = os.path.join(data_root, config.get("file_list", "FileList.csv"))
    
    if not os.path.exists(file_list_path):
        print(f"FileList.csv not found at {file_list_path}")
        print("Please generate FileList.csv first by running the dataset")
        return
    
    print("Generating SSL splits...")
    data = pd.read_csv(file_list_path)
    data["SPLIT"] = data["SPLIT"].str.upper()

    file_name_list = np.array(data[data['SPLIT'] == 'TRAIN']['FileName'])
    np.random.seed(0)
    np.random.shuffle(file_name_list)

    label_list = file_name_list[:rd_label]
    unlabel_list = file_name_list[rd_label:rd_label + rd_unlabel]

    data['SSL_SPLIT'] = "UNLABELED"
    data.loc[data['FileName'].isin(label_list), 'SSL_SPLIT'] = "LABELED"

    output_path = os.path.join(data_root, "FileList_ssl_{}_{}.csv".format(rd_label, rd_unlabel))
    data.to_csv(output_path, index=False)
    print(f"Saved SSL splits to: {output_path}")


# Import training functions - these are defined in ucvme_age.py
# We'll import them at runtime to avoid circular imports
import importlib.util
spec = importlib.util.spec_from_file_location("ucvme_age_module", "ucvme_age.py")
ucvme_age_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ucvme_age_module)
run_epoch = ucvme_age_module.run_epoch
run_epoch_val = ucvme_age_module.run_epoch_val


if __name__ == '__main__':
    run()

