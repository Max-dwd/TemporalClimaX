import os
from datetime import datetime

import dask.array as da
import math
import hydra
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr

from hydra.utils import to_absolute_path
from lightning.pytorch import LightningDataModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.callbacks import EarlyStopping

from _climate_kaggle_metric import score
from _test_kaggle_metric import test_metric_equivalence

try:
    import wandb  # Optional, for logging to Weights & Biases
except ImportError:
    wandb = None

from src.models import get_model
from src.utils import (
    Normalizer,
    calculate_weighted_metric,
    convert_predictions_to_kaggle_format,
    create_climate_data_array,
    create_comparison_plots,
    get_lat_weights,
    get_logger,
    get_trainer_config,
)

# Setup logging
log = get_logger(__name__)

# --- Climate Loss that aligns Kaggle score ---
def climate_loss(
        pred, target, lat_weights,
        var_weights = {"tas": 0.5, "pr": 0.5},
        metric_weights = {  "tas": dict(monthly=0.1, mean=1.0, std=1.0),
                            "pr" : dict(monthly=0.1, mean=1.0, std=0.75) },
        months_per_decade  = 120,
):
    """
    pred, target : [B,T,C,H,W]  (C: ['tas','pr'])
    """

    # --- Edge: this loss funciton handles 5-D tensor with new dimension T ---
    if pred.dim() == 4:                      
        pred, target = pred.unsqueeze(1), target.unsqueeze(1)

    # --- Edge: ensure lat_weights is a tensor and on the correct device ---
    if not isinstance(lat_weights, torch.Tensor):
        lat_weights = torch.as_tensor(np.asarray(lat_weights), dtype=pred.dtype, device=pred.device)
    else:
        lat_weights = lat_weights.to(pred.device, dtype=pred.dtype)

    # --- Edge: make lat_weights broadcastable to [B,T,C,H,W] ---
    if lat_weights.dim() == 1:             # (H)  →  (1,1,H,1)
        w_lat = lat_weights.view(1,1,-1,1)
    else:                                  
        w_lat = lat_weights

    # --- Compute variable‑specific components ---
    B,T,C,H,W = pred.shape                     
    total = 0.0
    
    for c,var in enumerate(["tas","pr"]):
        p,t = pred[:,:,c], target[:,:,c]        # [B,T,H,W]
            
        # monthly RMSE
        rmse = torch.sqrt(((p - t) ** 2 * w_lat).mean())

        # reshape into decades
        n_dec = max(1, T//months_per_decade)
        p_dec = p.view(B,n_dec,-1,H,W)          # [B,n_dec,months_per_decade,H,W]
        t_dec = t.view_as(p_dec)

        # decadal rmse and mae
        mean_rmse = torch.sqrt((((p_dec.mean(2)-t_dec.mean(2))**2) * w_lat).mean()) # [B,n_dec,H,W]
        std_mae   = (((p_dec.std(2,unbiased=False) -
                       t_dec.std(2,unbiased=False)).abs()) * w_lat).mean()    #[B, n_dec, H, W]

        coeff = metric_weights[var]
        loss_var = (coeff["monthly"] * rmse +
                    coeff["mean"]    * mean_rmse +
                    coeff["std"]     * std_mae)

        total += var_weights[var] * loss_var

    return total
    
    
# --- Data Handling: Time Window ---

# Preloads the entire (normalized) dataset into memory using Dask
class ClimateDataset(Dataset):
    def __init__(self, inputs_norm_dask, outputs_dask, output_is_normalized=True, block_len=10, valid_starts=None, stride=1, is_train=True):
        total_months = inputs_norm_dask.shape[0]
        self.stride = stride

        # --- Default stride-1 windows over entire dataset --- 
        if valid_starts is None:
            valid_starts = range(0, total_months - block_len + 1, stride) 

        # --- Decade bounding: allow overlapping within every 120 months ---  
        if not is_train:
            valid_starts = [
                s for s in valid_starts
                if (s // 120) == ((s + block_len - 1) // 120)   # stay inside a decade
            ]
        
        # --- Time window info ---
        self.valid_starts = np.asarray(valid_starts, dtype=np.int32)
        self.block_len = block_len
        self.size = len(self.valid_starts) # size of valid windows 

        # Log once with basic information
        log.info(
            f"Creating dataset: {self.size} samples, input shape: {inputs_norm_dask.shape}, normalized output: {output_is_normalized}"
        )

        # Precompute all tensors in one go
        inputs_np = inputs_norm_dask.compute()
        outputs_np = outputs_dask.compute()

        # Convert to PyTorch tensors
        self.input_tensors = torch.from_numpy(inputs_np).float()
        self.output_tensors = torch.from_numpy(outputs_np).float()

        # Handle NaN values (should not occur)
        if torch.isnan(self.input_tensors).any() or torch.isnan(self.output_tensors).any():
            raise ValueError("NaN values detected in dataset tensors")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # --- Pass data in time window slice ---
        start = self.valid_starts[idx]
        sl = slice(start, start + self.block_len)
        x_win = self.input_tensors[sl]   # (block_len, C, H, W)
        y_win = self.output_tensors[sl]  # (block_len, C, H, W)
        return x_win, y_win, start

# Loads and processes input and output variables for a single serie using Dask.
def _load_process_ssp_data(ds, ssp, input_variables, output_variables, member_id, spatial_template):
    """
    Args:
        ds (xr.Dataset): The opened xarray dataset.
        ssp (str): The SSP identifier (e.g., 'ssp126').
        input_variables (list): List of input variable names.
        output_variables (list): List of output variable names.
        member_id (int): The member ID to select.
        spatial_template (xr.DataArray): A template DataArray with ('y', 'x') dimensions
                                          for broadcasting global variables.
    Returns:
        tuple: (input_dask_array, output_dask_array)
               - input_dask_array: Stacked dask array of inputs (time, channels, y, x).
               - output_dask_array: Stacked dask array of outputs (time, channels, y, x).
    """
    ssp_input_dasks = []
    for var in input_variables:
        da_var = ds[var].sel(ssp=ssp)
        # Rename spatial dims if needed
        if "latitude" in da_var.dims:
            da_var = da_var.rename({"latitude": "y", "longitude": "x"})
        # Select member if applicable
        if "member_id" in da_var.dims:
            da_var = da_var.sel(member_id=member_id)
        # Process based on dimensions
        if set(da_var.dims) == {"time"}:  # Global variable, broadcast to spatial dims:
            # Broadcast like template, then transpose to ensure ('time', 'y', 'x')
            da_var_expanded = da_var.broadcast_like(spatial_template).transpose("time", "y", "x")
            ssp_input_dasks.append(da_var_expanded.data)
        elif set(da_var.dims) == {"time", "y", "x"}:  # Spatially resolved
            ssp_input_dasks.append(da_var.data)
        else:
            raise ValueError(f"Unexpected dimensions for variable {var} in SSP {ssp}: {da_var.dims}")

    # Stack inputs along channel dimension -> dask array (time, channels, y, x)
    stacked_input_dask = da.stack(ssp_input_dasks, axis=1)

    # Prepare output dask arrays for each output variable
    output_dasks = []
    for var in output_variables:
        da_output = ds[var].sel(ssp=ssp, member_id=member_id)

        # --- Log transform on precipitation ---
        if var == "pr":
            da_output = xr.apply_ufunc(np.log1p, da_output, dask="parallelized")
            
        # Ensure output also uses y, x if necessary`
        if "latitude" in da_output.dims:
            da_output = da_output.rename({"latitude": "y", "longitude": "x"})

        # Add time, y, x dimensions as a dask array
        output_dasks.append(da_output.data)

    # Stack outputs along channel dimension -> dask array (time, channels, y, x)
    stacked_output_dask = da.stack(output_dasks, axis=1)
    return stacked_input_dask, stacked_output_dask


# Data Module
class ClimateEmulationDataModule(LightningDataModule):
    def __init__(
        self,
        path: str,
        input_vars: list,
        output_vars: list,
        train_ssps: list,
        test_ssp: str,
        target_member_id: list[int],
        test_months: int = 360,
        batch_size: int = 32,
        eval_batch_size: int = None,
        num_workers: int = 0,
        seed: int = 42,
        n_months_per_series: int = 0,   # default: 0, use full length of a series
        block_len: int = 120,            # length of time window for training
        val_len: int = 120,              
        test_len: int = 120,
        stride: int = 1,                # stride of time window
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.path = to_absolute_path(path)
        self.normalizer = Normalizer()
        self.n_months_per_series = n_months_per_series
        self.block_len = block_len
        self.val_len = val_len
        self.test_len = test_len
        self.stride = stride

        # Set evaluation batch size to training batch size if not specified
        if eval_batch_size is None:
            self.hparams.eval_batch_size = batch_size

        # Placeholders
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.lat_coords, self.lon_coords, self._lat_weights_da = None, None, None

    def prepare_data(self):
        if not os.path.exists(self.hparams.path):
            raise FileNotFoundError(f"Data path not found: {self.hparams.path}")
        log.info(f"Data found at: {self.hparams.path}")

    def setup(self, stage: str | None = None):
        log.info(f"Setting up data module for stage: {stage} from {self.hparams.path}")
        
        # --- List of all idx allowed to start a time window slice ---
        valid_starts = []
        offset = 0

        # Use context manager for opening dataset
        with xr.open_zarr(self.hparams.path, consolidated=True, chunks={"time": 24}) as ds:
            # Create a spatial template ONCE using a variable guaranteed to have y, x
            # Extract the template DataArray before renaming for coordinate access
            spatial_template_da = ds["rsdt"].isel(time=0, ssp=0, drop=True)  # drop time/ssp dims

            #  Prepare Training and Validation Data 
            train_inputs_dask_list, train_outputs_dask_list = [], []
            val_input_dask, val_output_dask = None, None
            val_ssp = "ssp370"
            val_months = 120
            
            # Process all SSPs
            log.info(f"Loading data from SSPs: {self.hparams.train_ssps}")        
            
            for ssp in self.hparams.train_ssps:
                # Load the data for this SSP
                for i in self.hparams.target_member_id:
                    ssp_input_dask, ssp_output_dask = _load_process_ssp_data(
                        ds,
                        ssp,
                        self.hparams.input_vars,
                        self.hparams.output_vars,
                        i,      # Use more ensemble 
                        spatial_template_da,
                    )   

                    if ssp == val_ssp:
                        # Special handling for SSP 370: split into training and validation
                        # Last 120 months go to validation
                        val_input_dask = ssp_input_dask[-val_months:]
                        val_output_dask = ssp_output_dask[-val_months:]
                        # Early months go to training if there are any
                        train_inputs_dask_list.append(ssp_input_dask[:-val_months])
                        train_outputs_dask_list.append(ssp_output_dask[:-val_months])
                        
                    # --- Valid windows stay inside this member’s segment ---
                    series_len = ssp_input_dask.shape[0]
                    block_len = self.block_len
                    n_months = self.n_months_per_series or series_len

                    train_inputs_dask_list.append(ssp_input_dask)
                    train_outputs_dask_list.append(ssp_output_dask)
                    
                    # --- After loops finish, valid_starts should be like 
                    # [(0, ..., len_series_1 - block_len + 1), (len_series_1, ..., len_series_2 - block_len + 1), ...] ---
                    valid_starts.extend(range(offset, offset + n_months - block_len + 1))  
                    offset += n_months               # move pointer for next series
                    print(len(valid_starts))

            # --- Strided sampling ---
            valid_starts = valid_starts[::self.stride] # keeps only every‐nth start
            print(len(valid_starts))
                    
            # Concatenate training data only
            train_input_dask = da.concatenate(train_inputs_dask_list, axis=0)
            train_output_dask = da.concatenate(train_outputs_dask_list, axis=0)

            # Compute z-score normalization statistics using the training data
            input_mean = da.nanmean(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            input_std = da.nanstd(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            output_mean = da.nanmean(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()
            output_std = da.nanstd(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()

            self.normalizer.set_input_statistics(mean=input_mean, std=input_std)
            self.normalizer.set_output_statistics(mean=output_mean, std=output_std)

            # Define Normalized Training Dask Arrays 
            train_input_norm_dask = self.normalizer.normalize(train_input_dask, data_type="input")
            train_output_norm_dask = self.normalizer.normalize(train_output_dask, data_type="output")

            #  Define Normalized Validation Dask Arrays
            val_input_norm_dask = self.normalizer.normalize(val_input_dask, data_type="input")
            val_output_norm_dask = self.normalizer.normalize(val_output_dask, data_type="output")
            
            #  Prepare Test Data 
            full_test_input_dask, full_test_output_dask = _load_process_ssp_data(
                ds,
                self.hparams.test_ssp,
                self.hparams.input_vars,
                self.hparams.output_vars,
                0, # Use the first ensemble now to align with template
                spatial_template_da,
            )

            # Slice Test Data 
            test_slice = slice(-self.hparams.test_months, None)  # Last N months

            sliced_test_input_dask = full_test_input_dask[test_slice]
            sliced_test_output_raw_dask = full_test_output_dask[test_slice]

            # Define Normalized Test Input Dask Array
            test_input_norm_dask = self.normalizer.normalize(sliced_test_input_dask, data_type="input")  
            test_output_raw_dask = sliced_test_output_raw_dask  # Keep unnormed for evaluation

        # --- Create datasets with time windows ---
        self.train_dataset = ClimateDataset(train_input_norm_dask, train_output_norm_dask, output_is_normalized=True, 
                                            block_len=self.block_len, valid_starts=valid_starts)
        self.val_dataset = ClimateDataset(val_input_norm_dask, val_output_norm_dask, output_is_normalized=True, 
                                          block_len=self.val_len, is_train=False)
        self.test_dataset = ClimateDataset(test_input_norm_dask, test_output_raw_dask, output_is_normalized=False, 
                                           block_len=self.test_len, is_train=False)

        # Log dataset sizes in a single message
        log.info(
            f"Datasets created. Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)} (last months of {val_ssp}), Test: {len(self.test_dataset)}"
        )

    # Common DataLoader configuration
    def _get_dataloader_kwargs(self, is_train=False):
        """Return common DataLoader configuration as a dictionary"""
        return {
            "batch_size": self.hparams.batch_size if is_train else self.hparams.eval_batch_size,
            "shuffle": is_train,  # Only shuffle training data
            "num_workers": self.hparams.num_workers,
            "persistent_workers": self.hparams.num_workers > 0,
            "pin_memory": True,
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self._get_dataloader_kwargs(is_train=True))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self._get_dataloader_kwargs(is_train=False))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self._get_dataloader_kwargs(is_train=False))

    def get_lat_weights(self):
        """
        Returns area weights for the latitude dimension as an xarray DataArray.
        The weights can be used with xarray's weighted method for proper spatial averaging.
        """
        if self._lat_weights_da is None:
            with xr.open_zarr(self.hparams.path, consolidated=True) as ds:
                template = ds["rsdt"].isel(time=0, ssp=0)
                y_coords = template.y.values

                # Calculate weights based on cosine of latitude
                weights = get_lat_weights(y_coords)

                # Create DataArray with proper dimensions
                self._lat_weights_da = xr.DataArray(weights, dims=["y"], coords={"y": y_coords}, name="area_weights")

        return self._lat_weights_da

    def get_coords(self):
        """
        Returns: tuple: (y array, x array)
        """
        if self.lat_coords is None or self.lon_coords is None:
            # Get coordinates if they haven't been stored yet
            with xr.open_zarr(self.hparams.path, consolidated=True) as ds:
                template = ds["rsdt"].isel(time=0, ssp=0, drop=True)
                self.lat_coords = template.y.values
                self.lon_coords = template.x.values

        return self.lat_coords, self.lon_coords


# PyTorch Lightning Module
class ClimateEmulationModule(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float):
        super().__init__()
        self.model = model
        # Access hyperparams via self.hparams object after saving, e.g., self.hparams.learning_rate
        self.save_hyperparameters(ignore=["model"])
        # --- Replace: self.criterion = nn.MSELoss() with climate loss ---
        self.normalizer = None
        # Store evaluation outputs for time-mean calculation
        self.test_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self) -> None:
        self.normalizer = self.trainer.datamodule.normalizer  # Access the normalizer from the datamodule

    # --- Training with lat_weighted climate_loss  ----
    def training_step(self, batch, batch_idx):
        area_weights = self.trainer.datamodule.get_lat_weights()
        x, y_true_norm, start = batch 
        y_pred_norm = self(x)   # y_pred_norm: (B,T,C,H,W) 
        loss = climate_loss(y_pred_norm, y_true_norm, area_weights)
        self.log("train/loss", loss, prog_bar=True, batch_size=x.size(0))  
        return loss

    def validation_step(self, batch, batch_idx):
        area_weights = self.trainer.datamodule.get_lat_weights()
        x, y_true_norm, start = batch
        y_pred_norm = self(x) # y_pred_norm: (B,T,C,H,W)
        
        # Save unnormalized outputs for decadal mean/stddev calculation in validation_epoch_end
        y_pred_denorm = self.normalizer.inverse_transform_output(y_pred_norm.detach().cpu().numpy())
        y_true_denorm = self.normalizer.inverse_transform_output(y_true_norm.detach().cpu().numpy())
        
        # --- Inverse log transforamtion ---
        y_pred_denorm[..., 1, :, :] = np.expm1(y_pred_denorm[..., 1, :, :])
        y_true_denorm[..., 1, :, :] = np.expm1(y_true_denorm[..., 1, :, :]) 
        y_pred_denorm = torch.as_tensor(y_pred_denorm,
                                        dtype=y_pred_norm.dtype,
                                        device=y_pred_norm.device)
        y_true_denorm = torch.as_tensor(y_true_denorm,
                                        dtype=y_pred_norm.dtype,
                                        device=y_pred_norm.device)
        
        # --- Use climate loss as validation metric to align with Kaggle scores ---
        loss = climate_loss(y_pred_denorm, y_true_denorm, area_weights)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0), sync_dist=True)
        y_pred_denorm = y_pred_denorm.detach().cpu().numpy()
        y_true_denorm = y_true_denorm.detach().cpu().numpy()
        self.validation_step_outputs.append((y_pred_denorm, y_true_denorm))
        return loss
    
    def _evaluate_predictions(self, predictions, targets, is_test=False):
        """
        Args:
            predictions (np.ndarray): Prediction array with shape (time, channels, y, x)
            targets (np.ndarray): Target array with shape (time, channels, y, x)
            is_test (bool): Whether this is being called from test phase (vs validation)
        """
        phase = "test" if is_test else "val"
        log_kwargs = {"prog_bar": not is_test, "sync_dist": not is_test}

        # Get number of evaluation timesteps
        n_timesteps = predictions.shape[0]

        # Get area weights for proper spatial averaging
        area_weights = self.trainer.datamodule.get_lat_weights()

        # Get coordinates
        lat_coords, lon_coords = self.trainer.datamodule.get_coords()
        time_coords = np.arange(n_timesteps)
        output_vars = self.trainer.datamodule.hparams.output_vars

        # Process each output variable
        for i, var_name in enumerate(output_vars):
            # Extract channel data
            preds_var = predictions[:, i, :, :]
            trues_var = targets[:, i, :, :]

            var_unit = "mm/day" if var_name == "pr" else "K" if var_name == "tas" else "unknown"

            # Create xarray objects for weighted calculations
            preds_xr = create_climate_data_array(
                preds_var, time_coords, lat_coords, lon_coords, var_name=var_name, var_unit=var_unit
            )
            trues_xr = create_climate_data_array(
                trues_var, time_coords, lat_coords, lon_coords, var_name=var_name, var_unit=var_unit
            )
            
            # --- Prevent overflow on fp16 ---
            preds_xr = preds_xr.astype(np.float32)
            trues_xr = trues_xr.astype(np.float32)

            # 1. Calculate weighted month-by-month RMSE over all samples
            diff_squared = (preds_xr - trues_xr) ** 2
            overall_rmse = calculate_weighted_metric(diff_squared, area_weights, ("time", "y", "x"), "rmse")
            self.log(f"{phase}/{var_name}/avg/monthly_rmse", float(overall_rmse), **log_kwargs)

            # 2. Calculate time-mean (i.e. decadal, 120 months average) and calculate area-weighted RMSE for time means
            pred_time_mean = preds_xr.mean(dim="time")
            true_time_mean = trues_xr.mean(dim="time")
            mean_diff_squared = (pred_time_mean - true_time_mean) ** 2
            time_mean_rmse = calculate_weighted_metric(mean_diff_squared, area_weights, ("y", "x"), "rmse")
            self.log(f"{phase}/{var_name}/time_mean_rmse", float(time_mean_rmse), **log_kwargs)

            # 3. Calculate time-stddev (temporal variability) and calculate area-weighted MAE for time stddevs
            pred_time_std = preds_xr.std(dim="time")
            true_time_std = trues_xr.std(dim="time")
            std_abs_diff = np.abs(pred_time_std - true_time_std)
            time_std_mae = calculate_weighted_metric(std_abs_diff, area_weights, ("y", "x"), "mae")
            self.log(f"{phase}/{var_name}/time_stddev_mae", float(time_std_mae), **log_kwargs)

            # Extra logging of sample predictions/images to wandb for test phase (feel free to use this for validation)
            if is_test:
                # Generate visualizations for test phase when using wandb
                if isinstance(self.logger, WandbLogger):
                    # Time mean visualization
                    fig = create_comparison_plots(
                        true_time_mean,
                        pred_time_mean,
                        title_prefix=f"{var_name} Mean",
                        metric_value=time_mean_rmse,
                        metric_name="Weighted RMSE",
                    )
                    self.logger.experiment.log({f"img/{var_name}/time_mean": wandb.Image(fig)})
                    plt.close(fig)

                    # Time standard deviation visualization
                    fig = create_comparison_plots(
                        true_time_std,
                        pred_time_std,
                        title_prefix=f"{var_name} Stddev",
                        metric_value=time_std_mae,
                        metric_name="Weighted MAE",
                        cmap="plasma",
                    )
                    self.logger.experiment.log({f"img/{var_name}/time_Stddev": wandb.Image(fig)})
                    plt.close(fig)

                    # Sample timesteps visualization
                    if n_timesteps > 3:
                        timesteps = np.random.choice(n_timesteps, 3, replace=False)
                        for t in timesteps:
                            true_t = trues_xr.isel(time=t)
                            pred_t = preds_xr.isel(time=t)
                            fig = create_comparison_plots(true_t, pred_t, title_prefix=f"{var_name} Timestep {t}")
                            self.logger.experiment.log({f"img/{var_name}/month_idx_{t}": wandb.Image(fig)})
                            plt.close(fig)
        
        
    def on_validation_epoch_end(self):
        # Compute time-mean and time-stddev errors using all validation months
        if not self.validation_step_outputs:
            return

        # Stack all predictions and ground truths
        # --- all_preds_np: epoch*(B,T,C,H,W)
        #  -> concat:(B*epoch,T,C,H,W) -> reshape:(B*epoch*T,C,H,W) -> _evaluate_predictions ---
        all_preds_np = np.concatenate([pred for pred, _ in self.validation_step_outputs], axis=0
                                      ).reshape(-1, *self.validation_step_outputs[0][0].shape[2:])
        all_trues_np = np.concatenate([true for _, true in self.validation_step_outputs], axis=0
                                      ).reshape(-1, *self.validation_step_outputs[0][0].shape[2:])

        # Use the helper method to evaluate predictions
        self._evaluate_predictions(all_preds_np, all_trues_np, is_test=False)

        self.validation_step_outputs.clear()  # Clear the outputs list for next epoch

    
    def test_step(self, batch, batch_idx):
        x, y_true_denorm, start = batch
        y_pred_norm = self(x) # y_pred_norm: (B,block_len,C,H,W)
        y_pred_denorm = self.normalizer.inverse_transform_output(y_pred_norm.cpu().numpy()) 
        
        # --- Inverse log transforamtion ---
        y_pred_denorm[..., 1, :, :] = np.expm1(y_pred_denorm[..., 1, :, :])
        y_true_denorm = y_true_denorm.cpu().numpy()
        y_true_denorm[..., 1, :, :] = np.expm1(y_true_denorm[..., 1, :, :])

        # --- Averaging data for overlapping window prediction ---
        for i in range(x.size(0)):
            self.test_step_outputs.append({
                "start":   int(start[i].item()),
                "pred":    y_pred_denorm[i],   # (T, C, H, W)
                "target":  y_true_denorm[i], # (T, C, H, W)
            })

    def on_test_epoch_end(self):
        # 1) figure out shapes
        total_months = self.trainer.datamodule.hparams.test_months
        L  = self.trainer.datamodule.block_len
        # assume every entry["pred"] has shape (L, C, H, W)
        _, C, H, W  = self.test_step_outputs[0]["pred"].shape
    
        # 2) allocate accumulators
        sum_preds = np.zeros((total_months, C, H, W), dtype=float)
        sum_trues = np.zeros((total_months, C, H, W), dtype=float)
        count = np.zeros((total_months,), dtype=int)
    
        # 3) fill them
        for entry in self.test_step_outputs:
            s = entry["start"]
            p = entry["pred"]    # (L, C, H, W)
            t = entry["target"]  # (L, C, H, W)
            sum_preds[s:s+L] += p
            sum_trues[s:s+L] += t
            count   [s:s+L] += 1
    
        # 4) compute per‐month averages
        cnt = count[:,None,None,None]
        final_preds = sum_preds / cnt
        final_trues = sum_trues / cnt
    
        self._evaluate_predictions(final_preds, final_trues, is_test=True)
        self._save_kaggle_submission(final_preds)
        self.test_step_outputs.clear()

    def _save_kaggle_submission(self, predictions, suffix=""):
        """
        Create a Kaggle submission file from the model predictions.

        Args:
            predictions (np.ndarray): Predicted values with shape (time, channels, y, x)
        """
        # Get coordinates
        lat_coords, lon_coords = self.trainer.datamodule.get_coords()
        output_vars = self.trainer.datamodule.hparams.output_vars
        n_times = predictions.shape[0]
        time_coords = np.arange(n_times)

        # Convert predictions to Kaggle format
        submission_df = convert_predictions_to_kaggle_format(
            predictions, time_coords, lat_coords, lon_coords, output_vars
        )

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = to_absolute_path(f"submissions/kaggle_submission{suffix}_{timestamp}.csv")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure directory exists
        submission_df.to_csv(filepath, index=False)

        if wandb is not None and isinstance(self.logger, WandbLogger):
            pass
            # Optionally, uncomment the following line to save the submission to the wandb cloud
            # self.logger.experiment.log_artifact(filepath)  # Log to wandb if available

        log.info(f"Kaggle submission saved to {filepath}")
    
    # --- Optimizer with weight decay and linear warm-up
    def configure_optimizers(self):
        base_lr        = self.hparams.learning_rate      
        warmup_epochs  = 3
        max_epochs     = self.trainer.max_epochs
        steps_per_ep   = len(self.trainer.datamodule.train_dataloader())
    
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr = base_lr,
                                      betas = (0.9, 0.999),
                                      weight_decay = 5e-2)
    
        total_steps  = max_epochs * steps_per_ep
        warmup_steps = warmup_epochs * steps_per_ep
    
        def cosine_with_warmup(step: int):
            if step < warmup_steps:                # linear warm‑up
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))   # cosine decay
    
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)
    
        return {
            "optimizer":  optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",              # call every train step
            },
        }
    
# --- Main Execution with Hydra ---
@hydra.main(version_base=None, config_path="configs", config_name="main_config.yaml")
def main(cfg: DictConfig):
    # Optional
    torch.set_float32_matmul_precision('medium')

    # Print resolved configs
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Create data module with parameters from configs
    datamodule = ClimateEmulationDataModule(seed=cfg.seed, **cfg.data)
    model = get_model(cfg)

    # Create lightning module
    lightning_module = ClimateEmulationModule(model, learning_rate=cfg.training.lr)
    
    # Create lightning trainer
    trainer_config = get_trainer_config(cfg, model=model)

    # --- Early‑stopping ---
    early_stop = pl.callbacks.EarlyStopping(
        monitor   = "val/loss",
        mode      = "min",
        patience  = 2,
        verbose   = True,
    )
    
    lr_monitor  = pl.callbacks.LearningRateMonitor(logging_interval="step")
    trainer_config.setdefault("callbacks", [])
    trainer_config["callbacks"] += [early_stop, 
                                   lr_monitor,
                                    ]

    trainer = pl.Trainer(**trainer_config)
    
    if cfg.ckpt_path and isinstance(cfg.ckpt_path, str):
        cfg.ckpt_path = to_absolute_path(cfg.ckpt_path)

    # Train model
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    log.info("Training finished.")

    # Test model
    # IMPORTANT: Please note that the test metrics will be bad because the test targets have been corrupted on the public Kaggle dataset.
    # The purpose of testing below is to generate the Kaggle submission file based on your model's predictions.
    trainer_config["devices"] = 1  # Make sure you test on 1 GPU only to avoid synchronization issues with DDP
    eval_trainer = pl.Trainer(**trainer_config)
    eval_trainer.test(lightning_module, datamodule=datamodule, ckpt_path="best")

    test_metric_equivalence()

    # Save checkpoint
    final_ckpt = to_absolute_path("submissions/model1aaaa.ckpt")
    # 'trainer' here is the same Trainer you used for fit (or eval_trainer)
    #trainer.save_checkpoint(final_ckpt)
    log.info(f"Model checkpoint saved to {final_ckpt}")
    
    if cfg.use_wandb and isinstance(trainer_config.get("logger"), WandbLogger):
        wandb.finish()  # Finish the run if using wandb
    

if __name__ == "__main__":
    main()
