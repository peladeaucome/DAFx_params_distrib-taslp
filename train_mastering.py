import AEAFX
import AEAFX.models as models
import torch
import torch.utils.data
import lightning.pytorch as pl
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import os
import yaml

# torch.set_float32_matmul_precision("medium")

accelerator = "gpu"

if accelerator == "gpu":
    device = "cuda:0"
elif accelerator == "cpu":
    device = "cpu"

samplerate = 16000

from mastering_fx import get_anaFx, get_synFx


def get_dataset(
    name: str,
    synFx,
    Fx_norm=None,
    seed: int = 0,
    splits: list = [0.8, 0.1, 0.1],
    audio_length: int = 4,
):

    generator = torch.Generator().manual_seed(seed)
    if name.lower() == "millionsong":
        full_ds = AEAFX.data.MillionSong_2(
            ds_path="/home/ids/peladeau/Data/millionsong-50k",
            Fx=synFx,
            Fx_norm=Fx_norm,
            samplerate=samplerate,
            len_s=audio_length,
        )
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            full_ds, lengths=splits, generator=generator
        )
    if name.lower() == "medleydb":
        full_ds = AEAFX.data.MedleyDBLoaded_Dataset(
            root_dir="/tsi/mir/MedleyDB",
            Fx=synFx,
            samplerate=samplerate,
            audio_length_s=audio_length,
        )
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            full_ds, lengths=splits, generator=generator
        )
    if name.lower() == "msm":
        full_ds = AEAFX.data.MSMastering_Dataset(
            root_dir="/home/ids/peladeau/Data/mixing_secrets_mastering",
            audio_length=audio_length,
            samplerate=samplerate,
        )
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            full_ds, lengths=splits, generator=generator
        )
    if name.lower() == "msm+medleydb":
        full_ds = AEAFX.data.MSMastering_Dataset(
            root_dir="/home/ids/peladeau/Data/mixing_secrets_mastering",
            audio_length=audio_length,
            samplerate=samplerate,
        )
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            full_ds, lengths=splits, generator=generator
        )
        train_dataset = torch.utils.data.ConcatDataset(
            (
                train_dataset,
                AEAFX.data.MedleyDBLoaded_Dataset(
                    root_dir="/tsi/mir/MedleyDB",
                    Fx=synFx,
                    samplerate=samplerate,
                    audio_length_s=audio_length,
                ),
            )
        )

    # train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    #     full_ds, lengths=splits, generator=generator
    # )

    if name.lower() != "millionsong":
        num_cat = 40_000 // len(train_dataset)
        train_dataset = torch.utils.data.ConcatDataset(
            (train_dataset for _ in range(num_cat))
        )
        num_cat = 5_000 // len(valid_dataset)
        valid_dataset = torch.utils.data.ConcatDataset(
            (valid_dataset for _ in range(num_cat))
        )
        test_dataset = torch.utils.data.ConcatDataset((test_dataset for _ in range(20)))
    return train_dataset, valid_dataset, test_dataset


def get_model(cfg: dict, metrics_dict=None, audio_loss=None):
    model_name: str = cfg["model"]["name"]

    fx = get_anaFx(cfg["experiment"]["fx"]["ana"])
    synFx = get_synFx(cfg["experiment"]["fx"]["syn"])
    model_cfg = cfg
    if model_name == "synthperm":
        model = AEAFX.models.SynthPerm(
            fx=AEAFX.ddafx.DSPFX(synFx),
            frontend_args=model_cfg["model"]["frontend"],
            metrics_dict=metrics_dict,
            audio_loss_fn=audio_loss,
            mlp_depth=model_cfg["model"]["mlp"]["depth"],
            mlp_size=model_cfg["model"]["mlp"]["size"],
            mlp_type=model_cfg["model"]["mlp"]["type"],
            mlp_bn=model_cfg["model"]["mlp"]["bn"],
            vector_field_args=model_cfg["model"]["vector_field"],
            learning_rate=model_cfg["model"]["learning_rate"],
            minibatch_ot=model_cfg["model"]["minibatch_ot"],
        )
    elif model_name == "ssl":
        model = AEAFX.models.FX_SSL(
            frontend_args=model_cfg["model"]["frontend"],
            fx=fx,
            mlp_depth=model_cfg["model"]["mlp"]["depth"],
            mlp_size=model_cfg["model"]["mlp"]["size"],
            mlp_type=model_cfg["model"]["mlp"]["type"],
            learning_rate=model_cfg["model"]["learning_rate"],
            lr_sched_patience=model_cfg["experiment"]["lr_sched_patience"],
            metrics_dict=metrics_dict,
            loss_fn=audio_loss,
            params_loss_weight=1,
        )
    else:
        model = AEAFX.models.get_model(
            model_name=model_name.lower(),
            fx=fx,
            frontend_args=model_cfg["model"]["frontend"],
            metrics_dict=metrics_dict,
            loss_fn=audio_loss,
            start_beta=model_cfg["model"]["beta"]["start"],
            end_beta=cfg["model"]["beta"]["end"],
            context_size=model_cfg["model"]["flow"]["context_size"],
            mlp_depth=model_cfg["model"]["mlp"]["depth"],
            mlp_size=model_cfg["model"]["mlp"]["size"],
            mlp_type=model_cfg["model"]["mlp"]["type"],
            mlp_bn=model_cfg["model"]["mlp"]["bn"],
            flow_length=model_cfg["model"]["flow"]["length"],
            flow_layers_type=model_cfg["model"]["flow"]["layers"],
            flow_nl=model_cfg["model"]["flow"]["nl"]["name"],
            flow_nl_knots=model_cfg["model"]["flow"]["nl"]["knots"],
            flow_coupling=model_cfg["model"]["flow"]["coupling"],
            warmup_length=model_cfg["model"]["warmup_length"],
            num_mixtures=model_cfg["model"]["distrib"]["num_mixtures"],
            base_entropy=model_cfg["model"]["distrib"]["entropy"],
            distrib_type=cfg["model"]["distrib"]["type"],
            learning_rate=model_cfg["model"]["learning_rate"],
            lr_sched_patience=model_cfg["experiment"]["lr_sched_patience"],
            weight_decay=model_cfg["experiment"]["weight_decay"],
            estimation_or_usage=model_cfg["model"]["mode"],
            optim_only_flow=model_cfg["model"]["start_from_ae"],
        )
    return model


@hydra.main(version_base=None, config_path="conf", config_name="mastering")
def main(cfg: DictConfig) -> None:
    verbose = True
    print(OmegaConf.to_yaml(cfg))
    print(HydraConfig.get().job.name)

    output_dir = HydraConfig.get().runtime.output_dir

    fx = get_anaFx(cfg["experiment"]["fx"]["ana"])
    synFx = get_synFx(cfg["experiment"]["fx"]["syn"])

    if cfg["experiment"]["data"]["fx_norm"] is not None:
        fx_norm = AEAFX.data.fx.DAFx_Series(samplerate=44100)
        if "spectral" in cfg["experiment"]["data"]["fx_norm"]:
            fx_norm.append_multiple(
                AEAFX.data.fx.equalizer.ConstantPeak(2248, -0.169, 6.31, samplerate),
                AEAFX.data.fx.equalizer.ConstantPeak(3165, 0.263, 1.45, samplerate),
                AEAFX.data.fx.equalizer.ConstantHighShelf(
                    6889, -2.191, 3.73, samplerate
                ),
            )
        if "dynamic" in cfg["experiment"]["data"]["fx_norm"]:
            fx_norm.append(
                AEAFX.data.fx.compressor.DynamicAdjustment(
                    tshort_att=0.001,
                    tshort_rel=0.1,
                    time_ratio=10,
                    comp_ratio=0.714898,
                    samplerate=samplerate,
                )
            )
    else:
        fx_norm = None

    if verbose:
        print("Getting loss function")

    # mrstft = AEAFX.loss.MR_STFT_Loss(
    #     n_ffts=[1024, 512, 256],
    #     hop_lengths=[512, 256, 128],
    #     window_sizes=[1024, 512, 256],
    #     samplerate=samplerate,
    # ).to(device)

    mrstft_revisited = AEAFX.loss.MR_STFT_Revisited_Norm().to(device)

    mel_loss = AEAFX.loss.NormalizedLogMel_Loss(
        sr=samplerate, n_fft=4096, win_length=None, n_mels=128, hop_length=1024
    ).to(device)

    ldr_loss = AEAFX.loss.LDRLoss(long_t=1, short_t=0.05, samplerate=samplerate)

    sum_loss = AEAFX.loss.SumLosses(weights=[1, 1], loss_fns=[mel_loss, ldr_loss])

    if cfg["model"]["loss"] == "mel":
        audio_loss = mel_loss
    elif cfg["model"]["loss"] == "ldr":
        audio_loss = ldr_loss
    elif cfg["model"]["loss"] == "sum":
        audio_loss = sum_loss

    metrics_dict = {
        # "MR-STFT": mrstft,
        "MR-STFT rev": mrstft_revisited,
        "Mel": mel_loss,
        "LDR": ldr_loss,
        # "MSE": torch.nn.MSELoss(),
        "SI-SDR": AEAFX.loss.si_sdr,
        # "pimse": AEAFX.loss.pimse,
    }

    if verbose:
        print("Getting the neural network")

    model_name: str = cfg["model"]["name"]

    model_cfg = cfg

    model = get_model(cfg, metrics_dict, audio_loss)

    if cfg["resume_training"] is not None:
        model.load_state_dict(
            torch.load(
                os.path.join(cfg["resume_training"], "best.pt"),
                weights_only=True,
            ),
            strict=False,
        )

    if verbose:
        print(fx.ranges_parameters)
    if verbose:
        print("Getting datasets")

    if verbose:
        print("Getting dataloaders")

    train_dataset, valid_dataset, test_dataset = get_dataset(
        name=cfg["experiment"]["data"]["name"],
        synFx=synFx,
        splits=cfg["experiment"]["data"]["splits"],
        seed=cfg["experiment"]["data"]["seed"],
        audio_length=cfg["experiment"]["data"]["audio_length"],
        Fx_norm=fx_norm,
    )

    if cfg["experiment"]["data"]["name"].lower() == "msm":
        np.save(
            os.path.join(output_dir, "train_split"),
            np.array(train_dataset.datasets[0].indices),
        )
        np.save(
            os.path.join(output_dir, "valid_split"),
            np.array(valid_dataset.datasets[0].indices),
        )
        np.save(
            os.path.join(output_dir, "test_split"),
            np.array(test_dataset.datasets[0].indices),
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["experiment"]["batch_size"],
        shuffle=True,
        num_workers=8,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
    )

    if verbose:
        print("Setting the training")

    logger = pl.loggers.TensorBoardLogger(output_dir)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        filename="best",
        monitor="loss_total/valid",
        mode="min",
        save_last=True,
    )

    earlystop = EarlyStopping(
        monitor="loss_total/valid",
        mode="min",
        patience=cfg["experiment"]["early_stopping_patience"],
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_callback, earlystop, lr_monitor]

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        deterministic=False,
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=1,
        max_epochs=cfg["experiment"]["max_epochs"],
        enable_progress_bar=cfg["experiment"]["progress_bar"],
        callbacks=callbacks,
    )

    if verbose:
        print("Training")

    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
    )

    torch.save(model.state_dict(), os.path.join(output_dir, "last.pt"))

    trainer.test(model, test_loader, ckpt_path="last")
    trainer.test(model, test_loader, ckpt_path="best")

    model = model.__class__.load_from_checkpoint(
        os.path.join(output_dir, "lightning_logs/version_0/checkpoints/last.ckpt"),
        fx=fx,
        strict=False,
    )
    torch.save(model.state_dict(), os.path.join(output_dir, "last.pt"))
    model = model.__class__.load_from_checkpoint(
        os.path.join(output_dir, "lightning_logs/version_0/checkpoints/best.ckpt"),
        fx=fx,
        strict=False,
    )
    torch.save(model.state_dict(), os.path.join(output_dir, "best.pt"))


if __name__ == "__main__":
    main()
