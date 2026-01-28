import torch
import torch.utils.data
from torch import Tensor
import AEAFX

import lightning.pytorch as pl

from mastering_fx import get_anaFx, get_synFx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import os

import yaml
import train_mastering
import torch.nn as nn

from scipy.special import gamma


cm = 1 / 2.53


def dB20_np(x: np.ndarray) -> np.ndarray:
    return 20 * np.log(np.abs(x))


samplerate = 16_000
device = "cuda:0"


def nball_volume(d: int, R: float = 1.0):
    num = np.float_power(np.pi, d / 2) * np.power(R, d)
    den = gamma(d / 2 + 1)
    return num / den


def MMD_kernel(x: Tensor, y: Tensor, C: float = 1) -> Tensor:
    return C / (C + (x - y).square().sum(-1))


class RandomModel(AEAFX.models.BEAFX):
    def __init__(self, fx: AEAFX.ddafx.DDAFX, loss_fn):
        super().__init__(fx, loss_fn)
        self.fx = fx
        self.params_dim = self.fx.num_parameters

    def get_FXParams(self, y: Tensor, num_samples: int = 1):
        num_params = self.params_dim
        bs = y.size(0)

        if num_samples == 1:
            z = torch.rand((bs, num_params), device=y.device)
        else:
            z = torch.rand((bs, num_samples, num_params), device=y.device)

        return z

    def get_KLentropy_MMD_estimates(self, y, K):
        bs = y.size(0)
        d = self.params_dim

        z = self.get_FXParams(y, num_samples=K)

        ####################################
        # Computing the entropy using the Kozachenko Leonenko estimator

        z0 = z.unsqueeze(1)
        z1 = z.unsqueeze(2)
        R, _ = (z0 - z1).square().sum(3).sqrt().sort(dim=2)
        R = R[:, :, 1]
        Y = K * torch.pow(R, d)
        H_KL = torch.mean(torch.log(Y), dim=1) + 0.577 + np.log(nball_volume(d))

        ###################################
        # Computing the MMD
        C = 0.5 * d
        z_prior = torch.rand_like(z)
        mask = torch.ones(K, K, device=y.device) - torch.eye(K, K, device=y.device)
        mask = mask.unsqueeze(0) / (K * (K - 1))

        MMD = -2 * MMD_kernel(z.unsqueeze(1), z_prior.unsqueeze(2), C).mean((1, 2))

        temp = MMD_kernel(z.unsqueeze(1), z.unsqueeze(2), C)
        temp = temp * mask
        MMD = MMD + temp.sum((1, 2))

        temp = MMD_kernel(z_prior.unsqueeze(1), z_prior.unsqueeze(2), C)
        temp = temp * mask
        MMD = MMD + temp.sum((1, 2))

        return H_KL, MMD


class NoModel(AEAFX.models.BEAFX):
    def __init__(self, fx: AEAFX.ddafx.DDAFX, loss_fn):
        super().__init__(fx, loss_fn)
        self.fx = fx
        self.params_dim = self.fx.num_parameters


synFx = get_synFx()
anaFx = get_anaFx()

full_ds = AEAFX.data.MillionSong_2(
    ds_path="/home/ids/peladeau/Data/millionsong-50k",
    Fx=synFx,
    samplerate=samplerate,
    len_s=5,
    Fx_norm=None,
)

_, _, test_dataset = torch.utils.data.random_split(
    full_ds, lengths=[0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(0)
)

num_sets = 1
big_test_set = torch.utils.data.ConcatDataset((test_dataset for _ in range(num_sets)))

test_loader = torch.utils.data.DataLoader(big_test_set, batch_size=64, num_workers=16)


mrstft = AEAFX.loss.MR_STFT_Revisited_Norm(
    n_ffts=[1024, 512, 256],
    hop_lengths=[512, 256, 128],
    window_sizes=[1024, 512, 256],
    samplerate=44100,
).to("cuda:0")
mrstft_revisited = AEAFX.loss.MR_STFT_Revisited_Norm().to("cuda:0")
mel_loss = AEAFX.loss.NormalizedLogMel_Loss(
    sr=22050, n_fft=4096, win_length=None, n_mels=128, hop_length=1024
).to("cuda:0")
sisdr = AEAFX.loss.SISDR()
ldr_loss = AEAFX.loss.LDRLoss(long_t=1, short_t=0.05, samplerate=samplerate)
twof = AEAFX.loss.two_f_Model(samplerate=samplerate)

base_path = "logs/mastering/safe/strong_fx"


def get_model_from_path(path: str, exp_name: str, audio_loss=None, metrics_dict=None):
    full_path = os.path.join(path, exp_name)
    with open(full_path + ".yaml", "r") as f:
        cfg = yaml.safe_load(f)
    model: nn.Module = train_mastering.get_model(cfg)
    model.load_state_dict(torch.load(full_path + ".pt"), strict=False)
    return model


def compute_losses(
    tested_model,
    test_loader,
    metrics_dict: dict,
    nbest: int = 1,
    device: torch.device = device,
):
    with torch.no_grad():
        tested_model = tested_model.to(device).eval()
        results_dict = {}
        H_MC_list = None
        H_KL_list = None
        MMD_list = None
        anaFx = tested_model.fx
        for i, batch in enumerate(iter(test_loader)):
            x, y, v = batch
            x: Tensor = x.to(device)
            y: Tensor = y.to(device)
            v: Tensor = v.to(device)

            if isinstance(tested_model, AEAFX.models.FX_AE):
                tested_model: AEAFX.models.FX_AE
                z = tested_model.get_FXParams(y)
                yhat = anaFx(x, z)
                for m_key, metric in metrics_dict.items():
                    loss_batch: Tensor = metric(y, yhat)
                    loss_batch = loss_batch.cpu().numpy()
                    if m_key in results_dict.keys():
                        loss_previous: np.ndarray = results_dict[m_key]
                        loss_full = np.concatenate((loss_batch, loss_previous), axis=0)
                    else:
                        loss_full = loss_batch
                    results_dict[m_key] = loss_full
            if isinstance(tested_model, AEAFX.models.FX_Inference):
                tested_model: AEAFX.models.FX_Inference

                bs = x.size(0)
                dim = tested_model.params_dim

                zT, entropy = tested_model.get_FXParams_and_entropy(
                    y, num_samples=nbest
                )
                if nbest != 1:
                    zT = zT.flatten(0, 1)

                bs, _, N = x.size()

                x = x.expand(bs, nbest, N).flatten(0, 1).unsqueeze(1)
                y = y.expand(bs, nbest, N).flatten(0, 1).unsqueeze(1)

                yhat = anaFx(x, zT)

                for m_key, metric in metrics_dict.items():
                    loss_batch: Tensor = metric(y, yhat)
                    loss_batch = loss_batch.unflatten(0, (bs, nbest)).amin(dim=1)

                    loss_batch = loss_batch.cpu().numpy()

                    if m_key in results_dict.keys():
                        loss_previous: np.ndarray = results_dict[m_key]
                        loss_full = np.concatenate((loss_batch, loss_previous), axis=0)
                    else:
                        loss_full = loss_batch
                    results_dict[m_key] = loss_full

                entropy = entropy.cpu().numpy()

                if H_MC_list is None:
                    H_MC_list = entropy
                else:
                    H_MC_list = np.concatenate((H_MC_list, entropy), axis=0)

                results_dict["H_MC"] = H_MC_list

                if nbest == 1:
                    H_KL, MMD, _ = tested_model.get_KLentropy_MMD_estimates(y=y, K=100)
                    MMD = MMD.cpu().numpy()
                    H_KL = H_KL.cpu().numpy()
                    if MMD_list is None:
                        MMD_list = MMD
                    else:
                        MMD_list = np.concatenate((MMD_list, MMD), axis=0)
                    results_dict["MMD"] = MMD_list
                    if H_KL_list is None:
                        H_KL_list = H_KL
                    else:
                        H_KL_list = np.concatenate((H_KL_list, H_KL), axis=0)
                    results_dict["H_KL"] = H_KL_list

                    ### Computing results with the most likely params
                    bs = x.size(0)
                    zT = tested_model.get_FXParams_most_likely(y, K=10000)
                    yhat = anaFx(x, zT)

                    for m_key, metric in metrics_dict.items():
                        m_key = m_key + " most_likely"
                        loss_batch: Tensor = metric(y, yhat)

                        loss_batch = loss_batch.cpu().numpy()

                        if m_key in results_dict.keys():
                            loss_previous: np.ndarray = results_dict[m_key]
                            loss_full = np.concatenate(
                                (loss_batch, loss_previous), axis=0
                            )
                        else:
                            loss_full = loss_batch
                        results_dict[m_key] = loss_full

            if isinstance(tested_model, AEAFX.models.SynthPerm):
                tested_model: AEAFX.models.SynthPerm

                bs = x.size(0)
                dim = tested_model.params_dim

                z = tested_model.get_FXParams(y, num_steps=100, num_samples=nbest)
                bs, _, N = x.size()
                x = x.expand(bs, nbest, N).flatten(0, 1).unsqueeze(1)
                y = y.expand(bs, nbest, N).flatten(0, 1).unsqueeze(1)
                if nbest > 1:
                    z = z.flatten(0, 1)
                yhat = tested_model.fx(x, z)

                for m_key, metric in metrics_dict.items():
                    loss_batch: Tensor = metric(y, yhat)
                    loss_batch = loss_batch.unflatten(0, (bs, nbest)).amin(dim=1)

                    loss_batch = loss_batch.cpu().numpy()

                    if m_key in results_dict.keys():
                        loss_previous: np.ndarray = results_dict[m_key]
                        loss_full = np.concatenate((loss_batch, loss_previous), axis=0)
                    else:
                        loss_full = loss_batch
                    results_dict[m_key] = loss_full

                if nbest == 1:
                    H_KL, MMD = tested_model.get_KLentropy_MMD_estimates(y=y, K=100)
                    MMD = MMD.cpu().numpy()
                    H_KL = H_KL.cpu().numpy()
                    if MMD_list is None:
                        MMD_list = MMD
                    else:
                        MMD_list = np.concatenate((MMD_list, MMD), axis=0)
                    results_dict["MMD"] = MMD_list
                    if H_KL_list is None:
                        H_KL_list = H_KL
                    else:
                        H_KL_list = np.concatenate((H_KL_list, H_KL), axis=0)
                    results_dict["H_KL"] = H_KL_list

            if isinstance(tested_model, RandomModel):
                tested_model: RandomModel

                bs = x.size(0)
                dim = tested_model.params_dim

                z = tested_model.get_FXParams(y, num_samples=nbest)
                bs, _, N = x.size()
                x = x.expand(bs, nbest, N).flatten(0, 1).unsqueeze(1)
                y = y.expand(bs, nbest, N).flatten(0, 1).unsqueeze(1)
                if nbest > 1:
                    z = z.flatten(0, 1)
                yhat = tested_model.fx(x, z)

                for m_key, metric in metrics_dict.items():
                    loss_batch: Tensor = metric(y, yhat)
                    loss_batch = loss_batch.unflatten(0, (bs, nbest)).amin(dim=1)

                    loss_batch = loss_batch.cpu().numpy()

                    if m_key in results_dict.keys():
                        loss_previous: np.ndarray = results_dict[m_key]
                        loss_full = np.concatenate((loss_batch, loss_previous), axis=0)
                    else:
                        loss_full = loss_batch
                    results_dict[m_key] = loss_full

                if nbest == 1:
                    H_KL, MMD = tested_model.get_KLentropy_MMD_estimates(y=y, K=100)
                    MMD = MMD.cpu().numpy()
                    H_KL = H_KL.cpu().numpy()
                    if MMD_list is None:
                        MMD_list = MMD
                    else:
                        MMD_list = np.concatenate((MMD_list, MMD), axis=0)
                    results_dict["MMD"] = MMD_list
                    if H_KL_list is None:
                        H_KL_list = H_KL
                    else:
                        H_KL_list = np.concatenate((H_KL_list, H_KL), axis=0)
                    results_dict["H_KL"] = H_KL_list
            if isinstance(tested_model, NoModel):
                tested_model: NoModel

                bs, _, N = x.size()
                x = x.expand(bs, nbest, N).flatten(0, 1).unsqueeze(1)
                yhat = x
                y = y.expand(bs, nbest, N).flatten(0, 1).unsqueeze(1)
                for m_key, metric in metrics_dict.items():
                    loss_batch: Tensor = metric(y, yhat)
                    loss_batch = loss_batch.unflatten(0, (bs, nbest)).amin(dim=1)

                    loss_batch = loss_batch.cpu().numpy()

                    if m_key in results_dict.keys():
                        loss_previous: np.ndarray = results_dict[m_key]
                        loss_full = np.concatenate((loss_batch, loss_previous), axis=0)
                    else:
                        loss_full = loss_batch
                    results_dict[m_key] = loss_full
    return results_dict


# nbest=2


sisdr = AEAFX.loss.Neg_SISDR()
metrics_dict = {
    # "MR-STFT": mrstft,
    "Mel": mel_loss,
    "MR-STFT rev": mrstft_revisited,
    "SI-SDR": sisdr,
    "LDR": ldr_loss,
    "2f-model": twof
}
models_name_list = [
    # "deter",
    # "infer-gauss-flow0-sched",
    # "infer-gauss-flow1-sched",
    # "infer-gauss-flow2-sched",
    # "infer-mog-full-6-flow1-sched",
    # "infer-mog-unif-6-flow1-sched",
    # "infer-mog-unif-24-flow1-sched",
    # "infer-gauss-flow0-nosched",
    # "infer-gauss-flow1-nosched",
    # "infer-gauss-flow2-nosched",
    # "infer-mog-full-6-flow1-nosched",
    # "infer-mog-unif-6-flow1-nosched",
    # "infer-mog-unif-24-flow1-nosched",
    "mog-vae"
    # "random",
    # "none"
]


for model_name in models_name_list:
    print(model_name)
    if model_name not in ("random", "none"):
        full_model_path = os.path.join(base_path, model_name + ".ckpt")
        model = get_model_from_path(base_path, model_name)
        print(sum(p.numel() for p in model.parameters()))
    elif model_name == "none":
        model = NoModel(anaFx, mel_loss)
    else:
        model = RandomModel(anaFx, mel_loss)

    results_dict = compute_losses(model, test_loader, metrics_dict, 1, device)

    save_path = os.path.normpath(f"np_save/mastering/MSD/nbest_1/{model_name}.npz")
    np.savez_compressed(save_path, **results_dict)

print("\nNbest\n")

models_name_list = [
    # "deter",
    "infer-gauss-flow2-sched",
    "random",
    # "vae"
]

# metrics_dict = {
#     # "MR-STFT": mrstft,
#     "Mel": mel_loss,
#     "MR-STFT rev": mrstft_revisited,
#     "SI-SDR": sisdr,
#     "LDR": ldr_loss,
#     # "2f-model": twof
# }

# (1, 2, 3, 4, 5, 10)

# for nbest in (2, 3, 4, 5, 10):
#     for model_name in models_name_list:
#         print(model_name)
#         if model_name not in ("random", "none"):
#             full_model_path = os.path.join(base_path, model_name + ".ckpt")
#             model = get_model_from_path(base_path, model_name)
#             print(sum(p.numel() for p in model.parameters()))
#         elif model_name == "none":
#             model = NoModel(anaFx, mel_loss)
#         else:
#             model = RandomModel(anaFx, mel_loss)

#         results_dict = compute_losses(model, test_loader, metrics_dict, nbest, device)

#         save_path = os.path.normpath(
#             f"np_save/mastering/MSD/nbest_{nbest}/{model_name}.npz"
#         )
#         np.savez_compressed(save_path, **results_dict)

print("End.")
