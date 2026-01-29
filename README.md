# Estimating distributions in DDSP systems: applications to FM synthesis and audio effects estimation

Audio examples can be found on the [paper companion page](https://peladeaucome.github.io/DAFx_params_distrib-taslp/).


## How to use

There are models programmed for two tasks in this repository:
 - Estimation of FM synthesis parameters
 - Estimation of mastering audio effects

When downloading the repo, the needed libraries can be installed with the `requirements.txt`:
```bash
pip install -r requirements.txt
```

### FM synthesis estimation

Those models are trained using the `train_ddx7_56.py` file.

Examples of the commands used to train the various models can be found in the `_batch_files/ddx7_56.sh` file.

To train the Deterministic model:
```bash
python train_ddx7_56.py model=1bandeq/deter hydra=deter
```

In order to train one of the probabilistic models:
```bash
python train_ddx7_56.py model=1bandeq/infer hydra=infer
```
You then can chose between the various models we proposed in the paper. For instance, for the Gauss-L2 model:
```bash
python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.flow.length=2 model.distrib.type=gaussian_log
```

If you want to train the MoG-Unif-K24-L1 model:
```bash
python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=24 model.flow.length=1 model.distrib.type=unif
```

Or the Mog-Full-K6-L1 model:
```bash
python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=24 model.flow.length=1 model.distrib.type=unif
```

#### Simulated annealing

All the commands above were for the models trained *with* simulated annealing.
If you want to train a model without, you need to add the `model.beta.start=0.005` option.


### Audio effects estimation

The audio effects estimation experiment is launched using similar commands, such as:
```
python train_mastering.py hydra=deter model.name=deter
```
This is used to train the deterministic model

#### Evaluation

The models are evaluated using the `test_mastering.py` file.

# Citing this work

This repo was published when submitting a scientific paper (under review).
Please cite it if you use any of this code in a reseach project:
```bibtex
@misc{peladeauEstimatingDistributionsDDSP2026,
  title = {Estimating Distributions in {{DDSP}} Systems: Applications to {{FM}} Synthesis and Audio Effects Estimation},
  author = {Peladeau, Côme and Fourer, Dominique and Peeters, Geoffroy},
  date = {2026-01},
  pubstate = {prepublished},
  note = {Submitted to IEEE TASLP}
}
```