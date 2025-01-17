# Benchmark time series data sets for PyTorch (lightning)

<!-- [![PyPi](https://img.shields.io/pypi/v/torchtime)](https://pypi.org/project/torchtime) -->
<!-- [![Build status](https://img.shields.io/github/workflow/status/philipdarke/torchtime/build.svg)](https://github.com/philipdarke/torchtime/actions/workflows/build.yml) -->
<!-- ![Coverage](https://philipdarke.com/torchtime/assets/coverage-badge.svg?dummy=8484744) -->

[![License](https://img.shields.io/github/license/mauricekraus/torchtime.svg)](https://github.com/mauricekraus/torchtime/blob/main/LICENSE)

<!-- [![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2207.12503-blue)](https://doi.org/10.48550/arXiv.2207.12503) -->

## Notice

This is a fork of [torchtime](https://github.com/philipdarke/torchtime) which doesn't seem to be maintained anymore. Therefore I forked this repository and try to add new features from time to time.

### New Features

- PyTorch 2.0 ready
- Python <= 3.10 support
- UEA/UCR Lightning Module for ease of use.

### Roadmap

- PyPi release
- License Update
- PhysioNet Challenge Lightning Module
- Add new tests
- Include one hot encoding
- Update Docs

PyTorch data sets for supervised time series classification and prediction problems, including:

- All UEA/UCR classification repository data sets
- PhysioNet Challenge 2012 (in-hospital mortality)
- PhysioNet Challenge 2019 (sepsis prediction)
- A binary prediction variant of the 2019 PhysioNet Challenge

## Why use `torchtime`?

1. Saves time. You don't have to write your own PyTorch data classes.
2. Better research. Use common, reproducible implementations of data sets for a level playing field when evaluating models.

## Installation

This installs the torchtime 0.6.1 (PyTorch 2.0 ready)

```bash
$ pip install git+https://github.com/mauricekraus/torchtime
```

To install torchtime with PyTorch < 2.0 support please install the tag torch@v1 (0.6.0)

## Getting started

Data classes have a common API. The `split` argument determines whether training ("_train_"), validation ("_val_") or test ("_test_") data are returned. The size of the splits are controlled with the `train_prop` and (optional) `val_prop` arguments.

### PhysioNet data sets

Three [PhysioNet](https://physionet.org/) data sets are currently supported:

- [`torchtime.data.PhysioNet2012`](https://philipdarke.com/torchtime/api/data.html#torchtime.data.PhysioNet2012) returns the 2012 challenge (in-hospital mortality) [[link]](https://physionet.org/content/challenge-2012/1.0.0/).
- [`torchtime.data.PhysioNet2019`](https://philipdarke.com/torchtime/api/data.html#torchtime.data.PhysioNet2019) returns the 2019 challenge (sepsis prediction) [[link]](https://physionet.org/content/challenge-2019/1.0.0/).
- [`torchtime.data.PhysioNet2019Binary`](https://philipdarke.com/torchtime/api/data.html#torchtime.data.PhysioNet2019Binary) returns a binary prediction variant of the 2019 challenge.

For example, to load training data for the 2012 challenge with a 70/30% training/validation split and create a [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for model training:

```python
from torch.utils.data import DataLoader
from torchtime.data import PhysioNet2012

physionet2012 = PhysioNet2012(
    split="train",
    train_prop=0.7,
)
dataloader = DataLoader(physionet2012, batch_size=32)
```

### UEA/UCR repository data sets

The [`torchtime.data.UEA`](https://philipdarke.com/torchtime/api/data.html#torchtime.data.UEA) class returns the [UEA/UCR repository](https://www.timeseriesclassification.com/) data set specified by the `dataset` argument, for example:

```python
from torch.utils.data import DataLoader
from torchtime.data import UEA

arrowhead = UEA(
    dataset="ArrowHead",
    split="train",
    train_prop=0.7,
)
dataloader = DataLoader(arrowhead, batch_size=32)
```

### Using the DataLoader

Batches are dictionaries of tensors `X`, `y` and `length`:

- `X` are the time series data. The package follows the _batch first_ convention therefore `X` has shape (_n_, _s_, _c_) where _n_ is batch size, _s_ is (longest) trajectory length and _c_ is the number of channels. By default, the first channel is a time stamp.
- `y` are one-hot encoded labels of shape (_n_, _l_) where _l_ is the number of classes.
- `length` are the length of each trajectory (before padding if sequences are of irregular length) i.e. a tensor of shape (_n_).

For example, ArrowHead is a univariate time series therefore `X` has two channels, the time stamp followed by the time series (_c_ = 2). Each series has 251 observations (_s_ = 251) and there are three classes (_l_ = 3). For a batch size of 32:

```python
next_batch = next(iter(dataloader))
next_batch["X"].shape       # torch.Size([32, 251, 2])
next_batch["y"].shape       # torch.Size([32, 3])
next_batch["length"].shape  # torch.Size([32])
```

See [Using DataLoaders](https://philipdarke.com/torchtime/tutorials/getting_started.html#using-dataloaders) for more information.

## Advanced options

- Missing data can be imputed by setting `impute` to _mean_ (replace with training data channel means) or _forward_ (replace with previous observation). Alternatively a custom imputation function can be passed to the `impute` argument.
- A time stamp (added by default), missing data mask and the time since previous observation can be appended with the boolean arguments `time`, `mask` and `delta` respectively.
- Time series data are standardised using the `standardise` boolean argument.
- The location of cached data can be changed with the `path` argument, for example to share a single cache location across projects.
- For reproducibility, an optional random `seed` can be specified.
- Missing data can be simulated using the `missing` argument to drop data at random from UEA/UCR data sets.

See the [tutorials](https://philipdarke.com/torchtime/tutorials/) and [API](https://philipdarke.com/torchtime/api/) for more information.

## Other resources

If you're looking for the TensorFlow equivalent for PhysioNet data sets try [medical_ts_datasets](https://github.com/ExpectationMax/medical_ts_datasets).

## Acknowledgements

`torchtime` uses some of the data processing ideas in Kidger et al, 2020 [[1]](https://arxiv.org/abs/2005.08926) and Che et al, 2018 [[2]](https://doi.org/10.1038/s41598-018-24271-9).

This work is supported by the Engineering and Physical Sciences Research Council, Centre for Doctoral Training in Cloud Computing for Big Data, Newcastle University (grant number EP/L015358/1).

## Citing `torchtime`

If you use this software, please cite the [paper](https://doi.org/10.48550/arXiv.2207.12503):

```
@software{darke_torchtime_2022,
    author = Darke, Philip and Missier, Paolo and Bacardit, Jaume,
    title = "Benchmark time series data sets for {PyTorch} - the torchtime package",
    month = July,
    year = 2022,
    publisher={arXiv},
    doi = 10.48550/arXiv.2207.12503,
    url = https://doi.org/10.48550/arXiv.2207.12503,
}
```

DOIs are also available for each version of the package [here](https://doi.org/10.5281/zenodo.6402406).

## References

1. Kidger, P, Morrill, J, Foster, J, _et al_. Neural Controlled Differential Equations for Irregular Time Series. _arXiv_ 2005.08926 (2020). [[arXiv]](https://arxiv.org/abs/2005.08926)

1. Che, Z, Purushotham, S, Cho, K, _et al_. Recurrent Neural Networks for Multivariate Time Series with Missing Values. _Sci Rep_ 8, 6085 (2018). [[doi]](https://doi.org/10.1038/s41598-018-24271-9)

1. Silva, I, Moody, G, Scott, DJ, _et al_. Predicting In-Hospital Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012. _Comput Cardiol_ 2012;39:245-248 (2010). [[hdl]](http://hdl.handle.net/1721.1/93166)

1. Reyna, M, Josef, C, Jeter, R, _et al_. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge. _Critical Care Medicine_ 48 2: 210-217 (2019). [[doi]](https://doi.org/10.1097/CCM.0000000000004145)

1. Reyna, M, Josef, C, Jeter, R, _et al_. Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019 (version 1.0.0). _PhysioNet_ (2019). [[doi]](https://doi.org/10.13026/v64v-d857)

1. Goldberger, A, Amaral, L, Glass, L, _et al_. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. _Circulation_ 101 (23), pp. e215–e220 (2000). [[doi]](https://doi.org/10.1161/01.cir.101.23.e215)

1. Löning, M, Bagnall, A, Ganesh, S, _et al_. sktime: A Unified Interface for Machine Learning with Time Series. _Workshop on Systems for ML at NeurIPS 2019_ (2019). [[doi]](https://doi.org/10.5281/zenodo.3970852)

1. Löning, M, Bagnall, A, Middlehurst, M, _et al_. alan-turing-institute/sktime: v0.10.1 (v0.10.1). _Zenodo_ (2022). [[doi]](https://doi.org/10.5281/zenodo.6191159)

## License

Released under the MIT license.
