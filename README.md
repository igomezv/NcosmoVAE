# NcosmoVAE

[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![arXiv](https://img.shields.io/badge/arXiv-2209.02685-b31b1b.svg)](https://doi.org/10.48550/arXiv.2209.02685)
[![GitHub Repo stars](https://img.shields.io/github/stars/igomezv/NcosmoVAE?style=social)](https://github.com/igomezv/NcosmoVAE)

**`NcosmoVAE`** is a framework to train a Variational Autoencoder with N-body cosmological simulations.

Run with: tensorflow 2.15 and python 3.9.

## Documentation

For an introduction and API documentation, visit [ncosmovae` Docs](https://igomezv.github.io/NcosmoVAE).

## Installation

You can install **`ncosmovae`** directly from the source:

```bash
git clone https://github.com/igomezv/NcosmoVAE
cd NcosmoVAE
pip3 install -e .
```

After installation, you may remove the cloned repository as **`nnogada`** will be installed locally.

## Usage

### Conda Environment Setup

To set up a clean conda environment with the required dependencies:

1. Create the environment from the `environment.yml` file:

```bash
  conda env create -f environment.yml
  ```

2. Activate the environment:
  ```bash
  conda activate ncosmovae_env
  ```

Please check examples included in this repository (`example_1.py`, `example_2.py`).

## Citing `ncosmovae`

If you find **`ncosmovae`** useful in your research, please consider citing [our paper](https://arxiv.org/abs/2209.02685):

```bibtex
@article{ncosmovae2025,
  title={Variational Autoencoder generating realistic N-Body simulations for dark matter halos},
  author={Chacón-Lavanderos, J. and Gómez-Vargas, I. and Menchaca-Mendez, R. and Vázquez, J. A.},
  journal={arxiv},
  volume={0},
  number={0},
  pages={0},
  year={2025},
  url={https://doi.org/10.48550/arXiv.2209.02685}
}
```

## Contributions

Contributions to **`NcosmoVAE`** are very welcome! If you have suggestions for improvements or new features, feel free to create an issue or pull request.


## Project Structure

- **data/**: Sample data files.
- **docs/**: Documentation files. Use `docs_sphinx` for the latest Sphinx documentation.
- **ncosmovae/**: Core library files.
- **outputs/**: Model output and logs.
- **examples files**: Example scripts to demonstrate library usage.
- **requirements.txt**: List of package dependencies.
- **setup.py**, **setup.cfg**: Packaging and installation scripts.

## License

NcosmoVAE is licensed under the MIT license. See the [LICENSE](LICENSE) file for more details.
