
# Semi-Supervised Deep Regression with Uncertainty Consistency and Variational Model Ensembling via Bayesian Neural Networks



This is the implementation of UCVME for the paper ["Semi-Supervised Deep Regression with Uncertainty Consistency and Variational Model Ensembling via Bayesian Neural Networks"]().

![UCVME](intro_aaai_v4.PNG)

<br />
<br />

## Data

Researchers can get the UTKFace dataset from https://susanqq.github.io/UTKFace/ (Aligned&Cropped Faces). Extract the zip file and set up the files according to the example files in DATA_DIR/UTKFace

```
DATA_DIR
 |_ FileList.csv
 |_ UTKFace
    |_ 1_0_0_20161219140623097.jpg.chip.jpg
    |_ 1_0_0_20161219140627985.jpg.chip.jpg
    |_ 1_0_0_20161219140642920.jpg.chip.jpg
    ...
```




<br />
<br />

## Environment

It is recommended to use PyTorch `conda` environments for running the program. A requirements file has been included. 

### Docker Setup (Recommended)

The easiest way to run this codebase is using Docker. The repository includes a Dockerfile and docker-compose configuration.

#### Prerequisites
- Docker (version 20.10 or later)
- Docker Compose (version 1.29 or later)
- NVIDIA Docker runtime (for GPU support, optional)

#### Building the Docker Image

```bash
# Build using docker-compose
docker-compose build

# Or build directly with docker
docker build -t ucvme:latest .
```

#### Running with Docker Compose

```bash
# Start the container
docker-compose up -d

# Access the container shell
docker-compose exec ucvme bash

# Run training inside the container
python3 ucvme_age.py --output=/workspace/output

# Run testing
python3 ucvme_age.py --output=/workspace/output --test_only

# Stop the container
docker-compose down
```

#### Running with Docker directly

```bash
# Build the image
docker build -t ucvme:latest .

# Run the container (CPU only)
docker run -it --rm \
  -v $(pwd)/DATA_DIR:/workspace/DATA_DIR \
  -v $(pwd)/output:/workspace/output \
  ucvme:latest bash

# Run with GPU support (requires nvidia-docker)
docker run -it --rm \
  --gpus all \
  -v $(pwd)/DATA_DIR:/workspace/DATA_DIR \
  -v $(pwd)/output:/workspace/output \
  ucvme:latest bash
```

#### GPU Support

To enable GPU support in docker-compose, uncomment the GPU-related lines in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Or use the `runtime: nvidia` option for older docker-compose versions.

#### Notes
- The `DATA_DIR` and `output` directories are mounted as volumes, so your data and results persist outside the container
- Make sure your data is organized in `DATA_DIR/UTKFace/` as described in the Data section
- The container uses PyTorch 1.11.0 with CUDA 11.3 support

<br />
<br />

## Training and testing


### To perform training, run:

```
python3 ucvme_age.py --output=<OUTPUT_DIR> 
```


### To perform testing only, run:

```
python3 ucvme_age.py --output=<OUTPUT_DIR> --test_only
```


<br />
<br />



## Pretrained models

Trained checkpoints and models for the 10% labeled dataset setting can be downloaded from:
https://hkustconnect-my.sharepoint.com/:f:/g/personal/wdaiaj_connect_ust_hk/Epq-44XUV_lIoUe7IdkZo44B6vBgiqGIxo6tgCxMQsU48A?e=Uf3GQu 




To run with the pretrained model weights, replace the `.pts` files in the target output directory with the downloaded files. 

<br />

|  Experiments         | MAE   | R<sup>2</sup>   |
| ---------- | :-----------:  | :-----------: |
| 10% labeled dataset    | 5.26  &plusmn; 0.02  | 57.9%	&plusmn;  0.3 |

<br />
<br />

## Notes
* Contact: DAI Weihang (wdai03@gmail.com)
<br />
<br />

## Citation
If this code is useful for your research, please consider citing:


```
@article{dai2023semi,
  title={Semi-Supervised Deep Regression with Uncertainty Consistency and Variational Model Ensembling via Bayesian Neural Networks},
  author={Dai, Weihang and Li, Xiaomeng and Cheng, Kwang-Ting},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={6},
  pages={7304--7313},
  year={2023}
}

```