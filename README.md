# Are we certain it's anomalous?
_Alessandro Flaborea, Bardh Prenkaj, Bharti Munjal, Marco Aurelio Sterpa, Dario Aragona, Luca Podo and Fabio Galasso_

<p align="center">
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch-red?logo=pytorch&labelColor=gray"></a>
</p>

The official PyTorch implementation of the **IEEE/CVF CVPR Visual Anomaly and Novelty Detection (VAND) Workshop** paper [**Are we certain it's anomalous?**](https://openaccess.thecvf.com/content/CVPR2023W/VAND/papers/Flaborea_Are_We_Certain_Its_Anomalous_CVPRW_2023_paper.pdf).

Visit our [**webpage**](https://www.pinlab.org/hypad) for more details.

![teaser](assets/teaser_hypad.png)

## Content
```
.
├── assets
│   └── teaser_hypad.png
├── anomaly_detection.py
├── configs
│   └── univariate.yaml
│   └── multivariate.yaml
├── data
│   └── dataset.pickle
│   └── all univariate datasets
├── environment.yml
├── hyperspace
│   ├── hyrnn_nets.py
│   ├── losses.py
│   ├── poincare_distance.py
│   └── utils.py
├── LICENSE
├── main.py
├── math_.py
├── models
│   └── tadgan.py
├── README.md
├── train.py
└── utils
    ├── anomaly_detection_utils.py
    ├── dataloader_multivariate.py
    ├── dataloader.py
    ├── data.py
    └── utils.py
```
## Setup
### Environment
```
conda env create -f environment.yml
conda activate hypad
```

### Datasets
:warning: **UPDATE 21/11/2023** :warning:

The univariate datasets have been added to this repo.

:warning: **UPDATE 04/05/2023** :warning:

At present, the univariate datasets are not available for download. To stay informed about the status of the site, please track this [issue](https://github.com/sintel-dev/Orion/issues/415) for updates.

The YAHOO dataset can be requested here https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70



### **Training+detector** on univariate signals 

To run TadGAN and HypAD, you must pass one of the signals associated with each dataset. 
You can find the list of datasets and signals in data/datasets.pickle.

All the univariate datasets, except for 'SMAP' and 'MSL', use the same set for train and test. 
When train == test, the flag unique_dataset must be set to True. 

To run HypAD:
```
python main.py --config configs/univariate.yaml
```
To run TadGAN just set the parameter **hyperbolic** to False


#### Once trained, you can run the **Detector**

Run the anomaly detector with the config used during trained which is stored in trained_models/models_{eucl/hyper}\_{signal}\_{epochs}_{lr}
```
python anomaly_detection.py --config trained_models/models_{eucl/hyper}_{signal}_{epochs}_{lr}/config.yaml
```
additional flags you can use:
1. How to compute the reconstruction error **rec_error** [dtw/area/point] (dtw by default) (not used with HypAD)
2. How to combine critic_score and reconstruction_error **combination** [mult/sum/rec/critic/uncertainty] (mult by default, uncertainty only for HypAD)


### **Training+detector** on multivariate signals 

To run HypAD:
```
python main.py --config configs/multivariate.yaml
```
To run TadGAN just set the parameter **hyperbolic** to False


#### Once trained, you can run the only **Detector**

Run the anomaly detector with the config used during trained which is stored in trained_models/models_{eucl/hyper}\_{signal}\_{epochs}_{lr}
```
python anomaly_detection.py --config trained_models/models_{eucl/hyper}_{signal}_{epochs}_{lr}/config.yaml
```
list of signals: [fall, weakness, nocturia. moretimeinchair, slowerwalking]

Ask flaborea@di.uniroma1.it for the multivariate dataset.

## Citation
```
@InProceedings{Flaborea_2023_CVPR,
    author    = {Flaborea, Alessandro and Prenkaj, Bardh and Munjal, Bharti and Sterpa, Marco Aurelio and Aragona, Dario and Podo, Luca and Galasso, Fabio},
    title     = {Are We Certain It's Anomalous?},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {2897-2907}
}
```


## Acknowledgements
We build upon [TadGAN](https://arxiv.org/abs/2009.07769).




