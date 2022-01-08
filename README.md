# HOWTO run TadGAN & HypAD
 
### **Training+detector** on univariate signals 

To run TadGAN and HypAD, you must pass one of the signals associated with each dataset. 
You can find the list of datasets and signals in data/datasets.pickle.

All the univariate datasets, except for 'SMAP' and 'MSL', use the same set for train and test. 
When train == test, the flag --unique_dataset must be enabled. 

A TadGAN example:
```
python main.py --dataset 'MSL' --signal 'D-15' --epochs=40 --lr=0.0005
```
To try HypAD just add the flag --hyperbolic:
```
python main.py --dataset 'original' --signal 'D-15' --epochs=40 --lr=0.0005 --hyperbolic
```

The YAHOO dataset can be requested here https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70

#### Once trained, you can run the **Detector**
add the flag --hyperbolic if you want to test the HypAD model you've trained
```
python anomaly_detection.py --dataset 'M-6' --signal 'D-15' --epochs=40 --lr=0.0005
```
additional flags you can use:
1. How to compute the reconstruction error --rec_error [dtw/area/point] (dtw by default) (not used with HypAD)
2. How to combine critic_score and reconstruction_errror --combination [mult/sum/rec/critic/uncertainty] (mult by default, uncertainty only for HypAD)

### **Training+detector** on multivariate signals 
A TadGAN example:
```
python main.py --dataset 'new_CASAS' --signal 'fall' --epochs=30 --lr=0.0005 --signal_shape=150
```
HypAD:
```
python main.py --dataset 'new_CASAS' --signal 'fall' --epochs=30 --lr=0.0005 --signal_shape=150
```

#### Once trained, you can run the only **Detector**
add the flag --hyperbolic if you want to use the hyperspace train
```
python anomaly_detection.py --dataset 'new_CASAS' --signal 'fall' --signal_shape=150
```
list of signals: [fall, weakness, nocturia. moretimeinchair, slowerwalking]
Ask flaborea@di.uniroma1.it for the multivariate dataset.
