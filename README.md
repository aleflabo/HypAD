# TadGAN
 
### **Training+detector** on univariate signals 

```
python main.py --dataset 'original' --signal 'D-15' --epochs=20 --model 'tadgan' --lr=0.0005
```
with hyperbolic:
```
python main.py --dataset 'original' --signal 'D-15' --epochs=20 --model 'tadgan' --lr=0.005 --hyperbolic
```

The signals that are already present in the orion DB are here: https://sintel.dev/Orion/user_guides/data.html

#### Once trained, you can run the only **Detector**
add the flag --hyperbolic if you want to use the hyperspace train
```
python anomaly_detection.py --dataset 'original' --signal 'M-1' --model 'tadgan'
```

### **Training+detector** on multivariate signals 

```
python main.py --dataset 'new_CASAS' --signal 'fall' --epochs=10 --model 'tadgan' --lr=0.0005 --signal_shape=150
```
with hyperbolic:
```
python main.py --dataset 'new_CASAS' --signal 'fall' --epochs=10 --model 'tadgan' --lr=0.0005 --signal_shape=150
```

#### Once trained, you can run the only **Detector**
add the flag --hyperbolic if you want to use the hyperspace train
```
python anomaly_detection.py --dataset 'new_CASAS' --signal 'fall' --model 'tadgan'
```
list of signals: [fall, weakness, nocturia. moretimeinchair, slowerwalking]
