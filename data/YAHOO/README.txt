Dataset: ydata-labeled-time-series-anomalies-v1_0

Yahoo! Synthetic and real time-series with labeled anomalies, version 1.0

=====================================================================
This dataset is provided as part of the Yahoo! Webscope program, to be
used for approved non-commercial research purposes by recipients who 
have signed a Data Sharing Agreement with Yahoo!. This dataset is not
to be redistributed. No personally identifying information is available
in this dataset. More information about the Yahoo! Webscope program is
available at http://research.yahoo.com

=====================================================================

Full description:

This dataset contains 371 files:

A1Benchmark/real_(int).csv
A2Benchmark/synthetic_(int).csv
A3Benchmark/A3Benchmark-TS(int).csv
A4Benchmark/A4Benchmark-TS(int).csv

A1Benchmark is based on the real production traffic to some of the Yahoo! properties.
The other 3 benchmarks are based on synthetic time-series. A2 and A3 Benchmarks include outliers,
while the A4Benchmark includes change-point anomalies. The bechmarks based on real-data have property
and geos removed. Fields in each data file are delimited with (",") characters.

The content of the six files are as follows:

=====================================================================
(1) "A[1-2]Benchmark/[real,synthetic]_(int).csv" contains real and synthetic time-series with labeled anomalies.
    The synthetic data set contains time-series with random seasonality, trend and noise. The outliers in the
    synthetic dataset are inserted at random positions. Note that the timestamps of the A1Benchmark are replaced by 
    integers with the increment of 1, where each data-point represents 1 hour worth of data. The anomalies in A1Benchmark 
    are marked by humans and therefore may not be consistent, therefore during benchmarking A1Benchmark is best used to measure 
    recall.

    The fields are:
    
    0 timestamp
    1 value
    2 is_anomaly
    
    The is_anomaly field is a boolean indicating if the current value at a given timestamp is considered an anomaly.

Snippet:
  1,83,0
  2,605,0
  3,181,0
  4,37,0
  5,45,1

=====================================================================
(2) "A[3-4]Benchmark/A[3-4]Benchmark-TS(int).csv" contain synthetic time-series. 
     The A3Benchmark only contains outliers while the A4Benchmark also contains the anomalies
     that are marked as change-points. The synthetic time-series have varying noise and trends 
     with three pre-specified seasonalities. The anomalies in the synthetic time-series are inserted at random positions.
     The fields are:

    0 timestamps: the UNIX timestamp marks every hour (the data is hourly sampled)
    1 value: the value of time series at this timestamp
    2 anomaly: 1 if this stamp is an outlier
    3 changepoint: 1 if this stamp is a change point
    4 trend: the additive trend value for this timestamp 
    5 noise: the additive noise value for this timestamp
    6 seasonality1: the 12-hour seasonality value
    7 seasonality2: the daily seasonality value
    8 seasonality3: the weekly seasonality value
   
Snippet:

1422237600,4333.43325915382,0,0,4599,1.81512268926974,-190.958601534458,-128.864580083128,52.4413180821374
1422241200,4316.14322293657,0,0,4602,-14.6572208523525,-220.5,-105.217489040563,54.5179328294825
1422244800,4403.20006523115,0,0,4605,7.04036744752875,-190.958601534463,-74.3999999999969,56.5182993180795
1422248400,4531.20632084718,0,0,4608,13.5289749039305,-110.250000000009,-38.5122739112586,58.4396198545142
1422252000,4967.50678185938,1,0,4911,-3.7724254388993,-6.91288625182698e-12,-2.33251127952801e-12,60.279207
