
# Machine Learning Model that predicts flight delays
This is a binary classification model that predicts whether a flight will arrive on-time or late ("binary" because there are only two possible outputs).
<br> It utilizes:
- Pandas to clean and prepare data
- Scikit-learn to build a machine-learning model
- Matplotlib to visualize the results
<br>It uses one Scikit-learn's RandomForestClassifier, which fits multiple decision trees to the data and uses averaging to boost the overall accuracy and limit overfitting. It is one of the several classifers in Scikit-learn for implementing common machine learning models.

**Import all the stuff we need**


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
```

**Import dataset**


```python
#This is a Linux Bash command that downloads an CSV file from Azure Blob Storage and saves it using given name
!curl https://topcs.blob.core.windows.net/public/FlightData.csv -o flightdata.csv
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
      0     0    0     0    0     0      0      0 --:--:--  0:00:01 --:--:--     0
      0     0    0     0    0     0      0      0 --:--:--  0:00:02 --:--:--     0
      0     0    0     0    0     0      0      0 --:--:--  0:00:03 --:--:--     0
      3 1552k    3 48746    0     0  11262      0  0:02:21  0:00:04  0:02:17 11426
      8 1552k    8  127k    0     0  24309      0  0:01:05  0:00:05  0:01:00 25967
     12 1552k   12  191k    0     0  31083      0  0:00:51  0:00:06  0:00:45 39741
     14 1552k   14  223k    0     0  31378      0  0:00:50  0:00:07  0:00:43 45794
     18 1552k   18  287k    0     0  35363      0  0:00:44  0:00:08  0:00:36 60597
     22 1552k   22  351k    0     0  38726      0  0:00:41  0:00:09  0:00:32 62647
     25 1552k   25  399k    0     0  39206      0  0:00:40  0:00:10  0:00:30 55023
     28 1552k   28  447k    0     0  39856      0  0:00:39  0:00:11  0:00:28 50528
     31 1552k   31  495k    0     0  41166      0  0:00:38  0:00:12  0:00:26 55362
     36 1552k   36  559k    0     0  42252      0  0:00:37  0:00:13  0:00:24 53215
     39 1552k   39  607k    0     0  43473      0  0:00:36  0:00:14  0:00:22 52271
     42 1552k   42  655k    0     0  43709      0  0:00:36  0:00:15  0:00:21 53259
     45 1552k   45  703k    0     0  43999      0  0:00:36  0:00:16  0:00:20 53773
     48 1552k   48  751k    0     0  44495      0  0:00:35  0:00:17  0:00:18 52755
     48 1552k   48  751k    0     0  41992      0  0:00:37  0:00:18  0:00:19 41252
     48 1552k   48  751k    0     0  39789      0  0:00:39  0:00:19  0:00:20 29309
     48 1552k   48  751k    0     0  37833      0  0:00:42  0:00:20  0:00:22 19723
     48 1552k   48  751k    0     0  36060      0  0:00:44  0:00:21  0:00:23  9893
     48 1552k   48  751k    0     0  34422      0  0:00:46  0:00:22  0:00:24     0
     48 1552k   48  751k    0     0  32948      0  0:00:48  0:00:23  0:00:25     0
     48 1552k   48  751k    0     0  31575      0  0:00:50  0:00:24  0:00:26     0
     48 1552k   48  751k    0     0  30330      0  0:00:52  0:00:25  0:00:27     0
     48 1552k   48  751k    0     0  29180      0  0:00:54  0:00:26  0:00:28     0
     48 1552k   48  751k    0     0  28114      0  0:00:56  0:00:27  0:00:29     0
     48 1552k   48  751k    0     0  27109      0  0:00:58  0:00:28  0:00:30     0
     48 1552k   48  751k    0     0  26172      0  0:01:00  0:00:29  0:00:31     0
     48 1552k   48  751k    0     0  25312      0  0:01:02  0:00:30  0:00:32     0
     48 1552k   48  751k    0     0  24506      0  0:01:04  0:00:31  0:00:33     0
     48 1552k   48  751k    0     0  23749      0  0:01:06  0:00:32  0:00:34     0
     48 1552k   48  751k    0     0  23039      0  0:01:09  0:00:33  0:00:36     0
     48 1552k   48  751k    0     0  22369      0  0:01:11  0:00:34  0:00:37     0
     48 1552k   48  751k    0     0  21718      0  0:01:13  0:00:35  0:00:38     0
     48 1552k   48  751k    0     0  21122      0  0:01:15  0:00:36  0:00:39     0
     48 1552k   48  751k    0     0  20653      0  0:01:16  0:00:37  0:00:39     0
    curl: (56) Send failure: Connection was reset
    

**Load Dataset**


```python


#Create a Panda DataFrame - a two-dimensional labeled data structure
df = pd.read_csv('flightdata.csv')
df.head() # The DataFrame head function only returns the first five rows
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>QUARTER</th>
      <th>MONTH</th>
      <th>DAY_OF_MONTH</th>
      <th>DAY_OF_WEEK</th>
      <th>UNIQUE_CARRIER</th>
      <th>TAIL_NUM</th>
      <th>FL_NUM</th>
      <th>ORIGIN_AIRPORT_ID</th>
      <th>ORIGIN</th>
      <th>...</th>
      <th>CRS_ARR_TIME</th>
      <th>ARR_TIME</th>
      <th>ARR_DELAY</th>
      <th>ARR_DEL15</th>
      <th>CANCELLED</th>
      <th>DIVERTED</th>
      <th>CRS_ELAPSED_TIME</th>
      <th>ACTUAL_ELAPSED_TIME</th>
      <th>DISTANCE</th>
      <th>Unnamed: 25</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>DL</td>
      <td>N836DN</td>
      <td>1399</td>
      <td>10397</td>
      <td>ATL</td>
      <td>...</td>
      <td>2143</td>
      <td>2102.0</td>
      <td>-41.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>338.0</td>
      <td>295.0</td>
      <td>2182.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>DL</td>
      <td>N964DN</td>
      <td>1476</td>
      <td>11433</td>
      <td>DTW</td>
      <td>...</td>
      <td>1435</td>
      <td>1439.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>110.0</td>
      <td>115.0</td>
      <td>528.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>DL</td>
      <td>N813DN</td>
      <td>1597</td>
      <td>10397</td>
      <td>ATL</td>
      <td>...</td>
      <td>1215</td>
      <td>1142.0</td>
      <td>-33.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>335.0</td>
      <td>300.0</td>
      <td>2182.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>DL</td>
      <td>N587NW</td>
      <td>1768</td>
      <td>14747</td>
      <td>SEA</td>
      <td>...</td>
      <td>1335</td>
      <td>1345.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>196.0</td>
      <td>205.0</td>
      <td>1399.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>DL</td>
      <td>N836DN</td>
      <td>1823</td>
      <td>14747</td>
      <td>SEA</td>
      <td>...</td>
      <td>607</td>
      <td>615.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>247.0</td>
      <td>259.0</td>
      <td>1927.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
#gives (rows,columns) count
df.shape 
```




    (5442, 26)



## Clean and Prepare the data


```python
#Check for any missing values, if true then there are null values 
df.isnull().values.any()


```




    True




```python
#Function finds the mising values and shows number of missing values in each column 
df.isnull().sum()

```




    YEAR                      0
    QUARTER                   0
    MONTH                     0
    DAY_OF_MONTH              0
    DAY_OF_WEEK               0
    UNIQUE_CARRIER            0
    TAIL_NUM                  0
    FL_NUM                    0
    ORIGIN_AIRPORT_ID         0
    ORIGIN                    0
    DEST_AIRPORT_ID           0
    DEST                      0
    CRS_DEP_TIME              0
    DEP_TIME                 43
    DEP_DELAY                43
    DEP_DEL15                43
    CRS_ARR_TIME              0
    ARR_TIME                 46
    ARR_DELAY                83
    ARR_DEL15                83
    CANCELLED                 0
    DIVERTED                  0
    CRS_ELAPSED_TIME          1
    ACTUAL_ELAPSED_TIME      84
    DISTANCE                  1
    Unnamed: 25            5442
    dtype: int64




```python
#Filter out all the irrelevant columns so that you are only left with relevant features

#Fuction will remove the 26th column which has 11231 missing values because 
#it only contains commas usually found at the end of every line of CSV
df.drop('Unnamed: 25',axis=1)

#Filter out all the other columns so that you are only left with those relevant to the predictive model
df = df[["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "ORIGIN", "DEST", "CRS_DEP_TIME", "ARR_DEL15"]]
df.isnull().sum() #Check if they have been removed
```




    MONTH            0
    DAY_OF_MONTH     0
    DAY_OF_WEEK      0
    ORIGIN           0
    DEST             0
    CRS_DEP_TIME     0
    ARR_DEL15       83
    dtype: int64




```python
df[df.isnull().values.any(axis=1)].head()
#The null(Nan) values in ARR_DEL15 correspond to flights that were diverted or canceled
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MONTH</th>
      <th>DAY_OF_MONTH</th>
      <th>DAY_OF_WEEK</th>
      <th>ORIGIN</th>
      <th>DEST</th>
      <th>CRS_DEP_TIME</th>
      <th>ARR_DEL15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>177</th>
      <td>1</td>
      <td>9</td>
      <td>6</td>
      <td>MSP</td>
      <td>SEA</td>
      <td>701</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>179</th>
      <td>1</td>
      <td>10</td>
      <td>7</td>
      <td>MSP</td>
      <td>DTW</td>
      <td>1348</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>184</th>
      <td>1</td>
      <td>10</td>
      <td>7</td>
      <td>MSP</td>
      <td>DTW</td>
      <td>625</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>210</th>
      <td>1</td>
      <td>10</td>
      <td>7</td>
      <td>DTW</td>
      <td>MSP</td>
      <td>1200</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>478</th>
      <td>1</td>
      <td>22</td>
      <td>5</td>
      <td>SEA</td>
      <td>JFK</td>
      <td>2305</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#The fillna method fills all the null values with one to indicate taht there late by more than 15 minutes
df = df.fillna({'ARR_DEL15': 1})
df.iloc[177:185]

#The missing values have been replaced and the list of columns has been narrowed to those most relevant to the model
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MONTH</th>
      <th>DAY_OF_MONTH</th>
      <th>DAY_OF_WEEK</th>
      <th>ORIGIN</th>
      <th>DEST</th>
      <th>CRS_DEP_TIME</th>
      <th>ARR_DEL15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>177</th>
      <td>1</td>
      <td>9</td>
      <td>6</td>
      <td>MSP</td>
      <td>SEA</td>
      <td>701</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>178</th>
      <td>1</td>
      <td>9</td>
      <td>6</td>
      <td>DTW</td>
      <td>JFK</td>
      <td>1527</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>179</th>
      <td>1</td>
      <td>10</td>
      <td>7</td>
      <td>MSP</td>
      <td>DTW</td>
      <td>1348</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>180</th>
      <td>1</td>
      <td>10</td>
      <td>7</td>
      <td>DTW</td>
      <td>MSP</td>
      <td>1540</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>181</th>
      <td>1</td>
      <td>10</td>
      <td>7</td>
      <td>JFK</td>
      <td>ATL</td>
      <td>1325</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>182</th>
      <td>1</td>
      <td>10</td>
      <td>7</td>
      <td>JFK</td>
      <td>ATL</td>
      <td>610</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>183</th>
      <td>1</td>
      <td>10</td>
      <td>7</td>
      <td>JFK</td>
      <td>SEA</td>
      <td>1615</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>184</th>
      <td>1</td>
      <td>10</td>
      <td>7</td>
      <td>MSP</td>
      <td>DTW</td>
      <td>625</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#The CRS_DEP_TIME column contains 500 unique values...
#Performing binning or quantization leaves a maximum of 24 discrete values 

for index, row in df.iterrows():
    df.loc[index, 'CRS_DEP_TIME'] = math.floor(row['CRS_DEP_TIME'] / 100)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MONTH</th>
      <th>DAY_OF_MONTH</th>
      <th>DAY_OF_WEEK</th>
      <th>ORIGIN</th>
      <th>DEST</th>
      <th>CRS_DEP_TIME</th>
      <th>ARR_DEL15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>ATL</td>
      <td>SEA</td>
      <td>19</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>DTW</td>
      <td>MSP</td>
      <td>13</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>ATL</td>
      <td>SEA</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>SEA</td>
      <td>MSP</td>
      <td>8</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>SEA</td>
      <td>DTW</td>
      <td>23</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Generate indicator columns from the ORIGIN and DEST columns, while dropping the ORIGIN and DEST columns themselves
df = pd.get_dummies(df, columns=['ORIGIN', 'DEST'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MONTH</th>
      <th>DAY_OF_MONTH</th>
      <th>DAY_OF_WEEK</th>
      <th>CRS_DEP_TIME</th>
      <th>ARR_DEL15</th>
      <th>ORIGIN_ATL</th>
      <th>ORIGIN_DTW</th>
      <th>ORIGIN_JFK</th>
      <th>ORIGIN_MSP</th>
      <th>ORIGIN_SEA</th>
      <th>DEST_ATL</th>
      <th>DEST_DTW</th>
      <th>DEST_JFK</th>
      <th>DEST_MSP</th>
      <th>DEST_SEA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>19</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>13</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>9</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>8</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>23</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Build Machine Learning Model
Before training the model with the Random Forest classifier we:
- Split the DataFrame into a training set containing 80% of the original data, and a test set containing the remaining 20% using the sklearn train_test_split helper function. 
- Separate the DataFrame into input feature columns and output label columns.


```python
#Split the the dataset into two, that is, training and testing set.
#Separate the feature columns and label columns. 
train_x, test_x, train_y, test_y = train_test_split(df.drop('ARR_DEL15', axis=1), df['ARR_DEL15'], test_size=0.2, random_state=42)
```


```python
#Create a RandomForestClassifier object and train it by calling the fit method.
#The default values of the classifier can be overriden when creating the RandomForestClassifier object.
model = RandomForestClassifier(random_state=13)
model.fit(train_x, train_y)
```

    C:\Users\Sandra\Anaconda3\lib\site-packages\sklearn\ensemble\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
                oob_score=False, random_state=13, verbose=0, warm_start=False)




```python
#Call the predict method to test the model
predicted = model.predict(test_x)

#Determine the mean accuracy of the model using the score method
model.score(test_x, test_y)
```




    0.8833792470156107



## Determing accuracy further
The **mean accuracy** is 86%(calculated above) isn't always a reliable indicator of the accuracy of a classification model. 
<br>The score method reflects how many flights were predicted correctly. This score is skewed by the fact that our dataset contains many more rows representing on-time arrivals than rows representing late arrivals.
<br>A better overall measure for a binary classification model is **Area Under Receiver Operating Characteristic Curve** (sometimes referred to as "ROC AUC"), which essentially quantifies how often the model will make a correct prediction regardless of the outcome.
<br>Before computing the ROC AUC, we must generate prediction probabilities for the test set. These probabilities are estimates for each of the classes, or answers, the model can predict.
<br>Other measures of accuracy for a classification model include **precision and recall**. We'll use Scikit-learn  precision_score and recall_score methods for computing precision and recall respectively.


```python
#Generate a set of prediction probabilities from the test data
probabilities = model.predict_proba(test_x)

print(probabilities[0:6])
```

    [[1.  0. ]
     [0.9 0.1]
     [0.9 0.1]
     [1.  0. ]
     [1.  0. ]
     [0.8 0.2]]
    


```python
#Generate an ROC AUC score from the probabilities 
print("ROC AUC Score: {} %".format(100*roc_auc_score(test_y, probabilities[:, 1])))

#Quantify the precision of the model
train_predictions = model.predict(train_x)
print("Precision: {} %".format(100*precision_score(test_y, predicted)))

#Measure recall
print("Recall: {} %".format(100*recall_score(test_y, predicted)))
```

    ROC AUC Score: 66.49251132009752 %
    Precision: 58.620689655172406 %
    Recall: 12.878787878787879 %
    

##  The Model's Behavior
A confusion matrix, also known as an error matrix quantifies the number of false positives, false negatives, true positives, and true negatives. Simply, it quantifies the number of times each answer was classified correctly or incorrectly. 


```python
#Generate a confusion matrix for your model
print("Confusion Matrix:")
confusion_matrix = confusion_matrix(test_y, predicted)
print(confusion_matrix)

#Plot confusion matrix
LABELS = ["ARRIVED ON TIME",
         "ARRIVED LATE" ]
width = 12
height = 12
plt.figure(figsize=(width, height))
plt.imshow(
    confusion_matrix,
    interpolation='nearest',
    cmap=plt.cm.rainbow
)
plt.title("Confusion Matrix", fontsize=39)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, LABELS, fontsize=26, rotation=90)
plt.yticks(tick_marks, LABELS, fontsize=26)
plt.tight_layout()
plt.ylabel('True label', fontsize=19)
plt.xlabel('Predicted label', fontsize=19)
plt.show()

#Note : If you get an error when you run this cell...
#Make sure to rerun the cell where we import the libraries then rerun this ceel
```

    Confusion Matrix:
    [[945  12]
     [115  17]]
    


![png](output_24_1.png)


## Visualize Output of Model
Note that at the beginning of the notebook we configured the notebook to support inline Matplotlib output.


```python
#Configure Seaborn to enhance the output from Matplotlib.
sns.set()

#Plot the ROC curve for the model
fpr, tpr, _ = roc_curve(test_y, probabilities[:, 1])
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
```

The dotted line in the middle of the graph represents a 50-50 chance of obtaining a correct answer while blue curve represents the accuracy of our model.


```python
#Function that calls the model 
def predict_delay(departure_date_time, origin, destination):
    from datetime import datetime

    try:
        departure_date_time_parsed = datetime.strptime(departure_date_time, '%d/%m/%Y %H:%M:%S')
    except ValueError as e:
        return 'Error parsing date/time - {}'.format(e)

    month = departure_date_time_parsed.month
    day = departure_date_time_parsed.day
    day_of_week = departure_date_time_parsed.isoweekday()
    hour = departure_date_time_parsed.hour

    origin = origin.upper()
    destination = destination.upper()

    input = [{'MONTH': month,
              'DAY': day,
              'DAY_OF_WEEK': day_of_week,
              'CRS_DEP_TIME': hour,
              'ORIGIN_ATL': 1 if origin == 'ATL' else 0,
              'ORIGIN_DTW': 1 if origin == 'DTW' else 0,
              'ORIGIN_JFK': 1 if origin == 'JFK' else 0,
              'ORIGIN_MSP': 1 if origin == 'MSP' else 0,
              'ORIGIN_SEA': 1 if origin == 'SEA' else 0,
              'DEST_ATL': 1 if destination == 'ATL' else 0,
              'DEST_DTW': 1 if destination == 'DTW' else 0,
              'DEST_JFK': 1 if destination == 'JFK' else 0,
              'DEST_MSP': 1 if destination == 'MSP' else 0,
              'DEST_SEA': 1 if destination == 'SEA' else 0 }]

    return model.predict_proba(pd.DataFrame(input))[0][0]
```

This function takes as input a date (dd/mm/year) and time, an origin airport code, and a destination airport code, and returns a value between 0.0-1.0 indicating the probability that the flight will arrive at its destination on time.

**Now let's analyze some flights**


```python
predict_delay('1/10/2018 21:45:00', 'JFK', 'ATL')

```




    0.8




```python
predict_delay('2/10/2018 21:45:00', 'JFK', 'ATL')
```




    0.8




```python
predict_delay('2/10/2018 10:00:00', 'ATL', 'SEA')
```




    0.7



### Some more vizualization
Let's plot the probability of on-time arrivals for an evening flight from JFK to ATL over a range of days


```python
labels = ('Oct 1', 'Oct 2', 'Oct 3', 'Oct 4', 'Oct 5', 'Oct 6', 'Oct 7')
values = (predict_delay('1/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('2/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('3/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('4/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('5/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('6/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('7/10/2018 21:45:00', 'JFK', 'ATL'))
alabels = np.arange(len(labels))

plt.bar(alabels, values, align='center', alpha=0.5)
plt.xticks(alabels, labels)
plt.ylabel('Probability of On-Time Arrival')
plt.ylim((0.0, 1.0))
```




    (0.0, 1.0)




![png](output_35_1.png)


You can modify the code to produce a similar chart for different flights

