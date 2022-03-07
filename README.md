# Venture-Funding-With-Deep-Learning
Creating a binary classification model using a deep neural network with following steps: 

:fast_forward:Preprocess data for a neural network model.

:fast_forward:Use the model-fit-predict pattern to compile and evaluate a binary classification model.

:fast_forward:Optimize the model.

---

## Technologies

You’ll use [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/) and the following  **[python version 3.8.5](https://www.python.org/downloads/)** libraries:


* [pandas](https://pandas.pydata.org/docs/)

* [scikit-learn](https://scikit-learn.org/stable/)
    * [scikit metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) 
    *  [imbalanced-learn](https://imbalanced-learn.org/stable/) 
    *  [linear model](https://scikit-learn.org/stable/modules/linear_model.html)
    * [train test split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) 

    *  [Standard Scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
     [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)

---

## Installation Guide

 ### To check that scikit-learn and hvPlot are installed in your Conda dev environment, complete the following steps:

  ### 1. Activate your Conda dev environment (if it isn’t already) by running the following in your terminal:
```
conda activate dev
```
### 2. When the environment is active, run the following in your terminal to check if the scikit-learn and imbalance-learn libraries are installed on your machine:
```
conda list scikit-learn
conda list imbalanced-learn
```
### If you see scikit-learn and imbalance-learn listed in the terminal, you’re all set!

  ### 1. Install scikit-learn
```
pip install -U scikit-learn
```
### 2. Install imbalance-learn
```
 conda install -c conda-forge imbalanced-learn
```

### 3. Install tensorflow
```
 pip install --upgrade tensorflow
```

---



## Usage

To use this application, simply clone the repository and open jupyter lab from git bash by running the following command:

```jupyter lab```

After launching the application, navigate ``venture_funding_with_deep_learning.ipynb`` notebook in the repository. 

Then in your Jupyter notebook, import the required libraries and dependencies.

```
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder


```




### *Example*


*Original Model Result* 

![original_model](https://user-images.githubusercontent.com/94591580/156973548-e151c41e-b269-43da-b123-b9b37bc3c4b0.png)


*Alternative Model 1 Result*

![model_1](https://user-images.githubusercontent.com/94591580/156973543-9ce78b3d-b7be-4283-a017-66c8dfc4bb92.png)

*Alternative Model 2 Result*

![model_2](https://user-images.githubusercontent.com/94591580/156973546-7950532b-3356-4fa7-9888-797bc7f838bb.png)

*Alternative Model 3 Result*

![model_3](https://user-images.githubusercontent.com/94591580/156973547-77819b88-a0a6-485a-940b-861b576bc9b5.png)
---

## Contributors

[Nguyen Dao](https://www.linkedin.com/in/nguyen-dao-a55669215/)

daosynguyen21@gmail.com


---

## License

MIT
