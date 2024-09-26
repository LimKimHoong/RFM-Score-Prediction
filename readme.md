## RFM Score Prediction for Customer Transaction History in Bank

**Author** : [Lim Kim Hoong](https://github.com/LimKimHoong) ([kimhoong0324@gmail.com](mailto:kimhoong0324@gmail.com))

**Achievement:** Successfully developed and implemented an RFM (Recency, Frequency, Monetary) score prediction model utilizing machine learning techniques. The project effectively optimizes customer segmentation and enhances forecasting of future customer behavior. Achieved accurate predictions through a **Random Forest Regressor** with an **MSE of 0.0005** on testing datasets, leveraging customer transaction data to generate actionable insights for improving targeted marketing strategies.

**Keywords:** *RFM Score, Prediction, Customer Segmentation, Random Forest Regressor*

## How to Run

First, ensure you have [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html) installed and set up the project environment.

```shell
mkvirtualenv RFM_Score_Prediction
pip install -e 'git+https://github.com/LimKimHoong/RFM-Score-Prediction.git'
```

> For a private repository accessible only through SSH authentication, replace `git+https://github.com` with `git+ssh://git@github.com`.

## How to Contribute

To contribute, make sure you have [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html) installed and set up the project in development mode.

```shell
mkvirtualenv RFM_Score_Prediction
git clone https://github.com/LimKimHoong/RFM-Score-Prediction.git
cd RFM_Score_Prediction
pip install -r requirements.txt
pip install -e .
pip freeze | grep -v RFM_Score_Prediction > requirements.txt
```

> For a private repository accessible only through SSH authentication, replace `https://github.com/` with `git@github.com:`.

Then, create or select a GitHub branch and contribute your changes!