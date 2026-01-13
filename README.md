# Efficient Scaling

This aims to be a repository to have quick methodologies to build and fit scaling laws, given data.

The methods here are heavily based on Chinchilla ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)), and its three approaches.

For each of the approaches, we expect a dataframe with specific column names that represent certain entities. 

Typically, for the most general methodology, we expect a flattened data (usually a `pd.DataFrame`) with repeated rows for columns such as model sizes, hyperparameters, or broader training types.

For each approach, we expect suitable filtering and subsetting of this data, conditioned on the approach being used for the scaling law fit.

As a *minimal requirement*, we need a `parameters (N)` column, a `tokens/samples (D)` column, along with the final loss per `(N, D)`.


## Installation
To install the required packages, you can use pip. It is recommended to create a virtual environment first.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

Then, install the required packages:
```bash
python -m pip install -e .
```

## Download Data

Before downloading the data, create a `data/` directory in the root of the repository and navigate into it. 
```bash
mkdir data
cd  data
```

### Porian et al. data
To download the data from Porian et al., you can use the following command:
```
wget https://raw.githubusercontent.com/formll/resolving-scaling-law-discrepancies/main/data/experiment_results.pickle.xz -O porian_results.pickle.xz
```

It can be read in python as follows:
```python
import pandas as pd
porian_df = pd.read_pickle(
    "porian_experiments_results.pickle.xz",
    compression='xz'
)
```

### (Mis)Fitting Scaling Laws Data
To download the (Mis)Fitting Scaling Laws data, you can use the following command:
```bash
wget https://raw.githubusercontent.com/hadasah/scaling_laws/master/data/scaling_results.csv -O misfitting_results.csv
```

It can be read in python as follows:

```python
import pandas as pd

misfitting_df = pd.read_csv(
    "scaling_results.csv",
)
```

### Warmstarting Data
To download the warmstarting data, you can use the following command:
```bash
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1vr5Jq1TpTpnkb5CX5mcu2wWQQzbf-bmd' -O warmstarting_results.parquet
```
It can be read in python as follows:
```python
import pandas as pd
warmstarting_df = pd.read_parquet(
    "warmstarting_data.parquet",
)
```


