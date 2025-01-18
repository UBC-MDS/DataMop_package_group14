# datamop

`datamop` is a data cleaning and wrangling package designed to streamline the preprocessing of datasets. Whether you meet missing values, inconsistent categorical columns or need scaling for numeric columns when dealing with data. `datamop` provides a simple and consistent solution to automate and simplify these repetitive tasks. 
The following are core functions of this package:

* `sweep_nulls()`: Handle missing values such as imputation or removal, based on user preference.

* `column_encoder()`: Encodes categorical columns using either one-hot encoding or ordinal encoding, based on user preference.

* `column_scaler()`: Scales numerical columns, including Min-Max scaling and Z-score standardization, based on user preference.

`datamop` fits into Python data preprocessing ecosystem by offering a more lightweight and user-friendly alternative to complex libraries like `pandas`, `scikit-learn`. `datamop` focuses specifically on handling missing values, encoding categorical columns and normalizing numerical columns. `datamop` changes `scikit-learn` tasks performed by modules like `SimpleImputer`, `OneHotEncoder`, `OrdinalEncoder` and `StandardScaler` with fewer steps and easier customization.
Similar functionality can be found in:

* **pandas** (`fillna()`, etc.): [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/)

* **scikit-learn** (`SimpleImputer`, `OneHotEncoder`, `LabelEncoder`, `MinMaxScaler`, etc.): [scikit-learn preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

## Contributors

The authors of this project are Sepehr Heydarian, Ximin Xu, and Essie Zhang.

## Installation

```bash
$ pip install datamop
```

## Usage

`datamop` can be used to encode columns in a DataFrame using one-hot or ordinal encoding as follows:

```
import pandas as pd
import datamop

df = pd.DataFrame({
    'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton'],
    'Level': ['A', 'B', 'C', 'D']
})

encoded_df_onehot = datamop.column_encoder(df, columns=['Sport'], method='one-hot')
encoded_df_ordinal = datamop.column_encoder(df, columns=['Level'], method='ordinal', order={'Level': ['A', 'B', 'C', 'D']})

```

This package can also be used to handle missing values such as imputation or removal, based on user preference as following:

```
import numpy as np
df = pd.DataFrame({
    'a': [10, np.nan, 30],
    'b': [1.5, 2.5, np.nan],
    'c': ['x', np.nan, 'z']
    })
cleaned = datamop.sweep_nulls(df, strategy='mean')
```

Additionally, this package can be used to scale numerical columns as following:

```
df = pd.DataFrame({"price": [25, 50, 75]})
df_scaled = datamop.column_scaler(df, column = 'price', method='minmax', new_min=0, new_max=1)
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`datamop` was created by Sepehr Heydarian, Ximin Xu, Essie Zhang. It is licensed under the terms of the MIT license.

## Credits

`datamop` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
