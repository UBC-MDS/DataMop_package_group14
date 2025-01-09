# datamop

`datamop` is a data cleaning and wrangling package designed to streamline the preprocessing of datasets. Whether you meet missing values, inconsistent categorical columns or need scaling for numberic columns when dealing with data, `datamop` provides a simple and consistent solution to automate and simplify these repetitive tasks. 
`datamop` provides three core functions:

1. `sweep_nulls()`: Handle missing values such as imputation or removal, based on user preference.

2. `column_encoder()`: Encodes categorical columns using either one-hot encoding or ordinal encoding, based on user preference.

3. `column_scaler()`: Normalizes numerical columns, including Min-Max scaling and Z-score standardization, based on user preference.


## Contributors

The authors of this project are Sepehr Heydarian, Ximin Xu, and Essie Zhang.

## Installation

```bash
$ pip install datamop
```

## Usage

- TODO

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`datamop` was created by Sepehr Heydarian, Ximin Xu, Essie Zhang. It is licensed under the terms of the MIT license.

## Credits

`datamop` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
