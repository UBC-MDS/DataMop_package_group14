# datamop

A package containing tools for robust data cleaning.

## Contributors

The authors of this project are Sepehr Heydarian, Ximin Xu, and Essie Zhang.

## Installation

```bash
$ pip install datamop
```

## Usage

`datamop` can be used to encode categorical columns in a DataFrame using one-hot or ordinal encoding as follows:

```
import pandas as pd
from datamop.column_encoder import column_encoder

df = pd.DataFrame({
    'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton'],
    'Level': ['A', 'B', 'C', 'D']
})

encoded_df_onehot = column_encoder(df, columns=['Sport'], method='one-hot')
encoded_df_ordinal = column_encoder(encoded_df_onehot, columns=['Level'], method='ordinal')

```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`datamop` was created by Sepehr Heydarian, Ximin Xu, Essie Zhang. It is licensed under the terms of the MIT license.

## Credits

`datamop` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
