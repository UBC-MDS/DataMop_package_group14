# read version from installed package
from importlib.metadata import version
__version__ = version("datamop")

# populate package namespace
from datamop.column_encoder import column_encoder
from datamop.column_scaler import column_scaler
from datamop.sweep_nulls import sweep_nulls