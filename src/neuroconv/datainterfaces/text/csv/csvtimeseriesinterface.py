import pandas as pd


from ..timeseriesinterface import TimeSeriesInterface
from ....utils.types import FilePathType


class CsvTimeSeriesInterface(TimeSeriesInterface):
    """Interface for adding data from a .csv file as a TimeSeries object"""

    def _read_file(self, file_path: FilePathType, **read_kwargs):
        return pd.read_csv(file_path, **read_kwargs)
