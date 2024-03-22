from abc import abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
from pynwb import NWBFile

from ...basedatainterface import BaseDataInterface
from ...utils.dict import load_dict_from_file
from ...utils.types import FilePathType

from pynwb.epoch import TimeSeries

class TimeSeriesInterface(BaseDataInterface):
    """Abstract Interface for time series."""

    def __init__(
        self,
        file_path: FilePathType,
        read_kwargs: Optional[dict] = None,
        verbose: bool = True,
        series_name = 'TimeSeries',
    ):
        """
        Parameters
        ----------
        file_path : FilePath
        read_kwargs : dict, optional
        verbose : bool, default: True
        """
        read_kwargs = read_kwargs or dict()
        super().__init__(file_path=file_path)
        self.verbose = verbose

        self.series_name = series_name

        self._read_kwargs = read_kwargs
        self.dataframe = self._read_file(file_path, **read_kwargs)


    def get_metadata(self) -> dict:
        metadata = super().get_metadata()

#         metadata[self.series_name] =  {'name': self.series_name, 'description': 'A description', 'unit': 'a unit',
#    'resolution': 0.0, 'comments': 'a comment', 'continuity': 'either continuous, instantaneous or step',
#    'offset': 0.0}

        return metadata

    def get_metadata_schema(self) -> dict:
        fpath = Path(__file__).parent.parent.parent / "schemas" / "base_metadata_schema.json"
        return load_dict_from_file(fpath)

    def get_original_timestamps(self, column: str) -> np.ndarray:
        if not column.endswith("timestamps"):
            raise ValueError("Timing column on a TimeSeries table needs to be called 'timestamp'")

        return self._read_file(**self.source_data, **self._read_kwargs)[column].values

    def get_timestamps(self, column: str) -> np.ndarray:
        if not column.endswith("timestamp"):
            raise ValueError("Timing column on a TimeSeries table needs to be called 'timestamp'")

        return self.dataframe[column].values

    def set_aligned_starting_time(self, aligned_starting_time: float):
        timing_columns = [column for column in self.dataframe.columns if column.endswith("_time")]

        for column in timing_columns:
            self.dataframe[column] += aligned_starting_time

    def set_aligned_timestamps(
        self, aligned_timestamps: np.ndarray, column: str, interpolate_other_columns: bool = False
    ):
        if not column.endswith("timestamp"):
            raise ValueError("Timing column on a TimeSeries table needs to be called 'timestamp'")

        unaligned_timestamps = np.array(self.dataframe[column])
        self.dataframe[column] = aligned_timestamps

        if not interpolate_other_columns:
            return

        other_timing_columns = [
            other_column
            for other_column in self.dataframe.columns
            if other_column.endswith("_time") and other_column != column
        ]
        for other_timing_column in other_timing_columns:
            self.align_by_interpolation(
                unaligned_timestamps=unaligned_timestamps,
                aligned_timestamps=aligned_timestamps,
                column=other_timing_column,
            )

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: Optional[dict] = None,
    ) -> NWBFile:
        """
        Run the NWB conversion for the instantiated data interface.

        Parameters
        ----------
        nwbfile : NWBFile, optional
            An in-memory NWBFile object to write to the location.
        metadata : dict, optional
            Metadata dictionary with information used to create the NWBFile when one does not exist or overwrite=True.
        name: str, default: "dataa"
        tag : str, default: "trials"
        column_name_mapping: dict, optional
            If passed, rename subset of columns from key to value.
        column_descriptions: dict, optional
            Keys are the names of the columns (after renaming) and values are the descriptions. If not passed,
            the names of the columns are used as descriptions.

        """

        #metadata = metadata
        series_name = self.series_name

        time_series = TimeSeries(data=self.dataframe.drop('timestamps', axis=1).to_numpy(), 
                             timestamps=self.dataframe['timestamps'].to_numpy(),
                             name=series_name, 
                             description=metadata[series_name]['description'], 
                             comments = metadata[series_name]['comments'],
                             unit=metadata[series_name]['unit'],
                             resolution=metadata[series_name]['resolution'], 
                             offset = metadata[series_name]['offset']
                             )

        nwbfile = nwbfile.add_acquisition(time_series)

        return nwbfile

    @abstractmethod
    def _read_file(self, file_path: FilePathType, **read_kwargs):
        pass
