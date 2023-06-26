from datetime import datetime
from unittest import TestCase

import numpy as np
from pynwb import NWBHDF5IO

from neuroconv.datainterfaces import (
    BlackrockRecordingInterface,
    BlackrockSortingInterface,
    CellExplorerSortingInterface,
    NeuralynxSortingInterface,
    NeuroScopeSortingInterface,
    PhySortingInterface,
    PlexonSortingInterface,
)
from neuroconv.tools.testing.data_interface_mixins import (
    SortingExtractorInterfaceTestMixin,
)

try:
    from .setup_paths import ECEPHY_DATA_PATH as DATA_PATH
    from .setup_paths import OUTPUT_PATH
except ImportError:
    from setup_paths import ECEPHY_DATA_PATH as DATA_PATH
    from setup_paths import OUTPUT_PATH


class TestBlackrockSortingInterface(SortingExtractorInterfaceTestMixin, TestCase):
    data_interface_cls = BlackrockSortingInterface
    interface_kwargs = dict(file_path=str(DATA_PATH / "blackrock" / "FileSpec2.3001.nev"))

    associated_recording_cls = BlackrockRecordingInterface
    associated_recording_kwargs = dict(file_path=str(DATA_PATH / "blackrock" / "FileSpec2.3001.ns5"))

    save_directory = OUTPUT_PATH


class TestCellExplorerSortingInterfaceBuzCode(SortingExtractorInterfaceTestMixin, TestCase):
    """This corresponds to the Buzsaki old CellExplorerFormat or Buzcode format."""

    data_interface_cls = CellExplorerSortingInterface
    interface_kwargs = [
        dict(
            file_path=str(
                DATA_PATH / "cellexplorer" / "dataset_1" / "20170311_684um_2088um_170311_134350.spikes.cellinfo.mat"
            )
        ),
        dict(file_path=str(DATA_PATH / "cellexplorer" / "dataset_2" / "20170504_396um_0um_merge.spikes.cellinfo.mat")),
        dict(
            file_path=str(DATA_PATH / "cellexplorer" / "dataset_3" / "20170519_864um_900um_merge.spikes.cellinfo.mat")
        ),
    ]
    save_directory = OUTPUT_PATH


class TestCellEploreSortingInterface(SortingExtractorInterfaceTestMixin, TestCase):
    """This corresponds to the Buzsaki new CellExplorerFormat where a session.mat file with rich metadata is provided."""

    data_interface_cls = CellExplorerSortingInterface
    interface_kwargs = [
        dict(
            file_path=str(
                DATA_PATH
                / "cellexplorer"
                / "dataset_4"
                / "Peter_MS22_180629_110319_concat_stubbed"
                / "Peter_MS22_180629_110319_concat_stubbed.spikes.cellinfo.mat"
            )
        ),
        dict(
            file_path=str(
                DATA_PATH
                / "cellexplorer"
                / "dataset_4"
                / "Peter_MS22_180629_110319_concat_stubbed_hdf5"
                / "Peter_MS22_180629_110319_concat_stubbed_hdf5.spikes.cellinfo.mat"
            )
        ),
    ]
    save_directory = OUTPUT_PATH

    def test_writing_channel_metadata(self):
        channel_id = "1"
        expected_channel_properties_recorder = {
            "location": np.array([791.5, -160.0]),
            "brain_area": "CA1 - Field CA1",
            "group": "Group 5",
        }
        expected_channel_properties_electrodes = {
            "rel_x": 791.5,
            "rel_y": -160.0,
            "location": "CA1 - Field CA1",
            "group_name": "Group 5",
        }

        interface_kwargs = self.interface_kwargs
        for num, kwargs in enumerate(interface_kwargs):
            with self.subTest(str(num)):
                self.case = num
                self.test_kwargs = kwargs
                self.interface = self.data_interface_cls(**self.test_kwargs)
                self.nwbfile_path = str(self.save_directory / f"{self.data_interface_cls.__name__}_{num}.nwb")

                metadata = self.interface.get_metadata()
                metadata["NWBFile"].update(session_start_time=datetime.now().astimezone())
                self.interface.run_conversion(
                    nwbfile_path=self.nwbfile_path,
                    overwrite=True,
                    metadata=metadata,
                    write_ecephys_metadata=True,
                )

                # Test that the registered recording has the ``
                recording_extractor = self.interface.sorting_extractor._recording
                for key, expected_value in expected_channel_properties_recorder.items():
                    extracted_value = recording_extractor.get_channel_property(channel_id=channel_id, key=key)
                    if key == "location":
                        assert np.allclose(expected_value, extracted_value)
                    else:
                        assert expected_value == extracted_value

                # Test that the electrode table has the expected values
                with NWBHDF5IO(self.nwbfile_path, "r") as io:
                    nwbfile = io.read()
                    electrode_table = nwbfile.electrodes.to_dataframe()
                    electrode_table_row = electrode_table.query(f"channel_name=='{channel_id}'").iloc[0]
                    for key, value in expected_channel_properties_electrodes.items():
                        assert electrode_table_row[key] == value


class TestNeuralynxSortingInterface(SortingExtractorInterfaceTestMixin, TestCase):
    data_interface_cls = NeuralynxSortingInterface
    interface_kwargs = [
        dict(folder_path=str(DATA_PATH / "neuralynx" / "Cheetah_v5.5.1" / "original_data")),
        dict(folder_path=str(DATA_PATH / "neuralynx" / "Cheetah_v5.6.3" / "original_data")),
    ]
    save_directory = OUTPUT_PATH


class TestNeuroScopeSortingInterface(SortingExtractorInterfaceTestMixin, TestCase):
    data_interface_cls = NeuroScopeSortingInterface
    interface_kwargs = dict(
        folder_path=str(DATA_PATH / "neuroscope" / "dataset_1"),
        xml_file_path=str(DATA_PATH / "neuroscope" / "dataset_1" / "YutaMouse42-151117.xml"),
    )
    save_directory = OUTPUT_PATH

    def check_extracted_metadata(self, metadata: dict):
        assert metadata["NWBFile"]["session_start_time"] == datetime(2015, 8, 31, 0, 0)


class TestPhySortingInterface(SortingExtractorInterfaceTestMixin, TestCase):
    data_interface_cls = PhySortingInterface
    interface_kwargs = dict(folder_path=str(DATA_PATH / "phy" / "phy_example_0"))
    save_directory = OUTPUT_PATH


class TestPlexonSortingInterface(SortingExtractorInterfaceTestMixin, TestCase):
    data_interface_cls = PlexonSortingInterface
    interface_kwargs = dict(file_path=str(DATA_PATH / "plexon" / "File_plexon_2.plx"))
    save_directory = OUTPUT_PATH

    def check_extracted_metadata(self, metadata: dict):
        assert metadata["NWBFile"]["session_start_time"] == datetime(2000, 10, 30, 15, 56, 56)
