import unittest
from platform import python_version
from sys import platform
from packaging import version
from tempfile import mkdtemp
from pathlib import Path
from datetime import datetime

import numpy as np
import pytest
import spikeextractors as se
from hdmf.testing import TestCase
from pynwb import NWBHDF5IO

from neuroconv import NWBConverter, RecordingTutorialInterface, SortingTutorialInterface, CEDRecordingInterface
from neuroconv.datainterfaces.ecephys.basesortingextractorinterface import BaseSortingExtractorInterface


@pytest.mark.skipif(
    platform != "darwin" or version.parse(python_version()) >= version.parse("3.8"),
    reason="Only testing on MacOSX with Python 3.7!",
)
class TestAssertions(TestCase):
    def test_import_assertions(self):
        with self.assertRaisesWith(
            exc_type=AssertionError,
            exc_msg="The sonpy package (CED dependency) is not available on Mac for Python versions below 3.8!",
        ):
            CEDRecordingInterface.get_all_channels_info(file_path="does_not_matter.smrx")


def test_tutorials():
    class TutorialNWBConverter(NWBConverter):
        data_interface_classes = dict(
            RecordingTutorial=RecordingTutorialInterface, SortingTutorial=SortingTutorialInterface
        )

    duration = 10.0  # Seconds
    num_channels = 4
    num_units = 10
    sampling_frequency = 30000.0  # Hz
    stub_test = False
    test_dir = Path(mkdtemp())
    nwbfile_path = str(test_dir / "TestTutorial.nwb")
    source_data = dict(
        RecordingTutorial=dict(duration=duration, num_channels=num_channels, sampling_frequency=sampling_frequency),
        SortingTutorial=dict(duration=duration, num_units=num_units, sampling_frequency=sampling_frequency),
    )
    converter = TutorialNWBConverter(source_data=source_data)
    metadata = converter.get_metadata()
    metadata["NWBFile"]["session_description"] = "NWB Conversion Tools tutorial."
    metadata["NWBFile"]["session_start_time"] = datetime.now().astimezone()
    metadata["NWBFile"]["experimenter"] = ["My name"]
    metadata["Subject"] = dict(subject_id="Name of imaginary testing subject (required for DANDI upload)")
    conversion_options = dict(RecordingTutorial=dict(stub_test=stub_test), SortingTutorial=dict())
    converter.run_conversion(
        nwbfile_path=nwbfile_path,
        metadata=metadata,
        overwrite=True,
        conversion_options=conversion_options,
    )


class TestSortingInterface(unittest.TestCase):
    def setUp(self) -> None:
        self.sorting_start_frames = [100, 200, 300]
        self.num_frames = 1000
        sorting = se.NumpySortingExtractor()
        sorting.set_sampling_frequency(3000)
        sorting.add_unit(unit_id=1, times=np.arange(self.sorting_start_frames[0], self.num_frames))
        sorting.add_unit(unit_id=2, times=np.arange(self.sorting_start_frames[1], self.num_frames))
        sorting.add_unit(unit_id=3, times=np.arange(self.sorting_start_frames[2], self.num_frames))

        class TestSortingInterface(BaseSortingExtractorInterface):
            def __init__(self, verbose: bool = True):
                self.sorting_extractor = sorting
                self.source_data = dict()
                self.verbose = verbose

        class TempConverter(NWBConverter):
            data_interface_classes = dict(TestSortingInterface=TestSortingInterface)

        source_data = dict(TestSortingInterface=dict())
        self.test_sorting_interface = TempConverter(source_data)

    def test_sorting_stub(self):
        test_dir = Path(mkdtemp())
        minimal_nwbfile = test_dir / "stub_temp.nwb"
        conversion_options = dict(TestSortingInterface=dict(stub_test=True))
        metadata = self.test_sorting_interface.get_metadata()
        metadata["NWBFile"]["session_start_time"] = datetime.now().astimezone()
        self.test_sorting_interface.run_conversion(
            nwbfile_path=minimal_nwbfile, metadata=metadata, conversion_options=conversion_options
        )
        with NWBHDF5IO(minimal_nwbfile, "r") as io:
            nwbfile = io.read()
            start_frame_max = np.max(self.sorting_start_frames)
            for i, start_times in enumerate(self.sorting_start_frames):
                assert len(nwbfile.units["spike_times"][i]) == (start_frame_max * 1.1) - start_times

    def test_sorting_full(self):
        test_dir = Path(mkdtemp())
        minimal_nwbfile = test_dir / "temp.nwb"
        metadata = self.test_sorting_interface.get_metadata()
        metadata["NWBFile"]["session_start_time"] = datetime.now().astimezone()
        self.test_sorting_interface.run_conversion(nwbfile_path=minimal_nwbfile, metadata=metadata)
        with NWBHDF5IO(minimal_nwbfile, "r") as io:
            nwbfile = io.read()
            for i, start_times in enumerate(self.sorting_start_frames):
                assert len(nwbfile.units["spike_times"][i]) == self.num_frames - start_times
