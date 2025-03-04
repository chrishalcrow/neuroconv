"""Base Pydantic models for DatasetInfo and DatasetConfiguration."""

from typing import ClassVar, Dict, Literal, Type

from hdmf.container import DataIO
from pydantic import BaseModel, Field
from pynwb import NWBFile

from ._base_dataset_io import DatasetIOConfiguration
from .._dataset_configuration import get_default_dataset_io_configurations


class BackendConfiguration(BaseModel):
    """A model for matching collections of DatasetConfigurations to a specific backend."""

    backend: ClassVar[Literal["hdf5", "zarr"]] = Field(
        description="The name of the backend used to configure the NWBFile."
    )
    data_io_class: Type[DataIO] = Field(description="The DataIO class that is specific to this backend.")
    dataset_configurations: Dict[str, DatasetIOConfiguration] = Field(
        description=(
            "A mapping from object locations (e.g. `acquisition/TestElectricalSeriesAP/data`) "
            "to their DatasetConfiguration specification that contains all information "
            "for writing the datasets to disk using the specific backend."
        )
    )

    def __str__(self) -> str:
        """Not overriding __repr__ as this is intended to render only when wrapped in print()."""
        string = (
            f"\nConfigurable datasets identified using the {self.backend} backend"
            f"\n{'-' * (43 + len(self.backend) + 8)}"
        )

        for dataset_configuration in self.dataset_configurations.values():
            string += f"\n{dataset_configuration}"

        return string

    @classmethod
    def from_nwbfile(cls, nwbfile: NWBFile) -> "BackendConfiguration":
        default_dataset_configurations = get_default_dataset_io_configurations(nwbfile=nwbfile, backend=cls.backend)
        dataset_configurations = {
            default_dataset_configuration.dataset_info.location: default_dataset_configuration
            for default_dataset_configuration in default_dataset_configurations
        }

        return cls(dataset_configurations=dataset_configurations)
