import abc
import numpy as np
import numpy.typing as npt

from ._data import FrameData


__all__ = [
    "Vectorize",
]


class Vectorize(metaclass=abc.ABCMeta):
    """
    Interface for a functor that will vectorize input data into an embedding
    space for use in TCN training and inference.
    """

    @abc.abstractmethod
    def vectorize(self, data: FrameData) -> npt.NDArray[np.float32]:
        """
        Perform vectorization of the input data into an embedding space.

        :param data: Input data to generate an embedding from.
        """

    def __call__(self, data: FrameData) -> npt.NDArray[np.float32]:
        return self.vectorize(data)
