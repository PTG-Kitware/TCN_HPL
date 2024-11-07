import abc
import inspect
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
from pytorch_lightning.utilities.parsing import collect_init_args

from tcn_hpl.data.frame_data import FrameData


__all__ = [
    "Vectorize",
]


class Vectorize(metaclass=abc.ABCMeta):
    """
    Interface for a functor that will vectorize input data into an embedding
    space for use in TCN training and inference.
    """

    def __init__(self):
        # Collect parameters to checksum until we are out of the __init__
        # stack.
        init_args = {}
        # merge init args from the bottom up: higher stacks override.
        for local_args in collect_init_args(inspect.currentframe().f_back, []):
            init_args.update(local_args)
        # Instead of keeping around hyperparameter values forever, just
        # checksum now. This should be fine because, even if we retain them,
        # runtime
        self.__init_args = init_args

    def hparams(self) -> Mapping[str, Any]:
        """
        Return a deterministic checksum of hyperparameters this instance was
        constructed with.

        This may need to be overwritten if hyperparameters bed

        :returns: Hexadecimal digest of the SHA256 checksum of hyperparameters.
        """
        return self.__init_args

    @abc.abstractmethod
    def vectorize(self, data: FrameData) -> npt.NDArray[np.float32]:
        """
        Perform vectorization of the input data into an embedding space.

        :param data: Input data to generate an embedding from.
        """

    def __call__(self, data: FrameData) -> npt.NDArray[np.float32]:
        return self.vectorize(data)
