import os.path
from pathlib import Path

import hydra
import kwcoco
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from typing import Any, Optional

from .tcn_dataset import TCNDataset


def create_dataset_from_hydra(
    model_hydra_conf: Path,
    split: str = "test",
) -> "TCNDataset":
    """
    Create a TCNDataset for some specified split based on the Hydra
    configuration file.

    E.g. from a training run
        * `.hydra/config.yaml`
        * `csv/version_0/hparams.yaml`

    No data will have be loaded into the dataset as a result of this call. That
    is still for the user to perform.

    :param model_hydra_conf:
    :param split: Which split we should read the cofniguration for. This should
        be one of "train", "val", or "test".

    :return: New TCNDataset.
    """
    assert model_hydra_conf.is_file()
    # Apparently hydra requires that the path provided to initialize is a
    # relative path, even if it is just a bunch of `../../` etc. stuff. This
    # relative path must also be relative to **THIS FILE** that this function
    # is implemented in.
    dir_path = Path(os.path.relpath(model_hydra_conf.parent, Path(__file__).parent))
    file_name_stem = model_hydra_conf.stem
    with hydra.initialize(config_path=dir_path.as_posix(), version_base=None):
        cfg = hydra.compose(config_name=file_name_stem)
    return hydra.utils.instantiate(cfg.data[f"{split}_dataset"])


class PTGDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        train_dataset: TCNDataset,
        val_dataset: TCNDataset,
        test_dataset: TCNDataset,
        coco_train_activities: str,
        coco_train_objects: str,
        coco_train_poses: str,
        coco_validation_activities: str,
        coco_validation_objects: str,
        coco_validation_poses: str,
        coco_test_activities: str,
        coco_test_objects: str,
        coco_test_poses: str,
        batch_size: int,
        num_workers: int,
        target_framerate: float,
        epoch_length: int,
        pin_memory: bool,
    ) -> None:
        """Initialize a `PTGDataModule`.

        :param coco_train_activities: Path to the COCO file with train-split
            activity classification ground truth.
        :param coco_train_objects: Path to the COCO file with train-split
            object detections to use for training.
        :param coco_train_poses: Path to the COCO file with train-split pose
            estimations to use for training.
        :param coco_validation_activities: Path to the COCO file with
            validation-split activity classification ground truth.
        :param coco_validation_objects: Path to the COCO file with
            validation-split object detections to use for training.
        :param coco_validation_poses: Path to the COCO file with train-split
            pose estimations to use for training.
        :param coco_test_activities: Path to the COCO file with test-split
            activity classification ground truth.
        :param coco_test_objects: Path to the COCO file with test-split
            object detections to use for training.
        :param coco_test_poses: Path to the COCO file with test-split
            pose estimations to use for training.
        :vector_cache_dir: Directory path to store cache files related to
            dataset vectory computation.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=["train_dataset", "val_dataset", "test_dataset"]
        )

        self.data_train: Optional[TCNDataset] = train_dataset
        self.data_val: Optional[TCNDataset] = val_dataset
        self.data_test: Optional[TCNDataset] = test_dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train.load_data_offline(
                kwcoco.CocoDataset(self.hparams.coco_train_activities),
                kwcoco.CocoDataset(self.hparams.coco_train_objects),
                kwcoco.CocoDataset(self.hparams.coco_train_poses),
                self.hparams.target_framerate,
            )
            self.data_val.load_data_offline(
                kwcoco.CocoDataset(self.hparams.coco_validation_activities),
                kwcoco.CocoDataset(self.hparams.coco_validation_objects),
                kwcoco.CocoDataset(self.hparams.coco_validation_poses),
                self.hparams.target_framerate,
            )
            self.data_test.load_data_offline(
                kwcoco.CocoDataset(self.hparams.coco_test_activities),
                kwcoco.CocoDataset(self.hparams.coco_test_objects),
                kwcoco.CocoDataset(self.hparams.coco_test_poses),
                self.hparams.target_framerate,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        train_sampler = torch.utils.data.WeightedRandomSampler(
            self.data_train.window_weights,
            self.hparams.epoch_length,
            replacement=True,
            generator=None,
        )
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=train_sampler,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
