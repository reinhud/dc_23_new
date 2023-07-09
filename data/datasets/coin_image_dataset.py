import logging

from PIL import Image
from torch.utils.data import Dataset

_logger = logging.getLogger(__name__)

_ERROR_RETRY = 50


class CoinImageDataset(Dataset):
    def __init__(
        self,
        reader,
        load_bytes=False,
        img_mode="RGB",
        transform=None,
        target_transform=None,
    ) -> None:
        self.reader = reader
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.reader[index]

        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(
                f"Skipped sample (index {index},"
                "file {self.reader.filename(index)}). {str(e)}"
            )
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0

        if self.img_mode and not self.load_bytes:
            img = img.convert(self.img_mode)
        if self.transform is not None:
            img = self.transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)
