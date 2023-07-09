import os

from .reader import Reader


class CoinDataReader(Reader):
    def __init__(self, root, images_and_targets):
        super().__init__()

        self.root = root
        self.images_and_targets = images_and_targets

    def __getitem__(self, index):
        path, target = self.images_and_targets[index]
        return open(path, "rb"), target

    def __len__(self):
        return len(self.images_and_targets)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.images_and_targets[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename
