import torch
from torch.utils.data import Dataset
from models import program_map
import random

class ProgAdd2x2Dataset(Dataset):
    """
    Returns:
      - grid: Tensor of shape (4, 1, 28, 28)
      - program_id: int ∈ [0, 3]
      - target: float (sum based on selected program)
    """
    def __init__(self, mnist_dataset, which=None, size=1000):
        super().__init__()
        self.mnist = mnist_dataset
        self.size = size
        self.indices = random.choices(range(len(mnist_dataset)), k=4 * size)
        self.which = which

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Sample 4 images
        imgs, digits = [], []
        for i in range(4):
            img, label = self.mnist[self.indices[4 * idx + i]]
            imgs.append(img)
            digits.append(label)

        grid = torch.stack(imgs)  # shape: (4, 1, 28, 28)
        digits = torch.tensor(digits)

        # Sample a program_id and compute label
        program_id = self.which or random.randint(0, len(program_map)-1)
        # print("program_id: {}".format(program_id))
        i1, i2 = program_map[program_id]
        label = digits[i1] + digits[i2]

        return grid, program_id, label.float(), digits
