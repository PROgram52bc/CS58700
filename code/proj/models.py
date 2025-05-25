import torch
import torch.nn as nn
import torch.nn.functional as F

program_map = {
    0: (0, 1),
    1: (2, 3),
    2: (0, 2),
    3: (1, 3),
    # 4: (0, 3),
    # 5: (1, 2),
}

NUMPROG = len(program_map)

temperature = 0.1

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # (B,32,28,28)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # (B,64,28,28)
        self.pool = nn.MaxPool2d(2, 2)                # (B,64,14,14)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -------------
# add2x2 model
# -------------

class Add2x2Model(nn.Module):
    def __init__(self, digit_net):
        super().__init__()
        self.digit_net = digit_net

    def forward(self, grid):
        B = grid.size(0)
        flattened = grid.view(B * 4, 1, 28, 28)
        logits = self.digit_net(flattened)
        probs = F.softmax(logits / temperature, dim=-1)
        digit_vals = torch.arange(10).float().to(probs.device)
        expected_digits = torch.matmul(probs, digit_vals)
        expected_digits = expected_digits.view(B, 4)
        return expected_digits[:, 0] + expected_digits[:, 1]  # sum of row 0, can be parameterized later


class ProgAdd2x2Model(nn.Module):
    def __init__(self, digit_net, num_programs=NUMPROG):
        super().__init__()
        self.digit_net = digit_net

        # Learn to condition read head positions on program_id
        self.pointer_encoder = nn.Linear(num_programs, 16)
        self.readhead1 = nn.Linear(16, 4)  # logits over 4 grid cells
        self.readhead2 = nn.Linear(16, 4)

    def forward(self, grid, program_id_onehot):
        B = grid.size(0)
        device = grid.device

        # Run digit recognizer on all 4 positions.
        # digit_net outputs logits over 10 classes for each image.
        logits = self.digit_net(grid.view(B * 4, 1, 28, 28))  # (B*4, 10)
        # Compute digit probabilities using softmax.
        digit_probs = F.softmax(logits / temperature, dim=-1)  # shape: (B*4, 10)
        # Reshape to (B, 4, 10): each example has 4 cells, each with a 10-dimensional distribution.
        digit_probs = digit_probs.view(B, 4, 10)

        # Condition pointer logits on program ID.
        h = self.pointer_encoder(program_id_onehot)
        # print("self.pointer_encoder.weight: {}".format(self.pointer_encoder.weight))
        # print("program_id_onehot: {}".format(program_id_onehot))
        # print("h: {}".format(h))
        rh1 = self.readhead1(h)
        rh2 = self.readhead2(h)
        # print("rh1: {}".format(rh1))
        # print("rh2: {}".format(rh2))

        ptr1 = F.softmax(rh1, dim=-1)  # shape: (B, 4)
        ptr2 = F.softmax(rh2, dim=-1)  # shape: (B, 4)
        # print("ptr1: {}".format(ptr1))
        # print("ptr2: {}".format(ptr2))

        # should converge to top row
        self._latest_probs = (ptr1.detach().cpu(), ptr2.detach().cpu())
        self._latest_prog_id = program_id_onehot
        self._latest_digit_logits = digit_probs

        # Compute the weighted mixture of digit distributions for each pointer.
        # (B, 4) pointer weights combined with (B, 4, 10) digit probabilities.
        dist1 = torch.bmm(ptr1.unsqueeze(1), digit_probs).squeeze(1)  # shape: (B, 10)
        dist2 = torch.bmm(ptr2.unsqueeze(1), digit_probs).squeeze(1)  # shape: (B, 10)

        # Compute outer product of the two digit distributions.
        outer = torch.bmm(dist1.unsqueeze(2), dist2.unsqueeze(1))  # shape: (B, 10, 10)
        
        # Invariant layer
        outer_sym = 0.5 * (outer + outer.transpose(1,2)) # average to make it invariant

        # Convolve the two distributions to form the sum distribution over values 0 to 18.
        sum_distributions = []
        for k in range(19):  # possible sums: 0, 1, ..., 18
            mask = torch.zeros((10, 10), device=device)
            for i in range(10):
                j = k - i
                if 0 <= j < 10:
                    mask[i, j] = 1.0
            # Sum the probabilities on the diagonal corresponding to sum = k.
            sum_k = (outer_sym * mask).sum(dim=(1, 2))  # shape: (B,)
            sum_distributions.append(sum_k.unsqueeze(1))
        # Combine the contributions to obtain a distribution of shape (B, 19).
        sum_distribution = torch.cat(sum_distributions, dim=1)

        # Convert probabilities to log probabilities (small epsilon added for numerical stability).
        logits_out = torch.log(sum_distribution + 1e-10)
        return logits_out

class SoftPointerAdd2x2Model(nn.Module):
    def __init__(self, digit_net):
        super().__init__()
        self.digit_net = digit_net

        # Learnable logits over 4 grid cells (shared across examples)
        self.read_logits1 = nn.Parameter(torch.randn(4))
        self.read_logits2 = nn.Parameter(torch.randn(4))

    def forward(self, grid):
        B = grid.size(0)
        device = grid.device

        # Run digit recognizer on all 4 positions
        logits = self.digit_net(grid.view(B * 4, 1, 28, 28))  # (B*4, 10)
        probs = F.softmax(logits / temperature, dim=-1)
        probs = probs.view(B, 4, 10)

        # digit_vals = torch.arange(10).float().to(device)
        # expected_digits = torch.matmul(probs, digit_vals).view(B, 4)  # (B, 4)

        # Fixed (shared) read heads
        ptr1 = F.softmax(self.read_logits1 / temperature, dim=0).view(1, 4)
        ptr2 = F.softmax(self.read_logits2 / temperature, dim=0).view(1, 4)

        # should converge to top row
        self._latest_probs = (ptr1.detach().cpu(), ptr2.detach().cpu())

        # For each example, get the weighted digit distribution for each read head.
        # This is done by taking a weighted sum of the digit probability distributions across the 4 cells.
        dist1 = torch.matmul(ptr1, probs).squeeze(1)  # shape: (B, 10)
        dist2 = torch.matmul(ptr2, probs).squeeze(1)  # shape: (B, 10)

        # Compute the outer product of the two distributions for each example.
        # This yields a (B, 10, 10) tensor where element (i,j) corresponds to probability p(i)*q(j).
        outer = torch.bmm(dist1.unsqueeze(2), dist2.unsqueeze(1))  # shape: (B, 10, 10)
        outer_sym = 0.5 * (outer + outer.transpose(1,2)) # average to make it invariant

        # The final distribution over sums (from 0 to 18) is obtained by summing the appropriate diagonals.
        sum_distributions = []
        for k in range(19):
            # Create a mask for indices where i + j == k.
            mask = torch.zeros((10, 10), device=device)
            for i in range(10):
                j = k - i
                if 0 <= j < 10:
                    mask[i, j] = 1.0
            # Sum over the masked elements for each batch example.
            sum_k = (outer_sym * mask).sum(dim=(1, 2))  # shape: (B,)
            sum_distributions.append(sum_k.unsqueeze(1))
        # Concatenate along the class dimension so that we have (B, 19) distribution.
        sum_distribution = torch.cat(sum_distributions, dim=1)

        # Add a small constant for numerical stability and take the log for use as logits.
        logits_sum = torch.log(sum_distribution + 1e-10)  # shape: (B, 19)

        return logits_sum
