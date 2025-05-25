import os
import csv
from datetime import datetime

class ReadHeadLogger:
    def __init__(self, log_dir="logs", log_name=None):
        os.makedirs(log_dir, exist_ok=True)
        if log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"readhead_log_{timestamp}.csv"
        self.log_path = os.path.join(log_dir, log_name)
        self._init_log()

    def _init_log(self):
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["variant", "step", "head", "program_id", "probs"])

    def log(self, variant, step, head, program_ids, probs):
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            B = probs.shape[0]
            for i in range(B):
                writer.writerow([
                    variant,
                    step,
                    head,
                    int(program_ids[i].argmax()),
                    probs[i].tolist()
                ])

class DigitNetLogger:
    def __init__(self, log_dir="logs", log_name=None):
        os.makedirs(log_dir, exist_ok=True)
        if log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"digitnet_log_{timestamp}.csv"
        self.log_path = os.path.join(log_dir, log_name)
        self._init_log()

    def _init_log(self):
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["variant", "step", "predicted_logits", "true_digit_label"])

    def log(self, variant, step, predicted_logits, true_labels):
        """
        predicted_logits: Tensor of shape (B, 10)
        true_labels: Tensor of shape (B,)
        """
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            B = predicted_logits.shape[0]
            predicted_logits = predicted_logits.view(B, 4, 10)  # (B, 4, 10)

            for img_idx in range(B):
                for cell_idx in range(4):  # Each digit cell
                    writer.writerow([
                        variant,
                        step,
                        predicted_logits[img_idx, cell_idx].tolist(),  # 10 logits
                        int(true_labels[img_idx, cell_idx])            # True digit (0–9)
                    ])


class TrainingLogger:
    def __init__(self, log_dir="logs", log_name=None):
        os.makedirs(log_dir, exist_ok=True)
        if log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"training_log_{timestamp}.csv"
        self.log_path = os.path.join(log_dir, log_name)
        self._init_log()

    def _init_log(self):
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["variant", "epoch", "train_acc(%)", "test_acc(%)", "time_elapsed(s)"])

    def log(self, variant, epoch, train_acc, test_acc, elapsed_time):
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                variant,
                epoch,
                f"{train_acc:.2f}",
                f"{test_acc:.2f}",
                f"{elapsed_time:.2f}"
            ])
