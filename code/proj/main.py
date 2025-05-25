import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import time
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from datasets import ProgAdd2x2Dataset
from models import DigitCNN, ProgAdd2x2Model, SoftPointerAdd2x2Model, program_map
from loggers import ReadHeadLogger,DigitNetLogger,TrainingLogger

DEBUG=True
readhead_logger = ReadHeadLogger(log_dir="logs")
digitnet_logger = DigitNetLogger(log_dir="logs")
training_logger = TrainingLogger(log_dir="logs")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -----------------------
# Initialization Methods
# -----------------------
def initialize_weights(model, init_type="xavier"):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "he":
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_type == "zero":
                nn.init.constant_(m.weight, 0.0)
            elif init_type == "terpret":
                raise NotImplementedError("TerpreT-based init requires external integration.")
            else:
                raise ValueError(f"Unknown initialization: {init_type}")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def make_plot(xs, ys, xlabel, ylabel, filename, title, label):
    plt.figure()
    plt.plot(xs, ys, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)

# -----------------------
# Experiment Setup
# -----------------------
def manually_initialize_readheads(model):
    print("manually initializing readheads")
    with torch.no_grad():
        # Set pointer_encoder to identity: one-hot → same vector
        model.pointer_encoder.weight.copy_(torch.eye(model.pointer_encoder.out_features, 4))
        model.pointer_encoder.bias.zero_()

        # # Initialize readhead1 and readhead2 to pick these positions deterministically
        def construct_readhead_weights(index):
            weights = torch.full((4, 16), 0)  # (programs, positions)
            for prog_id, pos in program_map.items():
                selected_pos = pos[index] # E.g. program 0 -> selected_pos for readhead1 = 0, selected_pos for readhead2 = 1
                weights[selected_pos][prog_id] = 10
            return weights

        model.readhead1.weight.copy_(construct_readhead_weights(index=0))
        model.readhead2.weight.copy_(construct_readhead_weights(index=1))

        model.readhead1.bias.zero_()
        model.readhead2.bias.zero_()

def test_readhead_selection(model):
    model.eval()
    with torch.no_grad():
        for prog_id, (pos1, pos2) in program_map.items():
            program_onehot = F.one_hot(torch.tensor(prog_id), num_classes=4).float().to(next(model.parameters()).device)
            encoded = model.pointer_encoder(program_onehot)

            logits1 = model.readhead1(encoded)
            logits2 = model.readhead2(encoded)

            pred_pos1 = torch.argmax(logits1).item()
            pred_pos2 = torch.argmax(logits2).item()

            assert pred_pos1 == pos1, f"Readhead1 failed for program {prog_id}: got {pred_pos1}, expected {pos1}"
            assert pred_pos2 == pos2, f"Readhead2 failed for program {prog_id}: got {pred_pos2}, expected {pos2}"

    print("✅ All readhead selections are correct!")


def build_model(use_program, device, init_method="he",
                pretrained_digitnet=False, frozen_digitnet=False,
                pretrained_readhead=False, frozen_readhead=False):
    """
    Construct and configure the full model including digit_net and (if applicable) program read head.
    """

    # ----------------------------
    # Setup digit recognizer
    # ----------------------------
    digit_net = DigitCNN().to(device)

    # digit_net is properly initialized
    if pretrained_digitnet:
        # digit_net.load_state_dict(torch.load("net0_digit.pth"))
        digit_net.load_state_dict(torch.load("best_digitnet.pth"))
    else:
        initialize_weights(digit_net, init_method)

    if frozen_digitnet:
        for p in digit_net.parameters():
            p.requires_grad = False

    # ----------------------------
    # Setup full model
    # ----------------------------
    if use_program:
        model = ProgAdd2x2Model(digit_net).to(device)

        if pretrained_readhead:
            manually_initialize_readheads(model)
            # test_readhead_selection(model)
            print("model.readhead1.weight: {}".format(model.readhead1.weight))
            print("model.readhead2.weight: {}".format(model.readhead2.weight))
            print("model.pointer_encoder.weight: {}".format(model.pointer_encoder.weight))
        else:
            initialize_weights(model.readhead1, init_method)
            initialize_weights(model.readhead2, init_method)
            initialize_weights(model.pointer_encoder, init_method)

        if frozen_readhead:
            for name, param in model.named_parameters():
                if 'pointer_encoder' in name or 'readhead' in name:
                    param.requires_grad = False
    else:
        model = SoftPointerAdd2x2Model(digit_net).to(device)
        # there's no pretrained readhead for this variant
        # initialize_weights(model.readhead1, init_method)
        # initialize_weights(model.readhead2, init_method)
        # initialize_weights(model.pointer_encoder, init_method)

    return model


def train_and_evaluate(task="add2x2", variant="unnamed", use_program=False,
                       pretrained_digitnet=False, frozen_digitnet=False,
                       pretrained_readhead=False, frozen_readhead=False,
                       init_method="he", epochs=5, batch_size=64, lr=0.05):
    """
    task="add2x2", 
    variant="e2e",         # 'e2e', 'pretrained', 'program', 'frozen_pretrained', 'frozen_program'
        Meaning of variant:
        "e2e"                End-to-end training from scratch
        "pretrained"         Pretrained net0, fine-tune with soft pointer
        "frozen_pretrained"  Pretrained net0, frozen
        "program"            Conditioned on program ID (soft pointer learned)
        "frozen_program"     Program ID + frozen pretrained net0
        "apply2x2"           Placeholder for APPLY2x2 task (to be filled in)
    init_method="xavier", 
    epochs=5, 
    batch_size=64
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # Define transforms and dataset
    transform = transforms.ToTensor()
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if task == "add2x2":

        # Dataset selection based on variant
        if use_program:
            trainset = ProgAdd2x2Dataset(mnist_train, size=5000)
            testset  = ProgAdd2x2Dataset(mnist_test,  size=500)
        else:
            # Fixed movement pattern: always use top row for training
            trainset = ProgAdd2x2Dataset(mnist_train, which=0, size=5000)
            testset  = ProgAdd2x2Dataset(mnist_test,  which=0, size=500)

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        # Load or initialize digit recognizer
        model = build_model(use_program, device,
                            frozen_readhead=frozen_readhead,
                            frozen_digitnet=frozen_digitnet,
                            pretrained_readhead=pretrained_readhead)
        loss_fn = nn.CrossEntropyLoss()

        for name, param in model.named_parameters():
            print(f"{name} — requires_grad={param.requires_grad}")

        # Init weights if not loading
        if not pretrained_digitnet and not pretrained_readhead:
            initialize_weights(model, init_type=init_method)

        # Optimizer and loss
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        train_accs, test_accs, losses, times = train(model, optimizer, loss_fn, trainloader, testloader,
                       device=device, epochs=epochs, variant=variant,
                       use_program=("program" in variant), unfreeze="unfreeze" in variant)

        return test_accs[-1], losses[-1], times[-1], train_accs, test_accs, losses, times

    elif task == "apply2x2":
        raise NotImplementedError("Apply2x2 task support coming soon.")

def train(model, optimizer, criterion, trainloader, testloader, device,
          epochs=5, use_program=False, unfreeze=False, variant="unnamed"):
    start_time = time.time()
    print("{:<6} {:<18} {:<18} {:<10}".format("Epoch", "Train Acc (%)", "Test Acc (%)", "Time (s)"))
    print("-" * 70)

    train_accs, test_accs, times, losses = [], [], [], []
    # Early stopping
    best_test_acc = -float('inf')
    epochs_without_improvement = 0
    patience = 5
    best_model_state = None

    for epoch in range(epochs):
        model.train()

        if unfreeze and epoch == 3:
            print("unfreeze conv.0 at epoch: {}".format(epoch))
            for name, param in model.named_parameters():
                if 'conv.0' in name:
                    param.requires_grad = True
        if unfreeze and epoch == 5:
            print("unfreeze conv.2 at epoch: {}".format(epoch))
            for name, param in model.named_parameters():
                if 'conv.2' in name:
                    param.requires_grad = True
        if unfreeze and epoch == 8:
            print("unfreeze fc at epoch: {}".format(epoch))
            for name, param in model.named_parameters():
                if 'fc' in name:
                    param.requires_grad = True

        for batch in trainloader:
            if use_program:
                inputs, program_ids, labels, digits = batch
                program_onehot = F.one_hot(program_ids, num_classes=len(program_map)).float().to(device)
                preds = model(inputs.to(device), program_onehot)
                loss = criterion(preds, labels.long().to(device))

            else:
                inputs, _, labels, digits = batch
                preds = model(inputs.to(device))
                loss = criterion(preds, labels.long().to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            latest_labels = digits

        # log the digit recognized
        digitnet_logger.log(
            variant,
            epoch,
            model._latest_digit_logits.detach().cpu(),
            latest_labels.detach().cpu()
        )

        # log the read head
        probs1, probs2 = model._latest_probs
        if use_program:
            program_id_onehot = model._latest_prog_id 
        else:
            program_id_onehot = F.one_hot(torch.tensor([0,1]), num_classes=4)

        readhead_logger.log(variant, epoch, "h1", program_id_onehot, probs1)
        readhead_logger.log(variant, epoch, "h2", program_id_onehot, probs2)

        # Track metrics
        elapsed = time.time() - start_time
        train_acc, train_loss = evaluate(model, trainloader, criterion, device, use_program)
        test_acc, test_loss = evaluate(model, testloader, criterion, device, use_program)

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        losses.append(test_loss)
        times.append(elapsed)

        print("{:<6} {:<18.2f} {:<18.2f} {:<10.2f}".format(epoch, train_acc, test_acc, elapsed))
        training_logger.log(variant, epoch, train_acc, test_acc, elapsed)

        # EARLY STOPPING CHECK
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best test acc: {best_test_acc:.2f}%")
                break

    return train_accs, test_accs, losses, times

def evaluate(model, dataloader, criterion, device, use_program=False):
    model.eval()
    total, correct = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs, program_ids, labels, digits = batch
            labels = labels.to(device)
            if use_program:
                program_onehot = F.one_hot(program_ids, num_classes=4).float().to(device)
                outputs = model(inputs.to(device), program_onehot)
                loss = criterion(outputs, labels.long())
                total_loss += loss.item() * labels.size(0)

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
            else:
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.long())
                total_loss += loss.item() * labels.size(0)

                # Get predicted class: use argmax over logits dimension 1.
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
            
            total += labels.size(0)
        # if DEBUG:
        #     preds, labels = preds.detach().cpu().numpy(), labels.cpu().numpy()
        #     for p, l in zip(preds[:10], labels[:10]):
        #         print(f"Predicted: {p:.2f} → Rounded: {round(p)} | Label: {l}")

    accuracy = 100 * correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss


# -----------------------
# Run All Initializations
# -----------------------

def main():
    import pandas as pd

    program_variants = [
        # "program",  # e2e baseline
        # "program_pretrained_digitnet",
        # "program_pretrained_digitnet_frozen_digitnet",
        # "program_pretrained_readhead",
        # "program_pretrained_readhead_frozen_readhead",
        # "program_pretrained_digitnet_pretrained_readhead",
        "program_pretrained_digitnet_pretrained_readhead_frozen_readhead",
        # "program_pretrained_digitnet_frozen_digitnet_unfreeze",
    ]

    no_program_variants = [
        # "e2e",  # end-to-end baseline
        # "pretrained_digitnet",
        # "pretrained_digitnet_frozen_digitnet"
    ]
    variants = program_variants + no_program_variants

    variants = [ f"highacc_{variant}" for variant in variants ]
    print("variants: {}".format(variants))

    results = []
    init = "he"

    for variant in variants:
        for lr in [0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.003]:
            print(f"\n🔧 Running variant: {variant} with learning rate {lr}")

            final_acc, final_loss, duration, train_accs, test_accs, losses, times = train_and_evaluate(
                task="add2x2",
                variant=variant,
                init_method=init,
                epochs=50,
                batch_size=64,
                lr=lr,
                use_program=("program" in variant),
                frozen_readhead=("frozen_readhead" in variant),
                pretrained_readhead=("pretrained_readhead" in variant),
                frozen_digitnet=("frozen_digitnet" in variant),
                pretrained_digitnet=("pretrained_digitnet" in variant)
            )

            # Plot Loss vs. Time
            make_plot(
                xs=times,
                ys=losses,
                xlabel="Time (s)",
                ylabel="Loss",
                filename=f"loss_vs_time_{variant}_{lr}.png",
                title=f"Loss vs Time ({variant})",
                label="Test Loss"
            )

            # Plot Accuracy vs. Time
            make_plot(
                xs=times,
                ys=test_accs,
                xlabel="Time (s)",
                ylabel="Accuracy (%)",
                filename=f"acc_vs_time_{variant}_{lr}_{init}.png",
                title=f"Accuracy vs Time ({variant})",
                label="Test Accuracy"
            )

            results.append({
                "Variant": variant,
                "Learning Rate": lr,
                "Accuracy (%)": round(final_acc, 2),
                "Loss": round(final_loss, 4),
                "Time (s)": round(duration, 2)
            })

    # Format results into a summary table
    df = pd.DataFrame(results)
    print("\n📊 Summary:")
    print(df.to_string(index=False))

main()
