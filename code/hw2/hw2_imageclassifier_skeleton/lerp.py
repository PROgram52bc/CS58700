import torch
import matplotlib.pyplot as plt

def evaluate(
    model,
    criterion: torch.nn.Module,
    minibatcher: torch.utils.data.DataLoader,
    /,
    *,
    device: str,
) -> float:
    R"""
    Evaluate.
    """
    #
    model.eval()

    #
    buf_total = []
    buf_metric = []
    for (inputs, targets) in minibatcher:
        #
        inputs = inputs.to(device)
        targets = targets.to(device)

        #
        with torch.no_grad():
            #
            outputs = model.forward(inputs)
            total = len(targets)
            metric = criterion.forward(outputs, targets).item()
        buf_total.append(total)
        buf_metric.append(metric * total)
    return float(sum(buf_metric)) / float(sum(buf_total))

# Function to perform LERP and evaluate accuracy or loss
def interpolate_and_evaluate(model, initial_model, final_model, train_loader, val_loader, device, criterion, num_steps=10):

    train = []
    val = []
    
    for alpha in range(num_steps + 1):
        # Perform linear interpolation
        alpha = alpha / num_steps
        interpolate_model(model, initial_model, final_model, alpha)

        # Calculate train and validation data

        train_data = evaluate(model, criterion, train_loader, device=device)
        val_data = evaluate(model, criterion, val_loader, device=device)
        print("alpha: {}, train_data: {}, val_data: {}".format(alpha, train_data, val_data))

        train.append(train_data)
        val.append(val_data)

    return train, val

# Function to interpolate model weights
def interpolate_model(model, initial_model, final_model, alpha):
    # Interpolate between initial and final weights
    # print("alpha: {}".format(alpha))
    for param_initial, param_final, (name_model, param_model) in zip(initial_model.parameters(), final_model.parameters(), model.named_parameters()):
        param_model.data = alpha * param_final.data + (1 - alpha) * param_initial.data

def preview_model_params(model):
    for name_model, param_model in model.named_parameters():
        print("{}: {}".format(name_model, param_model.data.view(-1)[:5]))
