import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as SGD

import matplotlib.pyplot as plt
import seaborn as sns


class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.bf = nn.Parameter(torch.tensor(-16.0), requires_grad=False)

    def forward(self, input):
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = (
            scaled_top_relu_output + scaled_bottom_relu_output + self.bf
        )
        final_output = F.relu(input_to_final_relu)

        return final_output


class BasicNN_train(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.bf = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, input):
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = (
            scaled_top_relu_output + scaled_bottom_relu_output + self.bf
        )
        final_output = F.relu(input_to_final_relu)

        return final_output


def plot_model(model):
    plt_x_vals = torch.linspace(
        start=0,
        end=1,
        steps=25,
    )
    plt_y_vals = model(plt_x_vals).detach()

    sns.set_theme(style="whitegrid")
    sns.lineplot(
        x=plt_x_vals,
        y=plt_y_vals,
        color="green",
        linewidth=2.5,
    )
    plt.ylabel("Efficacy")
    plt.xlabel("Dose")

    plt.show()


def optimize_model(model, inputs, labels, num_epochs=100, threshold=0.01):
    optimizer = SGD.SGD(model.parameters(), lr=0.1)
    print(f"Final bias, before optimization: {str(model.bf)}")

    for epoch in range(num_epochs):
        epoch_loss = 0
        for x, y in zip(inputs, labels):
            out = model(x)
            residual = out - y
            loss = residual * residual
            loss.backward()  # will accumulate the gradient as we go through this loop
            epoch_loss += loss

        if epoch_loss < threshold:
            print(f"Threshold reached at epoch {epoch}")
            break

        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch}: Loss = {epoch_loss}, bias = {model.bf}")


model = BasicNN_train()
inputs = torch.tensor([0.0, 0.5, 1.0])
labels = torch.tensor([0.0, 1.0, 0.0])

print(f"input = {inputs}")
print(f"labels = {labels}")

optimize_model(model, inputs, labels, num_epochs=1000, threshold=0.01)
plot_model(model)
