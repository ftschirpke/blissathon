from app.model import OurTorchModel
from loading_and_saving import load_test_data, load_train_data, to_file
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from configuration import ROOT_PATH
from util_functions import plot_image, plot_x_y


def process_data(arr: torch.Tensor) -> torch.Tensor:
    arr = arr.to('cpu')
    arr = torch.unsqueeze(arr, dim=1).float()
    return arr


def main():
    train_loader = load_train_data(batch_size=10, upscaled=True)

    resnet = OurTorchModel()

    resnet = resnet.to('cpu')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Number of training epochs
    num_epochs = 10

# Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs = process_data(inputs)
            labels = process_data(labels)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

            print(f'Epoch [{epoch + 1}, Batch {i + 1}] Loss: {running_loss * 100:.3f}')
            running_loss = 0.0

            if epoch < num_epochs - 1:
                continue

            if i <= 15:
                continue

            plt.figure()

            plt.subplot(131)
            plt.axis("off")
            plt.imshow(inputs.squeeze().cpu()[0], cmap="copper")
            plt.title("input")

            plt.subplot(132)
            plt.axis("off")
            plt.imshow(labels.squeeze().cpu()[0], cmap="copper")
            plt.title("labels")

            plt.subplot(133)
            plt.axis("off")
            plt.imshow(outputs.detach().squeeze().cpu().numpy()[0], cmap="copper")
            plt.title("network output")

            plt.tight_layout()
            plt.show()

        print(f"Finished epoch {epoch + 1} of {num_epochs}")

    print('Finished Training')

    # torch.save(resnet.state_dict(), 'trained_resnet.pth')

    test_x = torch.load(ROOT_PATH / "data/torch/test_damaged_upscaled.pt")
    test_x = process_data(test_x)

    test_y = resnet(test_x)
    print(test_y.shape)
    to_file(test_y, "submission.csv")

    plot_x_y(test_x.squeeze().cpu()[0], test_y.detach().squeeze().cpu().numpy()[0], "test")


def not_main() -> None:
    model = OurTorchModel()
    model.load_state_dict(torch.load(ROOT_PATH / "trained_resnet.pth", map_location=torch.device('cpu')))
    model.eval()
    print("model loaded")

    test_loader = load_test_data(batch_size=10, upscaled=True)
    print("test data loaded")

    collect = []
    for i, data in enumerate(test_loader, 0):
        data = process_data(data)
        print(data.shape)
        pred = model(data)
        print(pred.shape)
        collect.append(pred.cpu())
        to_file(pred, f"prediction_{i}.csv")

    # test_x = torch.load(ROOT_PATH / "data/torch/test_damaged_upscaled.pt")
    # test_x = process_data(test_x)
    # print(test_x.shape)

    # batches = torch.split(test_x, 18, dim=0)
    # res_gen = (model(batch) for batch in batches)
    # print("model ran almost")

    # test_y = next(res_gen)

    # print("model ran")
    test_y = torch.cat(collect, dim=0)
    print(test_y.shape)

    to_file(test_y, "submission.csv")
    print("file saved")


if __name__ == "__main__":
    # main()
    not_main()
