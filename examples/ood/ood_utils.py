
import os

import numpy as np
import torch

from examples.mnist.utils import (
    get_mnist_dataloader,
    get_fmnist_dataloader,
)

MNIST = "mnist"
FMNIST = "fmnist"
CIFAR10 = "cifar10"
DATA_LIST = [MNIST, FMNIST, CIFAR10]


def compute_input_stats(dataloader: torch.utils.data.DataLoader):
    """
    Compute the mean and standard deviation of the input data.
    """
    # Initialize sum and sum of squares tensors
    sum_tensor = torch.zeros_like(next(iter(dataloader))[0][0])
    sum_sq_tensor = torch.zeros_like(sum_tensor)
    
    # Make sure the dimensions are (channel, height, width)
    assert len(sum_tensor.shape) == 3

    num_images = 0
    # Loop over the DataLoader
    for images, _ in dataloader:
        sum_tensor += images.sum(dim=0)  # Summing over the batch dimension
        sum_sq_tensor += (images ** 2).sum(dim=0)  # Sum of squares
        num_images += images.size(0)  # Counting the number of images

    # Calculate the mean and standard deviation
    mean_per_pixel = sum_tensor / num_images
    std_per_pixel = torch.sqrt((sum_sq_tensor / num_images) - (mean_per_pixel ** 2))
    
    # Make sure the dimensions are (channel, height, width)
    assert len(mean_per_pixel.shape) == 3
    
    return mean_per_pixel.detach().cpu().numpy(), std_per_pixel.detach().cpu().numpy()
    

def save_input_stats(dataloader: torch.utils.data.DataLoader, save_name: str):
    """
    Compute and save the mean and standard deviation of the input data.
    """
    mean_per_pixel, std_per_pixel = compute_input_stats(dataloader)
    np.save(f'{save_name}_mean.npy', mean_per_pixel)
    np.save(f'{save_name}_std.npy', std_per_pixel)


def get_ood_input_processor(source_data: str, target_model: str):
    """
    Process OOD input data for the target model.
    """
    assert source_data in DATA_LIST
    assert target_model in DATA_LIST
    for d in [source_data, target_model]:
        if not os.path.exists(f"{d}_train_mean.npy") or not os.path.exists(f"{d}_train_std.npy"):
            if d == MNIST:
                dataloader = get_mnist_dataloader(batch_size=10000, split="train", shuffle=False)
            elif d == FMNIST:
                dataloader = get_fmnist_dataloader(batch_size=10000, split="train", shuffle=False)
            else:
                raise ValueError(f"Unsupported data: {d}")    
            save_input_stats(dataloader, f"{d}_train")
        
    if source_data == "mnist":
        mnist_tr_mean, mnist_tr_std = np.load("mnist_train_mean.npy"), np.load("mnist_train_std.npy")
        source_tr_mean = torch.from_numpy(mnist_tr_mean)
        source_tr_std = torch.from_numpy(mnist_tr_std)
        if target_model == "fmnist":
            fmnist_tr_mean, fmnist_tr_std = np.load("fmnist_train_mean.npy"), np.load("fmnist_train_std.npy")
            target_tr_mean = torch.from_numpy(fmnist_tr_mean)
            target_tr_std = torch.from_numpy(fmnist_tr_std)
        else:
            raise ValueError(f"Unsupported target model: {target_model}")
    elif source_data == "fmnist":
        fmnist_tr_mean, fmnist_tr_std = np.load("fmnist_train_mean.npy"), np.load("fmnist_train_std.npy")
        source_tr_mean = torch.from_numpy(fmnist_tr_mean)
        source_tr_std = torch.from_numpy(fmnist_tr_std)
        if target_model == "mnist":
            mnist_tr_mean, mnist_tr_std = np.load("mnist_train_mean.npy"), np.load("mnist_train_std.npy")
            target_tr_mean = torch.from_numpy(mnist_tr_mean)
            target_tr_std = torch.from_numpy(mnist_tr_std)
        else:
            raise ValueError(f"Unsupported target model: {target_model}")
    else:
        raise ValueError(f"Unsupported source data: {source_data}")
    
    def ood_input_processor(x):
        device = x.device
        x_transform = (x - source_tr_mean.to(device)) / source_tr_std.to(device)
        x_transform = x_transform * target_tr_std.to(device) + target_tr_mean.to(device)
        return x_transform
    
    return ood_input_processor
    

if __name__ == "__main__":
    import sys
    
    import numpy as np
    import matplotlib.pyplot as plt 
    from examples.mnist.utils import (
        get_mnist_dataloader,
        get_fmnist_dataloader,
    )
    fmnist_te_loader = get_fmnist_dataloader(batch_size=10000, split="valid", shuffle=False)
    mnist_te_loader = get_mnist_dataloader(batch_size=10000, split="valid", shuffle=False)
    fmnist_tr_loader = get_fmnist_dataloader(batch_size=10000, split="train", shuffle=False)
    mnist_tr_loader = get_mnist_dataloader(batch_size=10000, split="train", shuffle=False)
    
    mnist_tr_mean, mnist_tr_std = np.load('mnist_tr_mean.npy'), np.load('mnist_tr_std.npy')
    mnist_tr_mean = torch.from_numpy(mnist_tr_mean)
    mnist_tr_std = torch.from_numpy(mnist_tr_std)
    
    fmnist_tr_mean, fmnist_tr_std = np.load('fmnist_tr_mean.npy'), np.load('fmnist_tr_std.npy')
    fmnist_tr_mean = torch.from_numpy(fmnist_tr_mean)
    fmnist_tr_std = torch.from_numpy(fmnist_tr_std)
    
    
    fmnist_to_mnist_fn = get_ood_input_processor(source_data="fmnist", target_model="mnist")
    for x, y in fmnist_te_loader:
        print(x.shape, y.shape)
        
        # x_transform = (x - fmnist_tr_mean) / fmnist_tr_std
        # x_transform = x_transform * mnist_tr_std + mnist_tr_mean
        x_transform = fmnist_to_mnist_fn(x)
        x_mean = torch.mean(x_transform, dim=0)
        x_std = torch.std(x_transform, dim=0)
        # print(x_mean)
        # print(x_std)

        # img = x_transform[0].squeeze()  # Remove the channel dimension
        # plt.imshow(img, cmap='gray')
        # plt.title(f"Label: {y[0]}")
        # plt.axis('off')
        # plt.show()
        break
    
    fig, axs = plt.subplots(1, 2, figsize=(5, 2))
    axs[0].plot(x_mean.flatten().numpy(), label="fmnist")
    axs[1].plot(x_std.flatten().numpy(), label="fmnist")
    
    for x, y in mnist_te_loader:
        print(x.shape, y.shape)
        x_mean = torch.mean(x, dim=0)
        x_std = torch.std(x, dim=0)
        # print(x_mean)
        # print(x_std)
        # img = x[0].squeeze()  # Remove the channel dimension
        # plt.imshow(img, cmap='gray')
        # plt.title(f"Label: {y[0]}")
        # plt.axis('off')
        # plt.show()
        break
    
    axs[0].plot(x_mean.flatten().numpy(), label="mnist")
    axs[1].plot(x_std.flatten().numpy(), label="mnist")
    
    mnist_mean, mnist_std = compute_input_stats(mnist_te_loader)
    breakpoint()
    
    axs[0].legend()
    axs[1].legend()
    plt.show()