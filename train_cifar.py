import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import multiprocessing
from tqdm import tqdm
import time

from sklearn.metrics import precision_score, recall_score, f1_score

from Galore.adamw_optimizer import AdamW

def count_flops_macs(model, input_shape, device):
    return None

def train():
    torch.manual_seed(35)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on {device}")
    # Define transforms for data augmentation and normalization
    transform_train = transforms.Compose([
        # transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)

    # Define ResNet-18 model
    model = models.vit_h_14(pretrained=False, num_classes=10)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
    # optimizer = optim.AdamW(model.parameters(), lr=0.001)
    param_groups = [
                    # {'params': model.parameters()},
                    {'params': model.parameters(), 'rank': 16, 'update_projection_step': 300, 'scale': 0.25}
                    ]
    optimizer = AdamW(param_groups, lr=0.005, weight_decay=0.005)

    scaler = torch.cuda.amp.GradScaler()

    epochs = 10
    mixed_precision = 'bf16'

    # Train the model
    model.train()

    # Calculate and log FLOPs and MACs
    input_shape = trainloader.dataset[0][0].unsqueeze(0).shape  # Get sample input size of your model by input_size = trainloader.dataset[0][0].size()ple input size of your model by input_size = trainloader.dataset[0][0].size()
    print(input_shape)
    # flops, macs = count_flops_macs(model, input_shape, device=device)
    # print(f"Total FLOPs: {flops:,}")
    # print(f"Total MACs: {macs:,}")

    if device == torch.device('mps'):
        print(f"Memory: {torch.mps.current_allocated_memory() / 1e9:.4f} GB")
    elif device == torch.device('cuda'):
        print(f"Memory: {torch.cuda.memory_allocated() / 1e9:.4f} GB")
    else:
        print("Memory: 0 GB")

    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")  # Sum the number of parameters
    print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for k,v in model.named_parameters():
        print(k, v.shape)

    # # Training...
    start_time = time.time()
    for epoch in range(epochs):  # Loop over the dataset multiple times
        running_loss = 0.0
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for i, data in pbar:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            if mixed_precision == 'fp16' or mixed_precision == 'bf16':
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16 if mixed_precision=='bf16' else torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # Scales the loss, and calls backward()
                # to create scaled gradients
                scaler.scale(loss).backward()

                # Unscales gradients and calls
                # or skips optimizer.step()
                scaler.step(optimizer)

                # Updates the scale for next iteration
                scaler.update()
            else:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

            # Print statistics
            running_loss += loss.item()

            if device == torch.device('mps'):
                memory_usage = torch.mps.current_allocated_memory() / 1e9
            elif device == torch.device('cuda'):
                memory_usage = torch.cuda.memory_allocated() / 1e9
            else: 
                memory_usage = 0.0
            
            pbar.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / (i + 1):.4f}, Mem GPU: {memory_usage:.4f} GB")
    end_time = time.time()
    total_time = end_time - start_time
    print('Finished Training. Time(minutes): %.2f' % (total_time/60))

    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        pbar = tqdm(testloader, total=len(testloader))
        for data in pbar:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
            
            accuracy = 100 * correct / total
            precision = precision_score(all_labels, all_predicted, average='macro')
            recall = recall_score(all_labels, all_predicted, average='macro')
            f1 = f1_score(all_labels, all_predicted, average='macro')
            
            pbar.set_description(f'Acc: {accuracy:.2f}%, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}')
            pbar.set_description(f'Acc: {accuracy:.2f}%')


    print('Accuracy of the network on the 10000 test images: %.2f %%' % accuracy)
    print('Precision: %.4f' % precision)
    print('Recall: %.4f' % recall)
    print('F1 Score: %.4f' % f1)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    train()
