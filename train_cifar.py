import mlflow.system_metrics
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from tqdm import tqdm
import time
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score
from Galore.adamw_optimizer import AdamW
import mlflow


def train(args: argparse.ArgumentParser):  
    
    # mlflow.set_tracking_uri(args.mlflow_tracking_url)
    mlflow.set_experiment(args.mlflow_experiment_name)

    torch.set_float32_matmul_precision(precision=args.precision)
    torch.manual_seed(args.seed)

    print("="*50)
    print("===Training config===")
    for k,v in args._get_kwargs():
        print(f"{k}: {v}")

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on {device}")
    # Define transforms for data augmentation and normalization
    transform_train = transforms.Compose([
        # transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)
    # Define ResNet-18 model
    model = models.resnet18(pretrained=False, num_classes=10)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.enable_galore:
        param_groups = [
                        {'params': model.parameters(), 'rank': args.rank, 'update_projection_step': args.update_projection_step, 'scale': args.galore_scale}
                        ]
    else:
        param_groups = [
            {'params': model.parameters()},
        ]
    
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler()
    # Train the model
    model.train()

    if device == torch.device('mps'):
        print(f"Memory: {torch.mps.current_allocated_memory() / 1e9:.4f} GB")
    elif device == torch.device('cuda'):
        print(f"Memory: {torch.cuda.memory_allocated() / 1e9:.4f} GB")
    else:
        print("Memory: 0 GB")

    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")  # Sum the number of parameters
    print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training...
    with mlflow.start_run(run_name=args.mlflow_run_name):
        for k,v in args._get_kwargs():
            mlflow.log_param(key=k, value=v)
        
        start_time = time.time()
        steps = 0
        for epoch in range(args.epochs):  # Loop over the dataset multiple times
            running_loss = 0.0
            pbar = tqdm(enumerate(trainloader), total=len(trainloader))
            for i, data in pbar:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

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

                steps += 1
                mlflow.log_metric('Mem usage', memory_usage, step=steps)
                mlflow.log_metric('Train Loss', loss, step=steps)

                pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}, Loss: {running_loss / (i + 1):.4f}, Mem GPU: {memory_usage:.4f} GB")

        end_time = time.time()
        total_time = end_time - start_time
        print('Finished Training. Time(minutes): %.2f' % (total_time/60))
        mlflow.log_metric('Time taken', total_time/60)

        # Evaluate the model on the test set
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predicted = []

        with torch.no_grad():
            running_test_loss = 0.0
            pbar = tqdm(enumerate(testloader), total=len(testloader))
            for i, data in pbar:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())
                
                accuracy = 100 * correct / total
                precision = precision_score(all_labels, all_predicted, average='macro')
                recall = recall_score(all_labels, all_predicted, average='macro')
                f1 = f1_score(all_labels, all_predicted, average='macro')
                
                running_test_loss += loss.item()

                pbar.set_description(f'Test Loss: {running_loss / (i + 1):.4f}, Acc: {accuracy:.2f}%, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}')


        print('Accuracy of the network on the 10000 test images: %.2f %%' % accuracy)
        print('Precision: %.4f' % precision)
        print('Recall: %.4f' % recall)
        print('F1 Score: %.4f' % f1)

        mlflow.log_metric('Test Accuracy', accuracy)
        mlflow.log_metric('Precision', precision)
        mlflow.log_metric('Recall', recall)
        mlflow.log_metric('F1 Score', f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--precision', type=str, default='highest', choices=['medium', 'high', 'highest'], help='Mixed precision mode')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay')
    parser.add_argument('--enable_galore', action='store_true', help='Enable Galore optimizer')
    parser.add_argument('--rank', type=int, default=16, help='Rank for Galore optimizer')
    parser.add_argument('--update_projection_step', type=int, default=200, help='Update projection step for Galore optimizer')
    parser.add_argument('--galore_scale', type=float, default=0.2, help='Scale for Galore optimizer')
    
    # mlflow related arguments
    parser.add_argument('--mlflow_experiment_name', type=str, help='mlflow experiment name')
    parser.add_argument('--mlflow_run_name', type=str, help='mlflow run name')
    parser.add_argument('--mlflow_tracking_url', type=str, default='http://127.0.0.1:5000')
    
    args = parser.parse_args()
    # multiprocessing.freeze_support()
    train(args)
