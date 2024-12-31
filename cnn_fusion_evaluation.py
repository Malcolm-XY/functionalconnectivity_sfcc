# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:40:30 2024

@author: usouu
"""
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def train_model_fusion(model, data2d, data3d, labels, batch_size=128, learning_rate=0.0005, weight_decay=1e-4, step_size=5, gamma=0.5, epochs=30):
    """
    Train a PyTorch model.

    Parameters:
        model (torch.nn.Module): The model to train.
        data (array-like or torch.Tensor): The input data.
        labels (array-like or torch.Tensor): The corresponding labels.
        batch_size (int, optional): Batch size for the DataLoader. Default is 128.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 0.0005.
        weight_decay (float, optional): Weight decay (L2 regularization) for the optimizer. Default is 1e-4.
        step_size (int, optional): Step size for the learning rate scheduler. Default is 5.
        gamma (float, optional): Multiplicative factor of learning rate decay. Default is 0.5.
        epochs (int, optional): Number of training epochs. Default is 30.

    Returns:
        torch.nn.Module: The trained model.
    """
    # Convert data and labels to tensors if not already
    data2d = torch.as_tensor(data2d, dtype=torch.float32).clone().detach()
    data3d = torch.as_tensor(data3d, dtype=torch.float32).clone().detach()
    labels = torch.as_tensor(labels, dtype=torch.long).clone().detach()

    # Create DataLoader for batching
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(data2d, data3d, labels),
        batch_size=batch_size,
        shuffle=True
    )

    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_data2d, batchdata3d, batch_labels in train_loader:
            batch_data2d, batchdata3d, batch_labels = batch_data2d.to(device), batchdata3d, batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data2d, batchdata3d)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Step the learning rate scheduler
        scheduler.step()

        # Log epoch loss
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    return model

def test_model_fusion(model, data2d, data3d, labels, batch_size=128):
    """
    Test a PyTorch model with 2D and 3D input data.

    Parameters:
        model (torch.nn.Module): The trained model to evaluate.
        data2d (array-like or torch.Tensor): The 2D input data for testing.
        data3d (array-like or torch.Tensor): The 3D input data for testing.
        labels (array-like or torch.Tensor): The corresponding labels for testing.
        batch_size (int, optional): Batch size for the DataLoader. Default is 128.

    Returns:
        dict: A dictionary containing:
            - 'average_loss': The average loss over the test dataset.
            - 'accuracy': The accuracy of the model on the test dataset.
            - 'precision': The overall precision of the model.
            - 'recall': The overall recall of the model.
            - 'f1_score': The overall F1 score of the model.
            - 'classification_report': A detailed classification report.
            - 'predictions': List of all predicted labels.
            - 'true_labels': List of all true labels.
    """
    import torch

    # Convert data and labels to tensors if not already
    data2d = torch.as_tensor(data2d, dtype=torch.float32).clone().detach()
    data3d = torch.as_tensor(data3d, dtype=torch.float32).clone().detach()
    labels = torch.as_tensor(labels, dtype=torch.long).clone().detach()

    # Create DataLoader for batching
    test_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(data2d, data3d, labels),
        batch_size=batch_size,
        shuffle=False
    )

    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Evaluation mode
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch_data2d, batch_data3d, batch_labels in test_loader:
            batch_data2d, batch_data3d, batch_labels = (
                batch_data2d.to(device),
                batch_data3d.to(device),
                batch_labels.to(device)
            )

            # Forward pass
            outputs = model(batch_data2d, batch_data3d)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()

            # Calculate predictions
            _, predicted = torch.max(outputs, dim=1)
            all_predictions.extend(predicted.cpu().tolist())
            all_true_labels.extend(batch_labels.cpu().tolist())

            # Calculate accuracy
            correct_predictions += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

    # Compute average loss and accuracy
    average_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples

    # Calculate additional metrics
    precision = precision_score(all_true_labels, all_predictions, average='weighted')
    recall = recall_score(all_true_labels, all_predictions, average='weighted')
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    class_report = classification_report(all_true_labels, all_predictions, output_dict=True)

    print(f"Test Loss: {average_loss:.4f}")
    print(f"Accuracy: {accuracy:.4%}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Return detailed results
    return {
        'average_loss': average_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': class_report,
        'predictions': all_predictions,
        'true_labels': all_true_labels
    }

def k_fold_evaluation_fusion(model, data2d, data3d, labels, k_folds=5, batch_size=128):
    """
    Perform k-fold cross-validation on a PyTorch fusion model with 2D and 3D input data.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        data2d (array-like or torch.Tensor): The 2D input data.
        data3d (array-like or torch.Tensor): The 3D input data.
        labels (array-like or torch.Tensor): The corresponding labels.
        k_folds (int, optional): Number of folds for cross-validation. Default is 5.
        batch_size (int, optional): Batch size for training and testing. Default is 128.

    Returns:
        dict: A dictionary containing:
            - 'average_accuracy': The average accuracy across all folds.
            - 'average_precision': The average precision across all folds.
            - 'average_recall': The average recall across all folds.
            - 'average_f1_score': The average F1 score across all folds.
            - 'fold_results': List of detailed results for each fold.
    """

    # Convert data and labels to tensors if not already
    data2d = torch.as_tensor(data2d, dtype=torch.float32).clone().detach()
    data3d = torch.as_tensor(data3d, dtype=torch.float32).clone().detach()
    labels = torch.as_tensor(labels, dtype=torch.long).clone().detach()

    # Calculate fold size and indices
    fold_size = len(data2d) // k_folds
    indices = list(range(len(data2d)))

    # Initialize results storage
    fold_results = []

    for fold in range(k_folds):
        print(f'Fold {fold + 1}/{k_folds}')
        
        # Split data into training and testing sets for the current fold
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < k_folds - 1 else len(data2d)
        test_indices = indices[test_start:test_end]
        train_indices = indices[:test_start] + indices[test_end:]

        data_train2d, data_test2d = data2d[train_indices], data2d[test_indices]
        data_train3d, data_test3d = data3d[train_indices], data3d[test_indices]
        labels_train, labels_test = labels[train_indices], labels[test_indices]

        # Reinitialize the model for each fold
        model_copy = type(model)()  # Create a new instance of the model
        model_copy.load_state_dict(model.state_dict())  # Copy weights from the original model

        # Train and evaluate the model
        trained_model = train_model_fusion(
            model_copy, data_train2d, data_train3d, labels_train, batch_size=batch_size
        )
        results = test_model_fusion(trained_model, data_test2d, data_test3d, labels_test, batch_size=batch_size)

        # Store results for the current fold
        fold_results.append({
            'fold': fold + 1,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score']
        })

        print(f"Fold {fold + 1}/{k_folds} Results:")
        print(f"  Accuracy: {results['accuracy']:.4%}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1 Score: {results['f1_score']:.4f}")

    # Calculate average metrics
    average_accuracy = sum([result['accuracy'] for result in fold_results]) / k_folds
    average_precision = sum([result['precision'] for result in fold_results]) / k_folds
    average_recall = sum([result['recall'] for result in fold_results]) / k_folds
    average_f1_score = sum([result['f1_score'] for result in fold_results]) / k_folds

    print(f'\nK-Fold Cross-Validation Results for {k_folds} Folds:')
    print(f'  Average Accuracy: {average_accuracy:.4%}')
    print(f'  Average Precision: {average_precision:.4f}')
    print(f'  Average Recall: {average_recall:.4f}')
    print(f'  Average F1 Score: {average_f1_score:.4f}\n')

    return {
        'average_accuracy': average_accuracy,
        'average_precision': average_precision,
        'average_recall': average_recall,
        'average_f1_score': average_f1_score,
        'fold_results': fold_results
    }

