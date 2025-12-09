import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import GraphConv, Sequential
from torch.nn.functional import mse_loss
from sklearn.model_selection import train_test_split
from Data_preparation import graphs

def train_val_test_split_graphs(graphs, val_frac=0.10, test_frac=0.15, random_state=42):
    """
    Split graphs into train, validation, and test sets.
    Fractions are relative to the full dataset.
    """
    indices = np.arange(len(graphs))

    # First split off the test set
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_frac,
        random_state=random_state,
    )

    # Then split the remaining into train and validation
    val_size_rel = val_frac / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size_rel,
        random_state=random_state,
    )

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    return train_graphs, val_graphs, test_graphs


def build_GNN_model(input_dim, hidden_dim, output_dim, num_layers):
    """
    Build a GNN model with edge attributes using GraphConv.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (usually 1 for regression)
        num_layers: Number of graph convolution layers
    
    Returns:
        model: A PyTorch Sequential model
    """
    layers = []
    
    # First layer: input_dim -> hidden_dim with signature
    # GraphConv signature: x, edge_index, edge_weight -> x
    layers.append((GraphConv(input_dim, hidden_dim), 'x, edge_index, edge_weight -> x'))
    layers.append(torch.nn.ReLU())
    
    # Middle layers: hidden_dim -> hidden_dim
    for _ in range(num_layers - 2):
        layers.append((GraphConv(hidden_dim, hidden_dim), 'x, edge_index, edge_weight -> x'))
        layers.append(torch.nn.ReLU())
    
    # Output layer: hidden_dim -> output_dim
    layers.append((GraphConv(hidden_dim, output_dim), 'x, edge_index, edge_weight -> x'))
    
    # Create Sequential model
    model = Sequential('x, edge_index, edge_weight', layers)
    
    return model


def train_model(model, data_list, optimizer, device):
    """
    Train the GNN model for one epoch.
    
    Args:
        model: The GNN model to train
        data_list: List of PyTorch Geometric Data objects for training
        optimizer: Optimizer (e.g., Adam)
        device: Device to run on (cpu or cuda)
    
    Returns:
        total_loss / len(data_list): Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for data in data_list:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass (GraphConv expects 1D edge_weight, use first feature)
        edge_weight = data.edge_attr[:, 0] if data.edge_attr.dim() > 1 else data.edge_attr
        out = model(data.x, data.edge_index, edge_weight)  # [N, output_dim]
        
        # Compute loss
        loss = mse_loss(out, data.y)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_list)


def evaluate_model(model, data_list, device):
    """
    Evaluate the GNN model on test data.
    
    Args:
        model: The GNN model to evaluate
        data_list: List of PyTorch Geometric Data objects for testing
        device: Device to run on (cpu or cuda)
    
    Returns:
        mae: Mean Absolute Error on the test set
    """
    model.eval()
    total_mae = 0.0
    
    with torch.no_grad():
        for data in data_list:
            data = data.to(device)
            
            # Forward pass (extract 1D edge weight from multi-dimensional edge_attr)
            edge_weight = data.edge_attr[:, 0] if data.edge_attr.dim() > 1 else data.edge_attr
            out = model(data.x, data.edge_index, edge_weight)  # [N, output_dim]
            
            # Compute MAE
            mae = torch.mean(torch.abs(out - data.y))
            total_mae += mae.item()
    
    return total_mae / len(data_list)


def evaluate_loss(model, data_list, device):
    """Compute average MSE loss on a dataset without gradients."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in data_list:
            data = data.to(device)
            edge_weight = data.edge_attr[:, 0] if data.edge_attr.dim() > 1 else data.edge_attr
            out = model(data.x, data.edge_index, edge_weight)
            loss = mse_loss(out, data.y)
            total_loss += loss.item()
    return total_loss / len(data_list)


if __name__ == "__main__":
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Split graphs into train and test
    train_graphs, test_graphs = train_test_split_graphs(graphs, test_size=0.2, random_state=42)
    print(f"Training graphs: {len(train_graphs)}, Test graphs: {len(test_graphs)}")
    
    # Get input dimension from the first graph
    input_dim = train_graphs[0].x.shape[1]
    output_dim = train_graphs[0].y.shape[1]
    hidden_dim = 128
    num_layers = 3
    
    print(f"Input dimension: {input_dim}")
    
    # Build model
    model = build_GNN_model(input_dim, hidden_dim, output_dim, num_layers)
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    num_epochs = 50
    print("\nTraining started...")
    
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_graphs, optimizer, device)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.6f}")
    
    # Evaluation
    print("\nTraining completed. Evaluating on test set...")
    test_mae = evaluate_model(model, test_graphs, device)
    print(f"Testing MAE: {test_mae:.6f}")

    # Save trained model weights
    # model_path = os.path.join(os.path.dirname(__file__), "gnn_model.pt")
    # torch.save(model.state_dict(), model_path)
    # print(f"Saved model weights to {model_path}")
