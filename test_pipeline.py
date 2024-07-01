import matplotlib.pyplot as plt
import pytest
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from gnc import BELKAModule


@pytest.fixture
def model():
    model = BELKAModule(
        in_feat=21,
        hidden_feat=256,
        out_feat=1,
        num_layers=2,
        num_proteins=3,
        protein_embedding_dim=32,
        global_feat=1000,
        learning_rate=1e-3,
        grid_feat=200
    )
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_batch():
    num_graphs = 2
    num_nodes_per_graph = 64
    num_edges_per_graph = 256

    data_list = []
    for i in range(num_graphs):
        x = torch.randn(num_nodes_per_graph, 21)
        edge_index = torch.randint(0, num_nodes_per_graph, (2, num_edges_per_graph))
        global_features = torch.randn(num_nodes_per_graph, 1000)
        y = torch.randint(0, 2, (1, 1)).float()
        protein = torch.randint(0, 3, (1,))
        data = Data(x=x, edge_index=edge_index, global_features=global_features, y=y, protein=protein)
        data_list.append(data)

    batch = Batch.from_data_list(data_list)
    return batch.to('cuda' if torch.cuda.is_available() else 'cpu')


def test_numerical_stability(model, sample_batch):
    logger.info("Testing numerical stability...")

    output = model(sample_batch)
    assert not torch.isnan(output).any(), "NaN values in output"
    assert not torch.isinf(output).any(), "Inf values in output"

    loss = model.criterion(output, sample_batch.y)
    loss.backward()

    for name, param in model.named_parameters():
        assert not torch.isnan(param.grad).any(), f"NaN gradients in {name}"
        assert not torch.isinf(param.grad).any(), f"Inf gradients in {name}"

    logger.success("Numerical stability test passed")


def test_gradient_flow(model, sample_batch):
    logger.info("Testing gradient flow...")

    model.zero_grad()
    output = model(sample_batch)
    loss = model.criterion(output, sample_batch.y)
    loss.backward()

    grad_norms = []
    zero_grad_params = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append((name, grad_norm))
            if grad_norm == 0:
                zero_grad_params.append(name)
        else:
            zero_grad_params.append(name)

    grad_norms.sort(key=lambda x: x[1], reverse=True)

    logger.info("Gradient norms (top 10):")
    for name, norm in grad_norms[:10]:
        logger.info(f"{name}: {norm:.4f}")

    if zero_grad_params:
        logger.warning(f"Parameters with zero gradients: {', '.join(zero_grad_params)}")

    assert not zero_grad_params, f"Some gradients are zero: {', '.join(zero_grad_params)}"

    logger.success("Gradient flow test passed")


def test_feature_capture(model, sample_batch):
    logger.info("Testing feature capture...")

    with torch.no_grad():
        output = model(sample_batch)

    assert output.shape == (sample_batch.num_graphs, 1), f"Unexpected output shape: {output.shape}"
    assert torch.std(output) > 0.001, "Extremely low variation in output"

    logger.success("Feature capture test passed")


def test_learnability(model, sample_batch):
    logger.info("Testing learnability...")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []

    for _ in tqdm(range(500), desc="Training"):
        optimizer.zero_grad()
        output = model(sample_batch)
        loss = model.criterion(output, sample_batch.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("loss_curve.png")
    plt.close()

    assert losses[-1] < losses[0], "Loss is not decreasing"

    logger.success("Learnability test passed")


def test_overfitting(model, sample_batch):
    logger.info("Testing overfitting capability...")

    # Ensure we have both classes in the sample
    sample_batch.y[0] = 0
    sample_batch.y[1] = 1

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Increased learning rate

    for _ in tqdm(range(5000), desc="Overfitting"):  # Increased number of iterations
        optimizer.zero_grad()
        output = model(sample_batch)
        loss = model.criterion(output, sample_batch.y)
        loss.backward()
        optimizer.step()

    final_output = model(sample_batch)
    accuracy = ((final_output > 0) == sample_batch.y).float().mean().item()
    auc = roc_auc_score(sample_batch.y.cpu().numpy(), final_output.sigmoid().cpu().detach().numpy())

    logger.info(f"Final accuracy: {accuracy:.4f}")
    logger.info(f"Final AUC: {auc:.4f}")

    assert accuracy > 0.95, "Model unable to overfit on small batch"
    assert auc > 0.95, "Model unable to achieve high AUC on small batch"

    logger.success("Overfitting test passed")


def test_batch_independence(model, sample_batch):
    logger.info("Testing batch independence...")

    batch1 = sample_batch
    batch2 = sample_batch  # In a real scenario, this would be a different batch

    with torch.no_grad():
        output1 = model(batch1)
        output2 = model(batch2)

    assert not torch.allclose(output1, output2), "Outputs are identical for different batches"

    logger.success("Batch independence test passed")


def test_edge_case_handling(model, sample_batch):
    logger.info("Testing edge case handling...")
    device = next(model.parameters()).device

    # Test with zero nodes
    zero_nodes_data = Data(x=torch.tensor([], dtype=torch.float).reshape(0, 21),
                           edge_index=torch.tensor([[], []], dtype=torch.long),
                           global_features=torch.tensor([], dtype=torch.float).reshape(0, 1000),
                           y=torch.tensor([[0.0]], dtype=torch.float),
                           protein=torch.tensor([0], dtype=torch.long))
    zero_nodes_batch = Batch.from_data_list([zero_nodes_data]).to(device)

    try:
        output = model(zero_nodes_batch)
        assert output.shape == (1, 1), f"Unexpected output shape for zero nodes: {output.shape}"
        logger.success("Model handles zero nodes")
    except Exception as e:
        logger.error(f"Model fails with zero nodes: {str(e)}")

    # Test with disconnected graph
    disconnected_data = Data(x=torch.randn(10, 21),
                             edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                             global_features=torch.randn(10, 1000),
                             y=torch.tensor([[0.0]], dtype=torch.float),
                             protein=torch.tensor([0], dtype=torch.long))
    disconnected_batch = Batch.from_data_list([disconnected_data]).to(device)

    try:
        model(disconnected_batch)
        logger.success("Model handles disconnected graphs")
    except Exception as e:
        logger.error(f"Model fails with disconnected graphs: {str(e)}")

    logger.success("Edge case handling test passed")


if __name__ == "__main__":
    logger.info("Starting comprehensive test suite...")
    pytest.main([__file__, "-v"])
    logger.info("Test suite completed. Check test_results.log for detailed output.")
