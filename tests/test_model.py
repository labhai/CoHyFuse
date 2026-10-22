import copy

import pytest
import torch

from cohyfuse import CoHyFuse
from cohyfuse.model import ConditionHypergraph, HypergraphConvolution


def test_incidence_memberships_and_temperature_gradients():
    torch.manual_seed(0)
    nodes = (torch.randn(2, 9, 6) * 0.2).requires_grad_()
    builder = ConditionHypergraph([3, 5])
    for q, k in enumerate([3, 5]):
        incidence = builder(nodes, q)
        torch.testing.assert_close(incidence.diagonal(dim1=1, dim2=2), torch.ones(2, 9))
        assert ((incidence > 0).sum(dim=1) == k + 1).all()
        torch.testing.assert_close(incidence.sum(dim=1), torch.full((2, 9), float(k + 1)))
    incidence.square().sum().backward()
    assert nodes.grad.isfinite().all()
    assert builder.raw_temperature.grad[1].abs() > 0
    assert builder.raw_temperature.grad[0] == 0


def test_hgnn_normalization_with_unequal_edge_degrees():
    incidence = torch.tensor([[[1.0, 0.3, 0.0], [0.2, 1.0, 0.4], [0.0, 0.1, 1.0]]])
    nodes = torch.tensor([[[1.0, -2.0], [-1.0, 0.5], [2.0, 1.0]]])
    h = incidence[0]
    dv = torch.diag(h.sum(dim=1).rsqrt())
    de = torch.diag(h.sum(dim=0).reciprocal())
    expected = dv @ h @ de @ h.T @ dv @ nodes[0]
    torch.testing.assert_close(HypergraphConvolution.propagate(incidence, nodes)[0], expected)
    layer = HypergraphConvolution(2, dropout=0)
    with torch.no_grad():
        layer.linear.weight.copy_(torch.eye(2))
    torch.testing.assert_close(layer(incidence, nodes)[0], expected.relu())


@pytest.mark.parametrize("classes", [None, 3])
def test_shared_model_forward_and_backward(classes):
    torch.manual_seed(1)
    model = CoHyFuse(num_rois=8, k_per_condition=[2, 4], hidden_dim=8,
                     fc_hidden_dim=16, heads=2, dropout=0, num_classes=classes)
    conditions = torch.randn(3, 2, 8, 8)
    whole = torch.randn(3, 28)
    prediction = model(conditions, whole)
    assert prediction.shape == ((3,) if classes is None else (3, classes))
    target = torch.randn(3) if classes is None else torch.tensor([0, 1, 2])
    criterion = torch.nn.MSELoss() if classes is None else torch.nn.CrossEntropyLoss()
    criterion(prediction, target).backward()
    for module in (model.fc_encoder, model.hgnn, model.pool_score, model.fusion,
                   model.whole_fc, model.head, model.hypergraph):
        assert any(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0
                   for p in module.parameters())


def test_condition_order_is_paired_with_k_and_temperature():
    torch.manual_seed(2)
    model = CoHyFuse(num_rois=8, k_per_condition=[2, 4], hidden_dim=8,
                     fc_hidden_dim=16, heads=2, dropout=0).eval()
    with torch.no_grad():
        model.hypergraph.raw_temperature.copy_(torch.tensor([-0.5, 1.0]))
    reordered = copy.deepcopy(model)
    reordered.hypergraph.k_per_condition = (4, 2)
    with torch.no_grad():
        reordered.hypergraph.raw_temperature.copy_(model.hypergraph.raw_temperature.flip(0))
        conditions, whole = torch.randn(2, 2, 8, 8), torch.randn(2, 28)
        torch.testing.assert_close(model(conditions, whole), reordered(conditions.flip(1), whole))


def test_classification_changes_only_the_head():
    regression = CoHyFuse()
    classification = CoHyFuse(num_classes=3)
    reg_shapes = {
        k: v.shape for k, v in regression.state_dict().items() if not k.startswith("head.4.")
    }
    cls_shapes = {
        k: v.shape for k, v in classification.state_dict().items() if not k.startswith("head.4.")
    }
    assert reg_shapes == cls_shapes


def test_invalid_k_is_rejected():
    with pytest.raises(ValueError, match="Each K"):
        CoHyFuse(num_rois=8, k_per_condition=[8])
