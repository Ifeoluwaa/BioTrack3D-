import torch
import pytest
import torch.nn.functional as F
from tracking_cellmot.models.simple_node_transformer import SimpleNodeTransformer

@pytest.mark.parametrize("sibling_aware", [False, True])
def test_simple_node_transformer_shapes(sibling_aware):
    # Setup model
    model = SimpleNodeTransformer(
        feat_dim=16,
        hidden_dim=32,
        n_heads=2,
        n_blocks=2,
        pair_chunk_size=4,
        sibling_aware=sibling_aware
    )
    
    # Check that model parameters exist and output dimensionality of pair_mlp is correct
    # hidden_dim = 32 -> input dim of pair_mlp = 32 * 3 + 9 (or 14) = 105 (or 110)
    expected_dim = 32 * 3 + (14 if sibling_aware else 9)
    assert model.pair_mlp[0].in_features == expected_dim
    
    # Test batched inputs
    B, N_t, N_t1, D = 2, 8, 6, 16
    feat_t = torch.randn(B, N_t, D)
    feat_t1 = torch.randn(B, N_t1, D)
    coords_t = torch.randn(B, N_t, 3)
    coords_t1 = torch.randn(B, N_t1, 3)
    
    logits = model(feat_t, feat_t1, coords_t, coords_t1)
    assert logits.shape == (B, N_t, N_t1)
    
    # Test unbatched inputs
    feat_t_unb = torch.randn(N_t, D)
    feat_t1_unb = torch.randn(N_t1, D)
    coords_t_unb = torch.randn(N_t, 3)
    coords_t1_unb = torch.randn(N_t1, 3)
    
    logits_unb = model(feat_t_unb, feat_t1_unb, coords_t_unb, coords_t1_unb)
    assert logits_unb.shape == (N_t, N_t1)

def test_chunking_equivalence():
    model = SimpleNodeTransformer(
        feat_dim=16,
        hidden_dim=32,
        n_heads=2,
        n_blocks=2,
        sibling_aware=True
    )
    model.eval()
    
    B, N_t, N_t1, D = 2, 10, 8, 16
    feat_t = torch.randn(B, N_t, D)
    feat_t1 = torch.randn(B, N_t1, D)
    coords_t = torch.randn(B, N_t, 3)
    coords_t1 = torch.randn(B, N_t1, 3)
    
    # Run with different chunk sizes
    model.pair_chunk_size = None
    logits_no_chunk = model(feat_t, feat_t1, coords_t, coords_t1)
    
    model.pair_chunk_size = 3
    logits_chunked = model(feat_t, feat_t1, coords_t, coords_t1)
    
    # Verify outputs are identical
    torch.testing.assert_close(logits_no_chunk, logits_chunked, rtol=1e-5, atol=1e-5)

def test_feature_construction_correctness():
    model = SimpleNodeTransformer(
        feat_dim=8,
        hidden_dim=16,
        n_heads=2,
        n_blocks=1,
        sibling_aware=False
    )
    
    B, N_c, N_t1, H = 1, 3, 4, 16
    qc = torch.randn(B, N_c, H)
    kk = torch.randn(B, N_t1, H)
    cc = torch.randn(B, N_c, 3)
    cc1 = torch.randn(B, N_t1, 3)
    
    features = model._build_pair_features(qc, kk, cc, cc1)
    
    # Compute manual reference features
    qe = qc.unsqueeze(2).expand(B, N_c, N_t1, H)
    ke = kk.unsqueeze(1).expand(B, N_c, N_t1, H)
    abs_diff = torch.abs(qe - ke)
    raw_delta = cc.unsqueeze(2) - cc1.unsqueeze(1)
    distance_sq = torch.sum(raw_delta ** 2, dim=-1, keepdim=True)
    distance = torch.sqrt(distance_sq + 1e-12)
    direction = raw_delta / (distance + 1e-8)
    scaled_delta = raw_delta / 100.0
    
    # Cosine similarity manual calculation
    cos_sim = F.cosine_similarity(qe, ke, dim=-1).unsqueeze(-1)
    
    expected = torch.cat([
        qe, ke, abs_diff, scaled_delta, distance, distance_sq, direction, cos_sim
    ], dim=-1)
    
    torch.testing.assert_close(features, expected, rtol=1e-5, atol=1e-5)

def test_sibling_feature_construction_correctness():
    # Dedicated test verifying vectorized sibling feature values are correct.
    model = SimpleNodeTransformer(
        feat_dim=8,
        hidden_dim=16,
        n_heads=2,
        n_blocks=1,
        sibling_aware=True
    )
    
    B, N_c, N_t1, H = 1, 1, 3, 16
    qc = torch.randn(B, N_c, H)
    kk = torch.randn(B, N_t1, H)
    
    # Parent is at [0, 0, 0]
    cc = torch.zeros(B, N_c, 3)
    # Targets are at:
    # Target 0: [1, 0, 0] (distance 1.0)
    # Target 1: [-1, 0, 0] (distance 1.0)
    # Target 2: [0, 5, 0] (distance 5.0)
    cc1 = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 5.0, 0.0]]], dtype=torch.float32)
    
    features = model._build_pair_features(qc, kk, cc, cc1)
    
    # Sibling verification for target 0 (index 0):
    # k1 is Target 0 (dist 1.0), k2 is Target 1 (dist 1.0).
    # Since j=0 is k1, its sibling is k2 (index 1).
    # Sibling coordinate for j=0 is cc1[0, 1] = [-1, 0, 0].
    # Midpoint of j=0 and sibling is (cc1[0, 0] + cc1[0, 1])/2 = [0, 0, 0].
    # Midpoint error for j=0 is distance between parent [0,0,0] and midpoint [0,0,0] = 0.0.
    # Angle between j=0 ([1,0,0]) and sibling ([-1,0,0]) is 180 deg (cos_theta = -1.0).
    # Sibling distance to parent: 1.0.
    # Sibling-sibling distance: 2.0.
    # Symmetric distance difference: |1.0 - 1.0| = 0.0.
    
    # Features shape is (1, 1, 3, 3*16 + 14 = 62). Sibling features are the last 5 dimensions.
    # Sibling features for target 0 (index 0):
    sib_features_j0 = features[0, 0, 0, -5:]
    expected_j0 = torch.tensor([0.0, 2.0, -1.0, 0.0, 1.0], dtype=torch.float32)
    torch.testing.assert_close(sib_features_j0, expected_j0, rtol=1e-4, atol=1e-4)

    # Sibling verification for target 2 (index 2):
    # k1 is Target 0 (dist 1.0), k2 is Target 1 (dist 1.0).
    # Since j=2 is not k1, its sibling is k1 (index 0).
    # Sibling coordinate for j=2 is cc1[0, 0] = [1, 0, 0].
    # Midpoint of j=2 and sibling is (cc1[0, 2] + cc1[0, 0])/2 = [0.5, 2.5, 0.0].
    # Midpoint error for j=2 is distance from parent [0,0,0] to [0.5, 2.5, 0] = sqrt(0.25 + 6.25) = sqrt(6.5) = 2.5495.
    # Sibling distance to parent: 1.0.
    # Sibling-sibling distance: distance from [0,5,0] to [1,0,0] = sqrt(1 + 25) = sqrt(26) = 5.0990.
    # Symmetric distance difference: |5.0 - 1.0| = 4.0.
    # Angle: cos_theta = sum([0, 5, 0] * [1, 0, 0]) / (5.0 * 1.0) = 0.0.
    sib_features_j2 = features[0, 0, 2, -5:]
    import math
    expected_j2 = torch.tensor([math.sqrt(6.5), math.sqrt(26.0), 0.0, 4.0, 1.0], dtype=torch.float32)
    torch.testing.assert_close(sib_features_j2, expected_j2, rtol=1e-4, atol=1e-4)

def test_gradient_flow():
    model = SimpleNodeTransformer(
        feat_dim=16,
        hidden_dim=32,
        n_heads=2,
        n_blocks=2,
        pair_chunk_size=2
    )
    
    B, N_t, N_t1, D = 1, 4, 3, 16
    feat_t = torch.randn(B, N_t, D, requires_grad=True)
    feat_t1 = torch.randn(B, N_t1, D, requires_grad=True)
    coords_t = torch.randn(B, N_t, 3, requires_grad=True)
    coords_t1 = torch.randn(B, N_t1, 3, requires_grad=True)
    
    logits = model(feat_t, feat_t1, coords_t, coords_t1)
    loss = logits.sum()
    loss.backward()
    
    # Check that gradients are computed for inputs
    assert feat_t.grad is not None
    assert feat_t1.grad is not None
    assert coords_t.grad is not None
    assert coords_t1.grad is not None
    
    # Also verify gradients of model parameters are non-zero
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.isnan(param.grad).sum() == 0, f"NaN gradient in {name}"
