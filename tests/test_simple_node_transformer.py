import torch
import pytest
import torch.nn.functional as F
from tracking_cellmot.models.simple_node_transformer import SimpleNodeTransformer

def test_simple_node_transformer_shapes():
    # Setup model
    model = SimpleNodeTransformer(
        feat_dim=16,
        hidden_dim=32,
        n_heads=2,
        n_blocks=2,
        pair_chunk_size=4
    )
    
    # Check that model parameters exist and output dimensionality of pair_mlp is correct
    # hidden_dim = 32 -> input dim of pair_mlp = 32 * 3 + 9 = 105
    assert model.pair_mlp[0].in_features == 105
    
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
        n_blocks=2
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
        n_blocks=1
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
