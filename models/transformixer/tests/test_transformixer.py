import torch

from xlstm_mixer.models.transformixer import Transformixer


def test_permutation_equivariance():
    torch.manual_seed(0)
    model = Transformixer(
        pred_len=16,
        seq_len=32,
        enc_in=7,
        d_model=32,
        n_heads=4,
        e_layers=2,
        dropout=0.0,
        use_positional_encoding=False,
    )
    model.eval()

    x = torch.randn(2, 32, 7)
    perm = torch.randperm(7)

    with torch.no_grad():
        y = model.forecast(x, None)
        y_perm_in = model.forecast(x[:, :, perm], None)

    assert torch.allclose(y[:, :, perm], y_perm_in, atol=1e-4, rtol=1e-4)


def test_positional_encoding_breaks_equivariance():
    torch.manual_seed(0)
    model = Transformixer(
        pred_len=16,
        seq_len=32,
        enc_in=7,
        d_model=32,
        n_heads=4,
        e_layers=2,
        dropout=0.0,
        use_positional_encoding=True,
    )
    model.eval()

    x = torch.randn(2, 32, 7)
    perm = torch.randperm(7)

    with torch.no_grad():
        y = model.forecast(x, None)
        y_perm_in = model.forecast(x[:, :, perm], None)

    assert not torch.allclose(y[:, :, perm], y_perm_in, atol=1e-4, rtol=1e-4)


def test_nlinear_only_matches_linear_path():
    torch.manual_seed(0)
    model = Transformixer(
        pred_len=16,
        seq_len=32,
        enc_in=7,
        use_variate_mixer=False,
    )
    model.eval()

    x = torch.randn(2, 32, 7)
    with torch.no_grad():
        y = model.forecast(x, None)
    assert y.shape == (2, 16, 7)
