import torch


def tensor_to_str(x: torch.Tensor):
    def get_non_finite_features(x: torch.Tensor):
        features = []
        n_nans = torch.isnan(x).sum().item()
        if n_nans > 0:
            features.append(f'n_nans={n_nans}')
        n_infs = torch.isinf(x).sum().item()
        if n_infs > 0:
            features.append(f'n_infs={n_infs}')
        return features

    def get_finite_features(x: torch.Tensor):
        features = []
        if not x.dtype == torch.bool:
            features.append(f'min={x.min().item()}')
            features.append(f'max={x.max().item()}')

        if not isinstance(x, torch.FloatTensor):
            features.append(f'sum={x.sum().item()}')
            return features

        features.append(f'mean={x.mean().item()}')
        if x.numel() > 1:
            features.append(f'std={x.std().item()}')

        n_zeroes = (x == 0.0).sum().item()
        if n_zeroes > 0:
            features.append(f'n_zeroes={n_zeroes}')

        return features

    def get_features(x: torch.Tensor):
        features = []
        features.append(f'shape={tuple(x.shape)}')

        if (~torch.isfinite(x)).sum().item() > 0:
            features.extend(get_non_finite_features(x))
        else:
            features.extend(get_finite_features(x))

        features.append(f'dtype={x.dtype}')
        features.append(f'device={x.device}')
        return features

    return 'Tensor(' + ', '.join(get_features(x)) + ')'


def print_tensor(x: torch.Tensor):
    print(tensor_to_str(x))
