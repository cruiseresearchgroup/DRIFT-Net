import numpy as np


def lp_error(preds: np.ndarray, targets: np.ndarray, p=1):
    num_samples, num_channels, _, _ = preds.shape
    preds = preds.reshape(num_samples, num_channels, -1)
    targets = targets.reshape(num_samples, num_channels, -1)
    errors = np.sum(np.abs(preds - targets) ** p, axis=-1)
    return np.sum(errors, axis=-1) ** (1 / p)


def relative_lp_error(
    preds: np.ndarray,
    targets: np.ndarray,
    p=1,
    return_percent=True,
):
    num_samples, num_channels, _, _ = preds.shape
    preds = preds.reshape(num_samples, num_channels, -1)
    targets = targets.reshape(num_samples, num_channels, -1)
    errors = np.sum(np.abs(preds - targets) ** p, axis=-1)
    normalization_factor = np.sum(np.abs(targets) ** p, axis=-1)

    # catch 0 division
    normalization_factor = np.sum(normalization_factor, axis=-1)
    normalization_factor = np.where(
        normalization_factor == 0, 1e-10, normalization_factor
    )

    errors = (np.sum(errors, axis=-1) / normalization_factor) ** (1 / p)

    if return_percent:
        errors *= 100

    return errors


def mean_relative_lp_error(
    preds: np.ndarray,
    targets: np.ndarray,
    p=1,
    return_percent=True,
):
    errors = relative_lp_error(preds, targets, p, return_percent)
    return np.mean(errors, axis=0)


def median_relative_lp_error(
    preds: np.ndarray,
    targets: np.ndarray,
    p=1,
    return_percent=True,
):
    errors = relative_lp_error(preds, targets, p, return_percent)
    return np.median(errors, axis=0)

def _to_nchw(x):
    # Allow (N,C,H,W) or (N,H,W)
    if x.ndim == 3:
        x = x[:, None, ...]
    return x

def relative_lp_error(preds, targets, p=1, return_percent=True):
    preds = _to_nchw(preds); targets = _to_nchw(targets)
    N, C, H, W = preds.shape
    a = preds.reshape(N, C, -1)
    b = targets.reshape(N, C, -1)
    num = np.sum(np.abs(a - b) ** p, axis=-1)
    den = np.sum(np.abs(b) ** p, axis=-1)
    den = np.where(den == 0, 1e-10, den)
    err = (np.sum(num, axis=-1) / np.sum(den, axis=-1)) ** (1.0 / p)
    if return_percent: err *= 100
    return err

# ---- 2D: Isotropic energy spectrum E(k) (scalar field; for KS-2D/NS-2D) ----
def radial_energy_spectrum(field):
    # field: (N,H,W) or (N,1,H,W)
    X = _to_nchw(field)[:,0]  # Take single channel
    N, H, W = X.shape
    fx = np.fft.fftfreq(W)*W
    fy = np.fft.fftfreq(H)*H
    kx, ky = np.meshgrid(fx, fy)
    k = np.sqrt(kx**2 + ky**2)
    kmax = int(np.max(k))
    bins = np.arange(0, kmax + 1, 1)
    E = np.zeros((N, len(bins)-1))
    for i in range(N):
        F = np.fft.fft2(X[i])
        P = np.abs(F)**2
        # Radial average
        for bi in range(len(bins)-1):
            mask = (k >= bins[bi]) & (k < bins[bi+1])
            if np.any(mask):
                E[i, bi] = P[mask].mean()
    return E  # (N, n_bins)

def spectrum_rel_l1_error(preds, targets):
    Ep = radial_energy_spectrum(preds)
    Et = radial_energy_spectrum(targets)
    den = np.maximum(Et, 1e-10)
    return np.mean(np.abs(Ep - Et) / den, axis=0) * 100.0  # Relative L1(%) for each shell

# ---- 2D: Second-order structure function S2(r) ----
def structure_function_S2(field, radii):
    X = _to_nchw(field)[:,0]
    N, H, W = X.shape
    S2 = np.zeros((N, len(radii)))
    for ri, r in enumerate(radii):
        # Take x/y direction differences to approximate isotropy
        dx = np.roll(X, -r, axis=2) - X
        dy = np.roll(X, -r, axis=1) - X
        S2[:, ri] = 0.5*( (dx**2).mean(axis=(1,2)) + (dy**2).mean(axis=(1,2)) )
    return S2  # (N, len(radii))

def s2_rel_l1_error(preds, targets, radii=(1,2,4,8,16,32)):
    Sp = structure_function_S2(preds, radii)
    St = structure_function_S2(targets, radii)
    den = np.maximum(St, 1e-10)
    return np.mean(np.abs(Sp - St) / den, axis=0) * 100.0

# ---- 2D: Total kinetic energy/vorticity (simplified "energy budget difference") ----
def kinetic_energy(field):
    # Use spectral method to approximately reconstruct u,v from vorticity w: -Δψ=w, u=∂y ψ, v=-∂x ψ; if input is not vorticity, treat as scalar energy approximation
    X = _to_nchw(field)[:,0]
    N, H, W = X.shape
    fx = np.fft.fftfreq(W)*W
    fy = np.fft.fftfreq(H)*H
    kx, ky = np.meshgrid(fx, fy)
    k2 = kx**2 + ky**2
    E = np.zeros(N)
    for i in range(N):
        Wh = np.fft.fft2(X[i])
        psi_hat = np.zeros_like(Wh, dtype=complex)
        mask = k2 != 0
        psi_hat[mask] = -Wh[mask] / k2[mask]
        uh = 1j*ky*psi_hat
        vh = -1j*kx*psi_hat
        u = np.fft.ifft2(uh).real
        v = np.fft.ifft2(vh).real
        E[i] = 0.5*np.mean(u*u + v*v)
    return E  # (N,)

def energy_gap_percent(preds, targets):
    Ep = kinetic_energy(preds)
    Et = kinetic_energy(targets)
    den = np.maximum(Et, 1e-10)
    return np.mean(np.abs(Ep - Et)/den)*100.0
