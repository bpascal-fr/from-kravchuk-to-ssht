# LOAD NECESSARY PYTHON LIBRARIES

import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import matplotlib.pyplot as plt
import cmocean


# LOAD NECESSARY PYTHON FUNCTIONS

from matplotlib.pyplot import figure


def the_stft_transform(x, time_t=[]):
    """
    Compute the discretized standard Short-Time Fourier Transform with a circular Gaussian window.

    Args:
        - x (numpy.ndarray): discrete signal, noisy or not, possibly complex valued.
        - time_t (numpy.ndarray, optional): vector of time stamps at which the signal is sampled (default: [0,1, ...,N]).

    Returns:
        - Vx (numpy.ndarray): Short-Time Fourier Transform of the discrete signal, complex-valued.
        - fx (numpy.ndarray): frequencies at which the Short-Time Fourier transform is computed.
        - time_t (numpy.ndarray): time stamps at which the Short-Time Fourier transform is computed.
    """

    if len(time_t) == 0:
        time_t = np.arange(len(x)) - int(len(x) // 2)

    dt = time_t[1] - time_t[0]

    # Gaussian window of unit energy
    Ng = len(x)
    g = signal.windows.gaussian(Ng, np.sqrt((Ng) / 2 / np.pi))
    
    # Compute the Gaussian Short-Time Fourier Transform
    fx, _, Vx = signal.stft(
        x, fs=1 / dt, window=g, nperseg=Ng, noverlap=Ng - 1, return_onesided=False, scaling='psd'
    )

    # remap the frequencies in a symmetric fashion
    fx = fft.fftshift(fx)
    Vx = fft.fftshift(Vx, axes=(0,))

    return Vx, time_t, fx


def the_stft_zeros(Vx, time_t, fx, feedback=False):
    """
    Localize the zeros of the Short-Time Fourier Transform.

    Args:
        - Vx (numpy.ndarray): Short-Time Fourier Transform of the discrete signal, complex-valued.
        - time_t (numpy.ndarray): vector of time stamps at which the signal is sampled (default: [0,1, ...,N]).
        - fx (numpy.ndarray): frequencies at which the Short-Time Fourier transform is computed.
        - feedback (boolean, optional): if True print the number of detected zeros.

    Returns:
        - zt (list of float): time coordinates of the zeros of the Gaussian spectrogram.
        - zf (list of float): frequency coordinates of the zeros of the Gaussian spectrogram.
    """
    
    # compute the zeros
    zx, zy = extr2min(np.abs(Vx))

    # coordinates of the zeros on the sphre
    zt = time_t[zy]
    zf = fx[zx]

    # give some feedback
    if feedback:
        print("Local minima method has found " + str(zx.shape[0]) + " zeros.")

    return zt, zf


def stft_display(Vx, time_t, fx, zt=[], zf=[]):

    """
    Display the standard Gaussian spectrogram and its zeros.

    Args:
        - Vx (numpy.ndarray): Short-Time Fourier Transform of the discrete signal, complex-valued.
        - time_t (numpy.ndarray): vector of time stamps at which the signal is sampled (default: [0,1, ...,N]).
        - fx (numpy.ndarray): frequencies at which the Short-Time Fourier transform is computed.
        - zt (list of float, optional): time coordinates of the zeros of the Gaussian spectrogram.
        - zf (list of float, optional): frequency coordinates of the zeros of the Gaussian spectrogram.
    """
    
    # turn to pulsation for display
    zw = 2 * np.pi * np.array(zf)

    figure(figsize=(7.5, 4))
    plt.pcolormesh(
        time_t,
        2 * np.pi * fx,
        np.log10(np.abs(Vx)),
        shading="gouraud",
        cmap=cmocean.cm.deep,
    )
    plt.scatter(zt, zw, s=15, color="white")
    plt.xlabel(r"$t$ (s)", fontsize=30)
    plt.ylabel(r"$\omega$ (rad.s$^{-1}$)", fontsize=30)
    plt.tight_layout()


def extr2min(M):

    """
    Find zeros of a nonnegative function by the Minimal Grid Neighbors method.

    Args:
        - M (numpy.ndarray): two-dimensional array of nonnegative real numbers.

    Returns:
        - x (list of integers): x-indices of the zeros of the input matrix.
        - y (list of intergers): y-indices of the zeros of the input matrix.
    """
    
    central = M[1:-1, 1:-1]
    mask = np.full(central.shape, True, dtype=bool)
    sub_indices = (
        (np.s_[2:], np.s_[1:-1]),
        (np.s_[:-2], np.s_[1:-1]),
        (np.s_[1:-1], np.s_[2:]),
        (np.s_[1:-1], np.s_[:-2]),
        (np.s_[:-2], np.s_[:-2]),
        (np.s_[:-2], np.s_[2:]),
        (np.s_[2:], np.s_[2:]),
        (np.s_[2:], np.s_[:-2]),
    )
    for I, J in sub_indices:
        np.logical_and(mask, central <= M[I, J], out=mask, where=mask)

    x, y = np.where(mask)
    return x + 1, y + 1
