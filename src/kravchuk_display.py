# LOAD NECESSARY PYTHON LIBRARIES

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import math
import cmocean


# LOAD NECESSARY PYTHON FUNCTIONS

from matplotlib.pyplot import figure

sapin = (0.0353, 0.3216, 0.1569)
tendre = (0.106, 0.463, 0.827)
tclair = (0.776, 0.866, 0.957)


# PLANAR DISPLAY

def planar_display(
    Kz,
    zt=[],
    zp=[],
    thetas=np.linspace(1e-10, np.pi, 500),
    phis=np.linspace(0, 2 * np.pi, 500),
    new=True,
):
    
    """
    Display the Kravchuk spectrogram of a discrete signal on a planar surface.

    Args:
        - Kz (numpy.ndarray): Kravchuk transform (in general complex valued).
        - zt (list of float,optional): polar angles of the zeros of the Kravchuk transform.
        - zp (list of float, optional): azimuthal angles of the zeros of the Kravchuk transform.
        - thetas (numpy.ndarray, optional): azimuthal angles at which the transform is computed.
        - phis (numpy.ndarray, optional): azimuthal angles at which the transform is computed.
        - new (boolean,optional): if True adapt the ticks to the new transform.
    """

    if not np.shape(Kz) == (len(thetas),len(phis)):
        raise NameError("The lengths of thetas and phis do not match the size of the Kravchuk transform.") 
    
    figure(figsize=(6, 4))
    plt.pcolormesh(
        thetas, phis, np.log10(np.abs(Kz)).T, shading="gouraud", cmap=cmocean.cm.deep
    )
    plt.scatter(zt, zp, s=15, color="white")
    plt.ylabel(r"$\varphi$", fontsize=30)
    plt.xlabel(r"$\vartheta$", fontsize=30)
    if new:
        plt.yticks(np.pi * np.array([-1, 0, 1]), [r"$-\pi$", "$0$", r"$\pi$"])
        plt.xticks(np.pi / 2 * np.arange(3), [r"$0$", r"$\pi/2$", r"$\pi$"])
    else:
        plt.yticks(np.pi * np.arange(3), ["$0$", r"$\pi$", r"$2\pi$"])
        plt.xticks(np.pi / 2 * np.arange(3), [r"$0$", r"$\pi/2$", r"$\pi$"])
    plt.tight_layout()


# SPHERICAL DISPLAY

def spherical_display(
    Kz,
    zt=[],
    zp=[],
    thetas=np.linspace(1e-10, np.pi, 500),
    phis=np.linspace(0, 2 * np.pi, 500),
    size=150,
):

    """
    Display the Kravchuk spectrogram of a discrete signal on the sphere.

    Args:
        - Kz (numpy.ndarray): Kravchuk transform (in general complex valued).
        - zt (list of float,optional): polar angles of the zeros of the Kravchuk transform.
        - zp (numpy.ndarray, optional): azimuthal angles of the zeros of the Kravchuk transform.
        - thetas (numpy.ndarray, optional): azimuthal angles at which the transform is computed.
        - phis (numpy.ndarray, optional): azimuthal angles at which the transform is computed.
        - size (float,optional): size of the markers indicating zeros.
    """

    if not np.shape(Kz) == (len(thetas),len(phis)):
        raise NameError("The lengths of thetas and phis do not match the size of the Kravchuk transform.") 
        
    # maps of angles
    Thetas, Phis = np.meshgrid(thetas, phis, indexing="ij")

    # Cartesian coordinates of the unit sphere
    XX = np.sin(Thetas) * np.cos(Phis)
    YY = np.sin(Thetas) * np.sin(Phis)
    ZZ = np.cos(Thetas)

    # Kravchuk transform of white noise
    test = np.log10(np.abs(Kz))
    tmin, tmax = test.min(), test.max()
    test = (test - tmin) / (tmax - tmin)

    # Plot the surface
    fig = plt.figure(figsize=4 * plt.figaspect(1.0))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.plot_surface(
        XX,
        YY,
        ZZ,
        rstride=1,
        cstride=1,
        facecolors=cmocean.cm.deep(test),
        shade=False,
        antialiased=True,
    )
    ax.set_axis_off()  # Turn off the axis planes

    elev = -60
    telev = math.radians(elev)
    azim = -145
    tazim = math.radians(azim)
    ax.view_init(elev, azim, roll=0)

    # Add zeros
    xs = []
    ys = []
    zs = []
    eps = 0  # 1e-3 # used to shift zeros towards the viewer, and bypass a rendering bug on zorder

    for i in range(len(zt)):
        # spherical coordinates
        phi = zp[i]
        theta = zt[i]

        # Cartesian coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        # np.array([np.sin(telev-np.pi/2)*np.cos(np.pi-tazim),np.sin(telev-np.pi/2)*np.sin(np.pi-tazim),np.cos(telev-np.pi/2)])
        for j in range(2, 3):
            npr.seed(j)
            n = npr.randn(3)
            if np.dot(n, np.array([x, y, z])) >= 0:
                #            if x-y+z>=0: # that is, if the zero is on the hemisphere we can see
                t = np.array([x, y, z]) + eps * n
                xs.append(t[0])  # shift the zeros towards the viewer
                ys.append(t[1])
                zs.append(t[2])

    ax.scatter(xs, ys, zs, s=size, color="white", alpha=1)
