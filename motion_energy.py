from __future__ import division
from __future__ import print_function

import multiprocessing
import numpy as np

from scipy.misc import factorial
from scipy.signal import fftconvolve

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import brewer2mpl as b2mpl
import seaborn as sns

import generate_dots as gd

sns.set_style('ticks')
sns.set_context('talk')


def temporalImpResponse(n, k, t):
    time_response = (k * t) ** n * np.exp(-k * t) * \
        (1 / factorial(n) - (k * t) ** 2 / factorial(n + 2))

    return time_response


def cauchySpatialFilters(X, Y, sigma_c, sigma_g, order=4):
    even_filter = np.cos(np.arctan(X / sigma_c)) ** order * np.cos(
        order * np.arctan(X / sigma_c)) * np.exp(-Y ** 2 / (2 * sigma_g ** 2))
    odd_filter = np.cos(np.arctan(X / sigma_c)) ** order * np.sin(
        order * np.arctan(X / sigma_c)) * np.exp(-Y ** 2 / (2 * sigma_g ** 2))

    return even_filter, odd_filter


def directionSelectiveFilters(
        even_spatial, odd_spatial,
        temporal_impulse_fast, temporal_impulse_slow):
    spatiotemporal_even_fast = even_spatial * temporal_impulse_fast
    spatiotemporal_even_slow = even_spatial * temporal_impulse_slow

    spatiotemporal_odd_fast = odd_spatial * temporal_impulse_fast
    spatiotemporal_odd_slow = odd_spatial * temporal_impulse_slow

    left1 = spatiotemporal_even_slow + spatiotemporal_odd_fast
    left2 = spatiotemporal_even_fast - spatiotemporal_odd_slow
    right1 = spatiotemporal_even_slow - spatiotemporal_odd_fast
    right2 = spatiotemporal_even_fast + spatiotemporal_odd_slow

    return right1, right2, left1, left2


def createFilters(
        X, Y, T, cauchy_sigma_c, cauchy_sigma_g,
        cauchy_order, temporal_k):
    temporal_impulse_fast = temporalImpResponse(3, temporal_k, T)
    temporal_impulse_slow = temporalImpResponse(5, temporal_k, T)
    even_spatial, odd_spatial = cauchySpatialFilters(
        X, Y, cauchy_sigma_c, cauchy_sigma_g, cauchy_order)
    right1, right2, left1, left2 = directionSelectiveFilters(
        even_spatial, odd_spatial,
        temporal_impulse_fast, temporal_impulse_slow)

    return right1, right2, left1, left2


def getFilters(ppd=15.08, mon_refresh=60,
               theta=0, k=60, order=4, sigma_c=0.35, sigma_g=.05):
    # set the filter settings
    xlen = np.ceil(2 * ppd)
    tlen = np.ceil(.333 * mon_refresh)
    x = np.linspace(-1, 1, xlen)
    y = np.linspace(1, -1, xlen)
    t = np.linspace(0, tlen - 1, tlen) / mon_refresh
    X, Y, T = np.meshgrid(x, y, t)

    # the following changes the preferred direction of the filter;
    # 45 deg is up and to the right
    # (we're rotating the meshgrid "matrix")
    Xprime = np.cos(np.deg2rad(theta)) * X + np.sin(np.deg2rad(theta)) * Y
    Yprime = np.sin(np.deg2rad(theta)) * X - np.cos(np.deg2rad(theta)) * Y
    # generate the filter
    right1, right2, left1, left2 = createFilters(
        Xprime, Yprime, t, sigma_c, sigma_g, order, k)

    return right1, right2, left1, left2


def filterDots(dots_info, theta=0, k=60, order=4, sigma_c=0.35, sigma_g=.05):
    if len(dots_info) == 0:
        return []

    mon_refresh = 60
    ppd = dots_info['pix_per_deg']

    # set the filter settings
    xlen = np.ceil(2 * ppd)
    tlen = np.ceil(.333 * mon_refresh)
    x = np.linspace(-1, 1, xlen)
    y = np.linspace(1, -1, xlen)
    t = np.linspace(0, tlen - 1, tlen) / mon_refresh
    X, Y, T = np.meshgrid(x, y, t)

    # the following changes the preferred direction of the filter;
    # 45 deg is up and to the right
    # (we're rotating the meshgrid "matrix")
    Xprime = np.cos(np.deg2rad(theta)) * X + np.sin(np.deg2rad(theta)) * Y
    Yprime = np.sin(np.deg2rad(theta)) * X - np.cos(np.deg2rad(theta)) * Y
    # generate the filter
    right1, right2, left1, left2 = createFilters(
        Xprime, Yprime, t, sigma_c, sigma_g, order, k)

    # dots stuff
    xpixels = np.ceil(ppd * dots_info['aperture'][2])
    ypixels = np.ceil(ppd * dots_info['aperture'][3])
    tpixels = dots_info['shown_frames']

    xyoffset = np.round(xpixels / 2)
    dots_pixelpos = dots_info['dots_pixelpos'] + xyoffset - 1

    stimulus = np.zeros([xpixels, ypixels, tpixels])
    for i in range(dots_pixelpos.shape[2]):  # for each frame
        ix, iy = np.where(~np.isnan(dots_pixelpos[:, :, i]))
        positions = np.int16(dots_pixelpos[ix, iy, i].reshape(-1, 2))
        stimulus[positions[:, 1], positions[:, 0],
                 np.repeat(i, positions.shape[0])] = 1

    # convolve dot stimulus with each of the 4 filters (using FFT)
    right1_response, right2_response, left1_response, left2_response = [
        fftconvolve(stimulus, i, mode='full')
        for i in [right1, right2, left1, left2]]

    right_response = right1_response ** 2 + right2_response ** 2
    left_response = left1_response ** 2 + left2_response ** 2
    right = right_response - left_response

    return np.squeeze(np.apply_over_axes(np.sum, right, [0, 1]))


def testFilterLinearCoh(n_iterations=5):
    cohs = np.array([-0.512, -0.256, -0.128, -0.064, -0.032,
                     0, 0.032, 0.064, 0.128, 0.256, 0.512])

    inputs = []
    for coh in cohs:
        for j in xrange(n_iterations):
            seed = [np.random.randint(999), np.random.randint(999)]
            if np.sign(coh) == 1:
                direction = 0
            else:
                direction = 180
            inputs.append(
                gd.generateDots(seed, 0, 0, 5, 60, direction, np.abs(coh), 5))
    pool = multiprocessing.Pool()
    out = pool.map(filterDots, inputs)
    pool.close()

    data = np.array(out)
    summed_me = np.sum(data, axis=1)
    mean_me = np.mean(summed_me.reshape(-1, n_iterations), axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = b2mpl.get_map('Set1', 'Qualitative', 3).mpl_colors
    ax.add_line(plt.Line2D(cohs, mean_me, color=colors[0]))
    ax.set_xlabel('Motion strength (proportion coh)')
    ax.set_ylabel('Total motion energy')
    ax.set_xlim(-.61, .61)
    ax.set_ylim(np.min(mean_me) * 1.2, np.max(mean_me) * 1.2)
    sns.despine(offset=5, trim=True)
    plt.tight_layout()

    return cohs, mean_me


def testFilterDirectionBandwidth(n_iterations=5):
    directions = np.arange(-180, 180, 15)
    inputs = []
    for direction in directions:
        for j in xrange(n_iterations):
            seed = [np.random.randint(999), np.random.randint(999)]
            # change direction of dots, not motion energy filter
            inputs.append(
                gd.generateDots(seed, 0, 0, 5, 36, direction, 1, 5))
    pool = multiprocessing.Pool()
    out = pool.map(filterDots, inputs)
    pool.close()

    data = np.array(out)
    summed_me = np.sum(data, axis=1)
    mean_me = np.mean(summed_me.reshape(-1, n_iterations), axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = b2mpl.get_map('Set1', 'Qualitative', 3).mpl_colors
    ax.add_line(plt.Line2D(directions, mean_me, color=colors[0]))
    ax.set_xlabel('Motion direction (degrees)')
    ax.set_ylabel('Total motion energy')

    majorLocator = MultipleLocator(30)
    ax.xaxis.set_major_locator(majorLocator)

    majorLocator = MultipleLocator(50)
    ax.yaxis.set_major_locator(majorLocator)

    ax.set_xlim(-180, 180)
    ax.set_ylim(np.min(mean_me) - 10, np.max(mean_me) + 10)
    sns.despine(offset=5, trim=True)
    plt.tight_layout()

    return directions, mean_me


def testFilterSpeedBandwidth(n_iterations=5):
    speeds = np.arange(1, 9.5, 0.5)
    inputs = []
    for speed in speeds:
        for j in xrange(n_iterations):
            seed = [np.random.randint(999), np.random.randint(999)]
            inputs.append(
                gd.generateDots(seed, 0, 0, 5, 60, 0, 1, speed))
    pool = multiprocessing.Pool()
    out = pool.map(filterDots, inputs)
    pool.close()

    data = np.array(out)
    summed_me = np.sum(data, axis=1)
    mean_me = np.mean(summed_me.reshape(-1, n_iterations), axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = b2mpl.get_map('Set1', 'Qualitative', 3).mpl_colors
    ax.add_line(plt.Line2D(speeds, mean_me, color=colors[0]))
    ax.set_xlabel('Motion speed (degrees per s))')
    ax.set_ylabel('Total motion energy')

    majorLocator = MultipleLocator(1)
    ax.xaxis.set_major_locator(majorLocator)

    majorLocator = MultipleLocator(50)
    ax.yaxis.set_major_locator(majorLocator)

    ax.set_xlim(1, 9)
    ax.set_ylim(np.min(mean_me) - 10, np.max(mean_me) + 10)
    sns.despine(offset=5, trim=True)
    plt.tight_layout()

    return speeds, mean_me


def plotFFTofME():
    ppd = 15.08
    mon_refresh = 60
    filt, _, _, _ = getFilters(ppd=ppd, mon_refresh=mon_refresh, k=60)
    filt = filt[15, :, :]

    nfft = 128
    filt_fft = np.fft.fftshift(np.abs(np.fft.fftn(filt, [nfft, nfft])))

    # want temporal frequency as y axis and lower temporal freqs down
    filt_fft = np.flipud(filt_fft.T)

    spatial_Fs = ppd
    spatial_df = spatial_Fs / nfft
    spatial_fs = np.arange(0, spatial_Fs, spatial_df) - spatial_Fs / 2

    temporal_Fs = mon_refresh
    temporal_df = temporal_Fs / nfft
    temporal_fs = np.arange(0, temporal_Fs, temporal_df) - temporal_Fs / 2

    # spatial frequency on the x axis
    tS, tF = np.meshgrid(spatial_fs, temporal_fs)

    fig = plt.figure()

    ncontours = 10
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.contourf(tS, tF, filt_fft, ncontours, cmap='bone')
    ax1.grid('on', color='white', alpha=.35)
    ax1.set_xlabel('Spatial frequency (cyc/deg)')
    ax1.set_ylabel('Temporal frequency (cyc/sec)')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.contourf(
        tS[nfft / 2 + 1:, nfft / 2 + 1:], tF[nfft / 2 + 1:, nfft / 2 + 1:],
        filt_fft[nfft / 2 + 1:, nfft / 2 + 1:], ncontours, cmap='bone')
    speeds = np.arange(1, 8)
    colors = b2mpl.get_map('YlOrRd', 'Sequential', len(speeds)).mpl_colors
    for i in xrange(len(speeds)):
        x = np.arange(0.25, 3.5, .1)
        y = x * speeds[i]
        ax2.add_line(plt.Line2D(x, y, color=colors[i]))
        ax2.text(
            x[-1], y[-1], ''.join([str(speeds[i]), ' deg/s']),
            color=colors[i])

    ax2.grid('on', color='white', alpha=.35)
    ax2.set_xlabel('Spatial frequency (cyc/deg)')
    ax2.set_ylabel('Temporal frequency (cyc/sec)')

    sns.despine()
    plt.tight_layout()

    plt.savefig('filt_frequency_domain.pdf')
