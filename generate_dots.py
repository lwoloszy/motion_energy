from __future__ import division
import numpy as np
MAX_DOTS_PER_FRAME = 200


def generateDots(
        rseed, ap_x, ap_y, ap_d, n_frames, direction, coherence, speed,
        mon_refresh=60.0, mon_width_cm=60.0, view_dist_cm=27.0):

    screen_info = {
        'mon_width_cm': mon_width_cm,
        'view_dist_cm': view_dist_cm,
        'mon_refresh': mon_refresh,
        'screen_rect': np.array([0, 0, 1920, 1080]),
    }

    screen_info['pix_per_deg'] = (
        screen_info['screen_rect'][2] / screen_info['mon_width_cm'] /
        (2 * np.arctan(1 / (2 * screen_info['view_dist_cm'])) * 180 / np.pi))

    dots_info = {
        'aperture': np.array([ap_x, ap_y, ap_d * 2, ap_d * 2]),
        'pix_per_deg': screen_info['pix_per_deg'],
        'direction': direction,
        'coherence': coherence,
        'speed': speed,
        'density': 16.7,
        'interval': 3,
        'dot_size': 1,
        'dots_pixelpos': np.array([], dtype='float64'),
        'dots_degreepos': np.array([], dtype='float64'),
        'shown_frames': 0,
    }

    np.random.seed(np.prod(rseed))

    ndots = np.min([MAX_DOTS_PER_FRAME, np.round(
        dots_info['density'] * dots_info['aperture'][2] *
        dots_info['aperture'][3] / screen_info['mon_refresh'])])

    # define the ellipse aperture in screen coordinates
    AP = {
        'type': 'ellipse',
        'spec_center': np.array([
            np.mean(screen_info['screen_rect'][[0, 2]] +
                    dots_info['aperture'][0] * screen_info['pix_per_deg']),
            np.mean(screen_info['screen_rect'][[1, 3]] -
                    dots_info['aperture'][1] * screen_info['pix_per_deg'])]),
        'spec_radius': dots_info['aperture'][[2, 3]] / 2 * screen_info['pix_per_deg'],
    }

    # make a copy of AP struct with aperture center set to zero (used
    # for defining the mask) and half the radius
    # (we generated dots for twice the aperture)
    AP_ = AP.copy()
    AP_['spec_center'] = np.array([0, 0])
    AP_['spec_radius'] = AP_['spec_radius'] / 2

    dot_angle = np.pi * dots_info['direction'] / 180

    # displacement in xy in pixels
    dxdy = np.tile(
        np.array([np.cos(dot_angle), -np.sin(dot_angle)]) *
        dots_info['speed'] * dots_info['interval'] / screen_info['mon_refresh'],
        (ndots, 1)) * screen_info['pix_per_deg']
    d_ppd = np.tile(AP['spec_radius'], (ndots, 1))

    # note the reshape to mimick Matlab's Fortran like scan order
    dot_pos = (np.random.rand(dots_info['interval'] * 2 * ndots) - 0.5) * 2
    dot_pos = dot_pos.reshape(ndots, 2, dots_info['interval'], order='F')
    for i in range(dots_info['interval']):
        dot_pos[:, :, i] = dot_pos[:, :, i] * d_ppd

    dots_info['dots_pixelpos'] = np.empty([ndots, 2, n_frames])
    dots_info['dots_pixelpos'][:] = np.nan
    dots_info['dots_degreepos'] = np.empty([ndots, 2, n_frames])
    dots_info['dots_degreepos'][:] = np.nan

    loopi = 0
    for f in range(int(n_frames)):
        # find the index of coherently moving dots in this frame
        L = np.random.rand(ndots) < dots_info['coherence']

        # move the coherent dots
        dot_pos[L, :, loopi] = dot_pos[L, :, loopi] + dxdy[L, :]

        # replace the other dots (note Fortran order)
        random_pos = (np.random.rand(np.sum(~L) * 2) - 0.5) * 2
        random_pos = random_pos.reshape(np.sum(~L), 2, order='F') * d_ppd[~L, :]
        dot_pos[~L, :, loopi] = random_pos

        # wrap around
        L = dot_pos[:, 0, loopi] > d_ppd[:, 0]
        dot_pos[L, 0, loopi] = dot_pos[L, 0, loopi] - 2 * d_ppd[L, 0]
        L = dot_pos[:, 0, loopi] < -d_ppd[:, 0]
        dot_pos[L, 0, loopi] = 2 * d_ppd[L, 0] - dot_pos[L, 0, loopi]

        L = dot_pos[:, 1, loopi] > d_ppd[:, 1]
        dot_pos[L, 1, loopi] = dot_pos[L, 1, loopi] - 2 * d_ppd[L, 1]
        L = dot_pos[:, 1, loopi] < -d_ppd[:, 1]
        dot_pos[L, 1, loopi] = 2 * d_ppd[L, 1] - dot_pos[L, 1, loopi]

        # find which dots are in elliptical region
        dist = (
            dot_pos[:, :, loopi] - np.tile(AP_['spec_center'], (ndots, 1))
        ) / np.tile(AP_['spec_radius'], (ndots, 1))

        L = np.sqrt(np.sum(dist ** 2, axis=1)) <= 1

        pixel_pos = np.round(dot_pos[L, :, loopi])
        dots_info['shown_frames'] += 1
        dots_info['dots_pixelpos'][L, :, dots_info['shown_frames'] - 1] = \
            pixel_pos
        dots_info['dots_degreepos'][L, :, dots_info['shown_frames'] - 1] = \
            pixel_pos / screen_info['pix_per_deg']

        loopi += 1
        if loopi == dots_info['interval']:
            loopi = 0

    dots_info['aperture'] = np.array([ap_x, ap_y, ap_d, ap_d])
    return dots_info
