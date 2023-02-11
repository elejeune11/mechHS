import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# function to produce simply supported beams

lw = 2

# plot triangle specified by top point.
def plot_triangle(ax, x0, y0, s):
    ang = 30.0 / 360.0 * 2.0 * np.pi
    x_vals = [x0, x0 + s * np.sin(ang), x0 - s * np.sin(ang), x0]
    y_vals = [y0, y0 - s * np.cos(ang), y0 - s * np.cos(ang), y0]
    ax.plot(x_vals, y_vals, "k", linewidth=lw)
    return x0 - s * np.sin(ang), y0 - s * np.cos(ang), x0 + s * np.sin(ang), y0 - s * np.cos(ang)


def plot_circle(ax, x0, y0, r):
    cx = x0
    cy = y0 - r
    num_pts = 100
    x_vals = []
    y_vals = []
    for kk in range(0, num_pts):
        ang = kk * np.pi * 2.0 / (num_pts - 1)
        x = cx + np.cos(ang) * r
        y = cy + np.sin(ang) * r
        x_vals.append(x)
        y_vals.append(y)
    ax.plot(x_vals, y_vals, "k", linewidth=lw)
    return


def plot_line(ax, x0, y0, x1, y1, lw=lw):
    x_vals = [x0, x1]
    y_vals = [y0, y1]
    ax.plot(x_vals, y_vals, "k", linewidth=lw)
    return


def plot_pin_fixed(ax, x0, y0, s):
    xe0, ye0, xe1, ye1 = plot_triangle(ax, x0, y0, s)
    num_lines = 7
    ll = s / 5.0
    ang = np.pi / 3.0
    for kk in range(0, num_lines + 1):
        xl0 = (xe1 - xe0) / num_lines * kk + xe0
        yl0 = (ye1 - ye0) / num_lines * kk + ye0
        xl1 = xl0 - ll * np.cos(ang)
        yl1 = yl0 - ll * np.sin(ang)
        plot_line(ax, xl0, yl0, xl1, yl1)
    return


def plot_roller(ax, x0, y0, r):
    plot_circle(ax, x0, y0, r)
    return y0 - 2.0 * r


def plot_pin_roller(ax, x0, y0, s):
    xe0, ye0, _, _ = plot_triangle(ax, x0, y0, s)
    num_rollers = 3
    r = s / (num_rollers * 2 - 1) / 2.0
    for kk in range(0, num_rollers):
        xr = (1 + 4.0 * kk) * r + xe0
        yr = ye0
        plot_roller(ax, xr, yr, r)
    return


def plot_load(ax, x0, y0, x1, y1, h_mean, h_range, num_pts, seed):
    np.random.seed(seed)
    x_pts = []
    y_pts = []
    for kk in range(0, num_pts):
        x = (x1 - x0) / (num_pts - 1) * kk + x0
        y = (y1 - y0) / (num_pts - 1) * kk + y0 + h_mean + 2.0 * (np.random.random() - 1.0) * h_range
        x_pts.append(x)
        y_pts.append(y)
    x = np.asarray(x_pts)
    y = np.asarray(y_pts)
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    num_pts_line = 100
    x_line = []
    y_line = []
    for kk in range(0, num_pts_line):
        x = (x1 - x0) / (num_pts_line - 1) * kk + x0
        y = p(x)
        x_line.append(x)
        y_line.append(y)
    ax.plot(x_line, y_line, "k", linewidth=lw)
    num_arrows_line = 11
    for kk in range(0, num_arrows_line):
        xa0 = (x1 - x0) / (num_arrows_line - 1) * kk + x0
        ya0 = p(xa0)
        xa1 = xa0
        ya1 = y0 - 0.0001
        off = 0.5
        ax.plot([xa0, xa1], [ya0, ya1 + off], "k", linewidth=lw)
        dx = 0
        dy = -0.0001
        ax.arrow(xa1, ya1 + off, dx, dy, fc="k", ec="k", linewidth=lw, width=.05, head_length=0.05)
    return


def composite_beam_seg(ax, x0, y0, x1, y1, s):
    # beam
    plot_line(ax, x0, y0, x1, y1)
    # fixed pin
    plot_pin_fixed(ax, x0, y0, s)
    # roller pin
    r = s * 0.05
    y0_new = plot_roller(ax, x1, y1, r)
    return y0_new


def ss_beam(ax, x0, y0, x1, y1, s):
    # beam
    plot_line(ax, x0, y0, x1, y1)
    # fixed pin
    plot_pin_fixed(ax, x0, y0, s)
    # roller pin
    plot_pin_roller(ax, x1, y1, s)
    return


def composite_beam(ax, x0, y0, x1, s, num_segs):
    y_val = y0
    for kk in range(0, num_segs):
        if kk == num_segs - 1:
            xs0 = (x1 - x0) / num_segs * kk + x0
            ys0 = y_val
            xs1 = (x1 - x0) / num_segs * (kk + 1) + x0
            ys1 = y_val
            ss_beam(ax, xs0, ys0, xs1, ys1, s)
        else:
            xs0 = (x1 - x0) / num_segs * kk + x0
            ys0 = y_val
            xs1 = (x1 - x0) / num_segs * (kk + 1) + x0
            ys1 = y_val
            y_val = composite_beam_seg(ax, xs0, ys0, xs1, ys1, s)
    return


def rectangular_device(ax, x_ll, y_ll, wid, dep, num_sensors, lw_rect, lc, fc):
    # plot a rectangle, fill it with gray
    ax.add_patch(Rectangle((x_ll, y_ll), wid, dep, fc=fc, ec=lc, linewidth=lw_rect))
    sw = 0.5
    gap = (wid - sw * num_sensors) / (num_sensors - 1)
    for kk in range(0, num_sensors):
        xc = sw * kk + gap * kk + x_ll
        x_min = xc
        x_max = xc + sw
        ax.plot([x_min, x_max], [y_ll, y_ll], "r", linewidth=lw*3)
    return x_ll + wid, y_ll + dep
