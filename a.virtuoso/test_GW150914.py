from .qptransform import *
import matplotlib.pyplot as plt
from gwosc.datasets import event_gps
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy
import gwpy.TimeSeries as TimeSeries
import scipy.signal


def percentage(new, old):
    return ((new - old) / old) * 100


event = "GW150914-v3"
event_label = "GW150914"
gps = event_gps(event)
dets = ["L1", "H1"]
dets_label = ["LIGO Livingston (L1)", "LIGO Hanford (H1)"]
ncols = len(dets)
nrows = 3

denoising_threshold = 7.0
energy_density_threshold = 7.0

f_s = 2048.0
trange = [gps - 0.15, gps + 0.05]
frange = [20.0, 500.0]
duration = trange[-1] - trange[0]
alpha = 0.05
alpha_find_Qp = 0.2
qrange = [2.0 * numpy.pi, 6.0 * numpy.pi]
filmethod = "highpass_threshold_filtering"


fbig, axbig = plt.subplots(figsize=(10.1 * 1.4, 7.6 * 1.4), ncols=ncols, nrows=nrows)


for col, detector in enumerate(dets):
    print("Detector", detector)

    segment = (gps - 10, gps + 10)
    data_init = TimeSeries.fetch_open_data(detector, *segment, verbose=True)
    downsampling = int(
        numpy.ceil(
            numpy.ceil(1.0 / (data_init.times.value[1] - data_init.times.value[0]))
            / f_s
        )
    )
    data_downsampled = scipy.signal.decimate(data_init, downsampling)

    times = numpy.arange(segment[0], segment[-1], 1.0 / f_s)
    data = TimeSeries(data_downsampled, x0=times[0], dx=times[1] - times[0], copy=False)
    data = data.whiten()
    data = data.crop(trange[0], trange[-1])
    times = data.times.value

    # Qp-transform
    qpinstance = QpTransform(
        data=data,
        times=times,
        whiten=False,
        frange=frange,
        alpha=alpha,
        alpha_find_Qp=alpha_find_Qp,
        energy_density_threshold=energy_density_threshold,
        qrange=qrange,
        prange=None,
        denoising_threshold=denoising_threshold,
        filmethod=filmethod,
    )
    print(
        event
        + ", detector "
        + detector
        + "\n"
        + f"Q={round(qpinstance.Q, 8)},  p={round(qpinstance.p,8)}, peak={qpinstance.peak}, energy density={round(qpinstance.energy_density,2)}, area={round(qpinstance.TF_area,2)}"
    )
    filseries_Qp = qpinstance.filseries

    # Q-transform
    qinstance = QpTransform(
        data=data,
        times=times,
        whiten=False,
        frange=frange,
        alpha=alpha,
        alpha_find_Qp=alpha_find_Qp,
        energy_density_threshold=energy_density_threshold,
        qrange=qrange,
        prange=[0.0, 0.0],
        denoising_threshold=denoising_threshold,
        filmethod=filmethod,
    )
    print(
        event
        + ", detector "
        + detector
        + "\n"
        + f"Q={round(qinstance.Q, 8)},  p={0}, peak={qinstance.peak}, energy density={round(qinstance.energy_density,2)}, area={round(qinstance.TF_area,2)}"
    )
    filseries_Q = qinstance.filseries

    # Qp-transform scan
    row = 0
    axbig[row][col].set_yscale("log")
    axbig[row][col].minorticks_off()
    vmax = qpinstance.qpspecgram.max()
    pcol = axbig[row][col].pcolormesh(
        qpinstance.series.times.value - qpinstance.series.times.value[0],
        qpinstance.qpspecgram.frequencies.value,
        numpy.transpose(numpy.asarray(qpinstance.qpspecgram.value)),
        vmin=0,
        vmax=vmax,
        cmap="viridis",
        shading="gouraud",
        rasterized=True,
    )

    axbig[row][col].set_ylim(frange[0], frange[-1])
    qp_suptitle = (
        rf"$Q={round(qpinstance.Q, 2)}$, $p={round(qpinstance.p, 3)}$, "
        + rf"$energy \, peak={round(qpinstance.peak['energy'],2)}$,"
        + "\n"
        + rf"$energy \, density={round(qpinstance.energy_density,2)}$, $TF \, area={round(qpinstance.TF_area,2)}$"
    )
    if row == 0:
        axbig[row][col].set_title(dets_label[col] + "\n" + qp_suptitle)
    else:
        axbig[row][col].set_title(qp_suptitle)
    divider = make_axes_locatable(axbig[row][col])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if col == 0:
        cbar_ticks = [0, 10, 20, 30, 40, 50, 60]
    if col == 1:
        cbar_ticks = [0, 15, 30, 45, 60, 75, 90, 105]
    fbig.colorbar(pcol, cax=cax, orientation="vertical", ticks=cbar_ticks)
    color_ticks = 10 * numpy.arange(0, numpy.ceil(vmax / 10))
    cax.tick_params(axis="both", direction="out", size=0.7, pad=0.15)

    axbig[row][col].set_yticks([32, 64, 128, 256])
    if col == 0:
        axbig[row][col].set_ylabel(r"$\textrm{Frequency~[Hz]}$")
        axbig[row][col].set_yticklabels([r"$32$", r"$64$", r"$128$", r"$256$"])
    if col == 1:
        axbig[row][col].set_yticklabels([])
        axbig[row][col].minorticks_off()

    axbig[row][col].grid(alpha=0.3)

    axbig[row][col].tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labeltop=False,
        labelbottom=False,
        direction="out",
    )

    # Q-transform scan
    row = 1
    axbig[row][col].set_yscale("log")
    axbig[row][col].minorticks_off()
    vmax = qpinstance.qpspecgram.max()
    pcol = axbig[row][col].pcolormesh(
        qinstance.series.times.value - qinstance.series.times.value[0],
        qinstance.qpspecgram.frequencies.value,
        numpy.transpose(numpy.asarray(qinstance.qpspecgram.value)),
        vmin=0,
        vmax=vmax,
        cmap="viridis",
        shading="gouraud",
        rasterized=True,
    )

    axbig[row][col].set_ylim(frange[0], frange[-1])
    q_suptitle = (
        rf"$Q={round(qinstance.Q, 2)}$, $p={0}$, "
        + rf"$energy \, peak={round(qinstance.peak['energy'],2)}$,"
        + "\n"
        + rf"$energy \, density={round(qinstance.energy_density,2)}$, $TF \, area={round(qinstance.TF_area,2)}$"
    )
    axbig[row][col].set_title(q_suptitle)
    divider = make_axes_locatable(axbig[row][col])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if col == 0:
        cbar_ticks = [0, 10, 20, 30, 40, 50, 60]
    if col == 1:
        cbar_ticks = [0, 15, 30, 45, 60, 75, 90, 105]
    fbig.colorbar(pcol, cax=cax, orientation="vertical", pad=0.2, ticks=cbar_ticks)
    color_ticks = 10 * numpy.arange(0, numpy.ceil(vmax / 10))
    cax.tick_params(axis="both", direction="out", size=0.7, pad=0.15)

    axbig[row][col].set_yticks([32, 64, 128, 256])
    if col == 0:
        axbig[row][col].set_ylabel(r"$\textrm{Frequency~[Hz]}$")
        axbig[row][col].set_yticklabels([r"$32$", r"$64$", r"$128$", r"$256$"])

    axbig[row][col].grid(alpha=0.3)
    if col == 1:
        axbig[row][col].set_yticklabels([])
        axbig[row][col].minorticks_off()

    axbig[row][col].tick_params(
        axis="x",  # changes apply to the x-axis
        which="major",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labeltop=False,
        labelbottom=False,
        direction="out",
    )

    row = 2

    ylim_high = numpy.ceil(max(qpinstance.series.value))  # 6#
    ylim_low = numpy.ceil(max(-qpinstance.series.value))  # 5#
    axbig[row][col].set_ylim(-ylim_low - 1, +ylim_high + 1)
    axbig[row][col].set_yticks([-5, -2.5, 0, 2.5, 5])
    axbig[row][col].plot(
        qpinstance.series.times.value - qpinstance.series.times.value[0],
        qpinstance.series.value,
        color="green",
        label="Data",
        linewidth=0.7,
        alpha=0.5,
    )
    axbig[row][col].plot(
        qpinstance.filseries.times.value - qpinstance.filseries.times.value[0],
        qinstance.filseries.value,
        color="blue",
        label="Filtered with Q",
        linewidth=0.9,
    )
    axbig[row][col].plot(
        qpinstance.filseries.times.value - qpinstance.filseries.times.value[0],
        qpinstance.filseries.value,
        color="red",
        label="Filtered with Qp",
        linewidth=0.9,
    )
    axbig[row][col].set_xlabel(r"$\textrm{Time [s]}$")
    axbig[row][col].legend()
    if col == 0:
        axbig[row][col].set_ylabel(r"$\textrm{Whitened amplitude}$")

    print(
        "Percentage Qp to Q energy peak ",
        round(percentage(qpinstance.peak["energy"], qinstance.peak["energy"]), 2),
    )
    print(
        "Percentage Qp to Q energy density ",
        round(percentage(qpinstance.energy_density, qinstance.energy_density), 2),
    )
    print(
        "Percentage Qp to Q TF area ",
        round(percentage(qpinstance.TF_area, qinstance.TF_area), 2),
    )


fbig.savefig(event_label + ".pdf", dpi=150)
