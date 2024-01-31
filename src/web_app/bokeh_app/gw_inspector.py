import sys
import os.path

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = os.path.join(PATH_TO_THIS, "../..")
sys.path.append(PATH_TO_MASTER)

from qp_transform.utils import preprocessing, transform, filter

from bokeh.transform import linear_cmap
from bokeh.models import ColorBar
from bokeh.plotting import figure, curdoc
import bokeh.models as models
from bokeh.layouts import column, row
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler

import gwosc.datasets

import numpy
import scipy
import cupy


def event_exists_for_run(detector, run):
    all_events = gwosc.datasets.find_datasets(detector=detector, type="event")
    return any(
        gwosc.datasets.run_at_gps(gwosc.datasets.event_gps(event)) == run
        for event in all_events
    )


DETECTOR_OPTIONS = [
    ("L1", "Ligo Livingston (L1)"),
    ("H1", "Ligo Hanford (H1)"),
    ("V1", "Virgo (V1)"),
]

DEFAULT_RUN_LIST = [
    (run, run)
    for run in gwosc.datasets.find_datasets(detector=DETECTOR_OPTIONS[0][0], type="run")
    if event_exists_for_run(DETECTOR_OPTIONS[0][0], run)
]

CACHED_TIME_SERIES = {}


def download_signal_and_preprocess(detector_id, event_name, runs, tau_min, tau_max):
    max_runs = 100

    try:
        signal = preprocessing.signal.get_data_from_gwosc(
            [event_name],
            [detector_id],
            segment_duration=20,
        )

        processed_signal = preprocessing.signal.preprocessing(
            signal[event_name][detector_id]["time_series"],
            signal[event_name][detector_id]["gps_time"],
            crop=True,
            left_dt_ms=tau_min,
            right_dt_ms=tau_max,
            resample=True,
            new_sampling_rate=4096,
            whitening=True,
        )

        strain_GPU = cupy.array(processed_signal.value)

        out_data = {
            "unprocessed_timeseries": signal[event_name][detector_id],
            "strain": strain_GPU,
            "time_axis": processed_signal.times.value,
            "sampling_rate": numpy.int32(
                1 / (processed_signal.times.value[1] - processed_signal.times.value[0])
            ),
            "fft_value_GPU": cupy.fft.fft(strain_GPU),
            "fft_freqs_GPU": (
                cupy.fft.fftfreq(len(strain_GPU))
                * numpy.int32(
                    1
                    / (
                        processed_signal.times.value[1]
                        - processed_signal.times.value[0]
                    )
                )
            ),
        }
        return {"success": True, "data": out_data}

    except:
        # TODO: error handling
        if runs < max_runs:
            print(f"Failed downloading {runs} / {max_runs} times.")
            return download_signal_and_preprocess(
                detector_id, event_name, runs + 1, tau_min, tau_max
            )
        else:
            print(f"Failed downloading too many times")
            # TODO: error handling
            return {"success": False}


def check_and_download_timeseries(detector_id, event_name, tau_min, tau_max):
    # Verifica se i dati sono già presenti nella cache
    if (
        detector_id in CACHED_TIME_SERIES
        and event_name in CACHED_TIME_SERIES[detector_id]
    ):
        # I dati sono già presenti nella cache, restituisci direttamente i dati esistenti
        data = CACHED_TIME_SERIES[detector_id][event_name]
        # TODO: error handling
        return
    else:
        # I dati non sono presenti nella cache, effettua il download e aggiornamento della cache
        result = download_signal_and_preprocess(
            detector_id, event_name, 1, tau_min, tau_max
        )

        if result["success"]:
            CACHED_TIME_SERIES.setdefault(detector_id, {})[event_name] = result["data"]
        else:
            # TODO: error handling
            return


def generate_energy_plane(
    detector_id,
    event_name,
    q_value,
    p_value,
    alpha_value,
    phi_range,
):
    sampling_rate = CACHED_TIME_SERIES[detector_id][event_name]["sampling_rate"]
    time_axis = CACHED_TIME_SERIES[detector_id][event_name]["time_axis"]
    signal_fft_GPU = CACHED_TIME_SERIES[detector_id][event_name]["fft_value_GPU"]
    fft_frequencies_GPU = CACHED_TIME_SERIES[detector_id][event_name]["fft_freqs_GPU"]

    time_series_duration = time_axis.max() - time_axis.min()
    phi_axis = preprocessing.build_frequency_axis(
        phi_range,
        numpy.float32(q_value),
        numpy.float32(p_value),
        time_series_duration,
        numpy.int32(sampling_rate),
        numpy.float32(alpha_value),
    )

    phi_axis_GPU = cupy.array(phi_axis)

    tau_phi_plane = transform.qp_transform(
        signal_fft_GPU,
        fft_frequencies_GPU,
        phi_axis_GPU,
        numpy.float32(q_value),
        numpy.float32(p_value),
        numpy.int32(sampling_rate),
    )

    energy_plane = cupy.abs(tau_phi_plane) ** 2

    result = {
        "energy_plane": energy_plane,
        "time_axis": time_axis,
        "phi_axis": phi_axis_GPU,
    }

    return result


def filtered_signal(
    detector_id,
    event_name,
    phi_axis_GPU,
    time_axis,
    Q,
    p,
    energy_plane_GPU,
    energy_threshold,
):
    time_axis = CACHED_TIME_SERIES[detector_id][event_name]["time_axis"]
    signal_fft_GPU = CACHED_TIME_SERIES[detector_id][event_name]["fft_value_GPU"]
    fft_frequencies_GPU = CACHED_TIME_SERIES[detector_id][event_name]["fft_freqs_GPU"]
    filtered_signal = filter.filter(
        signal_fft_GPU,
        fft_frequencies_GPU,
        phi_axis_GPU,
        time_axis,
        Q,
        p,
        energy_plane_GPU,
        energy_threshold,
    )

    signal_result = {"filtered_signal": filtered_signal, "time_axis": time_axis}

    return signal_result


def main_func(doc):
    # Funzione di aggiornamento del grafico
    def update_plot():
        result = generate_energy_plane(
            detector_menu.value,
            event_menu.value,
            Q_slider.value,
            p_slider.value,
            alpha_slider.value,
            phi_range=phi_range_slider.value,
        )

        CACHED_TIME_SERIES[detector_menu.value][event_menu.value][
            "energy_plane"
        ] = result["energy_plane"]
        CACHED_TIME_SERIES[detector_menu.value][event_menu.value]["phi_axis"] = result[
            "phi_axis"
        ]

        energy_plane = result["energy_plane"].get()
        phi_axis = result["phi_axis"].get()

        # Utilizza un Colormap lineare
        mapper = models.LinearColorMapper(
            palette="Viridis256",
            low=energy_plane.min(),
            high=energy_plane.max(),
        )

        # Rimuovi tutti i renderer associati alla figura
        qp_transform_plot.renderers = []

        qp_transform_plot.image(
            image=[energy_plane],
            x=result["time_axis"].min(),
            y=phi_axis.min(),
            dw=result["time_axis"].max() - result["time_axis"].min(),
            dh=phi_axis.max() - phi_axis.min(),
            color_mapper=mapper,
            # alpha=0.4,
        )

        # setto i limiti
        qp_transform_plot.x_range.range_padding = 0
        qp_transform_plot.y_range.range_padding = 0

        color_bar.color_mapper = mapper

        update_signal_plot()

    def update_signal_plot():
        energy_plane = CACHED_TIME_SERIES[detector_menu.value][event_menu.value][
            "energy_plane"
        ]
        phi_axis = CACHED_TIME_SERIES[detector_menu.value][event_menu.value]["phi_axis"]
        time_axis = CACHED_TIME_SERIES[detector_menu.value][event_menu.value][
            "time_axis"
        ]
        signal_result = filtered_signal(
            detector_menu.value,
            event_menu.value,
            phi_axis,
            time_axis,
            Q_slider.value,
            p_slider.value,
            energy_plane,
            energy_threshold_slider.value,
        )

        signal_plot.renderers = []

        signal_plot.line(
            x=time_axis,
            y=CACHED_TIME_SERIES[detector_menu.value][event_menu.value]["strain"].get(),
            alpha=0.4,
        )
        signal_plot.line(
            x=time_axis,
            y=signal_result["filtered_signal"].get(),
            color="red",
        )
        # # setto i limiti
        signal_plot.x_range.range_padding = 0
        signal_plot.y_range.range_padding = 0

    def update_run_list(attr, old, new):
        all_runs = gwosc.datasets.find_datasets(detector=new, type="run")
        runs = [(run, run) for run in all_runs if event_exists_for_run(new, run)]
        run_menu.options = runs
        run_menu.value = runs[0][0]
        update_event_list("", "", run_menu.value)

    def update_event_list(attr, old, new):
        all_events = gwosc.datasets.find_datasets(
            detector=detector_menu.value, type="event"
        )
        events = [
            (event, event)
            for event in all_events
            if gwosc.datasets.run_at_gps(gwosc.datasets.event_gps(event)) == new
        ]
        event_menu.options = events
        event_menu.value = events[0][0]
        check_and_download_timeseries(
            detector_menu.value,
            event_menu.value,
            -100,
            100,
        )

    def update_event(attr, old, new):
        event_menu.value = new
        check_and_download_timeseries(
            detector_menu.value,
            new,
            tau_range_slider.value[0],
            tau_range_slider.value[1],
        )
        update_plot()

    def update_tau_range(attr, old, new):
        tau_range_slider.value = new
        reprocess_data()
        update_plot()

    def update_sampling_rate(attr, old, new):
        sampling_rate.value = new
        reprocess_data()
        update_plot()

    def update_withening(attr, old, new):
        withening.active = new
        reprocess_data()
        update_plot()

    def update_Q(attr, old, new):
        Q_slider.value = new
        update_plot()

    def update_p(attr, old, new):
        p_slider.value = new
        update_plot()

    def update_alpha(attr, old, new):
        alpha_slider.value = new
        update_plot()

    def update_phi_range(attr, old, new):
        phi_range_slider.value = new
        update_plot()

    def update_energy_threshold(attr, old, new):
        energy_threshold_slider.value = new
        update_signal_plot()

    def reprocess_data():
        detector_id = detector_menu.value
        event_id = event_menu.value

        cached_signal = CACHED_TIME_SERIES[detector_id][event_id][
            "unprocessed_timeseries"
        ]

        processed_signal = preprocessing.signal.preprocessing(
            cached_signal["time_series"],
            cached_signal["gps_time"],
            crop=True,
            left_dt_ms=tau_range_slider.value[0],
            right_dt_ms=tau_range_slider.value[1],
            resample=True,
            new_sampling_rate=sampling_rate.value,
            whitening=withening.active,
        )

        strain_GPU = cupy.array(processed_signal.value)

        CACHED_TIME_SERIES[detector_id][event_id] = {
            "unprocessed_timeseries": cached_signal,
            "strain": strain_GPU,
            "time_axis": processed_signal.times.value,
            "sampling_rate": numpy.int32(
                1 / (processed_signal.times.value[1] - processed_signal.times.value[0])
            ),
            "fft_value_GPU": cupy.fft.fft(strain_GPU),
            "fft_freqs_GPU": (
                cupy.fft.fftfreq(len(strain_GPU))
                * numpy.int32(
                    1
                    / (
                        processed_signal.times.value[1]
                        - processed_signal.times.value[0]
                    )
                )
            ),
        }

    # CREATING WISGETS
    # Building dropdowns
    detector_menu = models.Select(
        title="Detector", value=DETECTOR_OPTIONS[0][0], options=DETECTOR_OPTIONS
    )
    run_menu = models.Select(title="Run")
    event_menu = models.Select(title="Event")

    # building sliders
    tau_range_slider = models.RangeSlider(
        start=-200, end=200, value=(-100, 100), step=1, title="Tau range"
    )
    # building sliders
    phi_range_slider = models.RangeSlider(
        start=1, end=1024, value=(30, 500), step=1, title="Phi range"
    )
    Q_slider = models.Slider(
        start=2 * numpy.pi, end=10 * numpy.pi, value=10, step=0.1, title="Q value"
    )
    p_slider = models.Slider(start=0, end=0.2, value=0.12, step=0.01, title="p value")
    alpha_slider = models.Slider(
        start=0.01, end=1, value=0.05, step=0.01, title="alpha value"
    )
    energy_threshold_slider = models.Slider(
        start=1, end=10, value=7, step=0.1, title="Energy threshold"
    )

    withening = models.Switch(active=True)
    sampling_rate = models.NumericInput(
        value=4096, low=128, high=4096, title="Sampling Rate"
    )

    # Crea una figura Bokeh
    qp_transform_plot = figure(
        title="Energy Plane",
        x_axis_label="GPS time [s]",
        y_axis_label="Frequency [Hz]",
        tools="pan,box_zoom, wheel_zoom,reset,save",
        y_axis_type="log",
        min_width=200,
        aspect_ratio=4 / 3,
        output_backend="webgl",
    )

    signal_plot = figure(
        title="Signal waveform",
        x_axis_label="GPS time [s]",
        y_axis_label="Amplitude",
        tools="pan,box_zoom, wheel_zoom,reset,save",
        min_width=200,
        aspect_ratio=3 / 2,
        output_backend="webgl",
    )

    # # Aggiungi una barra dei colori
    color_bar = ColorBar(width=8, location=(0, 0))
    qp_transform_plot.add_layout(color_bar, "right")

    # POPULATING START PARAMS
    update_run_list("", "", detector_menu.value)
    check_and_download_timeseries(
        detector_menu.value,
        event_menu.value,
        tau_range_slider.value[0],
        tau_range_slider.value[1],
    )

    # UPDATE HANDLER

    update_plot()
    detector_menu.on_change("value", update_run_list)
    run_menu.on_change("value", update_event_list)
    event_menu.on_change("value", update_event)
    tau_range_slider.on_change("value", update_tau_range)
    sampling_rate.on_change("value", update_sampling_rate)
    withening.on_change("active", update_withening)

    Q_slider.on_change("value", update_Q)
    p_slider.on_change("value", update_p)
    alpha_slider.on_change("value", update_alpha)
    phi_range_slider.on_change("value", update_phi_range)
    energy_threshold_slider.on_change("value", update_energy_threshold)

    layout = row(
        column(
            tau_range_slider,
            phi_range_slider,
            Q_slider,
            p_slider,
            alpha_slider,
            sampling_rate,
            withening,
        ),
        column(
            row(detector_menu, run_menu, event_menu),
            row(qp_transform_plot),
            row(energy_threshold_slider),
            row(signal_plot),
        ),
    )

    # Aggiorna il documento Bokeh
    doc.clear()
    doc.add_root(layout)


if __name__ == "__main__":
    # Imposta il server Bokeh
    handler = FunctionHandler(main_func)
    app = Application(handler)
    server = Server({"/": app}, num_procs=1)
    server.start()

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
