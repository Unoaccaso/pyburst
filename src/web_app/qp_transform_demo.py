import sys
import os.path

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = os.path.join(PATH_TO_THIS, "../")
sys.path.append(PATH_TO_MASTER)

import configparser
from flask import Flask, render_template, request, jsonify
import gwosc.datasets
import logging
from qp_transform.utils import preprocessing, transform
import numpy
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import cupy
import scipy.fft
from flask_socketio import SocketIO

PATH_TO_SETTINGS = os.path.join(PATH_TO_MASTER, "config.ini")
config = configparser.ConfigParser()
config.read(PATH_TO_SETTINGS)

DETECTOR_OPTIONS = [
    ("L1", "Ligo Livingston (L1)"),
    ("H1", "Ligo Hanford (H1)"),
    ("V1", "Virgo (V1)"),
]

CACHED_TIME_SERIES = {}

app = Flask(__name__)
app.secret_key = "your_secret_key"
socketio = SocketIO(app)


def run_options_by_detector(detector):
    all_runs = gwosc.datasets.find_datasets(detector=detector, type="run")
    runs = [(run, run) for run in all_runs if event_exists_for_run(detector, run)]
    return runs


def event_exists_for_run(detector, run):
    all_events = gwosc.datasets.find_datasets(detector=detector, type="event")
    return any(
        gwosc.datasets.run_at_gps(gwosc.datasets.event_gps(event)) == run
        for event in all_events
    )


def event_options_by_detector_run(detector, run):
    all_events = gwosc.datasets.find_datasets(detector=detector, type="event")
    events = [
        (event, event)
        for event in all_events
        if gwosc.datasets.run_at_gps(gwosc.datasets.event_gps(event)) == run
    ]
    return events


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
            new_sampling_rate=2048,
            whitening=True,
        )

        out_data = {
            "full_timeseries": signal[event_name][detector_id],
            "strain": processed_signal.value,
            "time_axis": processed_signal.times.value,
            "sampling_rate": numpy.int32(
                1 / (processed_signal.times.value[1] - processed_signal.times.value[0])
            ),
            "fft_value": scipy.fft.fft(processed_signal.value),
            "fft_freqs": (
                scipy.fft.fftfreq(len(processed_signal.value))
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

    except Exception as e:
        socketio.emit(
            "log_message",
            {
                "message",
                f"Failed downloading {runs} / {max_runs} times. Error: {str(e)}",
            },
        )

        if runs < max_runs:
            socketio.emit(
                "log_message",
                {
                    "message": f"Failed downloading {runs} / {max_runs} times. Error: {str(e)}"
                },
            )

            return download_signal_and_preprocess(
                detector_id, event_name, runs + 1, tau_min, tau_max
            )
        else:
            socketio.emit(
                "log_message",
                {
                    "message",
                    f"Failed downloading too many times",
                },
            )
            return {"success": False, "message": "Could not download data!"}


def reprocess_data(
    detector_id,
    event_name,
    tau_min=0,
    tau_max=0,
    new_sampling_rate=0,
    crop=False,
    resample=False,
    withen=False,
):
    cached_signal = CACHED_TIME_SERIES[detector_id][event_name]["uprocessed_timeseries"]

    if not resample:
        new_sampling_rate = cached_signal["sampling_rate"]

    processed_signal = preprocessing.signal.preprocessing(
        cached_signal["time_series"],
        cached_signal["gps_time"],
        crop=crop,
        left_dt_ms=tau_min,
        right_dt_ms=tau_max,
        resample=resample,
        new_sampling_rate=new_sampling_rate,
        whitening=withen,
    )

    out_data = {
        "unumpyrocessed_timeseries": cached_signal,
        "strain": processed_signal.value,
        "time_axis": processed_signal.times.value,
        "sampling_rate": numpy.int32(
            1 / (processed_signal.times.value[1] - processed_signal.times.value[0])
        ),
        "fft_value": scipy.fft.fft(processed_signal.value),
        "fft_freqs": (
            scipy.fft.fftfreq(len(processed_signal.value))
            * numpy.int32(
                1 / (processed_signal.times.value[1] - processed_signal.times.value[0])
            )
        ),
    }
    return {"success": True, "data": out_data}


def generate_energy_plane(
    detector_id,
    event_name,
    q_value,
    p_value,
    alpha_value,
):
    sampling_rate = CACHED_TIME_SERIES[detector_id][event_name]["sampling_rate"]
    time_axis = CACHED_TIME_SERIES[detector_id][event_name]["time_axis"]
    signal_fft = CACHED_TIME_SERIES[detector_id][event_name]["fft_value"]
    fft_frequencies = CACHED_TIME_SERIES[detector_id][event_name]["fft_freqs"]
    phi_range = [30, 500]

    time_series_duration = time_axis.max() - time_axis.min()
    phi_axis = preprocessing.build_frequency_axis(
        phi_range,
        q_value,
        p_value,
        time_series_duration,
        sampling_rate,
        alpha_value,
    )

    signal_fft_GPU = cupy.array(signal_fft)
    fft_frequencies_GPU = cupy.array(fft_frequencies)
    phi_axis_GPU = cupy.array(phi_axis)

    tau_phi_plane = transform.qp_transform(
        signal_fft_GPU,
        fft_frequencies_GPU,
        phi_axis_GPU,
        q_value,
        p_value,
        sampling_rate,
    )

    energy_plane = cupy.abs(tau_phi_plane) ** 2

    result = {
        "energy_plane": energy_plane.get(),
        "time_axis": time_axis,
        "phi_axis": phi_axis,
    }

    return result


@app.route("/")
def index():
    default_detector = "L1"
    default_run = run_options_by_detector(default_detector)
    default_event = event_options_by_detector_run(default_detector, default_run[0][0])

    data = {
        "title": "Qp Transform Demo",
        "detector_options": DETECTOR_OPTIONS,
        "run_options": default_run,
        "event_options": default_event,
        "default_selected_run": default_run[0][0],
    }

    return render_template("./index.html", data=data)


@app.route("/get_run_options", methods=["POST"])
def get_run_options():
    detector_name = request.form.get("detector")
    run_options = run_options_by_detector(detector_name)
    return jsonify({"run_options": run_options})


@app.route("/get_event_options", methods=["POST"])
def get_event_options():
    detector_name = request.form.get("detector")
    run_name = request.form.get("run")
    event_options = event_options_by_detector_run(detector_name, run_name)
    return jsonify({"event_options": event_options})


@app.route("/check_and_download_timeseries", methods=["POST"])
def check_and_download_timeseries():
    detector_id = request.form.get("detector")
    event_name = request.form.get("event")
    tau_min = numpy.int32(request.form.get("tau_min"))
    tau_max = numpy.int32(request.form.get("tau_max"))

    result = download_signal_and_preprocess(
        detector_id, event_name, 1, tau_min, tau_max
    )

    if result["success"]:
        CACHED_TIME_SERIES.setdefault(detector_id, {})[event_name] = result["data"]
        return jsonify({"success": True, "message": "Data processed"})
    else:
        return jsonify({"success": False, "message": result["message"]})


@app.route("/reselect_time_interval")
def reselect_time_interval():
    detector_id = request.form.get("detector")
    event_name = request.form.get("event")
    tau_min = numpy.int32(request.form.get("tau_min"))
    tau_max = numpy.int32(request.form.get("tau_max"))

    result = reprocess_data(detector_id, event_name, tau_min, tau_max)

    if result["success"]:
        CACHED_TIME_SERIES.setdefault(detector_id, {})[event_name] = result["data"]
        return jsonify({"success": True, "message": "New interval selected"})
    else:
        return jsonify({"success": False, "message": "Recropping failed"})


from bokeh.plotting import figure
from bokeh.models import LogColorMapper
from bokeh.embed import components


@app.route("/generate_energy_plane", methods=["POST"])
def generate_energy_plane_route():
    q_value = numpy.float32(request.form.get("qValue"))
    p_value = numpy.float32(request.form.get("pValue"))
    alpha_value = numpy.float32(request.form.get("alphaValue"))
    detector_id = request.form.get("detector")
    event_name = request.form.get("event")

    result = generate_energy_plane(
        detector_id, event_name, q_value, p_value, alpha_value
    )

    if result:
        energy_plane = result["energy_plane"]
        time_axis = result["time_axis"]
        phi_axis = result["phi_axis"]

        # Creazione del grafico Bokeh
        p = figure(
            title="Energy Plane Heatmap",
            x_axis_label="Time Axis",
            y_axis_label="Phi Axis",
            width=800,
            height=600,
            toolbar_location="above",
        )

        # Utilizza LogColorMapper per l'asse y in scala logaritmica
        mapper = LogColorMapper(
            palette="Viridis256",
            low=numpy.min(energy_plane),
            high=numpy.max(energy_plane),
        )
        p.image(
            image=[energy_plane],
            x=numpy.min(time_axis),
            y=numpy.min(phi_axis),
            dw=numpy.ptp(
                time_axis
            ),  # Peak-to-peak, equivalent to numpy.max(time_axis) - numpy.min(time_axis)
            dh=numpy.ptp(
                phi_axis
            ),  # Peak-to-peak, equivalent to numpy.max(phi_axis) - numpy.min(phi_axis)
            color_mapper=mapper,
        )

        # Salva il codice HTML e JavaScript del grafico Bokeh
        script, div = components(p)
        print(script)
        return {"script": script, "div": div}

    else:
        return jsonify({"error": "Failed to generate energy plane"})


if __name__ == "__main__":
    socketio.run(app, debug=True)
