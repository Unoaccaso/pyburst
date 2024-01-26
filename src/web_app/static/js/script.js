$(document).ready(function () {
    // Inizializza gli slider
    $("#timeRangeSlider").ionRangeSlider({
        type: "double",
        min: -200,
        max: 200,
        from: -100,
        to: 100,
        step: 1,
        grid: true,
        onFinish: function (data) {
            reselectTimeInterval();
            checkAndDownloadTimeseries();
        }
    });

    $("#qValueSlider").ionRangeSlider({
        type: "single",
        min: 5,
        max: 20,
        step: 0.01,
        from: 10,
        grid: true,
        onFinish: function (data) {
            checkAndDownloadTimeseries();
        }
    });

    $("#pValueSlider").ionRangeSlider({
        type: "single",
        min: 0,
        max: 0.5,
        step: 0.01,
        from: 0.1,
        grid: true,
        onFinish: function (data) {
            checkAndDownloadTimeseries();
        }
    });

    $("#alphaValueSlider").ionRangeSlider({
        type: "single",
        min: 0.01,
        max: 1,
        from: 0.05,
        step: 0.01,
        grid: true,
        onFinish: function (data) {
            checkAndDownloadTimeseries();
        }
    });

    // Gestisce l'input degli slider
    $(".slider-input").on("input", function () {
        updateSliderRange($(this));
        $("#loadingSpinner").show();
        $("#generatedImage").hide();
        checkAndDownloadTimeseries();
    });

    // Inizializza i dropdown
    handleDetectorChange();
    checkAndDownloadTimeseries();

    // Cambio detector dropdown
    $("#detectorDropdown").change(function () {
        handleDetectorChange();
        checkAndDownloadTimeseries();
    });

    // Cambio run dropdown
    $("#runDropdown").change(function () {
        handleRunChange();
        checkAndDownloadTimeseries();
    });

    // Cambio event dropdown
    $("#eventDropdown").change(function () {
        checkAndDownloadTimeseries();
    });

    // Funzione per gestire il cambio del detector
    function handleDetectorChange() {
        var selectedDetector = $("#detectorDropdown").val();

        $.ajax({
            url: "/get_run_options",
            method: "POST",
            data: { detector: selectedDetector },
            success: function (response) {
                var runOptions = response.run_options;
                updateDropdownOptions("#runDropdown", runOptions);
                $("#runDropdown").val(runOptions[0]);
                updateEventDropdown();
            }
        });
    }

    // Funzione per gestire il cambio della run
    function handleRunChange() {
        var selectedDetector = $("#detectorDropdown").val();
        var selectedRun = $("#runDropdown").val();

        $.ajax({
            url: "/get_event_options",
            method: "POST",
            data: { detector: selectedDetector, run: selectedRun },
            success: function (response) {
                var eventOptions = response.event_options;
                updateDropdownOptions("#eventDropdown", eventOptions);
            }
        });
    }

    // Funzione per aggiornare le opzioni del dropdown degli eventi
    function updateEventDropdown() {
        var selectedDetector = $("#detectorDropdown").val();
        var selectedRun = $("#runDropdown").val();

        $.ajax({
            url: "/get_event_options",
            method: "POST",
            data: { detector: selectedDetector, run: selectedRun },
            success: function (response) {
                var eventOptions = response.event_options;
                updateDropdownOptions("#eventDropdown", eventOptions);
            }
        });
    }

    // Funzione per aggiornare le opzioni di un dropdown
    function updateDropdownOptions(dropdownId, options) {
        var dropdown = $(dropdownId);
        dropdown.empty();

        for (var i = 0; i < options.length; i++) {
            var option = options[i];
            dropdown.append('<option value="' + option[0] + '">' + option[1] + '</option>');
        }
    }

    // Funzione per aggiornare il range degli slider
    function updateSliderRange(input) {
        var sliderId = input.attr("id").replace("Min", "").replace("Max", "");
        var slider = $("#" + sliderId + "Slider");

        if (slider.length) {
            var minVal = parseFloat($("#" + sliderId + "Min").val());
            var maxVal = parseFloat($("#" + sliderId + "Max").val());

            slider.data("ionRangeSlider").update({
                min: minVal,
                max: maxVal
            });
        } else {
            console.error("Lo slider con ID " + sliderId + " non è stato trovato.");
        }
    }

    // Funzione per controllare e scaricare la serie temporale
    function checkAndDownloadTimeseries() {
        // Nascondi l'immagine generata e mostra la rotellina di caricamento
        hideImage();
        showLoadingSpinner();

        var selectedDetector = $("#detectorDropdown").val();
        var selectedRun = $("#runDropdown").val();
        var selectedEvent = $("#eventDropdown").val();
        var fromValue = $("#timeRangeSlider").data("ionRangeSlider").result.from;
        var toValue = $("#timeRangeSlider").data("ionRangeSlider").result.to;

        $.ajax({
            url: "/check_and_download_timeseries",
            method: "POST",
            data: {
                detector: selectedDetector,
                run: selectedRun,
                event: selectedEvent,
                tau_min: fromValue,
                tau_max: toValue,
            },
            success: function (response) {
                if (response.success) {
                    // Utilizza la funzione per aggiornare il contenuto di main-image
                    updateEnergyPlane();
                } else {
                    // In caso di errore, mostra un messaggio o gestisci l'errore
                    downloadError();
                }
            },
            error: function (xhr, status, error) {
                console.error("AJAX request failed:", status, error);
                // Nascondi la rotellina di caricamento in caso di errore
                hideLoadingSpinner();
            },
            complete: function () {
                // Nascondi la rotellina di caricamento quando l'operazione è completa
                // Nota: questo verrà chiamato anche in caso di errore
                hideLoadingSpinner();
            }
        });
    }

    function reselectTimeInterval() {
        // Nascondi l'immagine generata e mostra la rotellina di caricamento
        hideImage();
        showLoadingSpinner();

        var selectedDetector = $("#detectorDropdown").val();
        var selectedEvent = $("#eventDropdown").val();
        var fromValue = $("#timeRangeSlider").data("ionRangeSlider").result.from;
        var toValue = $("#timeRangeSlider").data("ionRangeSlider").result.to;

        $.ajax({
            url: "/reselect_time_interval",
            method: "POST",
            data: {
                detector: selectedDetector,
                event: selectedEvent,
                tau_min: fromValue,
                tau_max: toValue,
            },
            success: function (response) {
                if (response.success) {
                    // Utilizza la funzione per aggiornare il contenuto di main-image
                    updateEnergyPlane();
                } else {
                    // In caso di errore, mostra un messaggio o gestisci l'errore
                    downloadError();
                }
            },
            error: function (xhr, status, error) {
                console.error("AJAX request failed:", status, error);
                // Nascondi la rotellina di caricamento in caso di errore
                hideLoadingSpinner();
            },
            complete: function () {
                // Nascondi la rotellina di caricamento quando l'operazione è completa
                // Nota: questo verrà chiamato anche in caso di errore
                hideLoadingSpinner();
            }
        });
    }

    // Funzione per generare e aggiornare il piano energetico
    async function updateEnergyPlane() {
        // Nascondi l'immagine generata e mostra la rotellina di caricamento
        hideImage();
        showLoadingSpinner();

        var fromValue = $("#timeRangeSlider").data("ionRangeSlider").result.from;
        var toValue = $("#timeRangeSlider").data("ionRangeSlider").result.to;
        var qValue = $("#qValueSlider").data("ionRangeSlider").result.from;
        var pValue = $("#pValueSlider").data("ionRangeSlider").result.from;
        var alphaValue = $("#alphaValueSlider").data("ionRangeSlider").result.from;
        var selectedDetector = $("#detectorDropdown").val();
        var selectedRun = $("#runDropdown").val();
        var selectedEvent = $("#eventDropdown").val();

        $.ajax({
            url: "/generate_energy_plane",
            method: "POST",
            data: {
                timeRangeMin: fromValue,
                timeRangeMax: toValue,
                qValue: qValue,
                pValue: pValue,
                alphaValue: alphaValue,
                detector: selectedDetector,
                run: selectedRun,
                event: selectedEvent
            },
            success: function (response) {
                // Utilizza la funzione per aggiornare il contenuto di main-image
                updateMainImage(response.image);
            },
            error: function (xhr, status, error) {
                console.error("AJAX request failed:", status, error);
                // Nascondi la rotellina di caricamento in caso di errore
                hideLoadingSpinner();
            },
            complete: function () {
                // Nascondi la rotellina di caricamento quando l'operazione è completa
                // Nota: questo verrà chiamato anche in caso di errore
                hideLoadingSpinner();
            }
        });
    }

    function updateMainImage(imageData) {
        $(".main-image").html('<img id="generatedImage" src="data:image/png;base64,' + imageData + '" alt="Generated Energy Plane">');
        // Mostra l'immagine generata
        showImage();
    }

    function showLoadingSpinner() {
        var loadingContent = '<img id="loadingSpinner" class="loading-spinner" src="' + loadingSpinnerUrl + '" alt="Loading Spinner">';

        $(".main-image").html(loadingContent);
    }

    function hideLoadingSpinner() {
        $("#loadingSpinner").hide();
    }

    function showImage() {
        $("#generatedImage").show();
    }

    function hideImage() {
        $("#generatedImage").hide();
    }

    function downloadError() {
        $(".main-image").html("Failed to download from GWOSC");
    }
});
