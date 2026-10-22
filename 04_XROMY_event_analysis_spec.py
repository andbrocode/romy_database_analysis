#!/usr/bin/env python3
"""
Automated ROMY Event Spectral Analysis Script

This script processes a list of events from the events/ directory and:
1. Loads waveform data for each event (same configuration logic as 04_XROMY_event_analysis_baz.py)
2. Computes amplitude spectra (FFT) per channel
3. Computes power spectral density (PSD) per channel
4. Optionally computes smoothed versions of FFT and PSD
   - via Savitzky-Golay smoothing, or
   - via median values in octave bands
5. Stores results in JSON format in each event's directory

No plots are created.
"""

import os
import json
import argparse
import yaml
import numpy as np
from obspy import UTCDateTime
from sixdegrees.sixdegrees import sixdegrees

try:
    from scipy.signal import welch, savgol_filter
except Exception:  # pragma: no cover - SciPy may not be available in all environments
    welch = None
    savgol_filter = None


def _safe_savgol(y, window_length, polyorder):
    """
    Safely apply Savitzky-Golay smoothing.

    Ensures odd window length and that it does not exceed data length.
    Falls back to original data if parameters are not suitable.
    """
    if savgol_filter is None:
        return y

    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return y

    # Ensure window_length is odd and not larger than data
    window_length = int(window_length)
    if window_length < 3:
        window_length = 3
    if window_length % 2 == 0:
        window_length += 1
    if window_length > n:
        window_length = n if n % 2 == 1 else n - 1
    if window_length <= polyorder:
        return y

    try:
        return savgol_filter(y, window_length=window_length, polyorder=polyorder)
    except Exception:
        return y


def _octave_band_medians(freqs, values, fmin, fmax, octave_fraction=1.0):
    """
    Compute median values in octave bands between fmin and fmax.

    Parameters
    ----------
    freqs : array_like
        Frequency array (Hz).
    values : array_like
        Spectral values (same length as freqs).
    fmin, fmax : float
        Minimum and maximum frequency to consider.
    octave_fraction : float, optional
        Number of bands per octave (1 = full octaves, 3 = third-octave bands, etc.).

    Returns
    -------
    dict
        Dictionary with 'frequency_center' and 'value_median' lists.
    """
    freqs = np.asarray(freqs, dtype=float)
    values = np.asarray(values, dtype=float)

    mask = (freqs > 0) & np.isfinite(freqs) & np.isfinite(values)
    freqs = freqs[mask]
    values = values[mask]

    if freqs.size == 0:
        return {
            "frequency_center": [],
            "value_median": [],
        }

    if fmin is None:
        fmin = freqs.min()
    if fmax is None:
        fmax = freqs.max()

    fmin = max(fmin, freqs.min())
    fmax = min(fmax, freqs.max())

    if fmin <= 0 or fmin >= fmax:
        return {
            "frequency_center": [],
            "value_median": [],
        }

    band_centers = []
    band_medians = []

    # Ratio between adjacent band edges for the requested fraction of octaves
    if octave_fraction is None or octave_fraction <= 0:
        octave_fraction = 1.0
    band_ratio = 2.0 ** (1.0 / float(octave_fraction))

    f_low = fmin
    while f_low < fmax:
        f_high = min(band_ratio * f_low, fmax)
        band_mask = (freqs >= f_low) & (freqs < f_high)
        if np.any(band_mask):
            band_vals = values[band_mask]
            band_centers.append(float(np.sqrt(f_low * f_high)))
            band_medians.append(float(np.median(band_vals)))
        f_low = f_high

    return {
        "frequency_center": band_centers,
        "value_median": band_medians,
    }


def _compute_fft_and_psd(data, sampling_rate, psd_nperseg=2048, psd_noverlap=None):
    """
    Compute amplitude spectrum (FFT) and PSD for a 1D time series.

    Returns
    -------
    dict with keys:
        - fft_frequency, fft_amplitude
        - psd_frequency, psd
    """
    data = np.asarray(data, dtype=float)
    n = data.size
    if n == 0 or not np.isfinite(data).any():
        return {
            "fft_frequency": [],
            "fft_amplitude": [],
            "psd_frequency": [],
            "psd": [],
        }

    # FFT amplitude spectrum
    freqs_fft = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    fft_vals = np.fft.rfft(data)
    amp_fft = np.abs(fft_vals)

    # PSD via Welch (if available), otherwise approximate from FFT
    if welch is not None:
        nperseg = int(min(psd_nperseg, n)) if psd_nperseg is not None else min(2048, n)
        if nperseg < 8:
            nperseg = n
        try:
            freqs_psd, psd_vals = welch(
                data,
                fs=sampling_rate,
                nperseg=nperseg,
                noverlap=psd_noverlap,
            )
        except Exception:
            freqs_psd = freqs_fft
            psd_vals = (amp_fft ** 2) / n
    else:
        freqs_psd = freqs_fft
        psd_vals = (amp_fft ** 2) / n

    return {
        "fft_frequency": freqs_fft,
        "fft_amplitude": amp_fft,
        "psd_frequency": freqs_psd,
        "psd": psd_vals,
    }


def _spectra_for_stream(stream, sampling_rate, smoothing, psd_nperseg, psd_noverlap):
    """
    Compute spectra for all traces in a stream.

    Parameters
    ----------
    stream : obspy.Stream or None
    sampling_rate : float
    smoothing : dict
        Contains keys:
            - method: 'none', 'savgol', or 'octave_median'
            - savgol_window, savgol_order
    """
    if stream is None:
        return {}

    results = {}

    # Optional explicit frequency bounds and band fraction for octave-median smoothing
    octave_fmin = smoothing.get("octave_fmin", None)
    octave_fmax = smoothing.get("octave_fmax", None)
    octave_fraction = smoothing.get("octave_fraction", 1.0)

    for tr in stream:

        tr.detrend("linear")
        tr.taper(0.05, type='cosine')

        spec = _compute_fft_and_psd(
            tr.data,
            sampling_rate=sampling_rate,
            psd_nperseg=psd_nperseg,
            psd_noverlap=psd_noverlap,
        )

        # Convert to Python lists
        fft_freq = np.asarray(spec["fft_frequency"], dtype=float)
        fft_amp = np.asarray(spec["fft_amplitude"], dtype=float)
        psd_freq = np.asarray(spec["psd_frequency"], dtype=float)
        psd_vals = np.asarray(spec["psd"], dtype=float)

        channel_result = {
            "fft": {
                "frequency": fft_freq.tolist(),
                "amplitude": fft_amp.tolist(),
            },
            "psd": {
                "frequency": psd_freq.tolist(),
                "psd": psd_vals.tolist(),
            },
        }

        method = smoothing.get("method", "none")
        if method == "savgol":
            window = smoothing.get("savgol_window", 51)
            order = smoothing.get("savgol_order", 3)

            fft_amp_smooth = _safe_savgol(fft_amp, window_length=window, polyorder=order)
            psd_smooth = _safe_savgol(psd_vals, window_length=window, polyorder=order)

            channel_result["fft_smooth"] = {
                "frequency": fft_freq.tolist(),
                "amplitude": np.asarray(fft_amp_smooth, dtype=float).tolist(),
            }
            channel_result["psd_smooth"] = {
                "frequency": psd_freq.tolist(),
                "psd": np.asarray(psd_smooth, dtype=float).tolist(),
            }

        elif method == "octave_median":
            # Use explicitly provided fmin/fmax for octave bands (if any);
            # otherwise, _octave_band_medians falls back to the data range.
            fft_oct = _octave_band_medians(
                fft_freq,
                fft_amp,
                fmin=octave_fmin,
                fmax=octave_fmax,
                octave_fraction=octave_fraction,
            )
            psd_oct = _octave_band_medians(
                psd_freq,
                psd_vals,
                fmin=octave_fmin,
                fmax=octave_fmax,
                octave_fraction=octave_fraction,
            )

            channel_result["fft_octave_median"] = fft_oct
            channel_result["psd_octave_median"] = psd_oct

        results[tr.id] = channel_result

    return results


def process_event(
    date,
    events_list,
    analysis_config,
):
    """
    Process a single event and compute spectra.

    Parameters
    ----------
    date : str
        Event date string (e.g., '20240314')
    events_list : list
        List of event dates to process
    analysis_config : dict
        Analysis configuration dictionary containing:
        - fmin / fmax: Frequency bounds (Hz) used for filtering and summaries
        - psd_nperseg / psd_noverlap: Parameters for Welch PSD estimation
        - method, savgol_window, savgol_order, octave_fmin, octave_fmax, octave_fraction: smoothing settings
    """
    print(f"\n{'=' * 60}")
    print(f"Processing event: {date}")
    print(f"{'=' * 60}")

    try:
        # Load configuration
        cfg_path = f"./events/event_{date}/config_romy_{date}.yml"
        if not os.path.exists(cfg_path):
            print(f"Warning: Config file not found: {cfg_path}")
            return None

        event_config = sixdegrees.load_from_yaml(cfg_path)

        # Setup sixdegrees object
        sd = sixdegrees(conf=event_config)

        # Get event information
        ev = sd.get_event_info(
            origin_time=event_config["origin_time"],
            base_catalog="USGS",
            magnitude=event_config["magnitude"],
        )

        print(f"Event: {ev.get('origin_time', 'N/A')}")
        print(f"Theoretical backazimuth: {ev.get('backazimuth', float('nan')):.2f}°")

        # Load data
        sd.load_data(event_config["tbeg"], event_config["tend"], resample_rate=10)

        # Ensure all traces have the same amount of samples
        sd.trim_stream(set_common=True, set_interpolate=True)

        # Filter data
        # sd.filter_data(fmin=config["fmin"], fmax=config["fmax"], output=False)

        # Apply polarity correction if needed
        if "pol_dict" in event_config:
            sd.polarity_stream(pol_dict=event_config["pol_dict"], raw=True)
        else:
            # Default polarity correction from notebook
            sd.polarity_stream(pol_dict={"HZ": -1, "JZ": 1}, raw=True)

        # Cut bad data if specified
        if "time_cuts" in event_config.keys():
            print(f"Cutting bad data for {date}...")
            for key, values in event_config["time_cuts"].items():
                for value in values:
                    sd.replace_values_in_window(
                        tbeg=UTCDateTime(value[0]),
                        tend=UTCDateTime(value[1]),
                        value=0,
                        channel_list=[key],
                    )

        # Determine sampling rate
        sampling_rate = getattr(sd, "sampling_rate", None)
        if sampling_rate is None:
            # Fallback: take from any available trace
            rotation_stream = sd.get_stream("rotation")
            translation_stream = sd.get_stream("translation")
            ref_stream = rotation_stream or translation_stream
            if ref_stream and len(ref_stream) > 0:
                sampling_rate = float(ref_stream[0].stats.sampling_rate)
            else:
                raise RuntimeError("Could not determine sampling rate from streams.")
        else:
            sampling_rate = float(sampling_rate)

        print(f"Using sampling rate: {sampling_rate} Hz")

        # Get rotation and translation streams
        rot_stream = sd.get_stream("rotation")
        tra_stream = sd.get_stream("translation")

        # Rotate streams to radial/transverse using theoretical backazimuth
        baz_theo = ev.get("backazimuth", None)
        rot_stream_rt = None
        tra_stream_rt = None
        if baz_theo is not None:
            try:
                if rot_stream is not None:
                    rot_stream_rt = rot_stream.copy().select(channel="*J*")
                    if len(rot_stream_rt) > 0:
                        rot_stream_rt.rotate("NE->RT", back_azimuth=baz_theo)
                if tra_stream is not None:
                    tra_stream_rt = tra_stream.copy().select(channel="*H*")
                    if len(tra_stream_rt) > 0:
                        tra_stream_rt.rotate("NE->RT", back_azimuth=baz_theo)
            except Exception as e:
                print(f"Warning: could not rotate streams to RT: {e}")

        # Prepare smoothing configuration for this event from analysis_config
        smoothing = {
            "method": analysis_config["method"],
            "savgol_window": analysis_config["savgol_window"],
            "savgol_order": analysis_config["savgol_order"],
            "octave_fmin": analysis_config["octave_fmin"],
            "octave_fmax": analysis_config["octave_fmax"],
            "octave_fraction": analysis_config["octave_fraction"],
        }

        print("Computing spectra for rotation stream...")
        rotation_spectra = _spectra_for_stream(
            rot_stream,
            sampling_rate=sampling_rate,
            smoothing=smoothing,
            psd_nperseg=analysis_config["psd_nperseg"],
            psd_noverlap=analysis_config["psd_noverlap"],
        )

        print("Computing spectra for translation stream...")
        translation_spectra = _spectra_for_stream(
            tra_stream,
            sampling_rate=sampling_rate,
            smoothing=smoothing,
            psd_nperseg=analysis_config["psd_nperseg"],
            psd_noverlap=analysis_config["psd_noverlap"],
        )

        # Spectra for RT-rotated streams (if available)
        rotation_rt_spectra = {}
        translation_rt_spectra = {}
        if rot_stream_rt is not None:
            print("Computing spectra for rotation stream (RT)...")
            rotation_rt_spectra = _spectra_for_stream(
                rot_stream_rt,
                sampling_rate=sampling_rate,
                smoothing=smoothing,
                psd_nperseg=analysis_config["psd_nperseg"],
                psd_noverlap=analysis_config["psd_noverlap"],
            )
        if tra_stream_rt is not None:
            print("Computing spectra for translation stream (RT)...")
            translation_rt_spectra = _spectra_for_stream(
                tra_stream_rt,
                sampling_rate=sampling_rate,
                smoothing=smoothing,
                psd_nperseg=analysis_config["psd_nperseg"],
                psd_noverlap=analysis_config["psd_noverlap"],
            )

        # Build result dictionary
        result = {
            "date": date,
            "event_info": {
                "origin_time": str(ev.get("origin_time", "")),
                "magnitude": ev.get("magnitude", None),
                "distance_km": ev.get("distance_km", None),
                "backazimuth": ev.get("backazimuth", None),
            },
            "config": {
                "fmin": analysis_config.get("fmin", None),
                "fmax": analysis_config.get("fmax", None),
                "tbeg": str(event_config.get("tbeg", "")),
                "tend": str(event_config.get("tend", "")),
            },
            "sampling_rate": sampling_rate,
            "smoothing": smoothing,
            "spectra": {
                "rotation": rotation_spectra,
                "translation": translation_spectra,
                "rotation_RT": rotation_rt_spectra,
                "translation_RT": translation_rt_spectra,
            },
        }

        # Write JSON output into event directory
        event_dir = f"./events/event_{date}"
        if not os.path.exists(event_dir):
            os.makedirs(event_dir)

        json_path = os.path.join(event_dir, f"spectral_analysis_{date}.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Spectral results written to: {json_path}")

        return result

    except Exception as e:
        print(f"Error processing event {date}: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_events_list(yaml_file=None, events_arg=None):
    """
    Load event list from YAML file, command-line argument, or return default list.

    This mirrors the behavior of 04_XROMY_event_analysis_baz.py.
    """
    # If events are provided directly as argument, use those
    if events_arg:
        events_list = [e.strip() for e in events_arg.split(",") if e.strip()]
        print(f"Using {len(events_list)} events from command-line argument")
        return events_list

    # Otherwise, try to load from YAML file
    if yaml_file and os.path.exists(yaml_file):
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)
            events_list = config.get("events", [])
            print(f"Loaded {len(events_list)} events from {yaml_file}")
            return events_list

    # Fallback: empty list
    print("No event list provided; nothing to process.")
    return []


def main():
    """
    Main function to process all events.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Process ROMY events for spectral analysis (FFT, PSD, and optional smoothing) "
            "and store results as JSON in each event directory."
        )
    )

    parser.add_argument(
        "--events-file",
        type=str,
        default=None,
        help="Path to YAML file containing list of events to process",
    )
    parser.add_argument(
        "--events",
        type=str,
        default=None,
        help=(
            'Comma-separated list of event dates (e.g., "20230908,20230912"). '
            "Takes precedence over --events-file."
        ),
    )
    args = parser.parse_args()

    # Load events list (--events takes precedence over --events-file)
    events_list = load_events_list(
        yaml_file=args.events_file,
        events_arg=args.events,
    )

    if not events_list:
        print("No events to process. Exiting.")
        return

    print("=" * 60)
    print("ROMY Event Analysis - Spectral Analysis")
    print("=" * 60)
    print(f"Processing {len(events_list)} events...")

    # Internal configuration for smoothing and PSD (edit here as needed)
    analysis_config = {

        # Parameters for Welch PSD estimation (nperseg and noverlap)
        "psd_nperseg": 2*4096,
        "psd_noverlap": 2*2048,

        # 'none', 'savgol', or 'octave_median'
        "method": "octave_median",

        # Parameters for Savitzky-Golay (used only if method == 'savgol')
        "savgol_window": None,  # 51,
        "savgol_order": None,   # 3,

        # Frequency bounds for octave-median smoothing (Hz); if None, use data range
        "octave_fmin": 0.001,
        "octave_fmax": 5.0,

        # Number of bands per octave (1 = full octave, 3 = third-octave, etc.)
        "octave_fraction":6.0,
    }


    all_results = []
    for date in events_list:
        result = process_event(
            date=date,
            events_list=events_list,
            analysis_config=analysis_config,
        )
        if result is not None:
            all_results.append(result)

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

