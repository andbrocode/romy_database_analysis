#!/usr/bin/env python3
"""
ROMY vs ADR Comparison Plot Script

This script loads ROMY and ADR data for events specified in events_list.yml and creates comparison plots showing:
- Waveforms for JZ, JN, JE components
- Residuals in percent deviation
- Cross-correlation values
"""

import os
import gc
import warnings
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import hilbert
from obspy import UTCDateTime
from obspy.signal.cross_correlation import correlate, xcorr_max
from obspy.signal.tf_misfit import em, pm, eg, pg, plot_tf_misfits
from sixdegrees.sixdegrees import sixdegrees


def compute_cc(trace1, trace2):
    """Compute cross-correlation between two traces."""
    if trace1 is None or trace2 is None:
        return None, None, None

    if trace1.stats.sampling_rate != trace2.stats.sampling_rate:
        print(f"Warning: Sampling rates are different for {trace1.id} and {trace2.id}")
        return None, None, None

    sps = trace1.stats.sampling_rate

    # Ensure same length (views; no copy)
    min_len = min(len(trace1.data), len(trace2.data))
    data1 = trace1.data[:min_len]
    data2 = trace2.data[:min_len]

    # Normalize data (only required allocations)
    data1_norm = (data1 - data1.mean()) / (data1.std() + 1e-10)
    data2_norm = (data2 - data2.mean()) / (data2.std() + 1e-10)

    # Compute full cross-correlation, extract max, then free the array
    shift_len = len(data1_norm)
    corr = correlate(data1_norm, data2_norm, shift=shift_len)
    shift_max, cc_max = xcorr_max(corr)
    del corr

    # Get zero-shift correlation, then free remaining intermediates
    corr_zero = correlate(data1_norm, data2_norm, shift=0)
    _, cc_zero = xcorr_max(corr_zero)
    del corr_zero, data1_norm, data2_norm

    return cc_max, shift_max / sps, cc_zero


def compute_deviation(trace1, trace2):
    """Compute deviation between two traces."""
    if trace1 is None or trace2 is None:
        return None, None

    # Ensure same length (views; no copy)
    min_len = min(len(trace1.data), len(trace2.data))
    data1 = trace1.data[:min_len]
    data2 = trace2.data[:min_len]

    # Envelopes via Hilbert transform (each call allocates a complex array internally)
    data1_env = np.abs(hilbert(data1))
    data2_env = np.abs(hilbert(data2))
    data1_env_max = data1_env.max()

    # dev = np.where(data1_env > 1e-9, abs(data2_env - data1_env) / data1_env * 100, 0)
    # Allocate the result once, then scale in place; release envelopes early.
    dev = np.abs(data2_env - data1_env)
    dev *= (100.0 / data1_env_max)
    del data1_env, data2_env

    return dev


def compute_rmse(trace1, trace2):
    """
    Compute Root Mean Square Error (RMSE) between two traces.
    
    Parameters:
    -----------
    trace1 : obspy.Trace
        First trace (reference, e.g., ROMY)
    trace2 : obspy.Trace
        Second trace (comparison, e.g., ADR)
    
    Returns:
    --------
    float or None
        RMSE value, or None if traces are invalid
    """
    if trace1 is None or trace2 is None:
        return None
    
    # Ensure same length
    min_len = min(len(trace1.data), len(trace2.data))
    data1 = trace1.data[:min_len]
    data2 = trace2.data[:min_len]
    
    # Compute RMSE: sqrt(mean((data1 - data2)^2))
    rmse = np.sqrt(np.mean((data1 - data2) ** 2))
    
    return rmse


def compute_r_squared(trace1, trace2):
    """
    Compute R-squared (coefficient of determination) based on linear regression.
    
    Performs linear regression: trace2 = a * trace1 + b
    and computes R² = 1 - (SS_res / SS_tot)
    
    Parameters:
    -----------
    trace1 : obspy.Trace
        First trace (independent variable, e.g., ROMY)
    trace2 : obspy.Trace
        Second trace (dependent variable, e.g., ADR)
    
    Returns:
    --------
    float or None
        R-squared value (0 to 1), or None if traces are invalid
    """
    if trace1 is None or trace2 is None:
        return None
    
    # Ensure same length
    min_len = min(len(trace1.data), len(trace2.data))
    data1 = trace1.data[:min_len]
    data2 = trace2.data[:min_len]
    
    # Check if data has variance (avoid division by zero)
    if np.std(data1) == 0 or np.std(data2) == 0:
        return None
    
    # Perform linear regression: y = a*x + b
    # where x = data1 (ROMY), y = data2 (ADR)
    slope, intercept, r_value, p_value, std_err = stats.linregress(data1, data2)
    
    # R-squared is the square of the correlation coefficient
    r_squared = r_value ** 2
    
    return r_squared


def _tf_misfit_inputs(romy_tr, adr_tr, noise_rel=1e-12, seed=0):
    """Build the float arrays fed into the obspy TF-misfit routines.

    A tiny white-noise floor (``noise_rel`` * peak amplitude, default 1e-12)
    is added to both signals so that exact zeros - introduced by ``time_cuts``
    windows or by purely silent leading/trailing samples - cannot make the
    wavelet-domain phase term ``angle(w_1 / w_2)`` divide by zero.  The
    perturbation is many orders of magnitude below the signal and does not
    affect the EM / PM / EG / PG values to within their reported precision.

    Returns ``(romy_arr, adr_arr, dt)`` or ``(None, None, None)`` if a trace
    is missing or has fewer than 2 samples.
    """
    if romy_tr is None or adr_tr is None:
        return None, None, None

    min_len = min(len(romy_tr.data), len(adr_tr.data))
    if min_len <= 1:
        return None, None, None

    romy_arr = np.asarray(romy_tr.data[:min_len], dtype=float)
    adr_arr = np.asarray(adr_tr.data[:min_len], dtype=float)

    peak_amp = max(float(np.abs(romy_arr).max()),
                   float(np.abs(adr_arr).max()))
    if peak_amp > 0.0:
        floor = noise_rel * peak_amp
        rng = np.random.default_rng(seed)
        romy_arr += rng.normal(0.0, floor, size=romy_arr.shape)
        adr_arr += rng.normal(0.0, floor, size=adr_arr.shape)

    dt = float(romy_tr.stats.delta)
    return romy_arr, adr_arr, dt


def process_event(event_date, fmin=0.02, fmax=0.1):
    """
    Process a single event: load data, compute correlations, create plots, and save results.
    
    Parameters:
    -----------
    event_date : str
        Event date in format 'YYYYMMDD'
    fmin : float
        Minimum frequency for filtering (default: 0.02)
    fmax : float
        Maximum frequency for filtering (default: 0.1)
    
    Returns:
    --------
    tuple : (cc_results, deviations, rmse_results, r_squared_results, tf_misfit_results, adr_config, event_info)
        - cc_results: Dictionary containing cross-correlation results for each component
        - deviations: Dictionary containing deviation arrays for each component
        - rmse_results: Dictionary containing RMSE values for each component
        - r_squared_results: Dictionary containing R-squared values for each component
        - tf_misfit_results: Dictionary containing em/pm/eg/pg values for each component
        - adr_config: ADR configuration dictionary
        - event_info: Dictionary with magnitude, distance_km and backazimuth
    """
    print(f"\n{'='*60}")
    print(f"Processing event: {event_date}")
    print(f"{'='*60}")
    
    # Components to compare
    components = ['JZ', 'JN', 'JE']
    
    # Load ROMY data
    print(f"Loading ROMY data for event {event_date}...")
    
    # Try config_romy_<date>.yml first, then config_<date>.yml
    romy_config_path = f"./events/event_{event_date}/config_romy_{event_date}.yml"
    if not os.path.exists(romy_config_path):
        romy_config_path = f"./events/event_{event_date}/config_{event_date}.yml"
    
    if not os.path.exists(romy_config_path):
        raise FileNotFoundError(f"ROMY config file not found: {romy_config_path}")
    
    # Load ROMY configuration
    romy_config = sixdegrees.load_from_yaml(romy_config_path)
    
    # Setup ROMY sixdegrees object
    sd_romy = sixdegrees(conf=romy_config)

    # Get event metadata from catalog (magnitude, distance, backazimuth)
    try:
        event_info = sd_romy.get_event_info(
            origin_time=romy_config['origin_time'],
            base_catalog=romy_config.get("base_catalog", "USGS"),
            magnitude=romy_config.get("magnitude"),
        )
        print(event_info.keys())
    except Exception as e:
        print(f"Warning: could not retrieve event metadata: {e}")
        event_info = {"magnitude": None, "distance_km": None, "backazimuth": None}
    
    # Load ROMY data
    sd_romy.load_data(romy_config['tbeg'], romy_config['tend'], resample_rate=5)
    
    # Ensure all traces have the same amount of samples
    sd_romy.trim_stream(set_common=True, set_interpolate=True)
    
    # Filter data
    sd_romy.filter_data(fmin=fmin, fmax=fmax, output=False)
    
    # Apply polarity correction if needed
    if 'pol_dict' in romy_config:
        sd_romy.polarity_stream(pol_dict=romy_config['pol_dict'], raw=True)
    else:
        # Default polarity correction
        sd_romy.polarity_stream(pol_dict={"HZ": -1, "JZ": 1}, raw=True)
    
    # Cut bad data if specified.  Replace with 0 (not NaN) so downstream
    # computations (mean/std, Hilbert envelope, cross-correlation, TF misfit)
    # do not propagate NaNs and fail silently.
    if 'time_cuts' in romy_config:
        for key, values in romy_config['time_cuts'].items():
            for value in values:
                sd_romy.replace_values_in_window(
                    tbeg=value[0], tend=value[1],
                    value=0, channel_list=[key]
                )
    
    sd_romy.st.trim(UTCDateTime(romy_config['tbeg']), UTCDateTime(romy_config['tend']), nearest_sample=False)
    
    print(f"ROMY data loaded successfully")
    print(f"ROMY channels: {[tr.id for tr in sd_romy.st]}")
    
    # Load ADR data
    print(f"Loading ADR data for event {event_date}...")
    
    # Try config_adr_<date>.yml first, then check if ADR mseed file exists
    adr_config_path = f"./events/event_{event_date}/config_adr_{event_date}.yml"
    
    # Load existing ADR config
    adr_config = sixdegrees.load_from_yaml(adr_config_path)
    
    # Setup ADR sixdegrees object
    sd_adr = sixdegrees(conf=adr_config)
    
    # Load ADR data
    sd_adr.load_data(romy_config['tbeg'], adr_config['tend'], resample_rate=5)
    
    # Ensure all traces have the same amount of samples
    sd_adr.trim_stream(set_common=True, set_interpolate=True)
    
    # Filter data
    sd_adr.filter_data(fmin=fmin, fmax=fmax, output=False)

    # Cut bad data if specified for ROMY for a fair comparison.  Use 0
    # instead of NaN so the comparison metrics remain finite.
    if 'time_cuts' in romy_config:
        for key, values in romy_config['time_cuts'].items():
            for value in values:
                sd_adr.replace_values_in_window(
                    tbeg=value[0], tend=value[1],
                    value=0, channel_list=[key]
                )

    # Trim ADR data to same length as ROMY data
    sd_adr.st.trim(UTCDateTime(romy_config['tbeg']), UTCDateTime(romy_config['tend']), nearest_sample=False)
    
    print(f"ADR data loaded successfully")
    print(f"ADR channels: {[tr.id for tr in sd_adr.st]}")

    # Synchronize ROMY and ADR onto a common sample grid (linear interpolation).
    all_traces = list(sd_romy.st) + list(sd_adr.st)
    sync_starttime = max(tr.stats.starttime for tr in all_traces)
    sync_endtime = min(tr.stats.endtime for tr in all_traces)
    sync_rate = sd_romy.st[0].stats.sampling_rate

    for st in (sd_romy.st, sd_adr.st):
        st.interpolate(sampling_rate=sync_rate, method="linear", starttime=sync_starttime)
        st.trim(starttime=sync_starttime, endtime=sync_endtime, nearest_sample=False)

    sync_npts = sd_romy.st[0].stats.npts

    print(f"ROMY (master): {sync_starttime} - {sync_endtime} @ {sync_rate} Hz, npts={sync_npts}")
    print("After sync:")
    for tr in list(sd_romy.st) + list(sd_adr.st):
        print(f"  {tr.id:20s}  start={tr.stats.starttime}  npts={tr.stats.npts}  dt={tr.stats.delta}")

    print(sd_romy.st)
    print(sd_adr.st)

    # Drop the temporary trace list so its references release with the loop scope
    del all_traces

    # Extract traces for each component
    romy_traces = {}
    adr_traces = {}
    
    for comp in components:
        # Get ROMY trace
        romy_tr = sd_romy.st.select(channel=f"*{comp}")
        romy_tr = romy_tr.taper(0.1, type='cosine')
        if len(romy_tr) > 0:
            romy_traces[comp] = romy_tr[0]
        else:
            print(f"Warning: ROMY trace for {comp} not found")
            romy_traces[comp] = None
        
        # Get ADR trace
        adr_tr = sd_adr.st.select(channel=f"*{comp}")
        adr_tr = adr_tr.taper(0.1, type='cosine')
        if len(adr_tr) > 0:
            adr_traces[comp] = adr_tr[0]
        else:
            print(f"Warning: ADR trace for {comp} not found")
            adr_traces[comp] = None

    # The sixdegrees objects hold the full pre-trim buffers, catalogs, etc.
    # We've kept references to the trimmed/tapered Trace objects, so the
    # underlying numpy arrays survive while the heavy wrappers can be freed.
    del sd_romy, sd_adr
    gc.collect()

    # Get origin time for time axis
    origin_time = UTCDateTime(romy_config['tbeg'])
    
    # Get time arrays relative to origin time
    if romy_traces['JZ'] is not None:
        times_romy = (np.array(romy_traces['JZ'].times()) + 
                      (romy_traces['JZ'].stats.starttime - origin_time)) / 3600.0  # Convert to hours
    else:
        times_romy = None
    
    # Compute cross-correlation and residuals for each component
    cc_results = {}
    deviations = {}
    rmse_results = {}
    r_squared_results = {}
    tf_misfit_results = {}
    
    for comp in components:
        romy_tr = romy_traces[comp]
        adr_tr = adr_traces[comp]
        
        if romy_tr is not None and adr_tr is not None:
            # Compute cross-correlation
            cc_max, shift_max, cc_zero = compute_cc(romy_tr, adr_tr)
            cc_results[comp] = {
                'cc_max': cc_max,
                'shift_max': shift_max,
                'cc_zero': cc_zero
            }
            
            # Compute absolute deviation and envelope
            dev = compute_deviation(romy_tr, adr_tr)
            deviations[comp] = dev
            
            # Compute RMSE
            rmse = compute_rmse(romy_tr, adr_tr)
            rmse_results[comp] = rmse
            
            # Compute R-squared
            r_squared = compute_r_squared(romy_tr, adr_tr)
            r_squared_results[comp] = r_squared
            
            r_squared_str = f"{r_squared:.4f}" if r_squared is not None else "None"
            peak_dev = float(np.max(dev)) if dev is not None else None
            peak_dev_str = f"{peak_dev:.2f}%" if peak_dev is not None else "None"
            print(f"{comp}: CC_max={cc_max:.2f}, CC_zero={cc_zero:.2f}, RMSE={rmse:.6e}, R²={r_squared_str}, peak_dev={peak_dev_str}")

            # Compute time-frequency misfits (TF misfit package)
            # Reference is ROMY (st2), comparison is ADR (st1).
            romy_arr = None
            adr_arr = None
            try:
                romy_arr, adr_arr, dt = _tf_misfit_inputs(romy_tr, adr_tr)
                if romy_arr is None:
                    raise ValueError("trace too short or missing for TF misfit")

                # EM / PM are misfits; EG / PG are goodness-of-fit values from
                # the same TF-misfit framework.  Suppress the residual
                # divide-by-zero warnings that obspy can still emit inside
                # angle(w_1/w_2) when an entire TF cell is near zero.
                with warnings.catch_warnings(), np.errstate(divide="ignore",
                                                            invalid="ignore"):
                    warnings.simplefilter("ignore", RuntimeWarning)
                    em_val = em(adr_arr, romy_arr, dt=dt, fmin=fmin, fmax=fmax)
                    pm_val = pm(adr_arr, romy_arr, dt=dt, fmin=fmin, fmax=fmax)
                    eg_val = eg(adr_arr, romy_arr, dt=dt, fmin=fmin, fmax=fmax)
                    pg_val = pg(adr_arr, romy_arr, dt=dt, fmin=fmin, fmax=fmax)

                tf_misfit_results[comp] = {
                    "em": round(em_val, 2) if np.isfinite(em_val) else None,
                    "pm": round(pm_val, 2) if np.isfinite(pm_val) else None,
                    "eg": round(eg_val, 2) if np.isfinite(eg_val) else None,
                    "pg": round(pg_val, 2) if np.isfinite(pg_val) else None,
                }

                print(f"{comp}: EM={em_val:.4f}, PM={pm_val:.4f}, EG={eg_val:.4f}, PG={pg_val:.4f}")
            except Exception as e:
                print(f"{comp}: Warning - TF misfit computation failed: {e}")
                tf_misfit_results[comp] = {"em": None, "pm": None, "eg": None, "pg": None}
            finally:
                # Release the float copies before the next component overwrites them
                del romy_arr, adr_arr
        else:
            cc_results[comp] = None
            deviations[comp] = None
            rmse_results[comp] = None
            r_squared_results[comp] = None
            tf_misfit_results[comp] = {"em": None, "pm": None, "eg": None, "pg": None}
    
    # Create comparison plot
    # Create figure with subplots: 3 rows (one for each component), full width
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    plt.subplots_adjust(hspace=0.01)
    
    yscale = 1e9
    yunit = 'nrad/s'
    
    fs = 14
    
    # Colors for ROMY and ADR
    color_romy = 'black'
    color_adr = 'tab:orange'
    color_dev = 'tab:gray'
    
    # Plot each component
    for i, comp in enumerate(components):
        romy_tr = romy_traces[comp]
        adr_tr = adr_traces[comp]
        
        if romy_tr is None or adr_tr is None:
            continue
        
        # Get time arrays
        times_romy_comp = romy_tr.times()
        times_adr_comp = adr_tr.times()

        # Ensure same length for plotting (all slices are views)
        min_len = min(len(times_romy_comp), len(times_adr_comp))
        times_plot = times_romy_comp[:min_len]
        romy_data = romy_tr.data[:min_len]
        adr_data = adr_tr.data[:min_len]

        # Main axis for waveforms
        ax = axes[i]

        # Plot ROMY waveform on left y-axis
        line1 = ax.plot(times_plot, romy_data * yscale,
                        color=color_romy, linewidth=1.5, alpha=0.8, label=f'ROMY-{comp[1]}')

        # Plot ADR waveform on left y-axis
        line2 = ax.plot(times_plot, adr_data * yscale,
                        color=color_adr, linewidth=1.5, alpha=0.8, label=f'ADR-{comp[1]}')

        # Add cross-correlation info (and peak envelope deviation)
        if cc_results[comp] is not None:
            cc_max = cc_results[comp]['cc_max']
            cc_zero = cc_results[comp]['cc_zero']
            shift_max = cc_results[comp]['shift_max']
            maxstr = r"$_{max}$"
            info_text = f'CC$_{0}$: {cc_zero:.2f} | CC{maxstr}: {cc_max:.2f} with {shift_max:.2f} s (sps={sync_rate} Hz)'
            if deviations[comp] is not None:
                peak_dev = float(np.max(deviations[comp]))
                info_text += f' | peak dev: {peak_dev:.1f}%'
            ax.text(0.04, 0.96, info_text,
                    transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.5))

        # Set left y-axis properties
        ax.set_ylabel(f'Rotation Rate ({yunit})', fontsize=fs, color='black')
        ax.tick_params(axis='y', labelcolor='black')
        ax.set_xlim(left=0)

        # set y-axis limits from min/max of both traces using numpy reductions
        # (max(abs(arr)) creates a temporary abs array; do one at a time so the
        # temporary is reclaimed before the next.)
        ymax = max(float(np.abs(romy_data).max()), float(np.abs(adr_data).max()))
        ax.set_ylim(bottom=-ymax * yscale, top=ymax * yscale)

        # Create second y-axis for absolute deviation (on the right)
        ax2 = ax.twinx()

        dev_arr = deviations[comp]
        if dev_arr is not None:
            # Plot absolute deviation (already length min_len)
            line3 = ax2.plot(times_plot, dev_arr,
                            color=color_dev, linewidth=1.5, alpha=0.7,
                            label=f"Deviation"
                            )
            ax2.fill_between(times_plot, dev_arr, color=color_dev, alpha=0.3)
            ax2.set_ylim(bottom=0)
            ax2.set_ylim(top=200)
            ax2.set_yticks(np.linspace(0, 100, 5))
            ax2.set_yticklabels([f'{int(y)}' for y in np.linspace(0, 100, 5)])

        # Set right y-axis properties
        ax2.set_ylabel('Deviation (%)', fontsize=fs, color=color_dev, ha='right')
        ax2.tick_params(axis='y', labelcolor=color_dev)

        # Combine legends from both axes
        lines = line1 + line2
        if dev_arr is not None:
            lines += line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=fs-4, framealpha=0.9)

        # Drop our local references to the per-component data views; matplotlib
        # keeps its own internal copies of the scaled line data until close().
        # (times_romy_comp is intentionally left alive — it is reused after the
        # loop to set the shared x-limit.)
        del romy_data, adr_data, dev_arr, times_adr_comp
    
    for _ax, label in zip(axes, ["(a)", "(b)", "(c)"]):
        _ax.set_xlim(0, max(times_romy_comp))
        _ax.text(0.005, 0.92, label, fontsize=fs-1, transform=_ax.transAxes)
    
    axes[2].set_xlabel(f'Time (s) from {origin_time.strftime("%Y-%m-%d %H:%M:%S")} UTC', fontsize=fs)
    
    # Add overall title
    fig.suptitle(f'ROMY vs ADR Comparison - Event {UTCDateTime(event_date).strftime("%Y-%m-%d")}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Create output directory if it doesn't exist
    output_dir = f'./events/event_{event_date}/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save TF misfit plots (EM/PM) for each component
    # Note: plot_tf_misfits uses ADR (st1) vs ROMY (st2) with ROMY as reference.
    # The TF representations are nf x ntime float arrays, so we close the
    # figure and force a GC pass before moving to the next component.
    for comp in components:
        romy_tr = romy_traces[comp]
        adr_tr = adr_traces[comp]

        romy_arr, adr_arr, dt = _tf_misfit_inputs(romy_tr, adr_tr)
        if romy_arr is None:
            continue

        fig_tf = None
        try:
            with warnings.catch_warnings(), np.errstate(divide="ignore",
                                                        invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                fig_tf = plot_tf_misfits(
                    adr_arr,
                    romy_arr,
                    dt=dt,
                    fmin=fmin,
                    fmax=fmax,
                    nf=100,
                    w0=6,
                    norm="global",
                    st2_isref=True,
                    show=False,
                )
            out_tf = os.path.join(output_dir, f"{event_date}_tf_misfits_{comp}.png")
            fig_tf.savefig(out_tf, dpi=300, bbox_inches="tight")
            print(f"Saved: {out_tf}")
        except Exception as e:
            print(f"{comp}: Warning - plot_tf_misfits failed: {e}")
        finally:
            if fig_tf is not None:
                plt.close(fig_tf)
            del romy_arr, adr_arr, fig_tf
            gc.collect()

    # Save figure
    output_file = os.path.join(output_dir, f'{event_date}_romy_adr_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close(fig)  # Close this specific figure to free its data

    # Explicitly release large per-event objects.  sd_romy / sd_adr were
    # released earlier, right after the per-component traces were extracted.
    del romy_traces, adr_traces, fig, axes, romy_config
    gc.collect()

    return cc_results, deviations, rmse_results, r_squared_results, tf_misfit_results, adr_config, event_info


def save_results_to_csv(results_data, results_file):
    """
    Save results to CSV file, appending or updating existing entries.
    
    Parameters:
    -----------
    results_data : list
        List of dictionaries containing results for each component
    results_file : str
        Path to output CSV file
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.dirname(results_file)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    
    # Create dataframe
    df_new = pd.DataFrame(results_data)
    
    # Load existing results if they exist
    if os.path.exists(results_file):
        df_existing = pd.read_csv(results_file)
        # Ensure event column is string type for consistent comparison
        df_existing['event'] = df_existing['event'].astype(str)
        df_new['event'] = df_new['event'].astype(str)
        
        # Remove rows that match both event and component (overwrite existing)
        # Create a MultiIndex for efficient filtering
        df_existing_indexed = df_existing.set_index(['event', 'component'])
        df_new_indexed = df_new.set_index(['event', 'component'])
        
        # Drop rows from existing that match new data
        df_existing_indexed = df_existing_indexed.drop(df_new_indexed.index, errors='ignore')
        
        # Reset index and append new results
        df_existing = df_existing_indexed.reset_index()
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new
    
    # Save to CSV
    df_final.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")
    print(f"Total events in results: {df_final['event'].nunique()}")


def main():
    """Main function to process all events from events_list.yml"""
    
    #--------------------------------
    # Configuration
    fmin = 0.02
    fmax = 0.1

    # Results file
    results_file = os.path.join('./output', 'results_romy_adr_comparison.csv')

    # Events list file
    events_list_file = 'events_list.yml'

    #--------------------------------

    # Load events list
    if not os.path.exists(events_list_file):
        raise FileNotFoundError(f"Events list file not found: {events_list_file}")
    
    with open(events_list_file, 'r') as f:
        events_config = yaml.safe_load(f)
    
    events = events_config.get('events', [])
    
    if not events:
        print("No events found in events_list.yml")
        return
    
    print(f"Found {len(events)} events to process: {events}")
    
    # Process each event
    components = ['JZ', 'JN', 'JE']
    processed_events = 0
    
    for event_date in events:
        try:
            # Process event
            cc_results, deviations, rmse_results, r_squared_results, tf_misfit_results, adr_config, event_info = process_event(event_date, fmin=fmin, fmax=fmax)
            
            # Prepare data for dataframe
            event_results = []
            for comp in components:
                if cc_results[comp] is not None and deviations[comp] is not None:
                    maxdev = np.max(deviations[comp])
                    rmse = rmse_results[comp] if rmse_results[comp] is not None else None
                    r_squared = r_squared_results[comp] if r_squared_results[comp] is not None else None
                    tfm = tf_misfit_results.get(comp, {})
                    em_val = tfm.get("em")
                    pm_val = tfm.get("pm")
                    eg_val = tfm.get("eg")
                    pg_val = tfm.get("pg")
                    event_results.append({
                        'event': event_date,
                        'component': comp,
                        'ccmax': round(cc_results[comp]['cc_max'], 2),
                        'cczero': round(cc_results[comp]['cc_zero'], 2),
                        'ccmaxshift': round(cc_results[comp]['shift_max'], 2),
                        'maxdev': round(maxdev, 2),
                        'rmse': round(rmse*1e9, 2) if rmse is not None else None, # scale to nrad/s
                        'r_squared': round(r_squared, 4) if r_squared is not None else None,
                        'em': round(em_val, 2) if em_val is not None else None,
                        'pm': round(pm_val, 2) if pm_val is not None else None,
                        'eg': round(eg_val, 2) if eg_val is not None else None,
                        'pg': round(pg_val, 2) if pg_val is not None else None,
                        'n_stations': len(adr_config['stations']),
                        'fmin': fmin, 
                        'fmax': fmax,
                        'magnitude': event_info.get('magnitude'),
                        'distance_km': event_info.get('distance_km'),
                        'backazimuth': event_info.get('backazimuth'),
                    })
            
            # Save results after each event (in case script is interrupted)
            if event_results:
                save_results_to_csv(event_results, results_file)
                print(f"\nResults for event {event_date}:")
                df_event = pd.DataFrame(event_results)
                print(df_event.to_string(index=False))
                processed_events += 1

            # Release event-level objects and force garbage collection
            del cc_results, deviations, rmse_results, r_squared_results, tf_misfit_results, adr_config, event_info, event_results
            gc.collect()
            
        except Exception as e:
            print(f"\nERROR processing event {event_date}: {str(e)}")
            import traceback
            traceback.print_exc()
            gc.collect()
            continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Processed {processed_events}/{len(events)} events")
    print(f"Results saved to: {results_file}")


if __name__ == '__main__':
    main()
