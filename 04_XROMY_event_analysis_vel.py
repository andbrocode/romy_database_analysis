#!/usr/bin/env python3
"""
Automated ROMY Event Analysis Script - Dispersion Curves

This script processes a list of events from the events/ directory and:
1. Computes dispersion curves for Love and Rayleigh waves
2. Computes for both 'mid' and 'max' CC methods
3. Stores dispersion curve results in CSV format
4. Saves polar backazimuth-vs-frequency plots per event and CC method
"""

import os
import gc
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from obspy import UTCDateTime
from sixdegrees.sixdegrees import sixdegrees

S_MIN, S_MAX = 25, 220
LEGEND_MS_MIN, LEGEND_MS_MAX = 6, 16


def _band_baz_and_count(band):
    """
    Per-band backazimuth and count for polar plot.

    - baz: median backazimuth of windows with valid velocity (post CC threshold)
    - baz_std: std of those backazimuths (angular error bar)
    - n_count: n_valid_regressions
    """
    fc = band.get("f_center")
    if fc is None:
        return None

    velocities = np.asarray(band.get("velocities", []), dtype=float)
    backazimuths = np.asarray(band.get("backazimuths", []), dtype=float)
    valid = np.isfinite(velocities)

    n_count = band.get("n_valid_regressions", int(np.sum(valid)))
    if not np.isfinite(n_count):
        n_count = int(np.sum(valid))

    baz_std = np.nan
    if len(backazimuths) == len(velocities) and np.any(valid):
        baz_valid = backazimuths[valid]
        baz = float(np.nanmedian(baz_valid))
        if baz_valid.size > 1:
            baz_std = float(np.nanstd(baz_valid, ddof=0))
        else:
            baz_std = 0.0
    else:
        baz = band.get("baz_used")
        baz = float(baz) if baz is not None and np.isfinite(baz) else np.nan

    if not np.isfinite(baz) or n_count <= 0:
        return None
    return float(fc), baz, float(n_count), float(baz_std) if np.isfinite(baz_std) else 0.0


def _baz_freq_from_dispersion(dispersion_results):
    """Collect filtered baz, std, f_center, n_valid_regressions per band."""
    baz_deg, baz_std_deg, freq_hz, n_counts = [], [], [], []
    for band in dispersion_results.get("frequency_bands", []):
        row = _band_baz_and_count(band)
        if row is None:
            continue
        fc, baz, n_count, baz_std = row
        baz_deg.append(baz)
        baz_std_deg.append(baz_std)
        freq_hz.append(fc)
        n_counts.append(n_count)
    return (
        np.asarray(baz_deg),
        np.asarray(baz_std_deg),
        np.asarray(freq_hz),
        np.asarray(n_counts),
    )


def _scatter_marker_sizes(n_counts, s_min=S_MIN, s_max=S_MAX):
    """Scatter marker area (pt^2), normalized within each panel."""
    sizes = np.full(n_counts.shape, s_min, dtype=float)
    valid = np.isfinite(n_counts) & (n_counts > 0)
    if not valid.any():
        return sizes
    ns = n_counts[valid]
    ns_norm = (ns - ns.min()) / (ns.max() - ns.min() + 1e-9)
    sizes[valid] = s_min + ns_norm * (s_max - s_min)
    return sizes


def _legend_markersize(n, n_lo, n_hi, ms_min=LEGEND_MS_MIN, ms_max=LEGEND_MS_MAX):
    """Legend marker size (pt) scaled to regression count."""
    if np.isclose(n_lo, n_hi):
        return 0.5 * (ms_min + ms_max)
    frac = (n - n_lo) / (n_hi - n_lo)
    return ms_min + frac * (ms_max - ms_min)


def plot_dispersion_baz_freq_polar(
    event_info,
    output_file,
    love_results_disp=None,
    rayleigh_results_disp=None,
    fmax=None,
):
    """
    Polar plot: theta = filtered baz, r = f_center; marker size ~ n_valid_regressions.
    """
    wave_panels = []
    if love_results_disp is not None:
        wave_panels.append((love_results_disp, "Love", "tab:blue"))
    if rayleigh_results_disp is not None:
        wave_panels.append((rayleigh_results_disp, "Rayleigh", "tab:red"))
    if not wave_panels:
        print(f"  Warning: no dispersion results for polar plot; skipping {output_file}")
        return None

    panel_data = [
        (title, color, *_baz_freq_from_dispersion(results))
        for results, title, color in wave_panels
    ]

    all_freq = [freq_hz for _, _, _, freq_hz, _, _ in panel_data if freq_hz.size]
    if not all_freq:
        print(f"  Warning: no polar-plot data; skipping {output_file}")
        return None

    r_max = float(np.max(np.concatenate(all_freq))) * 1.05
    r_lim = float(fmax) * 1.15 if fmax is not None and np.isfinite(fmax) else r_max
    ncols = len(panel_data)

    fig = plt.figure(figsize=(10, 5))
    for i, (title, color, baz_deg, baz_std_deg, freq_hz, n_counts) in enumerate(
        panel_data, start=1
    ):
        ax = fig.add_subplot(1, ncols, i, projection="polar")

        if baz_deg.size:
            theta = np.deg2rad(baz_deg)
            theta_err = np.deg2rad(np.where(np.isfinite(baz_std_deg), baz_std_deg, 0.0))

            ax.errorbar(
                theta,
                freq_hz,
                xerr=theta_err,
                fmt="none",
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
                capthick=1.0,
                alpha=0.5,
                zorder=2,
            )
            ax.scatter(
                theta,
                freq_hz,
                s=_scatter_marker_sizes(n_counts),
                c=color,
                alpha=0.75,
                edgecolors="k",
                linewidths=0.4,
                zorder=3,
            )

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, r_lim)
        ax.grid(True, alpha=0.35)
        ax.text(0.08, 1.05, title + " waves", transform=ax.transAxes, fontsize=14, va="top")
        ax.set_rlabel_position(25.5)
        ax.text(
            0.82,
            0.98,
            "Frequency\n(Hz)",
            transform=ax.transAxes,
            fontsize=12,
            ha="center",
            va="center",
        )

        theo_baz = event_info.get("backazimuth") if event_info else None
        if theo_baz is not None and np.isfinite(theo_baz):
            ax.axvline(
                np.deg2rad(theo_baz),
                color="green",
                ls="--",
                lw=1.5,
                alpha=0.85,
                label="Theoretical BAZ",
            )

    all_n_counts = np.concatenate([
        n_counts[np.isfinite(n_counts) & (n_counts > 0)]
        for _, _, _, _, _, n_counts in panel_data
        if n_counts.size
    ]) if panel_data else np.array([])

    if all_n_counts.size:
        n_lo, n_hi = float(all_n_counts.min()), float(all_n_counts.max())
        legend_values = (
            [n_lo, n_hi] if np.isclose(n_lo, n_hi) else [n_lo, 0.5 * (n_lo + n_hi), n_hi]
        )

        size_handles = []
        size_labels = []
        for n in legend_values:
            size_handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="None",
                    marker="o",
                    markersize=_legend_markersize(n, n_lo, n_hi),
                    markerfacecolor="0.55",
                    markeredgecolor="k",
                    markeredgewidth=0.4,
                )
            )
            size_labels.append(f"{int(round(n))}")

        fig.legend(
            size_handles,
            size_labels,
            title="No. Samples",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.1),
            ncol=1,
            frameon=True,
            fontsize=10,
            title_fontsize=10,
            handletextpad=0.8,
        )

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved polar plot: {output_file}")
    return output_file


def save_dispersion_baz_freq_polar_plots(result, config):
    """Save polar baz-vs-frequency plots for each computed CC method."""
    if result is None:
        return

    date = result["date"]
    ev = result.get("event_info", {})
    dispersion_results = result.get("dispersion_results", {})
    fig_dir = config.get("path_to_figures", f"./events/event_{date}/figures/")
    if not fig_dir.endswith("/"):
        fig_dir += "/"

    for cc_method in ("mid", "max"):
        love_key = f"love_{cc_method}"
        ray_key = f"rayleigh_{cc_method}"
        if love_key not in dispersion_results or ray_key not in dispersion_results:
            continue

        output_file = os.path.join(
            fig_dir, f"{date}_dispersion_baz_freq_polar_{cc_method}.png"
        )
        try:
            plot_dispersion_baz_freq_polar(
                ev,
                output_file,
                love_results_disp=dispersion_results[love_key]["results"],
                rayleigh_results_disp=dispersion_results[ray_key]["results"],
                fmax=config.get("fmax"),
            )
        except Exception as e:
            print(f"  Warning: could not save polar plot ({cc_method}): {e}")


def process_event(date, events_list):
    """
    Process a single event and compute dispersion curves.
    
    Parameters:
    -----------
    date : str
        Event date string (e.g., '20240314')
    events_list : list
        List of event dates to process
        
    Returns:
    --------
    dict : Dictionary containing dispersion curve results
    """
    print(f"\n{'='*60}")
    print(f"Processing event: {date}")
    print(f"{'='*60}")
    
    try:
        # Load configuration
        config_path = f"./events/event_{date}/config_romy_{date}.yml"
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found: {config_path}")
            return None
            
        config = sixdegrees.load_from_yaml(config_path)
        
        # Setup sixdegrees object
        sd = sixdegrees(conf=config)
        
        # Get event information
        ev = sd.get_event_info(
            origin_time=config['origin_time'],
            base_catalog="USGS",
            magnitude=config['magnitude']
        )
        
        print(f"Event: {ev.get('origin_time', 'N/A')}")
        print(f"Theoretical backazimuth: {ev.get('backazimuth', 'N/A'):.2f}°")
        
        # Load data
        sd.load_data(config['tbeg'], config['tend'], resample_rate=5)
        
        # Ensure all traces have the same amount of samples
        sd.trim_stream(set_common=True, set_interpolate=True)
        
        # Filter data
        sd.filter_data(fmin=config['fmin'], fmax=config['fmax'], output=False)
        
        # Apply polarity correction if needed
        if 'pol_dict' in config:
            sd.polarity_stream(pol_dict=config['pol_dict'], raw=True)
        else:
            # Default polarity correction from notebook
            sd.polarity_stream(pol_dict={"HZ": -1, "JZ": 1}, raw=True)
        
        # Cut bad data if specified
        if 'time_cuts' in config:
            for key, values in config['time_cuts'].items():
                for value in values:
                    sd.replace_values_in_window(
                        tbeg=value[0], tend=value[1], 
                        value=0, channel_list=[key]
                    )
        
        # Dispersion curve parameters (from notebook)
        params = {
            'fmin': 0.01,
            'fmax': 0.8,
            'octave_fraction': 3,
            'window_factor': 6.0,
            'cc_threshold': 0.8,
            'baz_step': 1,
            'time_window_overlap': 0.75,
            'velocity_method': 'odr',
            'velocity_threshold': 5000.0,  # exclude v > 5000 m/s from KDE only
            'zero_intercept': False, #True,
            'verbose': False,
            'n_jobs': 4,
        }
        
        # Compute dispersion curves
        print("\nComputing dispersion curves...")
        dispersion_results = {}
        
        # Process both 'mid' and 'max' methods
        for cc_method in ['mid', 'max']:
            print(f"  Computing with cc_method='{cc_method}'...")
            
            # Process Love and Rayleigh waves
            for wave_type in ['love', 'rayleigh']:
                try:
                    print(f"    Computing {wave_type} waves...")
                    
                    # Compute dispersion curve
                    results = sd.compute_dispersion_curve(
                        wave_type=wave_type,
                        fmin=params['fmin'],
                        fmax=params['fmax'],
                        octave_fraction=params['octave_fraction'],
                        window_factor=params['window_factor'],
                        cc_threshold=params['cc_threshold'],
                        cc_method=cc_method,
                        time_window_overlap=params['time_window_overlap'],
                        velocity_threshold=params['velocity_threshold'],
                        velocity_method=params['velocity_method'],
                        zero_intercept=params['zero_intercept'],
                        verbose=params['verbose'],
                        n_jobs=params['n_jobs']
                    )
                    
                    # Extract dispersion curve data
                    if results is not None:
                        key = f"{wave_type}_{cc_method}"
                        dispersion_results[key] = {
                            'wave_type': wave_type,
                            'cc_method': cc_method,
                            'results': results
                        }
                        
                        print(f"      Successfully computed {wave_type} dispersion curve")
                    
                except Exception as e:
                    print(f"      Warning: Could not compute dispersion curve for {wave_type} with {cc_method}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                gc.collect()
        gc.collect()

        result = {
            'date': date,
            'event_info': ev,
            'dispersion_results': dispersion_results,
        }
        save_dispersion_baz_freq_polar_plots(result, config)
        return result
    
    except Exception as e:
        print(f"Error processing event {date}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _band_get(band, key, default=None):
    if isinstance(band, dict):
        return band.get(key, default)
    return getattr(band, key, default)


def _valid_band_velocities(band):
    velocities = _band_get(band, 'velocities', [])
    if isinstance(velocities, np.ndarray):
        return velocities[np.isfinite(velocities)]
    return np.asarray(
        [v for v in velocities if v is not None and np.isfinite(v)],
        dtype=float,
    )


def _band_median_velocity(band):
    valid = _valid_band_velocities(band)
    return float(np.nanmedian(valid)) if valid.size else np.nan


def _band_median_velocity_std(band):
    valid = _valid_band_velocities(band)
    return float(np.nanstd(valid)) if valid.size else np.nan


def _extract_from_frequency_bands(frequency_bands):
    """Build per-band dispersion series from sixdegrees frequency_bands."""
    cc_values = []
    n_samples_values = []

    for band in frequency_bands:
        band_ccoefs = _band_get(band, 'ccoefs', [])
        if isinstance(band_ccoefs, np.ndarray):
            band_ccoefs = band_ccoefs.tolist()

        if band_ccoefs:
            valid_cc = [
                abs(v) for v in band_ccoefs if v is not None and np.isfinite(v)
            ]
            cc_values.append(float(np.nanmedian(valid_cc)) if valid_cc else np.nan)
        else:
            cc_values.append(np.nan)

        kde_stats = _band_get(band, 'kde_stats', {}) or {}
        n_samples = kde_stats.get(
            'n_samples', _band_get(band, 'n_valid_regressions', np.nan)
        )
        n_samples_values.append(n_samples)

    return {
        'frequency': [_band_get(band, 'f_center', np.nan) for band in frequency_bands],
        'velocity': [
            _band_get(band, 'kde_peak_velocity', np.nan) for band in frequency_bands
        ],
        'velocity_median': [_band_median_velocity(band) for band in frequency_bands],
        'velocity_uncertainty': [
            _band_get(band, 'kde_deviation', np.nan) for band in frequency_bands
        ],
        'velocity_median_uncertainty': [
            _band_median_velocity_std(band) for band in frequency_bands
        ],
        'cc': cc_values,
        'n_samples': n_samples_values,
    }


def extract_dispersion_data(dispersion_result):
    """
    Extract dispersion curve data from results object.
    
    Parameters:
    -----------
    dispersion_result : dict or object
        Dispersion curve results from compute_dispersion_curve
        
    Returns:
    --------
    dict : Extracted data with frequency, velocity, and uncertainty
    """
    extracted_data = {
        'frequency': [],
        'velocity': [],
        'velocity_median': [],
        'velocity_uncertainty': [],
        'velocity_median_uncertainty': [],
        'cc': [],
        'n_samples': []
    }

    try:
        frequency_bands = None
        if isinstance(dispersion_result, dict):
            frequency_bands = dispersion_result.get('frequency_bands')
        elif hasattr(dispersion_result, 'frequency_bands'):
            frequency_bands = getattr(dispersion_result, 'frequency_bands', None)

        if frequency_bands:
            extracted_data = _extract_from_frequency_bands(frequency_bands)
        elif isinstance(dispersion_result, dict):
            if 'frequencies' in dispersion_result:
                extracted_data['frequency'] = dispersion_result['frequencies']
            elif 'frequency' in dispersion_result:
                extracted_data['frequency'] = dispersion_result['frequency']

            if 'velocities' in dispersion_result:
                extracted_data['velocity'] = dispersion_result['velocities']
            elif 'velocity' in dispersion_result:
                extracted_data['velocity'] = dispersion_result['velocity']

            if 'velocity_uncertainty' in dispersion_result:
                extracted_data['velocity_uncertainty'] = dispersion_result['velocity_uncertainty']
            elif 'uncertainties' in dispersion_result:
                extracted_data['velocity_uncertainty'] = dispersion_result['uncertainties']
            elif 'velocity_std' in dispersion_result:
                extracted_data['velocity_uncertainty'] = dispersion_result['velocity_std']
            elif 'vel_std' in dispersion_result:
                extracted_data['velocity_uncertainty'] = dispersion_result['vel_std']

            if 'cc' in dispersion_result:
                extracted_data['cc'] = dispersion_result['cc']
            elif 'correlation' in dispersion_result:
                extracted_data['cc'] = dispersion_result['correlation']

            if 'n_samples' in dispersion_result:
                extracted_data['n_samples'] = dispersion_result['n_samples']
        elif hasattr(dispersion_result, '__dict__'):
            extracted_data['frequency'] = getattr(
                dispersion_result, 'frequencies',
                getattr(dispersion_result, 'frequency',
                       getattr(dispersion_result, 'freq', [])),
            )
            extracted_data['velocity'] = getattr(
                dispersion_result, 'velocities',
                getattr(dispersion_result, 'velocity',
                       getattr(dispersion_result, 'vel', [])),
            )
            extracted_data['velocity_uncertainty'] = getattr(
                dispersion_result, 'velocity_uncertainty',
                getattr(dispersion_result, 'uncertainties',
                       getattr(dispersion_result, 'velocity_std',
                              getattr(dispersion_result, 'vel_std', []))),
            )
            extracted_data['cc'] = getattr(
                dispersion_result, 'cc',
                getattr(dispersion_result, 'correlation', []),
            )
            extracted_data['n_samples'] = getattr(dispersion_result, 'n_samples', [])

        for key in extracted_data:
            if isinstance(extracted_data[key], np.ndarray):
                extracted_data[key] = extracted_data[key].tolist()
            elif extracted_data[key] is None:
                extracted_data[key] = []

    except Exception as e:
        print(f"Warning: Could not extract dispersion data: {e}")
        import traceback
        traceback.print_exc()

    return extracted_data


def save_dispersion_results(all_results, output_dir='./output'):
    """
    Save dispersion curve results to CSV.
    
    Parameters:
    -----------
    all_results : list
        List of result dictionaries from process_event
    output_dir : str
        Output directory for CSV files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare data for CSV
    rows = []
    for result in all_results:
        if result is None:
            continue
        
        date = result['date']
        ev = result['event_info']
        dispersion_results = result.get('dispersion_results', {})
        
        for key, disp_data in dispersion_results.items():
            wave_type = disp_data.get('wave_type', '')
            cc_method = disp_data.get('cc_method', '')
            results_obj = disp_data.get('results', None)
            
            if results_obj is not None:
                # Extract dispersion curve data
                extracted = extract_dispersion_data(results_obj)
                
                # Create rows for each frequency point
                frequencies = extracted.get('frequency', [])
                velocities = extracted.get('velocity', [])
                velocity_medians = extracted.get('velocity_median', [])
                velocity_uncertainties = extracted.get('velocity_uncertainty', [])
                velocity_median_uncertainties = extracted.get('velocity_median_uncertainty', [])
                ccs = extracted.get('cc', [])
                n_samples_list = extracted.get('n_samples', [])
                
                # Ensure all lists have the same length
                max_len = max(
                    len(frequencies), len(velocities), len(velocity_medians),
                    len(velocity_uncertainties), len(velocity_median_uncertainties),
                    len(ccs), len(n_samples_list),
                )
                
                for i in range(max_len):
                    freq_value = frequencies[i] if i < len(frequencies) else np.nan
                    if not pd.isna(freq_value):
                        freq_value = round(float(freq_value), 5)
                    cc_value = ccs[i] if i < len(ccs) else np.nan
                    if not pd.isna(cc_value):
                        cc_value = round(float(cc_value), 2)

                    rows.append({
                        'event_date': date,
                        'origin_time': ev.get('origin_time', ''),
                        'magnitude': ev.get('magnitude', ''),
                        'distance_km': ev.get('distance_km', ''),
                        'wave_type': wave_type,
                        'cc_method': cc_method,
                        'frequency': freq_value,
                        'velocity': velocities[i] if i < len(velocities) else np.nan,
                        'velocity_median': velocity_medians[i] if i < len(velocity_medians) else np.nan,
                        'velocity_uncertainty': velocity_uncertainties[i] if i < len(velocity_uncertainties) else np.nan,
                        'velocity_median_uncertainty': (
                            velocity_median_uncertainties[i]
                            if i < len(velocity_median_uncertainties) else np.nan
                        ),
                        'cc': cc_value,
                        'n_samples': n_samples_list[i] if i < len(n_samples_list) else np.nan,
                    })
    
    if rows:
        df = pd.DataFrame(rows)
        output_file = os.path.join(output_dir, 'results_dispersion.csv')
        df.to_csv(output_file, index=False)
        print(f"\nDispersion curve results saved to: {output_file}")
        print(f"Total data points: {len(df)}")
        return df
    else:
        print("\nNo dispersion curve results to save.")
        return None


def load_events_list(yaml_file=None):
    """
    Load event list from YAML file or return default list.
    
    Parameters:
    -----------
    yaml_file : str, optional
        Path to YAML file containing event list. If None, returns default list.
        
    Returns:
    --------
    list : List of event date strings
    """
    if yaml_file and os.path.exists(yaml_file):
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            events_list = config.get('events', [])
            print(f"Loaded {len(events_list)} events from {yaml_file}")
            return events_list
    else:
        # Default list of events to process
        events_list = [
            '20230908',
            # '20230912',
            # '20231202',
            # '20240314',
            # '20240322',
            # '20240323',
            # '20240414',
            # '20240422',
            # '20251206',
            # '20251212',
            # '20251227',
            # '20260107',
            # '20240815',
            # '20241023',
        ]
        # '20231203',  # Note: may have no data
        return events_list


def main():
    """
    Main function to process all events.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Process ROMY events for dispersion curve analysis'
    )
    parser.add_argument(
        '--events-file',
        type=str,
        default=None,
        help='Path to YAML file containing list of events to process (default: use hardcoded list)'
    )
    args = parser.parse_args()
    
    # Load events list
    events_list = load_events_list(args.events_file)
    
    print("="*60)
    print("ROMY Event Analysis - Dispersion Curves")
    print("="*60)
    print(f"Processing {len(events_list)} events...")
    
    # Process all events
    all_results = []
    for date in events_list:
        result = process_event(date, events_list)
        if result is not None:
            all_results.append(result)
    
    # Save results
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)
    
    save_dispersion_results(all_results)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
