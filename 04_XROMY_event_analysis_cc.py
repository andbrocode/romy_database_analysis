#!/usr/bin/env python3
"""
Automated ROMY Event Analysis Script — Correlation & Tilt Correction

This script processes events from the events/ directory and:
1. Applies tilt correction to horizontal translation components
2. Computes full-waveform cross-correlation (Love/Rayleigh) using theoretical backazimuth
3. Stores CC and tilt-correction RMS metrics in CSV format

Backazimuth estimation and per-window CC are omitted for faster processing.
Saves tilt-correction before/after plots per event (see 05_XROMY_plot_tilt_correction.ipynb).
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from obspy import UTCDateTime
from obspy.signal.cross_correlation import correlate, xcorr_max
from sixdegrees.sixdegrees import sixdegrees

PLOT_SCALE = 1e6
PLOT_COMPONENTS = (("HN", r"$a_N$"), ("HE", r"$a_E$"))


def _iter_time_cut_windows(config):
    """
    Yield (tbeg, tend) for every window in config['time_cuts'].

    Supports one or many windows per channel (e.g. JE with two tuples).
    """
    for entries in config.get('time_cuts', {}).values():
        if not entries:
            continue
        # Single window stored as [tbeg, tend] without an extra list level.
        if len(entries) == 2 and isinstance(entries[0], str):
            entries = [entries]
        for entry in entries:
            yield UTCDateTime(entry[0]), UTCDateTime(entry[1])


def _merged_cut_windows(config):
    """Union of all time_cuts windows across channels, sorted and merged."""
    windows = list(_iter_time_cut_windows(config))
    if not windows:
        return []

    # Drop exact duplicates (same window listed on JE, JN, …).
    unique = sorted({(float(a), float(b)): (a, b) for a, b in windows}.values(),
                    key=lambda w: w[0])
    merged = [unique[0]]
    for tbeg, tend in unique[1:]:
        if tbeg <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], tend))
        else:
            merged.append((tbeg, tend))
    return merged


def _segments_outside_cuts(stream, cut_windows):
    """
    (start, end) intervals between merged cut windows.

    N gaps produce up to N+1 segments; each is corrected independently.
    """
    if not cut_windows:
        tr = stream[0]
        return [(tr.stats.starttime, tr.stats.endtime)]

    dt = stream[0].stats.delta
    t_start = stream[0].stats.starttime
    t_end = stream[0].stats.endtime
    segments = []
    cursor = t_start
    for cut_beg, cut_end in cut_windows:
        if cut_beg > cursor:
            segments.append((cursor, cut_beg - dt))
        cursor = max(cursor, cut_end + dt)
    if cursor <= t_end:
        segments.append((cursor, t_end))
    return segments


def _cut_sample_mask(trace, cut_windows):
    """Boolean mask: True inside any cut window."""
    mask = np.zeros(len(trace.data), dtype=bool)
    if not cut_windows:
        return mask

    t0, dt = trace.stats.starttime, trace.stats.delta
    for tbeg, tend in cut_windows:
        i0 = int(np.ceil((tbeg - t0) / dt))
        i1 = int(np.floor((tend - t0) / dt))
        mask[max(0, i0):min(len(mask), i1 + 1)] = True
    return mask


def _paste_segment_correction(full_st, seg_before, seg_after):
    """
    Paste tilt-corrected HN/HE for one segment back into the full stream.

    Per-segment integration resets introduce a spurious DC; subtract the
    segment-mean correction so before/after traces stay aligned across gaps.
    """
    dt = full_st[0].stats.delta
    t0 = full_st[0].stats.starttime
    for channel in ('HN', 'HE'):
        full_tr = full_st.select(channel=f'*{channel}')[0]
        tr_b = seg_before.select(channel=f'*{channel}')[0]
        tr_a = seg_after.select(channel=f'*{channel}')[0]
        delta = tr_b.data - tr_a.data
        delta = delta - np.mean(delta)
        corrected = tr_b.data - delta
        i0 = int(round((tr_a.stats.starttime - t0) / dt))
        n = len(corrected)
        i0 = max(0, min(i0, len(full_tr.data) - n))
        full_tr.data[i0:i0 + n] = corrected


def correct_tilt_split_at_cuts(sd, config):
    """
    Split at each time cut, correct_tilt() per good segment, merge back.

    Cut windows are skipped; each continuous piece is integrated separately
  so ∫ω never runs through zero-filled gaps.
    """
    cut_windows = _merged_cut_windows(config)
    if not cut_windows:
        sd.correct_tilt()
        return

    full_st = sd.st.copy()
    segments = _segments_outside_cuts(full_st, cut_windows)
    print(
        f"  Tilt correction: {len(segments)} segment(s) outside "
        f"{len(cut_windows)} cut window(s)"
    )
    for seg_start, seg_end in segments:
        seg_st = full_st.slice(starttime=seg_start, endtime=seg_end).copy()
        if len(seg_st) == 0 or len(seg_st[0].data) < 2:
            continue
        seg_before = seg_st.copy()
        sd.st = seg_st
        sd.correct_tilt()
        _paste_segment_correction(full_st, seg_before, seg_st)

    sd.st = full_st


def _variance_reduction_pct(before, after):
    """Variance reduction in percent: 100 * (var_before - var_after) / var_before."""
    var_b = float(np.var(before))
    var_a = float(np.var(after))
    if var_b <= 0:
        return np.nan
    return 100.0 * (var_b - var_a) / var_b


def _trace_plot_stats(before_tr, after_tr, cut_mask=None, scale=PLOT_SCALE):
    """Tilt metrics for plots: RMS (µm/s²) and variance reduction (%)."""
    b, a = before_tr.data, after_tr.data
    if cut_mask is not None and len(cut_mask) == len(b):
        valid = ~cut_mask
        if np.any(valid):
            b, a = b[valid], a[valid]
    rms_b = float(np.sqrt(np.mean(b ** 2)) * scale)
    rms_a = float(np.sqrt(np.mean(a ** 2)) * scale)
    return {
        'rms_before': rms_b,
        'rms_after': rms_a,
        'var_reduction_pct': _variance_reduction_pct(b, a),
    }


def plot_tilt_correction_figure(
    date,
    config,
    ev,
    st_before,
    st_after,
    cut_mask=None,
    output_dir=None,
):
    """
    Save before/after tilt-correction figure (05_XROMY_plot_tilt_correction.ipynb).
    """
    ref_time = UTCDateTime(config['tbeg'])
    scale = PLOT_SCALE
    unit_label = r"$\mu$m/s$^{2}$"

    mag = config.get('magnitude', np.nan)
    baz = ev.get('backazimuth', np.nan)
    dist = ev.get('distance_km', np.nan)

    title_parts = [f"ROMY event {date[:4]}-{date[4:6]}-{date[6:]}", f"M {mag:.1f}"]
    if np.isfinite(baz):
        title_parts.append(f"BAz {baz:.0f}°")
    if np.isfinite(dist):
        title_parts.append(f"Δ = {dist:.0f} km")
    title_parts.append(f"{config['fmin']:.2f}–{config['fmax']:.2f} Hz")
    fig_title = " | ".join(title_parts)

    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'ytick.minor.visible': True,
        'xtick.minor.visible': True,
        'legend.frameon': False,
    })

    fig = plt.figure(figsize=(10, 6.5))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 3, 1], hspace=0.08)
    colors = {'before': 'black', 'after': 'tab:green', 'diff': 'gray'}

    ax_diff = None
    for i, (ch, label) in enumerate(PLOT_COMPONENTS):
        tr_b = st_before.select(channel=f"*{ch}")[0]
        tr_a = st_after.select(channel=f"*{ch}")[0]
        t_min = tr_b.times(reftime=ref_time) / 60.0
        y_b = tr_b.data * scale
        y_a = tr_a.data * scale
        y_d = y_b - y_a
        s = _trace_plot_stats(tr_b, tr_a, cut_mask=cut_mask, scale=scale)

        ax_main = fig.add_subplot(gs[2 * i])
        ax_diff = fig.add_subplot(gs[2 * i + 1], sharex=ax_main)

        ax_main.plot(t_min, y_b, color=colors['before'], lw=1.3, label='before tilt correction')
        ax_main.plot(t_min, y_a, color=colors['after'], lw=1.1, ls='--', label='after tilt correction')
        ax_main.set_xlim(0, t_min.max())
        ax_main.set_ylabel(f"{label}\n({unit_label})")
        ax_main.grid(True, alpha=0.25, lw=0.5)
        if i == 0:
            ax_main.set_xticklabels([])
        metric_text = (
            f"RMS$_{{before}}$ = {s['rms_before']:.3f} µm/s$^{{2}}$\n"
            f"RMS$_{{after}}$ = {s['rms_after']:.3f} µm/s$^{{2}}$\n"
            f"Var. reduction = {s['var_reduction_pct']:.2f}%"
        )
        ax_main.text(
            0.985, 0.95, metric_text,
            transform=ax_main.transAxes,
            va='top', ha='right',
            fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='0.8', alpha=0.9),
        )

        if i == 0:
            ax_main.legend(loc='upper left', fontsize=10)
            ax_main.set_title(fig_title, fontsize=12, pad=10)
        else:
            plt.setp(ax_main.get_xticklabels(), visible=False)

        ax_diff.plot(t_min, y_d, color=colors['diff'], lw=1.0)
        ax_diff.fill_between(t_min, 0, y_d, color=colors['diff'], alpha=0.25)
        ax_diff.set_ylabel(f"Δ{label}\n" + r"(µm s$^{-2}$)", fontsize=10)
        ax_diff.grid(True, alpha=0.25, lw=0.5, ls='--')

        if i < len(PLOT_COMPONENTS) - 1:
            plt.setp(ax_diff.get_xticklabels(), visible=False)

    if ax_diff is not None:
        ax_diff.set_xlabel(
            f"Time (min) since {str(ref_time)[:-8].replace('T', ' ')} UTC"
        )

    fig.align_ylabels()
    fig.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.08)

    if output_dir is None:
        output_dir = config.get('path_to_figures', './output/figures/tilt_correction/')
    if not output_dir.endswith('/'):
        output_dir += '/'
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'event_{date}_tilt_correction.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved tilt correction plot: {out_path}")
    return out_path


def compute_tilt_correction_metrics(sd, config=None, scale=1e6):
    """
    Apply tilt correction and compute metrics for horizontal translations.

    Follows the workflow in 05_XROMY_plot_tilt_correction.ipynb:
    split at time cuts, correct_tilt per segment, merge, compare HN/HE.

    Primary metric: variance reduction (%),
    100 * (var_before - var_after) / var_before (outside cut windows).

    Parameters
    ----------
    sd : sixdegrees
        Preprocessed sixdegrees object (filtered, polarity-corrected).
    scale : float
        Display/storage scale for acceleration RMS (default 1e6 -> µm/s²).

    Returns
    -------
    dict
        RMS before/after and variance reduction (%) for north (HN) and east (HE).
        sd.st is modified in place (tilt-corrected).
    """
    st_before = sd.st.copy()
    if config and config.get('time_cuts'):
        correct_tilt_split_at_cuts(sd, config)
    else:
        sd.correct_tilt()
    st_after = sd.st.copy()

    cut_mask = None
    if config and config.get('time_cuts'):
        hn_ref = st_after.select(channel='*HN')
        if hn_ref:
            cut_mask = _cut_sample_mask(hn_ref[0], _merged_cut_windows(config))

    metrics = {'rms_unit': 'um/s2'}
    for channel, key in (('HN', 'north'), ('HE', 'east')):
        tr_b = st_before.select(channel=f"*{channel}")
        tr_a = st_after.select(channel=f"*{channel}")
        if len(tr_b) == 0 or len(tr_a) == 0:
            print(f"Warning: Missing {channel} trace for tilt correction metrics")
            metrics[f'rms_before_{key}'] = np.nan
            metrics[f'rms_after_{key}'] = np.nan
            metrics[f'rms_residual_{key}'] = np.nan
            metrics[f'var_reduction_{key}'] = np.nan
            continue

        b, a = tr_b[0].data, tr_a[0].data
        if cut_mask is not None and len(cut_mask) == len(b):
            valid = ~cut_mask
            if not np.any(valid):
                valid = np.ones(len(b), dtype=bool)
            b, a = b[valid], a[valid]
        metrics[f'rms_before_{key}'] = float(np.sqrt(np.mean(b ** 2)) * scale)
        metrics[f'rms_after_{key}'] = float(np.sqrt(np.mean(a ** 2)) * scale)
        metrics[f'rms_residual_{key}'] = float(np.sqrt(np.mean((b - a) ** 2)) * scale)
        metrics[f'var_reduction_{key}'] = _variance_reduction_pct(b, a)

    return metrics, st_before, st_after, cut_mask


def compute_full_waveform_cc(sd, baz_theo):
    """
    Compute Love and Rayleigh full-waveform CC using theoretical backazimuth.

    Returns
    -------
    dict
        Per wave_type: cc_max, cc_max_lag, cc_zero, cc_zero_lag, backazimuth_theoretical,
        sampling_rate.
    """
    cc_full_waveform = {}
    rot_stream = sd.get_stream("rotation")
    tra_stream = sd.get_stream("translation")

    if rot_stream is None or tra_stream is None:
        return cc_full_waveform

    for wave_type in ['love', 'rayleigh']:
        try:
            if wave_type == 'love':
                rot_z = rot_stream.select(channel="*JZ")
                if len(rot_z) == 0:
                    continue
                rot_z = rot_z[0]

                tra_h_rt = tra_stream.copy().select(channel="*H*")
                tra_h_rt.rotate('NE->RT', back_azimuth=baz_theo)
                tra_t = tra_h_rt.select(channel="*T")
                if len(tra_t) == 0:
                    continue
                tra_t = tra_t[0]

                min_len = min(len(rot_z.data), len(tra_t.data))
                if min_len <= 0:
                    continue

                rot_data = rot_z.data[:min_len]
                tra_data = tra_t.data[:min_len]

            else:  # rayleigh
                tra_z = tra_stream.select(channel="*HZ")
                if len(tra_z) == 0:
                    continue
                tra_z = tra_z[0]

                rot_h_rt = rot_stream.copy().select(channel="*J*")
                rot_h_rt.rotate('NE->RT', back_azimuth=baz_theo)
                rot_t = rot_h_rt.select(channel="*T")
                if len(rot_t) == 0:
                    continue
                rot_t = rot_t[0]

                min_len = min(len(rot_t.data), len(tra_z.data))
                if min_len <= 0:
                    continue

                rot_data = rot_t.data[:min_len]
                tra_data = tra_z.data[:min_len]

            shift_len = len(rot_data)
            corr_shift = correlate(rot_data, tra_data, shift=shift_len)
            corr_zero = correlate(rot_data, tra_data, shift=0)
            shift_max, cc_max = xcorr_max(corr_shift)
            shift_zero, cc_zero = xcorr_max(corr_zero)

            cc_full_waveform[wave_type] = {
                'cc_max': round(cc_max, 2),
                'cc_max_lag': round(shift_max, 2),
                'cc_zero': round(cc_zero, 2),
                'cc_zero_lag': round(shift_zero, 2),
                'backazimuth_theoretical': baz_theo,
                'sampling_rate': sd.sampling_rate,
            }

        except Exception as e:
            print(f"Warning: Could not compute full waveform CC for {wave_type}: {e}")
            import traceback
            traceback.print_exc()

    return cc_full_waveform


def process_event(date, save_plots=True):
    """
    Process a single event: tilt correction and full-waveform CC only.

    Parameters
    ----------
    date : str
        Event date string (e.g., '20240314')
    save_plots : bool
        If True, save tilt-correction before/after figure per event.

    Returns
    -------
    dict or None
        Dictionary containing event info, tilt correction, and CC results.
    """
    print(f"\n{'='*60}")
    print(f"Processing event: {date}")
    print(f"{'='*60}")

    try:
        config_path = f"./events/event_{date}/config_romy_{date}.yml"
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found: {config_path}")
            return None

        config = sixdegrees.load_from_yaml(config_path)
        sd = sixdegrees(conf=config)

        ev = sd.get_event_info(
            origin_time=config['origin_time'],
            base_catalog="USGS",
            magnitude=config['magnitude'],
        )

        print(f"Event: {ev.get('origin_time', 'N/A')}")
        print(f"Theoretical backazimuth: {ev.get('backazimuth', 'N/A'):.2f}°")

        sd.load_data(config['tbeg'], config['tend'], resample_rate=5)
        sd.trim_stream(set_common=True, set_interpolate=True)
        sd.filter_data(fmin=config['fmin'], fmax=config['fmax'], output=False)

        if 'pol_dict' in config:
            sd.polarity_stream(pol_dict=config['pol_dict'], raw=True)
        else:
            sd.polarity_stream(pol_dict={"HZ": -1, "JZ": 1}, raw=True)

        if config.get('time_cuts'):
            print(f"Cutting bad data for {date}...")
            for key, entries in config['time_cuts'].items():
                if not entries:
                    continue
                if len(entries) == 2 and isinstance(entries[0], str):
                    entries = [entries]
                for entry in entries:
                    sd.replace_values_in_window(
                        tbeg=UTCDateTime(entry[0]),
                        tend=UTCDateTime(entry[1]),
                        value=0,
                        channel_list=[key],
                    )

        print("\nApplying tilt correction to translation components...")
        tilt_correction, st_before, st_after, cut_mask = compute_tilt_correction_metrics(
            sd, config=config
        )
        print(
            f"  North: RMS before={tilt_correction['rms_before_north']:.3f}, "
            f"after={tilt_correction['rms_after_north']:.3f}, "
            f"var. reduction={tilt_correction['var_reduction_north']:.2f}%"
        )
        print(
            f"  East:  RMS before={tilt_correction['rms_before_east']:.3f}, "
            f"after={tilt_correction['rms_after_east']:.3f}, "
            f"var. reduction={tilt_correction['var_reduction_east']:.2f}%"
        )

        if save_plots:
            plot_tilt_correction_figure(
                date, config, ev, st_before, st_after, cut_mask=cut_mask
            )

        print("\nComputing CC for entire waveforms using theoretical backazimuth...")
        cc_full_waveform = compute_full_waveform_cc(sd, ev['backazimuth'])
        for wave_type, cc_data in cc_full_waveform.items():
            print(
                f"  {wave_type.capitalize()}: cc_max={cc_data['cc_max']:.2f}, "
                f"cc_zero={cc_data['cc_zero']:.2f}"
            )

        return {
            'date': date,
            'event_info': ev,
            'tilt_correction': tilt_correction,
            'cc_full_waveform': cc_full_waveform,
        }

    except Exception as e:
        print(f"Error processing event {date}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_cc_results(all_results, output_dir='./output'):
    """
    Save full-waveform CC and tilt-correction metrics to CSV.

    Parameters
    ----------
    all_results : list
        List of result dictionaries from process_event
    output_dir : str
        Output directory for CSV files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cc_full_rows = []

    for result in all_results:
        if result is None:
            continue

        date = result['date']
        ev = result['event_info']
        cc_full_waveform = result.get('cc_full_waveform', {})
        tilt_correction = result.get('tilt_correction', {})
        tilt_cols = {
            'rms_before_north': tilt_correction.get('rms_before_north', np.nan),
            'rms_after_north': tilt_correction.get('rms_after_north', np.nan),
            'rms_residual_north': tilt_correction.get('rms_residual_north', np.nan),
            'var_reduction_north': tilt_correction.get('var_reduction_north', np.nan),
            'rms_before_east': tilt_correction.get('rms_before_east', np.nan),
            'rms_after_east': tilt_correction.get('rms_after_east', np.nan),
            'rms_residual_east': tilt_correction.get('rms_residual_east', np.nan),
            'var_reduction_east': tilt_correction.get('var_reduction_east', np.nan),
            'rms_unit': tilt_correction.get('rms_unit', 'um/s2'),
        }

        if cc_full_waveform:
            for wave_type, cc_data in cc_full_waveform.items():
                cc_full_rows.append({
                    'event_date': date,
                    'origin_time': ev.get('origin_time', ''),
                    'wave_type': wave_type,
                    'cc_max': cc_data.get('cc_max', np.nan),
                    'cc_max_lag': cc_data.get('cc_max_lag', np.nan),
                    'cc_zero': cc_data.get('cc_zero', np.nan),
                    'cc_zero_lag': cc_data.get('cc_zero_lag', np.nan),
                    'backazimuth_theoretical': cc_data.get('backazimuth_theoretical', np.nan),
                    'sampling_rate': cc_data.get('sampling_rate', np.nan),
                    **tilt_cols,
                })
        elif tilt_correction:
            cc_full_rows.append({
                'event_date': date,
                'origin_time': ev.get('origin_time', ''),
                'wave_type': '',
                'cc_max': np.nan,
                'cc_max_lag': np.nan,
                'cc_zero': np.nan,
                'cc_zero_lag': np.nan,
                'backazimuth_theoretical': ev.get('backazimuth', np.nan),
                'sampling_rate': np.nan,
                **tilt_cols,
            })

    if cc_full_rows:
        df_full = pd.DataFrame(cc_full_rows)
        output_file = os.path.join(output_dir, 'results_correlation.csv')
        df_full.to_csv(output_file, index=False)
        print(f"CC results saved to: {output_file}")
    else:
        print("No CC results to save.")


def load_events_list(yaml_file=None, events_arg=None):
    """
    Load event list from YAML file or command-line argument.

    Parameters
    ----------
    yaml_file : str, optional
        Path to YAML file containing event list.
    events_arg : str, optional
        Comma-separated string of event dates (e.g., '20230908,20230912').
        Takes precedence over yaml_file if provided.

    Returns
    -------
    list
        List of event date strings
    """
    if events_arg:
        events_list = [e.strip() for e in events_arg.split(',') if e.strip()]
        print(f"Using {len(events_list)} events from command-line argument")
        return events_list

    if yaml_file and os.path.exists(yaml_file):
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            events_list = config.get('events', [])
            print(f"Loaded {len(events_list)} events from {yaml_file}")
            return events_list

    return []


def main():
    """Process all events for tilt correction and full-waveform CC."""
    parser = argparse.ArgumentParser(
        description='Process ROMY events for correlation and tilt correction (no backazimuth)'
    )
    parser.add_argument(
        '--events-file',
        type=str,
        default=None,
        help='Path to YAML file containing list of events to process',
    )
    parser.add_argument(
        '--events',
        type=str,
        default=None,
        help=(
            'Comma-separated list of event dates (e.g., "20230908,20230912"). '
            'Takes precedence over --events-file'
        ),
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip saving tilt-correction figures',
    )
    args = parser.parse_args()

    events_list = load_events_list(yaml_file=args.events_file, events_arg=args.events)
    if not events_list:
        print("No events to process. Provide --events or --events-file.")
        return

    print("=" * 60)
    print("ROMY Event Analysis — CC & Tilt Correction")
    print("=" * 60)
    print(f"Processing {len(events_list)} events...")

    all_results = []
    for date in events_list:
        result = process_event(date, save_plots=not args.no_plots)
        if result is not None:
            all_results.append(result)

    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    save_cc_results(all_results)

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
