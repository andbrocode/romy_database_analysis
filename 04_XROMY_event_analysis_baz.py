#!/usr/bin/env python3
"""
Automated ROMY Event Analysis Script

This script processes a list of events from the events/ directory and:
1. Applies tilt correction to horizontal translation components
2. Computes backazimuth estimates with uncertainty for all wave types (Love, Rayleigh, Tangent)
3. Stores backazimuth results in CSV format
4. Computes CC values by windows from compare_backazimuth function
5. Computes one CC estimate for entire waveforms using theoretical backazimuth
6. Stores CC results (including tilt-correction RMS metrics for north/east) in CSV format

No plots are created.
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.signal.cross_correlation import correlate, xcorr_max
from sixdegrees.sixdegrees import sixdegrees


def compute_tilt_correction_metrics(sd, scale=1e6):
    """
    Apply tilt correction and compute RMS metrics for horizontal translations.

    Follows the workflow in 05_XROMY_plot_tilt_correction.ipynb:
    copy stream before correction, call sd.correct_tilt(), then compare HN/HE.

    Parameters
    ----------
    sd : sixdegrees
        Preprocessed sixdegrees object (filtered, polarity-corrected).
    scale : float
        Display/storage scale for acceleration RMS (default 1e6 -> µm/s²).

    Returns
    -------
    dict
        RMS before/after/residual for north (HN) and east (HE) components.
        sd.st is modified in place (tilt-corrected).
    """
    st_before = sd.st.copy()
    sd.correct_tilt()
    st_after = sd.st.copy()

    def _trace_rms_stats(before_tr, after_tr):
        b = before_tr.data
        a = after_tr.data
        d = b - a
        return {
            'rms_before': float(np.sqrt(np.mean(b ** 2)) * scale),
            'rms_after': float(np.sqrt(np.mean(a ** 2)) * scale),
            'rms_residual': float(np.sqrt(np.mean(d ** 2)) * scale),
        }

    metrics = {'rms_unit': 'um/s2'}
    for channel, key in (('HN', 'north'), ('HE', 'east')):
        tr_b = st_before.select(channel=f"*{channel}")
        tr_a = st_after.select(channel=f"*{channel}")
        if len(tr_b) == 0 or len(tr_a) == 0:
            print(f"Warning: Missing {channel} trace for tilt correction metrics")
            metrics[f'rms_before_{key}'] = np.nan
            metrics[f'rms_after_{key}'] = np.nan
            metrics[f'rms_residual_{key}'] = np.nan
            continue

        stats = _trace_rms_stats(tr_b[0], tr_a[0])
        metrics[f'rms_before_{key}'] = stats['rms_before']
        metrics[f'rms_after_{key}'] = stats['rms_after']
        metrics[f'rms_residual_{key}'] = stats['rms_residual']

    return metrics


def process_event(date, events_list):
    """
    Process a single event and extract backazimuth and CC results.
    
    Parameters:
    -----------
    date : str
        Event date string (e.g., '20240314')
    events_list : list
        List of event dates to process
        
    Returns:
    --------
    dict : Dictionary containing backazimuth and CC results
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
        if 'time_cuts' in config.keys():
            print(f"Cutting bad data for {date}...")
            for key, values in config['time_cuts'].items():
                for value in values:
                    sd.replace_values_in_window(
                        tbeg=UTCDateTime(value[0]), tend=UTCDateTime(value[1]), 
                        value=0, channel_list=[key]
                    )

        # Tilt correction for horizontal translations (before CC/backazimuth)
        print("\nApplying tilt correction to translation components...")
        tilt_correction = compute_tilt_correction_metrics(sd)
        print(
            f"  North: RMS before={tilt_correction['rms_before_north']:.3f}, "
            f"after={tilt_correction['rms_after_north']:.3f}, "
            f"residual={tilt_correction['rms_residual_north']:.4f} µm/s²"
        )
        print(
            f"  East:  RMS before={tilt_correction['rms_before_east']:.3f}, "
            f"after={tilt_correction['rms_after_east']:.3f}, "
            f"residual={tilt_correction['rms_residual_east']:.4f} µm/s²"
        )
        
        # Compute backazimuth using compare_backazimuth_methods for both 'mid' and 'max'
        print("\nComputing backazimuth estimates...")
        baz_results = {}
        cc_results_by_window = {}
        
        # Process both 'mid' and 'max' methods
        for cc_method in ['mid', 'max']:
            try:
                print(f"  Computing with cc_method='{cc_method}'...")
                compare_results = sd.compare_backazimuth_methods(
                    Twin=config['twin_sec'],
                    Toverlap=config['twin_overlap'],
                    cc_threshold=config['cc_threshold'],
                    baz_theo=ev['backazimuth'],
                    cc_method=cc_method,
                    plot=False,  # No plots
                    output=True
                )
                
                # Extract results for all wave types (love, rayleigh, tangent)
                wave_types = ['love', 'rayleigh', 'tangent']
                
                for wave_type in wave_types:
                    if wave_type in compare_results:
                        wave_result = compare_results[wave_type]
                        
                        # Extract backazimuth estimate and uncertainty
                        baz_est = wave_result.get('baz_estimate', np.nan)
                        baz_std = wave_result.get('baz_std', np.nan)
                        baz_n_samples = wave_result.get('n_samples', np.nan)
                    
                        # Store results with method identifier
                        key = f"{wave_type}_{cc_method}"
                        baz_results[key] = {
                            'wave_type': wave_type,
                            'cc_method': cc_method,
                            'n_samples': baz_n_samples,
                            'backazimuth_estimate': baz_est,
                            'backazimuth_uncertainty': baz_std,
                            'backazimuth_theoretical': ev['backazimuth'],
                            'cc_threshold': config['cc_threshold']
                        }
                        
                        # Try to extract time, CC, and backazimuth arrays
                        times = []
                        ccs = []
                        bazs = []
                        try:
                            times = wave_result.get('time', [])
                            ccs = wave_result.get('correlation', [])
                            bazs = wave_result.get('backazimuth', [])
                        except Exception as e:
                            print(f"Warning: Could not extract time, CC, and backazimuth arrays: {e}")
                            continue

                        # Convert to lists if numpy arrays
                        if isinstance(times, np.ndarray):
                            times = times.tolist()
                        if isinstance(ccs, np.ndarray):
                            ccs = ccs.tolist()
                        if isinstance(bazs, np.ndarray):
                            bazs = bazs.tolist()
                        
                        cc_results_by_window[key] = {
                            'wave_type': wave_type,
                            'cc_method': cc_method,
                            'time': times,
                            'correlation': ccs,
                            'backazimuth': bazs
                            }
                
            except Exception as e:
                print(f"Warning: Could not compute backazimuth with cc_method='{cc_method}': {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Compute CC for entire waveforms using theoretical backazimuth
        # This mimics the waveformc_cc functionality but for full waveforms
        print("\nComputing CC for entire waveforms using theoretical backazimuth...")
        cc_full_waveform = {}
        
        baz_theo = ev['backazimuth']
        
        for wave_type in ['love', 'rayleigh']:
            try:
                # Get rotation and translation streams
                rot_stream = sd.get_stream("rotation")
                tra_stream = sd.get_stream("translation")
                
                if rot_stream is None or tra_stream is None:
                    continue
                
                # For Love waves: compare JZ rotation with transverse translation
                if wave_type == 'love':
                    # Get JZ rotation component
                    rot_z = rot_stream.select(channel="*JZ")
                    if len(rot_z) == 0:
                        continue
                    rot_z = rot_z[0]
                    
                    # Get horizontal translation components
                    tra_h = tra_stream.select(channel="*H*")
                    tra_n = tra_h.select(channel="*HN")
                    tra_e = tra_h.select(channel="*HE")
                    
                    if len(tra_n) > 0 and len(tra_e) > 0:
                        # Rotate to radial/transverse components
                        tra_h_rt = tra_stream.copy().select(channel="*H*")
                        tra_h_rt.rotate('NE->RT', back_azimuth=baz_theo)
                        tra_t = tra_h_rt.select(channel="*T")
                        
                        if len(tra_t) > 0:
                            tra_t = tra_t[0]
                            # Ensure same length
                            min_len = min(len(rot_z.data), len(tra_t.data))
                            if min_len > 0:
                                # Use obspy cross-correlation
                                shift_len = len(rot_z.data[:min_len])
                                corr_shift = correlate(
                                    rot_z.data[:min_len], 
                                    tra_t.data[:min_len],
                                    shift=shift_len
                                )
                                corr_zero = correlate(
                                    rot_z.data[:min_len], 
                                    tra_t.data[:min_len],
                                    shift=0
                                )
                                # Maximum correlationw with lag
                                shift_max, cc_max = xcorr_max(corr_shift)
                                # Zero shift correlation (at center of correlation array)
                                shift_zero, cc_zero = xcorr_max(corr_zero)

                                # Prepare output
                                cc_full_waveform[wave_type] = {
                                    'cc_max': round(cc_max, 2),
                                    'cc_max_lag': round(shift_max, 2),
                                    'cc_zero': round(cc_zero, 2),
                                    'cc_zero_lag': round(shift_zero, 2),
                                    'backazimuth_theoretical': baz_theo,
                                    'sampling_rate': sd.sampling_rate,
                                }
                
                # For Rayleigh waves: compare horizontal rotation with vertical translation
                elif wave_type == 'rayleigh':
                    # Get horizontal rotation components
                    rot_h = rot_stream.select(channel="*J*")
                    rot_n = rot_h.select(channel="*JN")
                    rot_e = rot_h.select(channel="*JE")
                    
                    # Get vertical translation
                    tra_z = tra_stream.select(channel="*HZ")
                    
                    if len(rot_n) > 0 and len(rot_e) > 0 and len(tra_z) > 0:
                        # Rotate rotation to radial component
                        rot_h_rt = rot_stream.copy().select(channel="*J*")
                        rot_h_rt.rotate('NE->RT', back_azimuth=baz_theo)
                        rot_t = rot_h_rt.select(channel="*T")
                        
                        if len(rot_t) > 0:
                            rot_t = rot_t[0]
                            tra_z = tra_z[0]
                            # Ensure same length
                            if len(rot_t.data) != len(tra_z.data):
                                print(f"Warning: Rotation and translation data have different lengths: {len(rot_t.data)} != {len(tra_z.data)}")
                                min_len = min(len(rot_t.data), len(tra_z.data))
                            else:
                                min_len = len(rot_t.data)
                                
                            if min_len > 0:
                                # Use obspy cross-correlation
                                shift_len = len(rot_t.data[:min_len])
                                corr_shift = correlate(
                                    rot_t.data[:min_len], 
                                    tra_z.data[:min_len],
                                    shift=shift_len
                                )
                                corr_zero = correlate(
                                    rot_t.data[:min_len], 
                                    tra_z.data[:min_len],
                                    shift=0
                                )
                                # Maximum correlation
                                shift_max, cc_max = xcorr_max(corr_shift)
                                # Zero shift correlation (at center of correlation array)
                                shift_zero, cc_zero = xcorr_max(corr_zero)

                                # Prepare output
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
                continue
        
        return {
            'date': date,
            'event_info': ev,
            'tilt_correction': tilt_correction,
            'backazimuth_results': baz_results,
            'cc_by_window': cc_results_by_window,
            'cc_full_waveform': cc_full_waveform
        }
    
    except Exception as e:
        print(f"Error processing event {date}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_backazimuth_results(all_results, output_dir='./output'):
    """
    Save backazimuth results to CSV.
    
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
        baz_results = result['backazimuth_results']
        
        for key, baz_data in baz_results.items():
            # Calculate residual, handling NaN values
            baz_est = baz_data.get('backazimuth_estimate', np.nan)
            baz_theo = baz_data.get('backazimuth_theoretical', np.nan)
            if not np.isnan(baz_est) and not np.isnan(baz_theo):
                baz_residual = round(baz_est - baz_theo, 1)
            else:
                baz_residual = np.nan
            
            rows.append({
                'event_date': date,
                'origin_time': ev.get('origin_time', ''),
                'magnitude': ev.get('magnitude', ''),
                'distance_km': ev.get('distance_km', ''),
                'wave_type': baz_data.get('wave_type', ''),
                'cc_method': baz_data.get('cc_method', ''),
                'n_samples': baz_data.get('n_samples', np.nan),
                'backazimuth_estimate': baz_est,
                'backazimuth_uncertainty': baz_data.get('backazimuth_uncertainty', np.nan),
                'backazimuth_theoretical': baz_theo,
                'backazimuth_residual': baz_residual,
                'cc_threshold': baz_data.get('cc_threshold', np.nan)
            })
    
    if rows:
        df = pd.DataFrame(rows)
        output_file = os.path.join(output_dir, 'results_backazimuth.csv')
        df.to_csv(output_file, index=False)
        print(f"\nBackazimuth results saved to: {output_file}")
        return df
    else:
        print("\nNo backazimuth results to save.")
        return None


def save_cc_results(all_results, output_dir='./output'):
    """
    Save CC results to CSV files.
    
    Parameters:
    -----------
    all_results : list
        List of result dictionaries from process_event
    output_dir : str
        Output directory for CSV files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save CC by windows
    cc_window_rows = []
    cc_full_rows = []
    
    for result in all_results:
        if result is None:
            continue
        
        date = result['date']
        ev = result['event_info']
        
        # CC by windows
        cc_by_window = result.get('cc_by_window', {})
        for key, window_data in cc_by_window.items():
            times = window_data.get('time', [])
            ccs = window_data.get('correlation', [])
            bazs = window_data.get('backazimuth', [])
            wave_type = window_data.get('wave_type', '')
            cc_method = window_data.get('cc_method', '')
            
            for i, (t, cc, baz) in enumerate(zip(times, ccs, bazs)):
                cc_window_rows.append({
                    'event_date': date,
                    'origin_time': ev.get('origin_time', ''),
                    'wave_type': wave_type,
                    'cc_method': cc_method,
                    'window_index': i,
                    'time': t,
                    'correlation': cc,
                    'backazimuth': baz,
                    'backazimuth_theoretical': ev.get('backazimuth', np.nan)
                })
        
        # CC for full waveform (includes tilt-correction RMS metrics per event)
        cc_full_waveform = result.get('cc_full_waveform', {})
        tilt_correction = result.get('tilt_correction', {})
        tilt_cols = {
            'rms_before_north': tilt_correction.get('rms_before_north', np.nan),
            'rms_after_north': tilt_correction.get('rms_after_north', np.nan),
            'rms_residual_north': tilt_correction.get('rms_residual_north', np.nan),
            'rms_before_east': tilt_correction.get('rms_before_east', np.nan),
            'rms_after_east': tilt_correction.get('rms_after_east', np.nan),
            'rms_residual_east': tilt_correction.get('rms_residual_east', np.nan),
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
    
    # Save CC by windows
    if cc_window_rows:
        df_windows = pd.DataFrame(cc_window_rows)
        output_file_windows = os.path.join(output_dir, 'restuls_correlation_by_window.csv')
        df_windows.to_csv(output_file_windows, index=False)
        print(f"CC by windows saved to: {output_file_windows}")
    else:
        print("No CC by windows results to save.")
    
    # Save CC full waveform
    if cc_full_rows:
        df_full = pd.DataFrame(cc_full_rows)
        output_file_full = os.path.join(output_dir, 'results_correlation.csv')
        df_full.to_csv(output_file_full, index=False)
        print(f"CC full waveform saved to: {output_file_full}")
    else:
        print("No CC full waveform results to save.")


def load_events_list(yaml_file=None, events_arg=None):
    """
    Load event list from YAML file, command-line argument, or return default list.
    
    Parameters:
    -----------
    yaml_file : str, optional
        Path to YAML file containing event list. If None, returns default list.
    events_arg : str, optional
        Comma-separated string of event dates (e.g., '20230908,20230912').
        Takes precedence over yaml_file if provided.
        
    Returns:
    --------
    list : List of event date strings
    """
    # If events are provided directly as argument, use those
    if events_arg:
        events_list = [e.strip() for e in events_arg.split(',') if e.strip()]
        print(f"Using {len(events_list)} events from command-line argument")
        return events_list
    
    # Otherwise, try to load from YAML file
    if yaml_file and os.path.exists(yaml_file):
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            events_list = config.get('events', [])
            print(f"Loaded {len(events_list)} events from {yaml_file}")
            return events_list


def main():
    """
    Main function to process all events.
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Process ROMY events for backazimuth and correlation analysis'
    )
    parser.add_argument(
        '--events-file',
        type=str,
        default=None,
        help='Path to YAML file containing list of events to process'
    )
    parser.add_argument(
        '--events',
        type=str,
        default=None,
        help='Comma-separated list of event dates (e.g., "20230908,20230912"). Takes precedence over --events-file'
    )
    args = parser.parse_args()
    
    # Load events list (--events takes precedence over --events-file)
    events_list = load_events_list(yaml_file=args.events_file, events_arg=args.events)
    
    print("="*60)
    print("ROMY Event Analysis - Automated Processing")
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
    
    save_backazimuth_results(all_results)
    save_cc_results(all_results)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
