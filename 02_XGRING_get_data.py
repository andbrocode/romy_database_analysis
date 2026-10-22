#!/usr/bin/env python3
"""
Fetch Gring Data for ROMY Events

This script downloads Gring (GR.WET translation, BW.RLAS rotation) data for all events
listed in events_list.yml. For each event, it downloads data from -5 min to +150 min
relative to the event origin time and saves it as gring_<date>.mseed in the
respective event directory.
"""

import os
import argparse
import yaml
from datetime import timedelta
from obspy import UTCDateTime
from sixdegrees.sixdegrees import sixdegrees


def load_events_list(yaml_file='events_list.yml'):
    """
    Load event list from YAML file.
    
    Parameters:
    -----------
    yaml_file : str
        Path to YAML file containing event list.
        
    Returns:
    --------
    list : List of event date strings
    """
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"Events list file not found: {yaml_file}")
    
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
        events_list = config.get('events', [])
        print(f"Loaded {len(events_list)} events from {yaml_file}")
        return events_list


def get_event_config(date):
    """
    Load event configuration from YAML file.
    
    Parameters:
    -----------
    date : str
        Event date string (e.g., '20240314')
        
    Returns:
    --------
    dict : Event configuration dictionary
    """
    config_path = f"./events/event_{date}/config_romy_{date}.yml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config


def calculate_time_window(origin_time_str, t_minus_min=5, t_plus_min=150):
    """
    Calculate time window relative to origin time.
    
    Parameters:
    -----------
    origin_time_str : str
        Origin time string (e.g., '2024-03-14 21:10:24')
    t_minus_min : int
        Minutes before origin time (default: 5)
    t_plus_min : int
        Minutes after origin time (default: 150)
        
    Returns:
    --------
    tuple : (tbeg, tend) as UTCDateTime objects
    """
    origin_time = UTCDateTime(origin_time_str)
    tbeg = origin_time - timedelta(minutes=t_minus_min)
    tend = origin_time + timedelta(minutes=t_plus_min)
    return tbeg, tend


def fetch_gring_data(date, event_config):
    """
    Fetch Gring data for a single event.
    
    Parameters:
    -----------
    date : str
        Event date string (e.g., '20240314')
    event_config : dict
        Event configuration dictionary
        
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Processing event: {date}")
    print(f"{'='*60}")
    
    try:
        # Get origin time and magnitude from event config
        origin_time_str = event_config.get('origin_time')
        magnitude = event_config.get('magnitude', 0.0)
        
        if not origin_time_str:
            print(f"Warning: No origin_time found in config for event {date}")
            return False
        
        # Calculate time window (-5 min to +150 min)
        tbeg, tend = calculate_time_window(origin_time_str, t_minus_min=5, t_plus_min=150)
        
        print(f"Origin time: {origin_time_str}")
        print(f"Time window: {tbeg} to {tend}")
        print(f"Magnitude: {magnitude}")
        
        # Setup configuration for Gring data
        config = {
            # Time window parameters
            'tbeg': str(tbeg),
            'tend': str(tend),
            
            # Event information
            'origin_time': origin_time_str,
            'magnitude': magnitude,
            
            # SEED identifiers for Gring
            'seed': "XX.RLAS..",     # seed ID of output stream
            'rot_seed': ["BW.RLAS..BJZ"],  # Rotation sensor seed ID
            'tra_seed': ["GR.WET..BHZ", "GR.WET..BHN", "GR.WET..BHE"],  # Translation sensor seed ID
            
            # Data source
            'data_source': "fdsn",  # Use FDSN to fetch data
            
            # FDSN clients
            'fdsn_client_rot': "LMU",  # For rotation data
            'fdsn_client_tra': "BGR",  # For translation data
            
            # Remove response
            'tra_remove_response': True,
            'rot_remove_response': True,
            
            # Path configurations
            'workdir': "./",
            'path_to_data_out': f"./events/event_{date}/",  # Output directory
            
            # Project information
            'project': "G_Analysis",
            
            # Optional parameters
            'verbose': True,
        }
        
        # Setup sixdegrees object
        print("\nSetting up sixdegrees object...")
        sd = sixdegrees(conf=config)
        
        # Load waveform data
        print("\nLoading waveform data from FDSN...")
        sd.load_data(config['tbeg'], config['tend'])
        
        # Ensure all traces have the same amount of samples
        print("\nTrimming stream to common time window...")
        sd.trim_stream(set_common=True, set_interpolate=False)
        
        # Create output directory if it doesn't exist
        output_dir = f"./events/event_{date}/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Write data to mseed file
        output_file = os.path.join(output_dir, f"gring_{date}.mseed")
        print(f"\nWriting data to: {output_file}")
        sd.st.write(output_file, format="MSEED")
        
        print(f"Successfully saved Gring data for event {date}")
        print(f"  File: {output_file}")
        print(f"  Traces: {len(sd.st)}")
        for tr in sd.st:
            print(f"    {tr.id} | {tr.stats.starttime} - {tr.stats.endtime} | {tr.stats.sampling_rate} Hz")
        
        return True
        
    except Exception as e:
        print(f"Error processing event {date}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function to fetch Gring data for all events.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Fetch Gring data for ROMY events'
    )
    parser.add_argument(
        '--events-file',
        type=str,
        default='events_list.yml',
        help='Path to YAML file containing list of events to process (default: events_list.yml)'
    )
    parser.add_argument(
        '--events',
        type=str,
        default=None,
        help='Comma-separated list of event dates (e.g., "20230908,20230912"). Takes precedence over --events-file'
    )
    args = parser.parse_args()
    
    # Load events list
    if args.events:
        events_list = [e.strip() for e in args.events.split(',') if e.strip()]
        print(f"Using {len(events_list)} events from command-line argument")
    else:
        events_list = load_events_list(args.events_file)
    
    print("="*60)
    print("ROMY Gring Data Fetching")
    print("="*60)
    print(f"Processing {len(events_list)} events...")
    print(f"Time window: -5 min to +150 min relative to origin time")
    
    # Process all events
    successful = 0
    failed = 0
    
    for date in events_list:
        try:
            # Load event configuration
            event_config = get_event_config(date)
            
            # Fetch Gring data
            if fetch_gring_data(date, event_config):
                successful += 1
            else:
                failed += 1
                
        except FileNotFoundError as e:
            print(f"\nSkipping event {date}: {e}")
            failed += 1
        except Exception as e:
            print(f"\nError with event {date}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"Total events: {len(events_list)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("="*60)


if __name__ == "__main__":
    main()
