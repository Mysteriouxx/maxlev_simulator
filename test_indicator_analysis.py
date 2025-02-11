import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import List, Dict
import numpy as np
import yaml
import sqlite3
from simulator_triple import Simulator, process_trade
from rolling_window import RollingWindowTester
from calculator import IndicatorCalculator
from utils import get_regime_data


# Add main directory to Python path
main_dir = Path(__file__).parent.parent
sys.path.append(str(main_dir))

def load_and_prepare_data(symbol: str, start_date: str, end_date: str) -> tuple:
    """Load and prepare both 15m and 1h data for a symbol"""
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_path_15m = Path('data/raw') / f'{symbol}_15m_{config["parameters"]["start_date"]}_{config["parameters"]["end_date"]}.csv'
    data_path_1h = Path('data/raw') / f'{symbol}_1h_{config["parameters"]["start_date"]}_{config["parameters"]["end_date"]}.csv'
    
    # Load and prepare 15m data
    df_15m = pd.read_csv(data_path_15m)
    df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'])
    df_15m.set_index('timestamp', inplace=True)
    df_15m = df_15m[start_date:end_date]
    
    # Load and prepare 1h data
    df_1h = pd.read_csv(data_path_1h)
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
    df_1h.set_index('timestamp', inplace=True)
    df_1h = df_1h[start_date:end_date]
    
    return df_15m, df_1h

def analyze_pair_timeframe(symbol: str, start_date: str, end_date: str, window_sizes: List[int]) -> Dict:
    """Analyze single pair across both timeframes"""
    try:
        print(f"\n=== Testing {symbol} ===\n")
        config_path = Path('config.yaml')
        
        # Validate window sizes
        valid_window_sizes = [24, 72, 120]  # Define valid window sizes
        if not all(size in valid_window_sizes for size in window_sizes):
            print("Warning: Invalid window sizes provided. Using default values.")
            window_sizes = valid_window_sizes
        
        # Load data for both timeframes
        df_15m, df_1h = load_and_prepare_data(symbol, start_date, end_date)
        
        # Reindex 1h data to match 15m timestamps
        df_1h_reindexed = df_1h.reindex(df_15m.index)
        df_1h_reindexed = df_1h_reindexed.fillna(method='ffill')
        
        print(f"Data loaded successfully:")
        print(f"15m Shape: {df_15m.shape}")
        print(f"1h Original Shape: {df_1h.shape}")
        print(f"1h Reindexed Shape: {df_1h_reindexed.shape}")
        
        # Add symbol column
        df_15m['symbol'] = symbol
        df_1h_reindexed['symbol'] = symbol
        
        # Calculate indicators
        calculator = IndicatorCalculator(df_15m, '15m', str(config_path))
        df_15m = calculator.calculate_all_indicators()
        
        calculator = IndicatorCalculator(df_1h_reindexed, '1h', str(config_path))
        df_1h_reindexed = calculator.calculate_all_indicators()
        
        # Initialize tester
        tester = RollingWindowTester(
            df=df_15m.copy(),
            timeframe='15m',
            config_path=str(config_path),
            window_sizes=window_sizes,
            use_gpu=True
        )
        
        # Define indicators to test
        indicators_config = {
            'bf_indicators': ['bf_RSI', 'bf_STOCH_K', 'bf_STOCH_D', 
            'bf_ADX', 'bf_VPCI', 'bf_VZO',
            'bf_OBV', 'bf_ROC_OBV', 'bf_N_OBV', 'bf_MACD_HG']
        }

        all_indicators = [ind for group in indicators_config.values() for ind in group]
        
        # Test each indicator
        results_dict = {}
        thresholds = {size: {} for size in window_sizes}  # Initialize thresholds structure
        
        for indicator in all_indicators:
            print(f"\nAnalyzing {indicator}...")
            thresholds_df = tester.analyze_windows(df_15m, df_1h_reindexed, indicator)
            
            # Debug print to verify window_idx is present
            if not thresholds_df.empty:
                print("\nDebug - First few rows of thresholds_df:")
                print(thresholds_df[['window_size', 'timeframe', 'direction', 'window_idx']].head())
            
            if not thresholds_df.empty:
                results_dict[indicator] = thresholds_df
                
                # Store thresholds with window_idx
                for _, row in thresholds_df.iterrows():
                    window_size = int(row['window_size'])
                    if window_size not in thresholds:
                        thresholds[window_size] = {}
                    
                    window_end = row['window_end']
                    if window_end not in thresholds[window_size]:
                        thresholds[window_size][window_end] = {}
                    
                    indicator_key = f"{row['timeframe']}_{indicator}"
                    if indicator_key not in thresholds[window_size][window_end]:
                        thresholds[window_size][window_end][indicator_key] = {}
                    
                    thresholds[window_size][window_end][indicator_key][row['direction']] = {
                        'window_idx': row['window_idx'],  # Make sure to include window_idx
                        'threshold': (row['threshold'], row['condition']),
                        'win_rate': row['win_rate'],
                        'is_optimal': row['win_rate'] > 0
                    }
        
        # Debug print to verify thresholds structure
        print("\nDebug - Thresholds structure:")
        for size in thresholds:
            print(f"\nWindow size {size}:")
            for timestamp in thresholds[size]:
                print(f"  Timestamp {timestamp}:")
                for indicator_key in thresholds[size][timestamp]:
                    print(f"    {indicator_key}: {thresholds[size][timestamp][indicator_key]}")
        
        return {
            'symbol': symbol,
            'data': df_15m,
            'data_15m': df_15m,
            'data_1h': df_1h_reindexed,
            'tester': tester,
            'thresholds': thresholds,
            'results': results_dict
        }
        
    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def compile_simulation_data(simulation_data):
    """Prepare simulation data from all symbols' results and F3 data"""
    try:
        # Extract components
        all_results = simulation_data['analysis_results']
        f3_results = simulation_data['f3_results']
        
        # Debug print initial data
        print("\nDEBUG - Initial Results Structure:")
        for idx, result in enumerate(all_results):
            print(f"\nResult {idx} - Symbol: {result['symbol']}")
            print(f"Threshold keys: {list(result['thresholds'].keys())}")
            for window_size in result['thresholds']:
                print(f"\nWindow {window_size}:")
                timestamps = list(result['thresholds'][window_size].keys())
                print(f"Number of timestamps: {len(timestamps)}")
                if timestamps:
                    print(f"First timestamp: {timestamps[0]} ({type(timestamps[0])})")
                    print(f"Sample data: {next(iter(result['thresholds'][window_size][timestamps[0]].items()))}")
        
        # First, aggregate results from all symbols
        aggregated_data = {
            'data_15m': {},  # Will store all symbols' 15m data
            'data_1h': {},   # Will store all symbols' 1h data
            'thresholds': {}, # Will store all thresholds by window size
            'F3A': f3_results['F3A'],  # F3A results
            'F3B': f3_results['F3B']   # F3B results
        }
        
        # Debug print before aggregation
        print("\nDEBUG - Before Aggregation:")
        print(f"Number of symbols to process: {len(all_results)}")
        
        # Aggregate data from each symbol
        for symbol_result in all_results:
            symbol = symbol_result['symbol']
            print(f"\nProcessing symbol: {symbol}")
            
            # Store price data by symbol
            aggregated_data['data_15m'][symbol] = symbol_result['data_15m']
            aggregated_data['data_1h'][symbol] = symbol_result['data_1h']
            
            # Merge thresholds
            symbol_thresholds = symbol_result['thresholds']
            for window_size in symbol_thresholds:
                if window_size not in aggregated_data['thresholds']:
                    aggregated_data['thresholds'][window_size] = {}
                    
                for window_end in symbol_thresholds[window_size]:
                    if window_end not in aggregated_data['thresholds'][window_size]:
                        aggregated_data['thresholds'][window_size][window_end] = {}
                        
                    # Store thresholds with symbol prefix
                    for indicator_key, indicator_data in symbol_thresholds[window_size][window_end].items():
                        symbol_indicator_key = f"{symbol}_{indicator_key}"
                        aggregated_data['thresholds'][window_size][window_end][symbol_indicator_key] = indicator_data

            # Debug print after processing each symbol's thresholds
            print(f"\nAfter processing {symbol} thresholds:")
            for window_size in aggregated_data['thresholds']:
                timestamps = list(aggregated_data['thresholds'][window_size].keys())
                print(f"Window {window_size}: {len(timestamps)} timestamps")
                if timestamps:
                    print(f"Sample indicator keys: {list(aggregated_data['thresholds'][window_size][timestamps[0]].keys())[:3]}")

        print("\nAggregated Data Summary:")
        print(f"Total symbols: {len(aggregated_data['data_15m'])}")
        print(f"Window sizes: {list(aggregated_data['thresholds'].keys())}")
        print(f"F3A pools: {list(aggregated_data['F3A'].keys())}")
        print(f"F3B pools: {list(aggregated_data['F3B'].keys())}")
        
        # Debug print before RollingWindowTester initialization
        first_symbol = list(aggregated_data['data_15m'].keys())[0]
        print("\nDEBUG - Before RollingWindowTester:")
        print(f"First symbol: {first_symbol}")
        print(f"Data shape: {aggregated_data['data_15m'][first_symbol].shape}")
        print("Threshold structure:")
        for window_size in aggregated_data['thresholds']:
            print(f"\nWindow {window_size}:")
            timestamps = list(aggregated_data['thresholds'][window_size].keys())
            print(f"Timestamps: {len(timestamps)}")
            if timestamps:
                print(f"First timestamp: {timestamps[0]} ({type(timestamps[0])})")
                print(f"Keys in first timestamp: {list(aggregated_data['thresholds'][window_size][timestamps[0]].keys())[:3]}")
        
        # Initialize results structure
        simulation_results = {
            'S2A': {
                'regime_sensitive_continuous': [],
                'regime_sensitive_target_triggered': [],
                'regime_independent_continuous': [],
                'regime_independent_target_triggered': []
            },
            'S2B': {
                'regime_sensitive_continuous': [],
                'regime_independent_continuous': [],
                'regime_sensitive_target_triggered': [],
                'regime_independent_target_triggered': []
            }
        }
        
        # Create RollingWindow instance once
        # Use the first symbol's data for initialization
        first_symbol = list(aggregated_data['data_15m'].keys())[0]
        first_symbol_data = aggregated_data['data_15m'][first_symbol]
        
        rw_tester = RollingWindowTester(
            df=first_symbol_data,  # Initialize with first symbol's data
            timeframe='15m',
            config_path='config.yaml',
            window_sizes=[24, 72, 120],
            use_gpu=True
        )
        
        # Add threshold setting and verification
        print("\nDEBUG - Setting thresholds in RollingWindowTester:")
        rw_tester.set_thresholds(aggregated_data['thresholds'])

        print("\nDEBUG - Verifying thresholds in RollingWindowTester:")
        for size in rw_tester.thresholds:
            timestamps = list(rw_tester.thresholds[size].keys())
            print(f"Window {size}: {len(timestamps)} timestamps")
            if timestamps:
                print(f"First timestamp: {timestamps[0]} ({type(timestamps[0])})")
                print(f"Sample data: {next(iter(rw_tester.thresholds[size][timestamps[0]].items()))}")

        # Process each account type ===============================================================================================
        for account_type in ['S2A','S2B']:
            print(f"\nProcessing {account_type} simulations...")

            # Get appropriate F3 data
            f3_data = aggregated_data[f'F3{account_type[-1]}']  # F3A for S2A, F3B for S2B
            
            # Debug print before test_combinations_parallel
            print("\nDEBUG - Before test_combinations_parallel:")
            print("Window data structure:")
            print(f"data_15m keys: {list(aggregated_data['data_15m'].keys())}")
            print(f"data_1h keys: {list(aggregated_data['data_1h'].keys())}")
            print(f"thresholds window sizes: {list(aggregated_data['thresholds'].keys())}")
            
            # Run trials through RollingWindow
            trial_results = rw_tester.test_combinations_parallel(
                window_data={
                    'data_15m': aggregated_data['data_15m'],
                    'data_1h': aggregated_data['data_1h'],
                    'thresholds': aggregated_data['thresholds'],
                    'regime_data': aggregated_data.get('regime_data'),
                    'f3_data': f3_data,
                    'account_type': account_type,
                    'strategy_mode': simulation_data['strategy_mode']
                }, 
                debug=False
            )
            
            # Accumulate results
            for trial_type in simulation_results[account_type].keys():
                if trial_results and trial_type in trial_results:
                    # Direct assignment since the structure is already correct
                    simulation_results[account_type][trial_type] = trial_results[trial_type]

        # Save results to database
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_path = f'results/simulation_results_{timestamp}.db'
        
        with sqlite3.connect(db_path) as conn:
            # Create all tables
            conn.executescript("""
                -- S2A Configuration Tables
                CREATE TABLE IF NOT EXISTS s2a_config_regime_independent_continuous (
                    block_id INTEGER,
                    slot_id INTEGER,
                    coin_name TEXT,
                    combo_id TEXT,
                    window_size INTEGER,
                    direction TEXT,
                    regime TEXT DEFAULT 'independent',
                    signal_1 TEXT,
                    signal_2 TEXT,
                    signal_3 TEXT,
                    signal_4 TEXT,
                    signal_5 TEXT,
                    signal_6 TEXT,
                    PRIMARY KEY (block_id, slot_id)
                );
                
                CREATE TABLE IF NOT EXISTS s2a_config_regime_independent_target_triggered (
                    block_id INTEGER,
                    slot_id INTEGER,
                    coin_name TEXT,
                    combo_id TEXT,
                    window_size INTEGER,
                    direction TEXT,
                    regime TEXT DEFAULT 'independent',
                    signal_1 TEXT,
                    signal_2 TEXT,
                    signal_3 TEXT,
                    signal_4 TEXT,
                    signal_5 TEXT,
                    signal_6 TEXT,
                    PRIMARY KEY (block_id, slot_id)
                );
                
                CREATE TABLE IF NOT EXISTS s2a_config_regime_sensitive_continuous (
                    block_id INTEGER,
                    slot_id INTEGER,
                    coin_name TEXT,
                    combo_id TEXT,
                    window_size INTEGER,
                    direction TEXT,
                    regime TEXT,
                    signal_1 TEXT,
                    signal_2 TEXT,
                    signal_3 TEXT,
                    signal_4 TEXT,
                    signal_5 TEXT,
                    signal_6 TEXT,
                    PRIMARY KEY (block_id, slot_id, regime)
                );
                
                CREATE TABLE IF NOT EXISTS s2a_config_regime_sensitive_target_triggered (
                    block_id INTEGER,
                    slot_id INTEGER,
                    coin_name TEXT,
                    combo_id TEXT,
                    window_size INTEGER,
                    direction TEXT,
                    regime TEXT,
                    signal_1 TEXT,
                    signal_2 TEXT,
                    signal_3 TEXT,
                    signal_4 TEXT,
                    signal_5 TEXT,
                    signal_6 TEXT,
                    PRIMARY KEY (block_id, slot_id, regime)
                );
                
                -- Same tables for S2B
                CREATE TABLE IF NOT EXISTS s2b_config_regime_independent_continuous (
                    block_id INTEGER,
                    slot_id INTEGER,
                    coin_name TEXT,
                    combo_id TEXT,
                    window_size INTEGER,
                    direction TEXT,
                    regime TEXT DEFAULT 'independent',
                    signal_1 TEXT,
                    signal_2 TEXT,
                    signal_3 TEXT,
                    signal_4 TEXT,
                    signal_5 TEXT,
                    signal_6 TEXT,
                    PRIMARY KEY (block_id, slot_id)
                );

                CREATE TABLE IF NOT EXISTS s2b_config_regime_independent_target_triggered (
                    block_id INTEGER,
                    slot_id INTEGER,
                    coin_name TEXT,
                    combo_id TEXT,
                    window_size INTEGER,
                    direction TEXT,
                    regime TEXT DEFAULT 'independent',
                    signal_1 TEXT,
                    signal_2 TEXT,
                    signal_3 TEXT,
                    signal_4 TEXT,
                    signal_5 TEXT,
                    signal_6 TEXT,
                    PRIMARY KEY (block_id, slot_id)
                );
                
                CREATE TABLE IF NOT EXISTS s2b_config_regime_sensitive_continuous (
                    block_id INTEGER,
                    slot_id INTEGER,
                    coin_name TEXT,
                    combo_id TEXT,
                    window_size INTEGER,
                    direction TEXT,
                    regime TEXT,
                    signal_1 TEXT,
                    signal_2 TEXT,
                    signal_3 TEXT,
                    signal_4 TEXT,
                    signal_5 TEXT,
                    signal_6 TEXT,
                    PRIMARY KEY (block_id, slot_id, regime)
                );
                
                CREATE TABLE IF NOT EXISTS s2b_config_regime_sensitive_target_triggered (
                    block_id INTEGER,
                    slot_id INTEGER,
                    coin_name TEXT,
                    combo_id TEXT,
                    window_size INTEGER,
                    direction TEXT,
                    regime TEXT,
                    signal_1 TEXT,
                    signal_2 TEXT,
                    signal_3 TEXT,
                    signal_4 TEXT,
                    signal_5 TEXT,
                    signal_6 TEXT,
                    PRIMARY KEY (block_id, slot_id, regime)
                );             

                -- Account Report Tables for S2A
                CREATE TABLE IF NOT EXISTS s2a_account_regime_independent_continuous (
                    date TEXT,
                    block_id INTEGER,
                    capital REAL,
                    trades_taken INTEGER,
                    target_hits INTEGER,
                    margin_used REAL,
                    free_margin REAL
                );
                
                CREATE TABLE IF NOT EXISTS s2a_account_regime_independent_target_triggered (
                    date TEXT,
                    block_id INTEGER,
                    capital REAL,
                    trades_taken INTEGER,
                    target_hits INTEGER,
                    margin_used REAL,
                    free_margin REAL
                );
                
                CREATE TABLE IF NOT EXISTS s2a_account_regime_sensitive_continuous (
                    date TEXT,
                    block_id INTEGER,
                    capital REAL,
                    trades_taken INTEGER,
                    target_hits INTEGER,
                    margin_used REAL,
                    free_margin REAL
                );
                
                CREATE TABLE IF NOT EXISTS s2a_account_regime_sensitive_target_triggered (
                    date TEXT,
                    block_id INTEGER,
                    capital REAL,
                    trades_taken INTEGER,
                    target_hits INTEGER,
                    margin_used REAL,
                    free_margin REAL
                );
                
                -- Slot State Tables for S2A
                CREATE TABLE IF NOT EXISTS s2a_slots_regime_independent_continuous (
                    block_id INTEGER,
                    slot_id INTEGER,
                    day INTEGER,
                    active BOOLEAN,
                    size REAL,
                    multiplier REAL,
                    slot_capital REAL,
                    won BOOLEAN,
                    triple_progress INTEGER,
                    trades INTEGER,
                    minutes_active INTEGER,
                    PRIMARY KEY (block_id, slot_id, day)
                );
                
                CREATE TABLE IF NOT EXISTS s2a_slots_regime_independent_target_triggered (
                    block_id INTEGER,
                    slot_id INTEGER,
                    day INTEGER,
                    active BOOLEAN,
                    size REAL,
                    multiplier REAL,
                    slot_capital REAL,
                    won BOOLEAN,
                    triple_progress INTEGER,
                    trades INTEGER,
                    minutes_active INTEGER,
                    PRIMARY KEY (block_id, slot_id, day)
                );
                
                CREATE TABLE IF NOT EXISTS s2a_slots_regime_sensitive_continuous (
                    block_id INTEGER,
                    slot_id INTEGER,
                    day INTEGER,
                    active BOOLEAN,
                    size REAL,
                    multiplier REAL,
                    slot_capital REAL,
                    won BOOLEAN,
                    triple_progress INTEGER,
                    trades INTEGER,
                    minutes_active INTEGER,
                    PRIMARY KEY (block_id, slot_id, day)
                );
                
                CREATE TABLE IF NOT EXISTS s2a_slots_regime_sensitive_target_triggered (
                    block_id INTEGER,
                    slot_id INTEGER,
                    day INTEGER,
                    active BOOLEAN,
                    size REAL,
                    multiplier REAL,
                    slot_capital REAL,
                    won BOOLEAN,
                    triple_progress INTEGER,
                    trades INTEGER,
                    minutes_active INTEGER,
                    PRIMARY KEY (block_id, slot_id, day)
                );
                
                -- Account Report Tables for S2B
                CREATE TABLE IF NOT EXISTS s2b_account_regime_independent_continuous (
                    date TEXT,
                    block_id INTEGER,
                    capital REAL,
                    trades_taken INTEGER,
                    target_hits INTEGER,
                    margin_used REAL,
                    free_margin REAL
                );
                
                CREATE TABLE IF NOT EXISTS s2b_account_regime_independent_target_triggered (
                    date TEXT,
                    block_id INTEGER,
                    capital REAL,
                    trades_taken INTEGER,
                    target_hits INTEGER,
                    margin_used REAL,
                    free_margin REAL
                );
                
                CREATE TABLE IF NOT EXISTS s2b_account_regime_sensitive_continuous (
                    date TEXT,
                    block_id INTEGER,
                    capital REAL,
                    trades_taken INTEGER,
                    target_hits INTEGER,
                    margin_used REAL,
                    free_margin REAL
                );
                
                CREATE TABLE IF NOT EXISTS s2b_account_regime_sensitive_target_triggered (
                    date TEXT,
                    block_id INTEGER,
                    capital REAL,
                    trades_taken INTEGER,
                    target_hits INTEGER,
                    margin_used REAL,
                    free_margin REAL
                );
                
                -- Slot State Tables for S2B
                CREATE TABLE IF NOT EXISTS s2b_slots_regime_independent_continuous (
                    block_id INTEGER,
                    slot_id INTEGER,
                    day INTEGER,
                    active BOOLEAN,
                    size REAL,
                    multiplier REAL,
                    slot_capital REAL,
                    won BOOLEAN,
                    triple_progress INTEGER,
                    trades INTEGER,
                    minutes_active INTEGER,
                    PRIMARY KEY (block_id, slot_id, day)
                );
                
                CREATE TABLE IF NOT EXISTS s2b_slots_regime_independent_target_triggered (
                    block_id INTEGER,
                    slot_id INTEGER,
                    day INTEGER,
                    active BOOLEAN,
                    size REAL,
                    multiplier REAL,
                    slot_capital REAL,
                    won BOOLEAN,
                    triple_progress INTEGER,
                    trades INTEGER,
                    minutes_active INTEGER,
                    PRIMARY KEY (block_id, slot_id, day)
                );
                
                CREATE TABLE IF NOT EXISTS s2b_slots_regime_sensitive_continuous (
                    block_id INTEGER,
                    slot_id INTEGER,
                    day INTEGER,
                    active BOOLEAN,
                    size REAL,
                    multiplier REAL,
                    slot_capital REAL,
                    won BOOLEAN,
                    triple_progress INTEGER,
                    trades INTEGER,
                    minutes_active INTEGER,
                    PRIMARY KEY (block_id, slot_id, day)
                );
                
                CREATE TABLE IF NOT EXISTS s2b_slots_regime_sensitive_target_triggered (
                    block_id INTEGER,
                    slot_id INTEGER,
                    day INTEGER,
                    active BOOLEAN,
                    size REAL,
                    multiplier REAL,
                    slot_capital REAL,
                    won BOOLEAN,
                    triple_progress INTEGER,
                    trades INTEGER,
                    minutes_active INTEGER,
                    PRIMARY KEY (block_id, slot_id, day)
                );
            """)
            
            # Process results with new table organization
            for account_type in simulation_results.keys():
                prefix = 's2a' if account_type == 'S2A' else 's2b'
                
                for trial_type, trial_data in simulation_results[account_type].items():
                    if not trial_data:  # Skip if no data
                        continue
                        
                    trial_suffix = trial_type.replace('-', '_')
                    
                    # Process configurations from metadata
                    metadata = trial_data.get('metadata', {})
                    units_data = metadata.get('units', {})
                    
                    for unit_id, unit_data in units_data.items():
                        regime_data = unit_data.get('regime_data', {})
                        
                        print(f"\nDebug - Processing configs for {account_type} {trial_type}")
                        print(f"Unit {unit_id} regime data: {regime_data}")
                        
                        # Process each regime's configuration
                        for regime, config in regime_data.items():
                            if account_type == 'S2A':
                                # S2A: One combo with multiple coins
                                combo_id = config['combo_id']
                                window_size = config['window_size']
                                direction = config['direction']
                                signals = config['signals']  # Already in correct format from test_combinations_parallel
                                signal_names = signals + [''] * (6 - len(signals))  # Pad to 6
                                
                                # For each coin in top_3_coins
                                for slot_id, coin in enumerate(config.get('top_3_coins', [])):
                                    config_record = (
                                        int(unit_id),  # block_id
                                        int(slot_id),
                                        coin,
                                        str(combo_id),
                                        int(window_size),
                                        'long' if direction == 1 else 'short',
                                        regime,
                                        *signal_names
                                    )
                                    conn.execute(f"INSERT OR REPLACE INTO {prefix}_config_{trial_suffix} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                                               config_record)
                            else:  # S2B
                                # S2B: One coin with multiple combos
                                coin = config['coin']
                                
                                # For each combo in top_3_combos
                                for slot_id, combo in enumerate(config.get('top_3_combos', [])):
                                    signals = combo['signals']  # Already in correct format
                                    signal_names = signals + [''] * (6 - len(signals))  # Pad to 6
                                    
                                    config_record = (
                                        int(unit_id),  # block_id
                                        int(slot_id),
                                        coin,
                                        str(combo['combo_id']),
                                        int(combo['window_size']),
                                        'long' if combo['direction'] == 1 else 'short',
                                        regime,
                                        *signal_names
                                    )
                                    conn.execute(f"INSERT OR REPLACE INTO {prefix}_config_{trial_suffix} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                                               config_record)
                    
                    # Process performance results
                    results_data = trial_data.get('results', [])
                    for result in results_data:
                        if isinstance(result, dict) and 'summary' in result:
                            # Insert into account table
                            for day_summary in result['summary']:
                                mother_account = day_summary['mother_account']
                                date = day_summary['date']['test_day_start'].strftime('%Y-%m-%d')
                                
                                summary_record = (
                                    date,
                                    int(day_summary['unit']),
                                    float(mother_account['current_capital']),
                                    int(len([t for t in result['trades'] if t['day'] == day_summary['day']])),
                                    int(mother_account['targets_reached']),
                                    float(mother_account['margin_used']),
                                    float(mother_account['free_margin'])
                                )
                                conn.execute(f"INSERT INTO {prefix}_account_{trial_suffix} VALUES (?, ?, ?, ?, ?, ?, ?)", 
                                           summary_record)
                                
                            # Insert into slot table
                            for day_summary in result['summary']:
                                unit = day_summary['unit']
                                day = day_summary['day']
                                
                                day_slots = {
                                    (t['slot'].split('_')[1], t['day']): t 
                                    for t in result['trades'] 
                                    if t['unit'] == unit and t['day'] == day
                                }
                                
                                for slot_id in range(3):
                                    slot_data = day_slots.get((str(slot_id), day), {
                                        'active': False,
                                        'trade_size': 0.0,
                                        'return_multiplier': 0.0,
                                        'slot_capital': 0.0,
                                        'won': False,
                                        'triple_progress': 0,
                                        'completed_trades': 0,
                                        'minutes_active': 0
                                    })
                                    
                                    slot_record = (
                                        int(unit),
                                        int(slot_id),
                                        int(day),
                                        bool(slot_data['active']),
                                        float(slot_data['trade_size']),
                                        float(slot_data['return_multiplier']),
                                        float(slot_data['slot_capital']),
                                        bool(slot_data['won']),
                                        int(slot_data['triple_progress']),
                                        int(slot_data['completed_trades']),
                                        int(slot_data['minutes_active'])
                                    )
                                    
                                    conn.execute(f"INSERT INTO {prefix}_slots_{trial_suffix} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                                               slot_record)
                            
                            conn.commit()
        
        return db_path
        
    except Exception as e:
        print(f"Error in compile_simulation_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    start_date = '2024-02-01' #inclusive
    end_date = '2024-12-31' #inclusive

    # Load pre-computed results instead of analyzing
    from loading_threshold import load_thresholds
    db_path = "data/thresholds/thresholds_Sim.db"
    all_results = load_thresholds(db_path, start_date, end_date)
    
    if not all_results:
        print("Failed to load threshold results")
        return None

    # Get F3 results from filter.py
    from filter import filter_results
    f3_results = filter_results()
    print("\nDebugging F3 Results:")
    print("\nF3 Results Structure:", type(f3_results))
    
    if isinstance(f3_results, dict):
        print("\nF3A Structure:")
        if 'F3A' in f3_results:
            print("- Pool 1 (Regime-sensitive):")
            for regime in ['bull', 'bear', 'flat']:
                if regime in f3_results['F3A']['pool_1']:
                    print(f"  {regime}: {len(f3_results['F3A']['pool_1'][regime])} combos")
            print("- Pool 2 (Regime-independent):")
            if 'combos' in f3_results['F3A']['pool_2']:
                print(f"  {len(f3_results['F3A']['pool_2']['combos'])} combos")
        
        print("\nF3B Structure:") 
        if 'F3B' in f3_results:
            print("- Pool 1 (Regime-sensitive):")
            for regime in ['bull', 'bear', 'flat']:
                if regime in f3_results['F3B']['pool_1']:
                    print(f"  {regime}: {len(f3_results['F3B']['pool_1'][regime])} coins")
            print("- Pool 2 (Regime-independent):")
            if 'coins' in f3_results['F3B']['pool_2']:
                print(f"  {len(f3_results['F3B']['pool_2']['coins'])} coins")
    else:
        print("F3 results is not a dictionary")
    
    if f3_results is None:
        print("Failed to get F3 results")
        return None
    
    # Compile final results to database
    if all_results:
        # Add F3 results to the simulation data
        simulation_data = {
            'analysis_results': all_results,
            'f3_results': f3_results,  # This contains the structured F3A and F3B data
            'strategy_mode': 'triple'  # Add strategy mode to simulation data
        }
        

        db_path = compile_simulation_data(simulation_data)
        print(f"Final results saved to: {db_path}")
        return db_path
    else:
        print("No results to compile")
        return None

if __name__ == "__main__":
    main()