from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
from tqdm import tqdm
from indicator_tester import IndicatorTester, process_trade, process_trade_kernel
from simulator_triple import Simulator
from numba import jit, cuda
from itertools import combinations, product
from pathlib import Path
from utils import get_regime_data

class RollingWindowTester:
    def __init__(self, df: pd.DataFrame, timeframe: str, config_path='config/config.yaml', 
                 window_sizes: List[int] = [24, 72, 120], use_gpu: bool = True):
        """Initialize the rolling window tester
        
        Args:
            df: DataFrame with price and indicator data
            timeframe: Data timeframe (e.g. '15m', '1h')
            config_path: Path to config file
            window_sizes: List of window sizes in candles (24=1day, 72=3days, 120=5days)
            use_gpu: Whether to use GPU acceleration
        """
        self.df = df
        self.timeframe = timeframe
        self.window_sizes = window_sizes  # Already in candles, no conversion needed
        self.use_gpu = use_gpu
        self.indicator_tester = IndicatorTester(df, timeframe, config_path)
        
        # Initialize global index tracking
        self.global_start_idx = 0  # Start from beginning of dataframe
        self.total_candles = len(df)
        self.candles_per_day = 96  # For 15m timeframe (24 hours * 4 candles per hour)
        
        with open('config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        
        # GPU diagnostics
        print("\nGPU Diagnostics:")
        print(f"CUDA Available: {cuda.is_available()}")
        if cuda.is_available():
            print(f"CUDA Device: {cuda.get_current_device().name}")
            print(f"Compute Capability: {cuda.get_current_device().compute_capability}")
        print(f"Using GPU: {self.use_gpu}")
        
        # Initialize thresholds with empty dicts
        self.thresholds = {size: {} for size in self.window_sizes}
        
        # Debug print thresholds structure after loading
        print("\nInitial Threshold Structure:")
        for size in self.window_sizes:
            print(f"Window {size}: {len(self.thresholds[size])} entries")
        
        self.unresolved_trades = {
            size: {'long': [], 'short': []} for size in self.window_sizes
        }
        self.combination_results = {}
        self.current_combos = None
        
        self._load_config(config_path)
        
        if self.use_gpu:
            # Pre-load data to GPU memory
            self.gpu_prices = cuda.to_device(self.df['close'].values.astype(np.float32))
            self.gpu_highs = cuda.to_device(self.df['high'].values.astype(np.float32))
            self.gpu_lows = cuda.to_device(self.df['low'].values.astype(np.float32))
        
        # Add new structures for tracking results and pending trades
        self.window_results = {}  # Store results per combo per window
        self.pending_trades = {
            size: {} for size in self.window_sizes
        }

        
        self.debug = False
        
    def _load_config(self, config_path: str) -> None:
        """Load trading parameters from config"""
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        self.target_increase = config['parameters']['target_increase']
        self.max_retracement = config['parameters']['max_retracement']
        self.stage_increment = config['parameters']['stage_increment']
        # Add new parameter with default value of 1
        self.max_simultaneous_trades = config['parameters'].get('max_simultaneous_trades', 1)

        # Calculate stage levels multipliers
        self.stage_multipliers = [
            1 + (self.stage_increment * (i + 1)) 
            for i in range(int(self.target_increase / self.stage_increment))
        ]
    
    def set_thresholds(self, thresholds: Dict):
        """Set thresholds after loading from database"""
        self.thresholds = thresholds
        
        # Debug print threshold structure after setting
        print("\nThreshold Structure After Setting:")
        for size in self.window_sizes:
            if size in self.thresholds:
                dates = sorted(self.thresholds[size].keys())
                print(f"\nWindow {size}:")
                print(f"Total dates: {len(dates)}")
                if dates:
                    print(f"Date range: {dates[0]} to {dates[-1]}")
    
    def _get_valid_test_days(self, data_15m, window_1_start, test_end_date):
        """
        Determine valid test days and establish the authoritative simulation period.
        Returns both valid test days and the definitive simulation period.
        """
        valid_test_days = []
        
        # Get all threshold dates for each window size
        threshold_dates_by_size = {}
        for size in self.window_sizes:
            if size in self.thresholds:
                threshold_dates_by_size[size] = sorted([
                    date for date in self.thresholds[size].keys()
                    if isinstance(date, pd.Timestamp)
                ])
        
        if not all(threshold_dates_by_size.values()):
            print("ERROR: Missing thresholds for some window sizes")
            return [], None, None
        
        # Find the common valid date range
        latest_start = max(dates[0] for dates in threshold_dates_by_size.values())
        earliest_end = min(dates[-1] for dates in threshold_dates_by_size.values())
        
        # Normalize intended start date
        intended_start = pd.Timestamp(window_1_start).normalize()
        
        # Check if intended start date has valid thresholds
        intended_threshold_date = intended_start - pd.Timedelta(minutes=15)
        intended_start_valid = all(
            intended_threshold_date in self.thresholds[size]
            for size in self.window_sizes
        )
        
        # Use intended start if valid, otherwise use latest threshold start + 1 day
        actual_sim_start = intended_start if intended_start_valid else (latest_start + pd.Timedelta(days=1)).normalize()
        actual_sim_end = min(pd.Timestamp(test_end_date).normalize(), earliest_end + pd.Timedelta(days=1))
        
        print(f"\n=== Authoritative Simulation Period ===")
        print(f"Original requested period: {window_1_start} to {test_end_date}")
        print(f"Adjusted simulation start: {actual_sim_start}")
        print(f"Adjusted simulation end: {actual_sim_end}")
        
        # Generate valid test days
        current_date = actual_sim_start
        while current_date.date() <= actual_sim_end.date():
            test_day_start = current_date.normalize()
            test_day_end = test_day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
            threshold_date = test_day_start - pd.Timedelta(minutes=15)
            
            try:
                test_day_global_start = data_15m.index.get_loc(test_day_start)
                test_day_global_end = test_day_global_start + self.candles_per_day
                
                # Verify threshold availability for all window sizes
                threshold_available = all(
                    threshold_date in self.thresholds[size]
                    for size in self.window_sizes
                )
                
                if threshold_available:
                    test_day_info = {
                        'test_day_start': test_day_start,
                        'test_day_end': test_day_end,
                        'global_start_idx': test_day_global_start,
                        'global_end_idx': test_day_global_end,
                        'threshold_date': threshold_date
                    }
                    valid_test_days.append(test_day_info)
                    print(f"Added valid test day: {test_day_start.date()} (threshold date: {threshold_date})")
            
            except KeyError:
                print(f"Data not found for date {test_day_start}")
            
            current_date += pd.Timedelta(days=1)
        
        print(f"\nFound {len(valid_test_days)} valid test days")
        
        return valid_test_days, actual_sim_start, actual_sim_end
        
    def _prepare_combo_data(self, window_data, signals, coins, start_date, end_date, current_rw_results, regime_data):
        """Prepare data for combo testing with proper timeframe mapping"""
        try:
            print(f"\n=== Preparing Data for Authorized Simulation Period ===")
            sim_start = pd.to_datetime(start_date)
            sim_end = pd.to_datetime(end_date)
            
            # Define earliest_window_0_end at the start
            earliest_window_0_end = sim_start - pd.Timedelta(days=1)
            
            # Define window size mapping
            window_size_to_idx = {24: 0, 72: 1, 120: 2}
            
            # Calculate threshold period
            threshold_start = (sim_start - pd.Timedelta(days=1)).normalize()  # Start of day before first sim day
            threshold_end = (sim_end - pd.Timedelta(days=1)).normalize() + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
            
            print(f"\nSimulation Period:")
            print(f"Sim days: {sim_start.date()} to {sim_end.date()}")
            print(f"\nRequired Threshold Period:")
            print(f"From: {threshold_start} to {threshold_end}")
            print(f"(Need thresholds ending at 23:45 for each day in this period)")
            
            # Verify we have all required threshold timestamps
            expected_timestamps = []
            current_date = threshold_start
            while current_date <= threshold_end:
                threshold_timestamp = current_date + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
                expected_timestamps.append(threshold_timestamp)
                current_date += pd.Timedelta(days=1)
            
            # Collect timestamps within our required period and verify completeness
            all_timestamps = set()
            missing_timestamps = set()
            
            for window_size in [24, 72, 120]:
                if window_size in current_rw_results:
                    for expected_ts in expected_timestamps:
                        # Check if this timestamp exists in current_rw_results for this window size
                        if expected_ts not in current_rw_results[window_size]:
                            missing_timestamps.add((window_size, expected_ts))
                        else:
                            # Only add to valid timestamps if it exists for all window sizes
                            if all(
                                expected_ts in current_rw_results[other_size]
                                for other_size in [24, 72, 120]
                                if other_size in current_rw_results
                            ):
                                all_timestamps.add(expected_ts)
            
            all_timestamps = sorted(all_timestamps)
            
            
            if missing_timestamps:
                print("\nMissing thresholds:")
                for window_size, ts in sorted(missing_timestamps):
                    print(f"Window {window_size}: {ts}")
                raise ValueError("Missing required thresholds for simulation period")
            
            # Debug the input data
            print("\n=== Input Data Check ===")
            print(f"Window data keys: {window_data.keys()}")
            print(f"Window data '15m' keys: {window_data['15m'].keys()}")
            print(f"Window data '1h' keys: {window_data['1h'].keys()}")
            print(f"First coin data shape: {next(iter(window_data['15m'].values())).shape}")
            # Print columns for each timeframe
            print("\nColumns in 15m data:")
            print(next(iter(window_data['15m'].values())).columns.tolist())
            
            print("\nColumns in 1h data:")
            print(next(iter(window_data['1h'].values())).columns.tolist())
            
            # Debug signals format
            print("\nSignals to prepare:", signals)
            
            # Create signal mappings for both directions at the beginning
            # Sort signals first to ensure consistent indexing across runs
            sorted_signals = sorted(signals)
            signal_map = {}
            all_signals_with_direction = []
            
            # Assign indices sequentially based on sorted signals
            for idx, base_signal in enumerate(sorted_signals):
                # Long signal gets even index
                long_idx = idx * 2
                signal_map[f'{base_signal}_long'] = {
                    'name': f'{base_signal}_long', 
                    'idx': long_idx,
                    'base_signal': base_signal,
                    'direction': 'long'
                }
                all_signals_with_direction.append(f'{base_signal}_long')
                
                # Short signal gets odd index
                short_idx = idx * 2 + 1
                signal_map[f'{base_signal}_short'] = {
                    'name': f'{base_signal}_short',
                    'idx': short_idx, 
                    'base_signal': base_signal,
                    'direction': 'short'
                }
                all_signals_with_direction.append(f'{base_signal}_short')
            
            print("\nSignal Mapping Created:")
            for signal_name, signal_info in sorted(signal_map.items()):
                print(f"{signal_name}: idx={signal_info['idx']}")

            print(f"\n=== Preparing Full Period Combo Data ===")
            print(f"Full Period: {start_date} to {end_date}")
            
            window_sizes = [24, 72, 120]
            # Start from the day before simulation start to capture first day's threshold
            n_signal_minutes = int((end_date - earliest_window_0_end).total_seconds() / 60)
            
            print(f"Earliest window end: {earliest_window_0_end}")
            print(f"Number of signal minutes: {n_signal_minutes}")
            
            # 1. PREPARE PRICE DATA FOR ALL COINS
            price_data = {}
            atr_values = {}
            sliced_regime_data = {}
            
            print(f"\nPreparing price, ATR, and regime data for {len(coins)} coins:")
            for symbol in sorted(coins):
                print(f"Processing {symbol}...")
                try:
                    # Load 1m data for the symbol
                    data_path_1m = Path('data/raw') / f'{symbol}_1m_{self.config["parameters"]["simulation_start_date"]}_{self.config["parameters"]["end_date"]}.csv'
                    df_1m_symbol = pd.read_csv(data_path_1m)
                    df_1m_symbol['timestamp'] = pd.to_datetime(df_1m_symbol['timestamp'])
                    df_1m_symbol.set_index('timestamp', inplace=True)
                    
                    # Get the relevant period
                    symbol_future_df = df_1m_symbol[earliest_window_0_end:end_date]
                    
                    # Prepare price array
                    reindexed_values = np.zeros((n_signal_minutes, 5), dtype=np.float32)
                    for i, col in enumerate(['open', 'high', 'low', 'close', 'volume']):
                        values = symbol_future_df[col].values
                        if len(values) > n_signal_minutes:
                            values = values[:n_signal_minutes]
                        elif len(values) < n_signal_minutes:
                            pad_length = n_signal_minutes - len(values)
                            values = np.pad(values, (0, pad_length), 'edge')
                        reindexed_values[:, i] = values
                    
                    price_data[symbol] = reindexed_values
                    
                    # Slice regime data for this symbol
                    if symbol in regime_data:
                        full_regime = regime_data[symbol]
                        sliced_regime = full_regime[-n_signal_minutes:]  # Take last n_signal_minutes
                        if len(sliced_regime) < n_signal_minutes:
                            pad_length = n_signal_minutes - len(sliced_regime)
                            sliced_regime = np.pad(sliced_regime, (0, pad_length), 'edge')
                        sliced_regime_data[symbol] = sliced_regime
                    
                    # Get 1H ATR values and reindex to 1-minute
                    atr_1h = window_data['1h'][symbol]['ATR'][earliest_window_0_end:end_date].values
                    reindexed_atr = np.repeat(atr_1h, 60)  # 1h to 1m
                    
                    if len(reindexed_atr) > n_signal_minutes:
                        reindexed_atr = reindexed_atr[:n_signal_minutes]
                    elif len(reindexed_atr) < n_signal_minutes:
                        pad_length = n_signal_minutes - len(reindexed_atr)
                        reindexed_atr = np.pad(reindexed_atr, (0, pad_length), 'edge')
                    
                    atr_values[symbol] = reindexed_atr
                    print(f"Prepared {symbol} price data shape: {reindexed_values.shape}, ATR shape: {reindexed_atr.shape}")
                except Exception as e:
                    print(f"Error preparing data for {symbol}: {str(e)}")
                    continue

            # 2. PREPARE SIGNAL DATA FOR ALL COINS
            print("\nPreparing signal data for all coins...")
            # Initialize as [n_signals*2, n_coins, n_signal_minutes] to match threshold order
            signal_values_np = np.zeros((len(signals) * 2, len(coins), n_signal_minutes), dtype=np.float32)
            
            for coin_idx, symbol in enumerate(sorted(coins)):
                try:
                    # Get signal data for current symbol
                    signal_df_15m = window_data['15m'][symbol][earliest_window_0_end:end_date]
                    signal_df_1h = window_data['1h'][symbol][earliest_window_0_end:end_date]
                    
                    # Process each signal
                    for signal in signals:
                        try:
                            # Split timeframe and base name
                            timeframe, *rest = signal.split('_', 1)
                            base_signal = rest[0]
                            
                            # Get correct dataframe and multiplier
                            if timeframe == '15m':
                                df = signal_df_15m
                                multiplier = 15
                            else:  # '1h'
                                df = signal_df_1h
                                multiplier = 60
                            
                            if base_signal in df.columns:
                                # Get original values and reindex to 1m
                                orig_values = df[base_signal].values
                                reindexed_values = np.repeat(orig_values, multiplier)
                                
                                # Ensure correct length
                                if len(reindexed_values) > n_signal_minutes:
                                    reindexed_values = reindexed_values[:n_signal_minutes]
                                elif len(reindexed_values) < n_signal_minutes:
                                    pad_length = n_signal_minutes - len(reindexed_values)
                                    reindexed_values = np.pad(reindexed_values, (0, pad_length), 'edge')
                                
                                # Store values at correct signal indices for this coin
                                long_idx = signal_map[f'{signal}_long']['idx']
                                short_idx = signal_map[f'{signal}_short']['idx']
                                signal_values_np[long_idx, coin_idx] = reindexed_values
                                signal_values_np[short_idx, coin_idx] = reindexed_values
                            else:
                                print(f"Warning: {base_signal} not found in data for {symbol}")
                                
                        except Exception as e:
                            print(f"Error processing signal {signal} for {symbol}: {str(e)}")
                            continue
                    
                    print(f"Prepared signals for {symbol}")
                    
                except Exception as e:
                    print(f"Error preparing signals for {symbol}: {str(e)}")
                    continue

            print(f"\nFinal signal values shape: {signal_values_np.shape}")
            print("Signal values organization:")
            print(f"Dimension 0 (signals): {signal_values_np.shape[0]} signal directions")
            print(f"Dimension 1 (coins): {signal_values_np.shape[1]} coins")
            print(f"Dimension 2 (time): {signal_values_np.shape[2]} minutes")

            # Create timeframe mapping array of timestamps
            timeframe_map = np.array([
                start_date + pd.Timedelta(minutes=i) 
                for i in range(n_signal_minutes)
            ], dtype='datetime64[ns]')
            
            print("\n=== Data Preparation Summary ===")
            print(f"Number of coins: {len(price_data)}")
            print(f"Number of signals: {len(signals)}")
            print(f"Signal period: {earliest_window_0_end} to {end_date}")

            # Process thresholds for all window sizes and signals
            print("\nProcessing thresholds for all window sizes and signals...")

            # Get simulation period boundaries
            sim_start = pd.to_datetime(start_date)
            sim_end = pd.to_datetime(end_date)
            
            # Threshold calculation timestamps (23:45 of each day)
            threshold_start = sim_start - pd.Timedelta(days=1)  # Start from day before
            threshold_dates = [
                threshold_start + pd.Timedelta(days=i) 
                for i in range((sim_end - threshold_start).days + 1)  # +1 to include end date
            ]
            threshold_times = [
                date.replace(hour=23, minute=45) 
                for date in threshold_dates
            ]
            
            print(f"\nThreshold Preparation:")
            print(f"Signal Period: {sim_start} to {sim_end}")
            print(f"Threshold Calculations:")
            for t in threshold_times:
                print(f"  {t}")
            
            # Create threshold array matching signal minutes
            thresholds = np.zeros((
                len(coins),
                3,  # window sizes
                len(signals) * 2,  # signals * directions
                n_signal_minutes,  # Same as signal array
                2  # [value, condition]
            ), dtype=np.float32)
            
            print(f"\nAllocated threshold array for {n_signal_minutes} minutes")
            print(f"Starting from: {threshold_start}")
            print(f"Ending at: {threshold_end}")
            
            # Map window sizes to indices
            window_size_to_idx = {24: 0, 72: 1, 120: 2}
            
            print("\nProcessing thresholds:")
            # Process each window size
            for window_size in [24, 72, 120]:
                if window_size not in current_rw_results:
                    print(f"Warning: No data for window size {window_size}")
                    continue
                    
                window_idx = window_size_to_idx[window_size]
                print(f"\nProcessing window size: {window_size}h")
                
                # Track last valid thresholds for each combination
                last_valid_thresholds = {}  # (coin, signal, direction) -> (value, condition)
                
                # Process each timestamp in chronological order
                for timestamp in sorted(current_rw_results[window_size].keys()):
                    timestamp_dt = pd.to_datetime(timestamp)
                    minute_start_idx = int((timestamp_dt - earliest_window_0_end).total_seconds() / 60)
                    
                    # Fill from previous minute to current with last known threshold
                    for coin in sorted(coins):
                        for base_signal in signals:
                            full_signal = f"{coin}_{base_signal}"
                            
                            for direction in ['long', 'short']:
                                key = (coin, base_signal, direction)
                                if key in last_valid_thresholds:
                                    prev_value, prev_condition = last_valid_thresholds[key]
                                    # Fill gap with previous threshold
                                    coin_idx = sorted(coins).index(coin)
                                    signal_with_direction = f'{base_signal}_{direction}'
                                    signal_idx = signal_map[signal_with_direction]['idx']
                                    
                                    thresholds[
                                        coin_idx,
                                        window_idx,
                                        signal_idx,
                                        :minute_start_idx,  # Fill up to current minute
                                        0  # value
                                    ] = prev_value
                                    
                                    thresholds[
                                        coin_idx,
                                        window_idx,
                                        signal_idx,
                                        :minute_start_idx,  # Fill up to current minute
                                        1  # condition
                                    ] = prev_condition
                    
                    # Process current timestamp's thresholds
                    minute_end_idx = min(minute_start_idx + 1440, n_signal_minutes)
                    
                    # Get the results for this timestamp
                    timestamp_results = current_rw_results[window_size][timestamp]
                    
                    # Process each coin and base signal
                    for coin in sorted(coins):
                        for base_signal in signals:
                            full_signal = f"{coin}_{base_signal}"
                            
                            # Check if we have data for this signal
                            if full_signal not in timestamp_results:
                                continue
                            
                            signal_data = timestamp_results[full_signal]
                            coin_idx = sorted(coins).index(coin)
                            
                            # Process both directions
                            for direction in ['long', 'short']:
                                if direction not in signal_data:
                                    continue
                                
                                result = signal_data[direction]
                                if result['threshold'][0] is not None and not np.isnan(result['threshold'][0]):
                                    threshold_value = result['threshold'][0]
                                    threshold_condition = 0 if result['threshold'][1] == 'greater' else 1
                                    
                                    # Store current threshold for future gap filling
                                    key = (coin, base_signal, direction)
                                    last_valid_thresholds[key] = (threshold_value, threshold_condition)
                                    
                                    # Get the correct signal index from signal_map
                                    signal_with_direction = f'{base_signal}_{direction}'
                                    signal_idx = signal_map[signal_with_direction]['idx']
                                    
                                    # Fill threshold for this specific coin and time period
                                    thresholds[
                                        coin_idx,
                                        window_idx,
                                        signal_idx,
                                        minute_start_idx:minute_end_idx,
                                        0  # value
                                    ] = threshold_value
                                    
                                    thresholds[
                                        coin_idx,
                                        window_idx,
                                        signal_idx,
                                        minute_start_idx:minute_end_idx,
                                        1  # condition
                                    ] = threshold_condition
                
                print(f"Processed {len(current_rw_results[window_size].keys())} timestamps for window {window_size}h")
                print(f"Non-zero thresholds for window {window_size}h:", 
                      np.count_nonzero(thresholds[:, window_idx, :, :, 0]))
            
            # Print sample values for each dimension with names
            print("\nSample values for each dimension:")
            print("Coins (dim 0):")
            for i, coin in enumerate(sorted(coins)):
                has_data = np.any(thresholds[i,...,0] != 0)
                print(f"  {coin}: {'✓' if has_data else '✗'}")
            
            print("\nSignals (dim 2):")
            for i, signal in enumerate(all_signals_with_direction):
                has_data = np.any(thresholds[:,:,i,...,0] != 0)
                print(f"  {signal}: {'✓' if has_data else '✗'}")
            
            print("\nWindows (dim 1):")
            window_sizes = [24, 72, 120]
            for i, size in enumerate(window_sizes):
                has_data = np.any(thresholds[:,i,...,0] != 0)
                print(f"  {size}h: {'✓' if has_data else '✗'}")
            
            if np.any(thresholds[...,0] != 0):
                print("\nThreshold Info:")
                print("  - First 3 threshold values:", thresholds[...,0][thresholds[...,0] != 0][:3])
                print("  - First 3 conditions:", thresholds[...,1][thresholds[...,1] != 0][:3])

            # Add verification of signal order
            print("\nVerifying signal order consistency:")
            print("Signal Values Array Shape:", signal_values_np.shape)

            # Convert all data to numpy arrays before returning
            # 1. Convert price_data to numpy array [n_coins, n_candles, 5(OHLCV)]
            price_data_np = np.zeros((len(coins), n_signal_minutes, 5), dtype=np.float32)
            for idx, (coin, data) in enumerate(sorted(price_data.items())):
                price_data_np[idx] = data
            
            # 2. ATR values to numpy array [n_coins, n_candles]
            atr_values_np = np.zeros((len(coins), n_signal_minutes), dtype=np.float32)
            for idx, (coin, data) in enumerate(sorted(atr_values.items())):
                atr_values_np[idx] = data
            
            # 3. Regime data to numpy array [n_coins, n_candles]
            regime_data_np = np.zeros((len(coins), n_signal_minutes), dtype=np.int32)
            
            # Process regime data for each coin
            for coin_idx, coin in enumerate(sorted(coins)):
                if coin in sliced_regime_data:
                    # Get this coin's regime data
                    coin_regime = sliced_regime_data[coin]
                    
                    # Fill regime data for this coin's timepoints
                    for t in range(n_signal_minutes):
                        if t < len(coin_regime):
                            regime_data_np[coin_idx, t] = coin_regime[t]
                        else:
                            # Use last known regime for this coin if beyond data
                            regime_data_np[coin_idx, t] = regime_data_np[coin_idx, t-1] if t > 0 else 0
                else:
                    print(f"{coin} is not included in this trial")
            
            print("\nRegime data shape:", regime_data_np.shape)
            print("Sample regime distributions:")
            for coin_idx, coin in enumerate(sorted(coins)):
                unique, counts = np.unique(regime_data_np[coin_idx], return_counts=True)
                print(f"{coin}: {dict(zip(unique, counts))}")
            
            # 4. Timeframe map is already numpy array
            timeframe_map_np = timeframe_map  # Just rename for consistency
            
            # 5. signal_values_np is already in correct format [n_signals*2, n_coins, n_candles]
            
            # 6. thresholds is already in correct format
            
            print("\nFinal Array Shapes:")
            print(f"Price Data: {price_data_np.shape}")
            print(f"Signal Values: {signal_values_np.shape}")
            print(f"ATR Values: {atr_values_np.shape}")
            print(f"Regime Data: {regime_data_np.shape}")
            print(f"Timeframe Map: {timeframe_map_np.shape}")
            print(f"Thresholds: {thresholds.shape}")
            
            # Print sample values to verify data
            print("\nSample Values Check:")
            print(f"First coin, first signal long values (first 5): {signal_values_np[0, 0, :5]}")
            print(f"First coin, first signal short values (first 5): {signal_values_np[1, 0, :5]}")
            print(f"First coin thresholds (first window, first signal): {thresholds[0, 0, 0, :5]}")

            print("\nDEBUG: current_rw_results structure:")
            print("Window sizes:", list(current_rw_results.keys()))

            return {
                'price_data': price_data_np,
                'signal_values': signal_values_np,
                'atr_values': atr_values_np,
                'timeframe_map': timeframe_map_np,
                'thresholds': thresholds,
                'regime_data': regime_data_np,  # Now correctly shaped
                'coins': sorted(coins),
                'signals': sorted(signals),
                'signal_map': signal_map,
                'all_signals_with_direction': all_signals_with_direction,
                'window_sizes': self.window_sizes
            }
            
        except Exception as e:
            print(f"Error in _prepare_combo_data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_f3a_combo(self, combo_data: Dict) -> Dict:
        """Process F3A combo data to extract signals and parameters"""
        try:
            # Extract basic combo info
            combo_id = combo_data['combo_id']
            direction = combo_data['direction']
            window_size = combo_data['window_size']
            
            # Extract signals
            signals = []
            for signal_name in combo_data['signals']:
                parts = signal_name.split('_')
                timeframe = parts[0]
                base_name = '_'.join(parts[1:-1]) if parts[-1] in ['long', 'short'] else '_'.join(parts[1:])
                
                signals.append({
                    'name': signal_name,
                    'timeframe': timeframe,
                    'base_name': base_name,
                    'direction': direction
                })
            
            return {
                'combo_id': combo_id,
                'direction': direction,
                'window_size': window_size,
                'signals': signals,
                'top_3_coins': combo_data['top_3_coins']
            }
            
        except Exception as e:
            print(f"Error processing F3A combo: {str(e)}")
            return None
    
    def _process_f3b_coin(self, coin_data: Dict) -> Dict:
        """Process F3B coin data to extract combos and parameters"""
        try:
            # Extract basic coin info
            coin_symbol = coin_data['coin']
            
            # Process each combo for this coin
            combos = []
            for combo_data in coin_data['top_3_combos']:
                combo = combo_data['combo']  # Get the nested combo data
                combo_id = combo['combo_id']
                direction = combo['direction']
                window_size = combo['window_size']
                
                # Extract signals for this combo
                signals = []
                for signal_name in combo['signals']:
                    parts = signal_name.split('_')
                    timeframe = parts[0]
                    base_name = '_'.join(parts[1:-1]) if parts[-1] in ['long', 'short'] else '_'.join(parts[1:])
                    
                    signals.append({
                        'name': signal_name,
                        'timeframe': timeframe,
                        'base_name': base_name,
                        'direction': direction
                    })
                
                combos.append({
                    'combo_id': combo_id,
                    'direction': direction,
                    'window_size': window_size,
                    'signals': signals
                })
            
            return {
                'symbol': coin_symbol,
                'combos': combos
            }
            
        except Exception as e:
            print(f"Error processing F3B coin: {str(e)}")
            return None


    def test_combinations_parallel(self, window_data: Dict, debug: bool = False):
        """Main orchestrator for parallel combo testing"""
        print("\n=== Starting Parallel Combination Testing ===")
        
        try:
            # Extract input data
            data_15m = window_data['data_15m']  # Dictionary of DataFrames
            data_1h = window_data['data_1h']    # Dictionary of DataFrames
            current_rw_results = window_data['thresholds']
            f3_data = window_data['f3_data']
            account_type = window_data['account_type']

            print("\nF3 Data Structure:")
            print("Raw F3 data keys:", list(f3_data.keys()) if f3_data else "Empty")
            print(f3_data)

            print("\nCurrent RW Results Structure:")
            print("Window sizes:", list(current_rw_results.keys()))
            
            if account_type == 'S2A':
                print("\nF3A Structure:")
                print("F3A keys:", list(f3_data.keys()))
                if 'pool_1' in f3_data:
                    print("\nF3A Pool 1:")
                    for regime in ['bull', 'bear', 'flat']:
                        if regime in f3_data['pool_1']:
                            print(f"{regime} combos:", len(f3_data['pool_1'][regime]))
                            if f3_data['pool_1'][regime]:
                                print(f"First combo example:", f3_data['pool_1'][regime][0])
                if 'pool_2' in f3_data:
                    print("\nF3A Pool 2:")
                    if 'combos' in f3_data['pool_2']:
                        print(f"Total combos:", len(f3_data['pool_2']['combos']))
                        if f3_data['pool_2']['combos']:
                            print(f"First combo example:", f3_data['pool_2']['combos'][0])
            
            elif account_type == 'S2B':
                print("\nF3B Structure:")
                print("F3B keys:", list(f3_data.keys()))
                if 'pool_1' in f3_data:
                    print("\nF3B Pool 1:")
                    for regime in ['bull', 'bear', 'flat']:
                        if regime in f3_data['pool_1']:
                            print(f"{regime} coins:", len(f3_data['pool_1'][regime]))
                            if f3_data['pool_1'][regime]:
                                print(f"First coin example:", f3_data['pool_1'][regime][0])
                if 'pool_2' in f3_data:
                    print("\nF3B Pool 2:")
                    if 'coins' in f3_data['pool_2']:
                        print(f"Total coins:", len(f3_data['pool_2']['coins']))
                        if f3_data['pool_2']['coins']:
                            print(f"First coin example:", f3_data['pool_2']['coins'][0])

            # Get the first symbol's data for reference timestamps
            first_symbol = list(data_15m.keys())[0]
            first_symbol_data = data_15m[first_symbol]
            window_1_start = first_symbol_data.index[0]
            test_end_date = first_symbol_data.index[-1]
            
            # Calculate valid test days using first symbol's data
            valid_test_days, sim_start, sim_end = self._get_valid_test_days(first_symbol_data, window_1_start, test_end_date)
            
            if not valid_test_days:
                print("No valid test days found")
                return None
            
            print(f"\nAuthoritative Simulation Period:")
            print(f"Start: {sim_start}")
            print(f"End: {sim_end}")
            print(f"Found {len(valid_test_days)} valid test days")
            
            # Get strategy mode from window_data
            strategy_mode = window_data.get('strategy_mode', 'triple')  # default to triple
            if strategy_mode == 'double':
                from simulator_double import Simulator
            elif strategy_mode == 'single':
                from simulator_single import Simulator
            else:
                from simulator_triple import Simulator  

            simulator = Simulator()
            success = simulator.initialize_historical_pools(n_test_days=len(valid_test_days))
            if not success:
                print("Failed to initialize historical pools")
                return None
            
            # Initialize trial results
            trial_results = {
                'regime_sensitive_continuous': [],
                'regime_sensitive_target_triggered': [],
                'regime_independent_continuous': [],
                'regime_independent_target_triggered': []
            }
            
            # Process each trial configuration
            for regime_type in ['regime_sensitive', 'regime_independent']:
                for tp_mode in ['continuous', 'target_triggered']:
                    print(f"\nRunning {regime_type} - {tp_mode} trial")
                    
                    # Get trial F3 data based on account type and regime type
                    if account_type == 'S2A':
                        trial_f3_data = {}
                        if regime_type == 'regime_sensitive':
                            for regime in ['bull', 'bear', 'flat']:
                                if regime in f3_data['pool_1']:
                                    # Each combo in the list already has window_size
                                    trial_f3_data[regime] = f3_data['pool_1'][regime][:14]  # Top 10 combos per regime
                        else:  # regime_independent
                            trial_f3_data['independent'] = f3_data['pool_2']['combos'][:14]
                            
                    else:  # S2B
                        trial_f3_data = {}
                        if regime_type == 'regime_sensitive':  
                            for regime in ['bull', 'bear', 'flat']:
                                if regime in f3_data['pool_1']:
                                    # For each coin, include its top 3 combos with their window sizes
                                    trial_f3_data[regime] = [
                                        {
                                            'coin': coin_data['symbol'],
                                            'top_3_combos': [
                                                {
                                                    'combo': combo,
                                                    'window_size': combo['window_size']  # Window size is in combo data
                                                }
                                                for combo in coin_data['top_3_combos']
                                            ]
                                        }
                                        for coin_data in f3_data['pool_1'][regime][:14]  # Top 10 coins per regime
                                    ]
                        else:  # regime_independent
                            trial_f3_data = {
                                'independent': [
                                    {
                                        'coin': coin_data['symbol'],
                                        'top_3_combos': [
                                            {
                                                'combo': combo,
                                                'window_size': combo['window_size']  # Window size is in combo data
                                            }
                                            for combo in coin_data['top_3_combos']
                                        ]
                                    }
                                    for coin_data in f3_data['pool_2']['coins'][:14]  # Top 10 regime-independent coins
                                ]
                            }
                    
                    print("\nTrial F3 Data Structure:")
                    for regime, candidates in trial_f3_data.items():
                        print(f"{regime}: {len(candidates)} candidates")
                        if candidates:
                            if account_type == 'S2A':
                                print(f"Sample S2A combo: {candidates[0]}")
                            else:
                                print(f"Sample S2B coin: {candidates[0]}")
                                print(f"Sample S2B combo: {candidates[0]['top_3_combos'][0] if candidates[0]['top_3_combos'] else 'No combos'}")
                    
                    # Initialize regime data structure
                    regime_data = {}
                    candidates_dict = {}  # Use dictionary to group by candidate index
                    candidate_idx = 0

                    if account_type == 'S2A':
                        if regime_type == 'regime_sensitive':
                            # Get all regime combos
                            bull_combos = trial_f3_data.get('bull', [])
                            bear_combos = trial_f3_data.get('bear', [])
                            flat_combos = trial_f3_data.get('flat', [])
                            
                            # Find max number of combos across regimes
                            max_combos = max(len(bull_combos), len(bear_combos), len(flat_combos))
                            
                            # Process each index position
                            for idx in range(max_combos):
                                candidate_data = {'regime_data': {}}
                                
                                # Add bull combo if available
                                if idx < len(bull_combos):
                                    bull_combo = bull_combos[idx]
                                    candidate_data['regime_data']['bull'] = {
                                        'signals': [signal.strip() for signal in bull_combo['signals']],
                                        'top_3_coins': bull_combo['top_3_coins'],
                                        'direction': bull_combo['direction'],
                                        'window_size': bull_combo['window_size'],
                                        'combo_id': bull_combo['combo_id']
                                    }
                                    # Get regime data for bull coins
                                    for coin in bull_combo['top_3_coins']:
                                        if coin not in regime_data:
                                            regime_data[coin] = get_regime_data(
                                                symbol=coin,
                                                start_date=sim_start.strftime('%Y-%m-%d'),
                                                end_date=sim_end.strftime('%Y-%m-%d')
                                            )
                                
                                # Add bear combo if available
                                if idx < len(bear_combos):
                                    bear_combo = bear_combos[idx]
                                    candidate_data['regime_data']['bear'] = {
                                        'signals': [signal.strip() for signal in bear_combo['signals']],
                                        'top_3_coins': bear_combo['top_3_coins'],
                                        'direction': bear_combo['direction'],
                                        'window_size': bear_combo['window_size'],
                                        'combo_id': bear_combo['combo_id']
                                    }
                                    # Get regime data for bear coins
                                    for coin in bear_combo['top_3_coins']:
                                        if coin not in regime_data:
                                            regime_data[coin] = get_regime_data(
                                                symbol=coin,
                                                start_date=sim_start.strftime('%Y-%m-%d'),
                                                end_date=sim_end.strftime('%Y-%m-%d')
                                            )
                                
                                # Add flat combo if available
                                if idx < len(flat_combos):
                                    flat_combo = flat_combos[idx]
                                    candidate_data['regime_data']['flat'] = {
                                        'signals': [signal.strip() for signal in flat_combo['signals']],
                                        'top_3_coins': flat_combo['top_3_coins'],
                                        'direction': flat_combo['direction'],
                                        'window_size': flat_combo['window_size'],
                                        'combo_id': flat_combo['combo_id']
                                    }
                                    # Get regime data for flat coins
                                    for coin in flat_combo['top_3_coins']:
                                        if coin not in regime_data:
                                            regime_data[coin] = get_regime_data(
                                                symbol=coin,
                                                start_date=sim_start.strftime('%Y-%m-%d'),
                                                end_date=sim_end.strftime('%Y-%m-%d')
                                            )
                                
                                # Only add candidate if it has at least one regime
                                if candidate_data['regime_data']:
                                    candidates_dict[candidate_idx] = candidate_data
                                    candidate_idx += 1
                        
                        else:  # regime_independent
                            # Process independent combos
                            independent_combos = trial_f3_data.get('independent', [])
                            for combo in independent_combos:
                                # Get regime data for combo coins
                                for coin in combo['top_3_coins']:
                                    if coin not in regime_data:
                                        regime_data[coin] = get_regime_data(
                                            symbol=coin,
                                            start_date=sim_start.strftime('%Y-%m-%d'),
                                            end_date=sim_end.strftime('%Y-%m-%d')
                                        )
                                
                                # Initialize candidate with combo data
                                candidates_dict[candidate_idx] = {'regime_data': {
                                    'independent': {
                                        'signals': [signal.strip() for signal in combo['signals']],
                                        'top_3_coins': combo['top_3_coins'],
                                        'direction': combo['direction'],
                                        'window_size': combo['window_size'],
                                        'combo_id': combo['combo_id']
                                    }
                                }}
                                candidate_idx += 1

                    else:  # S2B
                        if regime_type == 'regime_sensitive':
                            # Get coins from each regime
                            bull_coins = trial_f3_data.get('bull', [])
                            bear_coins = trial_f3_data.get('bear', [])
                            flat_coins = trial_f3_data.get('flat', [])
                            
                            # Process each rank position (0-9 for top 10)
                            for idx in range(10):
                                candidate_data = {'regime_data': {}}
                                
                                # Add bull coin with ALL its combos if available at this rank
                                if idx < len(bull_coins):
                                    bull_coin = bull_coins[idx]
                                    processed_bull = self._process_f3b_coin({
                                        'coin': bull_coin['coin'],
                                        'top_3_combos': bull_coin['top_3_combos']
                                    })
                                    if processed_bull:
                                        candidate_data['regime_data']['bull'] = {
                                            'coin': processed_bull['symbol'],
                                            'signals': [signal['name'] for signal in processed_bull['combos'][0]['signals']],
                                            'direction': processed_bull['combos'][0]['direction'],
                                            'window_size': processed_bull['combos'][0]['window_size'],
                                            'top_3_combos': [{
                                                'signals': [signal['name'] for signal in combo['signals']],
                                                'direction': combo['direction'],
                                                'window_size': combo['window_size'],
                                                'combo_id': combo['combo_id']
                                            } for combo in processed_bull['combos']]
                                        }
                                        # Get regime data
                                        if processed_bull['symbol'] not in regime_data:
                                            regime_data[processed_bull['symbol']] = get_regime_data(
                                                symbol=processed_bull['symbol'],
                                                start_date=sim_start.strftime('%Y-%m-%d'),
                                                end_date=sim_end.strftime('%Y-%m-%d')
                                            )
                                
                                # Add flat coin with ALL its combos if available at this rank
                                if idx < len(flat_coins):
                                    flat_coin = flat_coins[idx]
                                    processed_flat = self._process_f3b_coin({
                                        'coin': flat_coin['coin'],
                                        'top_3_combos': flat_coin['top_3_combos']
                                    })
                                    if processed_flat:
                                        candidate_data['regime_data']['flat'] = {
                                            'coin': processed_flat['symbol'],
                                            'signals': [signal['name'] for signal in processed_flat['combos'][0]['signals']],
                                            'direction': processed_flat['combos'][0]['direction'],
                                            'window_size': processed_flat['combos'][0]['window_size'],
                                            'top_3_combos': [{
                                                'signals': [signal['name'] for signal in combo['signals']],
                                                'direction': combo['direction'],
                                                'window_size': combo['window_size'],
                                                'combo_id': combo['combo_id']
                                            } for combo in processed_flat['combos']]
                                        }
                                        # Get regime data
                                        if processed_flat['symbol'] not in regime_data:
                                            regime_data[processed_flat['symbol']] = get_regime_data(
                                                symbol=processed_flat['symbol'],
                                                start_date=sim_start.strftime('%Y-%m-%d'),
                                                end_date=sim_end.strftime('%Y-%m-%d')
                                            )
                                
                                # Add bear coin with ALL its combos if available at this rank
                                if idx < len(bear_coins):
                                    bear_coin = bear_coins[idx]
                                    processed_bear = self._process_f3b_coin({
                                        'coin': bear_coin['coin'],
                                        'top_3_combos': bear_coin['top_3_combos']
                                    })
                                    if processed_bear:
                                        candidate_data['regime_data']['bear'] = {
                                            'coin': processed_bear['symbol'],
                                            'signals': [signal['name'] for signal in processed_bear['combos'][0]['signals']],
                                            'direction': processed_bear['combos'][0]['direction'],
                                            'window_size': processed_bear['combos'][0]['window_size'],
                                            'top_3_combos': [{
                                                'signals': [signal['name'] for signal in combo['signals']],
                                                'direction': combo['direction'],
                                                'window_size': combo['window_size'],
                                                'combo_id': combo['combo_id']
                                            } for combo in processed_bear['combos']]
                                        }
                                        # Get regime data
                                        if processed_bear['symbol'] not in regime_data:
                                            regime_data[processed_bear['symbol']] = get_regime_data(
                                                symbol=processed_bear['symbol'],
                                                start_date=sim_start.strftime('%Y-%m-%d'),
                                                end_date=sim_end.strftime('%Y-%m-%d')
                                            )
                                
                                # Only add candidate if it has at least one regime
                                if candidate_data['regime_data']:
                                    candidates_dict[candidate_idx] = candidate_data
                                    candidate_idx += 1

                        else:  # regime_independent
                            independent_coins = trial_f3_data.get('independent', [])
                            for coin_data in independent_coins[:14]:  # Top 10 regime-independent coins
                                processed_coin = self._process_f3b_coin({
                                    'coin': coin_data['coin'],
                                    'top_3_combos': coin_data['top_3_combos']
                                })
                                if processed_coin:
                                    # Get regime data for independent coin
                                    if processed_coin['symbol'] not in regime_data:
                                        regime_data[processed_coin['symbol']] = get_regime_data(
                                            symbol=processed_coin['symbol'],
                                            start_date=sim_start.strftime('%Y-%m-%d'),
                                            end_date=sim_end.strftime('%Y-%m-%d')
                                        )
                                    
                                    # Add all combos for this coin
                                    candidates_dict[candidate_idx] = {'regime_data': {
                                        'independent': {
                                            'coin': processed_coin['symbol'],
                                            'signals': [signal['name'] for signal in processed_coin['combos'][0]['signals']],
                                            'direction': processed_coin['combos'][0]['direction'],
                                            'window_size': processed_coin['combos'][0]['window_size'],
                                            'top_3_combos': [{
                                                'signals': [signal['name'] for signal in combo['signals']],
                                                'direction': combo['direction'],
                                                'window_size': combo['window_size'],
                                                'combo_id': combo['combo_id']
                                            } for combo in processed_coin['combos']]
                                        }
                                    }}
                                candidate_idx += 1

                    candidates = list(candidates_dict.values())
                    
                    print("\nPrepared Candidates Structure:")
                    print(f"Total candidates: {len(candidates)}")
                    for idx, candidate in enumerate(candidates):
                        print(f"\nCandidate {idx}:")
                        for regime, data in candidate['regime_data'].items():
                            print(f"  {regime}: {data}")
                    
                    # Before prepare_combo_data, collect all unique signals and coins from selected combos
                    all_unique_signals = set()
                    all_unique_coins = set()
                    
                    if account_type == 'S2A':
                        # Extract signals and coins from regime-sensitive combos
                        for regime_list in trial_f3_data.values():
                            for combo_data in regime_list:
                                processed_combo = self._process_f3a_combo(combo_data)
                                if processed_combo:
                                    for signal_dict in processed_combo['signals']:
                                        timeframe = signal_dict['timeframe'].strip()
                                        base_name = signal_dict['base_name'].strip()
                                        if base_name.endswith('_long') or base_name.endswith('_short'):
                                            base_name = base_name.rsplit('_', 1)[0]
                                        signal_name = f"{timeframe}_{base_name}"
                                        all_unique_signals.add(signal_name)
                                    all_unique_coins.update(processed_combo['top_3_coins'])
                    else:  # S2B
                        # Extract signals and coins from coins' combos
                        for regime_list in trial_f3_data.values():
                            for coin_data in regime_list:
                                processed_coin = self._process_f3b_coin({
                                    'coin': coin_data['coin'],
                                    'top_3_combos': coin_data['top_3_combos']
                                })
                                if processed_coin:
                                    all_unique_coins.add(processed_coin['symbol'])
                                    for combo in processed_coin['combos']:
                                        for signal_dict in combo['signals']:
                                            timeframe = signal_dict['timeframe'].strip()
                                            base_name = signal_dict['base_name'].strip()
                                            if base_name.endswith('_long') or base_name.endswith('_short'):
                                                base_name = base_name.rsplit('_', 1)[0]
                                            signal_name = f"{timeframe}_{base_name}"
                                            all_unique_signals.add(signal_name)
                    
                    # Also add all coins from window_data using correct key
                    all_unique_coins.update(window_data['data_15m'].keys())
                    
                    print(f"\nTotal unique signals to prepare: {len(all_unique_signals)}")
                    print(f"Total unique coins to prepare: {len(all_unique_coins)}")
                    if debug:
                        print("Signals:", sorted(list(all_unique_signals)))
                        print("Coins:", sorted(list(all_unique_coins)))
                    
                    # Prepare combo data with regime data
                    combo_data = self._prepare_combo_data(
                        window_data={
                            '15m': window_data['data_15m'],
                            '1h': window_data['data_1h']
                        },
                        signals=list(all_unique_signals),
                        coins=list(all_unique_coins),
                        start_date=sim_start,  # Use authoritative start date
                        end_date=sim_end,      # Use authoritative end date
                        current_rw_results=current_rw_results,
                        regime_data=regime_data
                    )
                    
                    if not combo_data:
                        print("Failed to prepare combo data for simulation")
                        continue

                    print(f'candidates: {candidates}')
                    
                    # Create simulation input package
                    simulation_input = {
                        'account_type': account_type,
                        'regime_type': regime_type,
                        'tp_mode': tp_mode,
                        'trial_f3_data': trial_f3_data,
                        'combo_data': combo_data,
                        'valid_test_days': valid_test_days,
                        'candidates': candidates
                    }
                    
                    # Call simulator's simulate_combinations method
                    simulation_success = simulator.simulate_combinations(
                        simulation_input,
                        debug=debug
                    )
                    
                    # Store trial results if simulation was successful
                    if simulation_success:
                        # Get results from historical pool
                        pool = simulator.historical_pools[account_type][regime_type][tp_mode]
                        
                        # Calculate total days from authoritative period
                        total_days = len(valid_test_days)
                        
                        # Process results for this trial configuration
                        trial_results[f"{regime_type}_{tp_mode}"] = {
                            'results': [],  # Store performance results here
                            'metadata': {   # Store configuration data here
                                'units': {}
                            }
                        }
                        
                        # Store candidate configuration data
                        for unit_idx, candidate in enumerate(candidates):
                            trial_results[f"{regime_type}_{tp_mode}"]['metadata']['units'][str(unit_idx)] = {
                                'regime_data': candidate['regime_data']
                            }
                        
                        # Process each unit's performance data (maintaining existing structure)
                        for unit in range(len(candidates)):
                            unit_result = {
                                'summary': [],
                                'trades': []
                            }
                            
                            # Process each day's data
                            for day in range(total_days):
                                day_data = pool[f'unit_{unit}'][f'day_{day}']
                                
                                # Add mother account summary
                                unit_result['summary'].append({
                                    'unit': unit,
                                    'day': day,
                                    'date': valid_test_days[day],
                                    'mother_account': day_data['mother_account']
                                })
                                
                                # Add trade data from slots
                                for slot_name, slot_data in day_data['slots'].items():
                                    if slot_data['active'] or slot_data['completed_trades'] > 0:
                                        unit_result['trades'].append({
                                            'unit': unit,
                                            'day': day,
                                            'date': valid_test_days[day],
                                            'slot': slot_name,
                                            **slot_data  # Include all slot data
                                        })
                            
                            # Add this unit's results to the trial results
                            trial_results[f"{regime_type}_{tp_mode}"]['results'].append(unit_result)
                        
                        print(f"\nStored results for {regime_type}_{tp_mode}:")
                        print(f"Number of units processed: {len(trial_results[f'{regime_type}_{tp_mode}']['results'])}")
                        for unit_idx, unit_result in enumerate(trial_results[f"{regime_type}_{tp_mode}"]['results']):
                            print(f"\nUnit {unit_idx}:")
                            print(f"Summary entries: {len(unit_result['summary'])}")
                            print(f"Trade entries: {len(unit_result['trades'])}")
                    
                    # Force cleanup of GPU memory after each trial
                    # This ensures we don't accumulate memory between trials
                    simulator.cleanup_block_memory_pools()
                    
                    # Force Python garbage collection
                    import gc
                    gc.collect()
                    
                    # Synchronize CUDA operations
                    from numba import cuda
                    cuda.synchronize()

            return trial_results
            
        except Exception as e:
            print(f"Error in test_combinations_parallel: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            