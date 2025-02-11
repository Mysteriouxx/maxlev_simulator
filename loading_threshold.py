import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
import yaml
from rolling_window import RollingWindowTester
from calculator import IndicatorCalculator

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

def reconstruct_window_indices(windows_df):
    """
    Reconstruct window indices chronologically for each coin and window size.
    
    Args:
        windows_df: DataFrame with columns [window_id, symbol, window_size, window_start, window_end]
    
    Returns:
        Dictionary mapping (symbol, window_size, window_end) to chronological index
    """
    window_index_map = {}
    
    for (symbol, window_size), group in windows_df.groupby(['symbol', 'window_size']):
        # Sort windows by end timestamp
        sorted_windows = group.sort_values('window_end')
        
        # Assign sequential indices
        for idx, row in enumerate(sorted_windows.itertuples()):
            key = (row.symbol, row.window_size, row.window_end)
            window_index_map[key] = idx
            
    return window_index_map

def load_thresholds(db_path, sim_start_date, sim_end_date):
    """
    Load thresholds and reconstruct all_results structure from database file.
    """
    sim_start = pd.to_datetime(sim_start_date)
    sim_end = pd.to_datetime(sim_end_date)
    print(f"\nSimulation period: {sim_start} to {sim_end}")

    with sqlite3.connect(db_path) as conn:
        windows_df = pd.read_sql("SELECT * FROM windows", conn)
        thresholds_df = pd.read_sql("SELECT * FROM thresholds", conn)
        
        windows_df['window_start'] = pd.to_datetime(windows_df['window_start'])
        windows_df['window_end'] = pd.to_datetime(windows_df['window_end'])
        
        print(f"\nInitial data:")
        print(f"Total windows: {len(windows_df)}")
        print(f"Total thresholds: {len(thresholds_df)}")
        print(f"Unique symbols: {windows_df['symbol'].unique().tolist()}")
        print(f"Window sizes: {sorted(windows_df['window_size'].unique().tolist())}")
        print(f"Date range: {windows_df['window_end'].min()} to {windows_df['window_end'].max()}")
        
        # Debug print first row of threshold data
        print("\nSample threshold data:")
        print(thresholds_df.head(1))
        
        # Reconstruct window indices for each symbol and window size
        window_index_map = {}
        all_results = []
        
        for symbol in windows_df['symbol'].unique():
            print(f"\nProcessing {symbol}")
            symbol_windows = windows_df[windows_df['symbol'] == symbol]
            thresholds = {}
            
            # Process each window size separately
            for window_size in sorted(symbol_windows['window_size'].unique()):
                size_windows = symbol_windows[symbol_windows['window_size'] == window_size]
                thresholds[window_size] = {}
                
                # Sort windows chronologically
                sorted_windows = size_windows.sort_values('window_end')
                window_ends = sorted_windows['window_end'].dt.strftime('%Y-%m-%d %H:%M').tolist()
                total_windows = len(window_ends)
                
                print(f"\n  Window size {window_size}:")
                print(f"    Available windows: {total_windows}")
                
                if total_windows <= 9:
                    print(f"    Window ends: {window_ends}")
                else:
                    first_3 = window_ends[:3]
                    middle_3 = window_ends[total_windows//2-1:total_windows//2+2]
                    last_3 = window_ends[-3:]
                    print(f"    Window ends: {first_3} ... {middle_3} ... {last_3}")
                
                # Assign chronological indices and load thresholds
                for idx, row in enumerate(sorted_windows.itertuples()):
                    window_index_map[(symbol, window_size, row.window_end)] = idx
                    
                    # Get thresholds for this window
                    window_thresholds = thresholds_df[thresholds_df['window_id'] == row.window_id]
                    
                    if not window_thresholds.empty:
                        processed_data = {}
                        for _, threshold in window_thresholds.iterrows():
                            indicator_key = f"{threshold['timeframe']}_{threshold['indicator']}"
                            direction = threshold['direction']
                            
                            if indicator_key not in processed_data:
                                processed_data[indicator_key] = {}
                            
                            processed_data[indicator_key][direction] = {
                                'window_idx': idx,
                                'threshold': (threshold['threshold_value'], threshold['condition']),
                                'win_rate': threshold['win_rate'],
                                'is_optimal': bool(threshold['is_optimal'])
                            }
                        
                        thresholds[window_size][row.window_end] = processed_data
                        # Debug print first threshold content
                        if idx == 0:
                            first_indicator = next(iter(processed_data))
                            first_direction = next(iter(processed_data[first_indicator]))
                            print(f"\n    Sample threshold content for window {idx}:")
                            print(f"    {first_indicator} ({first_direction}): {processed_data[first_indicator][first_direction]}")
            
            # Load price data and initialize tester
            df_15m, df_1h = load_and_prepare_data(symbol, sim_start_date, sim_end_date)
            
            calculator = IndicatorCalculator(df_15m, '15m', 'config.yaml')
            df_15m = calculator.calculate_all_indicators()
            
            calculator = IndicatorCalculator(df_1h, '1h', 'config.yaml')
            df_1h = calculator.calculate_all_indicators()
            
            # Initialize tester with thresholds
            tester = RollingWindowTester(
                df=df_15m.copy(),
                timeframe='15m',
                config_path='config.yaml',
                window_sizes=[24, 72, 120],
                use_gpu=True
            )
            
            # Debug print thresholds before setting
            print(f"\nThresholds before setting for {symbol}:")
            for size in thresholds:
                print(f"Window {size}: {len(thresholds[size])} entries")
                if thresholds[size]:
                    first_date = next(iter(thresholds[size].keys()))
                    print(f"Sample data: {thresholds[size][first_date]}")
            
            # Set thresholds after initialization
            tester.set_thresholds(thresholds)
            
            result = {
                'symbol': symbol,
                'data': df_15m,
                'data_15m': df_15m,
                'data_1h': df_1h,
                'tester': tester,
                'thresholds': thresholds,
                'results': {}
            }
            
            all_results.append(result)
        
        print(f"\nProcessed {len(all_results)} symbols")
        return all_results

def main():
    db_path = "data/thresholds/thresholds_20250131_023916.db"
    sim_start_date = "2025-01-31"
    sim_end_date = "2025-02-01"
    all_results = load_thresholds(db_path, sim_start_date, sim_end_date)
    
    # Verify structure matches test_indicator_analysis.py
    for result in all_results:
        print(f"\nVerifying {result['symbol']}:")
        print(f"data_15m shape: {result['data_15m'].shape}")
        print(f"data_1h shape: {result['data_1h'].shape}")
        print(f"Window sizes: {list(result['thresholds'].keys())}")

if __name__ == "__main__":
    main()
