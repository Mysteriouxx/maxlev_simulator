import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from calculator import IndicatorCalculator

def get_regime_data(symbol: str, start_date: str, end_date: str) -> np.ndarray:
    """Get market regime data reindexed to 1m timeframe"""
    try:
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Load 1d and 1m data
        data_path_1d = Path('data/raw') / f'{symbol}_1d_{config["parameters"]["start_date"]}_{config["parameters"]["end_date"]}.csv'
        data_path_1m = Path('data/raw') / f'{symbol}_1m_{config["parameters"]["simulation_start_date"]}_{config["parameters"]["end_date"]}.csv'
        
        if not data_path_1d.exists():
            print(f"Warning: Daily data file not found for {symbol}: {data_path_1d}")
            return np.full(1440, 3, dtype=np.int32)  # Return flat regime if no data
            
        if not data_path_1m.exists():
            print(f"Warning: Minute data file not found for {symbol}: {data_path_1m}")
            return np.full(1440, 3, dtype=np.int32)  # Return flat regime if no data
        
        # Load and prepare 1d data for regime calculation
        df_1d = pd.read_csv(data_path_1d)
        df_1d['timestamp'] = pd.to_datetime(df_1d['timestamp'])
        df_1d.set_index('timestamp', inplace=True)
        df_1d = df_1d[start_date:end_date]
        
        if df_1d.empty:
            print(f"Warning: No daily data in range for {symbol}")
            return np.full(1440, 3, dtype=np.int32)  # Return flat regime if no data
        
        # Load 1m data for reindexing
        df_1m = pd.read_csv(data_path_1m)
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])
        df_1m.set_index('timestamp', inplace=True)
        df_1m = df_1m[start_date:end_date]
        
        if df_1m.empty:
            print(f"Warning: No minute data in range for {symbol}")
            return np.full(1440, 3, dtype=np.int32)  # Return flat regime if no data
        
        # Calculate market regime using 1d data
        calculator = IndicatorCalculator(df_1d, '1d', 'config.yaml')
        regime_df = calculator.calculate_market_regime()
        
        if regime_df is None or regime_df.empty:
            print(f"Warning: Failed to calculate regime for {symbol}")
            return np.full(1440, 3, dtype=np.int32)  # Return flat regime if calculation fails
        
        # Initialize the mapped regime series
        mapped_regime = pd.Series(index=regime_df.index, dtype=np.int32)
        last_major = 0  # Start with no major regime
        
        # Process each day's regime
        for idx in regime_df.index:
            current_regime = regime_df.loc[idx, 'regime']
            
            # Update last major if we hit a major regime
            if current_regime == 2.0:  # Major Bull
                last_major = 2.0
            elif current_regime == -2.0:  # Major Bear
                last_major = -2.0
                
            # Map current regime based on rules
            if last_major == 2.0:  # In Bull phase
                if current_regime >= 1.0:  # Major or Minor Bull
                    mapped_regime[idx] = 1  # Bull (matches memory index)
                else:  # Flat or any Bear
                    mapped_regime[idx] = 3  # Flat (matches memory index)
            elif last_major == -2.0:  # In Bear phase
                if current_regime <= -1.0:  # Major or Minor Bear
                    mapped_regime[idx] = 2  # Bear (matches memory index)
                else:  # Flat or any Bull
                    mapped_regime[idx] = 3  # Flat (matches memory index)
            else:  # No major regime yet
                mapped_regime[idx] = 3  # Flat (matches memory index)
        
        # Reindex to 1m timeframe
        regime_1m = mapped_regime.reindex(df_1m.index, method='ffill')
        
        print(f"\nMarket Regime Distribution for {symbol}:")
        regime_counts = regime_1m.value_counts()
        for regime, count in regime_counts.items():
            regime_name = {1: 'bull', 2: 'bear', 3: 'flat'}[regime]
            percentage = (count / len(regime_1m)) * 100
            print(f"{regime_name}: {count} candles ({percentage:.2f}%)")
        
        return regime_1m.values
        
    except Exception as e:
        print(f"Error getting regime data for {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return np.full(1440, 3, dtype=np.int32)  # Return flat regime on error 