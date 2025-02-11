import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np

def create_thresholds_db_connection(db_path: str = None):
    """Create or connect to thresholds database"""
    if not db_path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        db_path = f'data/thresholds/thresholds_{timestamp}.db'
    
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn

def initialize_thresholds_tables(conn: sqlite3.Connection):
    """Create tables for storing threshold data"""
    cursor = conn.cursor()
    
    # Window metadata table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS windows (
            window_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            window_size INTEGER NOT NULL,
            window_start TIMESTAMP NOT NULL,
            window_end TIMESTAMP NOT NULL,
            UNIQUE(symbol, window_size, window_start, window_end)
        )
    ''')
    
    # Thresholds table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS thresholds (
            threshold_id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_id INTEGER NOT NULL,
            timeframe TEXT NOT NULL,
            indicator TEXT NOT NULL,
            direction TEXT NOT NULL,
            threshold_value REAL,
            condition TEXT NOT NULL,
            win_rate REAL NOT NULL,
            trades INTEGER NOT NULL,
            is_optimal BOOLEAN NOT NULL,
            FOREIGN KEY (window_id) REFERENCES windows(window_id)
        )
    ''')
    
    # Create indexes for faster querying
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_windows_symbol ON windows(symbol)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_thresholds_indicator ON thresholds(indicator)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_thresholds_window ON thresholds(window_id)
    ''')
    
    conn.commit()

def export_thresholds_to_db(analysis_results: List[Dict], db_path: str = None) -> str:
    """Export threshold results from analyze_pair_timeframe to SQLite database"""
    # Create database path if not provided
    if not db_path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        db_path = f'data/thresholds/thresholds_{timestamp}.db'
    
    conn = create_thresholds_db_connection(db_path)
    initialize_thresholds_tables(conn)
    
    try:
        cursor = conn.cursor()
        
        for result in analysis_results:
            if not result or 'thresholds' not in result:
                continue
                
            symbol = result['symbol']
            thresholds = result['thresholds']
            
            # Process each window size
            for window_size, window_data in thresholds.items():
                # Process each timestamp in window data
                for window_end, indicators in window_data.items():
                    # First insert window metadata
                    window_start = pd.Timestamp(window_end) - pd.Timedelta(days=window_size/24)
                    cursor.execute('''
                        INSERT OR IGNORE INTO windows 
                        (symbol, window_size, window_start, window_end)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        str(symbol),
                        int(window_size),
                        str(window_start),
                        str(window_end)
                    ))
                    
                    # Get the window_id
                    cursor.execute('''
                        SELECT window_id FROM windows
                        WHERE symbol = ? 
                        AND window_size = ?
                        AND window_end = ?
                    ''', (str(symbol), int(window_size), str(window_end)))
                    
                    window_row = cursor.fetchone()
                    if not window_row:
                        continue
                    window_id = window_row[0]
                    
                    # Insert thresholds data
                    threshold_batch = []
                    for indicator_key, directions in indicators.items():
                        if indicator_key == 'window_info':
                            continue
                            
                        # Split timeframe and indicator name
                        if '_' not in indicator_key:
                            continue
                        timeframe, indicator = indicator_key.split('_', 1)
                        
                        for direction, data in directions.items():
                            if not isinstance(data, dict) or 'threshold' not in data:
                                continue
                                
                            threshold_value = data['threshold'][0] if isinstance(data['threshold'], tuple) else None
                            condition = data['threshold'][1] if isinstance(data['threshold'], tuple) else 'unknown'
                            
                            if threshold_value is None or np.isnan(threshold_value):
                                continue
                                
                            threshold_batch.append((
                                int(window_id),
                                str(timeframe),
                                str(indicator),
                                str(direction),
                                float(threshold_value),
                                str(condition),
                                float(data.get('win_rate', 0.0)),
                                int(data.get('trades', 0)),
                                bool(data.get('is_optimal', False))
                            ))
                    
                    # Batch insert thresholds
                    if threshold_batch:
                        cursor.executemany('''
                            INSERT INTO thresholds (
                                window_id, timeframe, indicator, direction,
                                threshold_value, condition, win_rate,
                                trades, is_optimal
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', threshold_batch)
            
            conn.commit()
            
        print(f"Successfully exported {len(analysis_results)} results to database")
        return db_path  # Return the actual database path
        
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Error exporting thresholds: {str(e)}")
    finally:
        conn.close()

# Test the export functionality
if __name__ == "__main__":
    from test_indicator_analysis import analyze_pair_timeframe
    
    # Test parameters
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT', 
               'ADAUSDT', 'APTUSDT', 'TRXUSDT', 'PEPEUSDT', 
               'SHIBUSDT', 'BONKUSDT', 'DOGEUSDT', 'XRPUSDT', 
               'XLMUSDT', 'AAVEUSDT', 'UNIUSDT', 'SANDUSDT', 'LINKUSDT', 'HBARUSDT']
    start_date = '2023-09-01'
    end_date = '2024-02-01'
    window_sizes = [24, 72, 120]
    
    # Collect results
    test_results = []
    for symbol in symbols:
        try:
            result = analyze_pair_timeframe(symbol, start_date, end_date, window_sizes)
            if result is not None:
                test_results.append(result)
                print(f"Analysis completed for {symbol}")
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
    
    # Export results
    if test_results:
        try:
            db_path = export_thresholds_to_db(test_results)
            print(f"Results exported successfully to: {db_path}")
            
            # Verify database exists
            if Path(db_path).exists():
                print(f"Database file created successfully at: {db_path}")
                # Optional: Print database size
                print(f"Database size: {Path(db_path).stat().st_size / 1024:.2f} KB")
            else:
                print("Warning: Database file not found after export!")
                
        except Exception as e:
            print(f"Error exporting results: {e}")
    else:
        print("No results to export") 