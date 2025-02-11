import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
from datetime import datetime
from tabulate import tabulate
import itertools

def calculate_capital_multiple(df):
    """Calculate how many times the initial capital was multiplied"""
    initial_capital = 500
    final_capital = df['capital'].iloc[-1]
    multiple = final_capital / initial_capital
    print(f"Debug - Initial: {initial_capital}, Final: {final_capital}, Multiple: {multiple}")  # Debug
    return multiple

def calculate_target_count(df):
    """Count how many targets were reached"""
    return df['target_hits'].sum()

def calculate_return_stats(df):
    """Calculate weekly and monthly return statistics"""
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate daily returns first
    df['returns'] = df['capital'].pct_change().fillna(0)
    
    # Resample to weekly and monthly, handling empty periods
    weekly_returns = df.set_index('date')['returns'].resample('W').apply(
        lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
    ).fillna(0)
    
    monthly_returns = df.set_index('date')['returns'].resample('M').apply(
        lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
    ).fillna(0)
    
    return {
        'weekly_std': weekly_returns.std(),
        'monthly_std': monthly_returns.std()
    }

def calculate_avg_time_to_target(df):
    """Calculate average time to target"""
    target_times = df[df['target_hit'] == 1]['minutes_active'].diff()
    return target_times.mean() if not target_times.empty else 0

def calculate_slot_utilization(df):
    """Calculate slot utilization based on margin usage"""
    # Consider a slot utilized when margin_used is non-zero
    total_time = len(df)
    active_time = len(df[df['margin_used'] != 0])
    return active_time / total_time if total_time > 0 else 0

def calculate_metrics_for_block(df, conn, table_name, block_id):
    """Calculate all metrics for a single block"""
    df = df.copy()
    
    initial_capital = 500
    # Adjust final capital for target-triggered trials
    if 'target_triggered' in table_name.lower():
        target_rewards = df.iloc[-1]['target_hits'] * 9500  # Changed from 2000 to 9500
        final_capital = df['capital'].iloc[-1] + target_rewards
    else:
        final_capital = df['capital'].iloc[-1]
    
    net_profit = final_capital - initial_capital
    
    # Basic calculations
    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    df.loc[:, 'returns'] = df['capital'].pct_change().fillna(0)
    df.loc[:, 'trade_returns'] = df['capital'].diff()
    
    # For target-triggered trials, adjust the returns calculation
    if 'target_triggered' in table_name.lower():
        df['adjusted_capital'] = df['capital'] + (df['target_hits'] * 9500)  # Changed from 2000 to 9500
        df['returns'] = df['adjusted_capital'].pct_change().fillna(0)
        df['trade_returns'] = df['adjusted_capital'].diff()
    
    # Calculate drawdown
    df['drawdown'] = (df['capital'] - df['capital'].cummax()) / df['capital'].cummax()
    max_drawdown = df['drawdown'].min()
    
    # Win/Loss calculations
    winning_trades = df[df['trade_returns'] > 0]['trade_returns']
    losing_trades = df[df['trade_returns'] < 0]['trade_returns']
    win_rate = len(winning_trades) / len(df[df['trade_returns'] != 0]) if len(df[df['trade_returns'] != 0]) > 0 else 0
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
    
    # Sortino Ratio calculation - fixed
    downside_returns = df['returns'][df['returns'] < 0]
    if len(downside_returns) > 0 and downside_returns.std() != 0:
        sortino = (df['returns'].mean() * 252) / (downside_returns.std() * np.sqrt(252))
    else:
        sortino = 0  # or could use np.nan if you prefer
    
    # Time to Target calculation - convert to days (minutes / 1440) and round to 1 decimal
    target_hits = df[df['target_hits'] > df['target_hits'].shift(1)].copy()
    if len(target_hits) > 1:
        target_hits['time_diff'] = (target_hits['date'].diff().dt.total_seconds() / 60).fillna(0)  # First get minutes
        time_to_target = target_hits['time_diff'].mean() / 1440  # Convert minutes to days
    else:
        time_to_target = None
    
    # Calculate slot utilization
    slots_table = table_name.replace('account', 'slots')
    query = f"""
    SELECT slot_id, MAX(minutes_active) as last_active
    FROM {slots_table}
    WHERE block_id = {block_id}
    GROUP BY slot_id
    """
    slots_df = pd.read_sql_query(query, conn)
    total_simulation_time = 476640
    avg_slot_utilization = slots_df['last_active'].mean() / total_simulation_time * 100
    
    # Monthly returns calculation
    monthly_returns = df.set_index('date')['returns'].resample('M').apply(
        lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
    ).fillna(0)
    
    # VaR calculation
    var_95 = np.percentile(df['returns'], 5)
    es_95 = df[df['returns'] <= var_95]['returns'].mean()
    
    return {
        # Basic Metrics
        'capital_multiple': final_capital / initial_capital,
        'targets_hit': df.iloc[-1]['target_hits'],
        'slot_utilization': avg_slot_utilization,
        
        # Risk-Adjusted Performance
        'sharpe_ratio': (np.mean(df['returns']) * 252) / (np.std(df['returns']) * np.sqrt(252)) if np.std(df['returns']) != 0 else 0,
        'max_drawdown': max_drawdown * 100,
        'calmar_ratio': ((final_capital/initial_capital - 1) / abs(max_drawdown)) if max_drawdown != 0 else 0,
        'sortino_ratio': sortino,
        
        # Trading Efficiency
        'win_rate': win_rate * 100,
        'profit_factor': winning_trades.sum() / abs(losing_trades.sum()) if len(losing_trades) > 0 else float('inf'),
        'time_to_target': time_to_target,
        
        # Consistency
        'monthly_win_rate': len(monthly_returns[monthly_returns > 0]) / len(monthly_returns) * 100,
        'longest_win_streak': max((sum(1 for _ in group) for key, group in itertools.groupby(df['trade_returns'] > 0) if key), default=0),
        'longest_lose_streak': max((sum(1 for _ in group) for key, group in itertools.groupby(df['trade_returns'] > 0) if not key), default=0),
        
        # Risk Distribution
        'var_95': var_95 * 100,
        'expected_shortfall': es_95 * 100
    }

def calculate_btc_metrics(btc_path):
    """Calculate metrics for Bitcoin buy & hold strategy"""
    btc_df = pd.read_csv(btc_path)
    btc_df['date'] = pd.to_datetime(btc_df['timestamp'])
    
    # Filter data to match simulation period
    start_date = '2024-02-01'
    end_date = '2024-12-31'
    btc_df = btc_df[(btc_df['date'] >= start_date) & (btc_df['date'] <= end_date)].copy()
    
    # Calculate how much BTC we could buy with initial capital
    initial_capital = 500
    initial_btc_price = btc_df['close'].iloc[0]
    btc_amount = initial_capital / initial_btc_price
    
    # Calculate portfolio value over time
    btc_df['capital'] = btc_amount * btc_df['close']
    final_capital = btc_df['capital'].iloc[-1]
    
    # Now calculate returns based on actual capital changes
    btc_df['returns'] = btc_df['capital'].pct_change().fillna(0)
    
    # Rest of metrics calculations...
    capital_multiple = final_capital / initial_capital
    
    # Calculate drawdown
    btc_df['drawdown'] = (btc_df['capital'] - btc_df['capital'].cummax()) / btc_df['capital'].cummax()
    max_drawdown = btc_df['drawdown'].min()
    
    # Win/Loss calculations
    daily_returns = btc_df['returns']
    winning_days = daily_returns[daily_returns > 0]
    losing_days = daily_returns[daily_returns < 0]
    win_rate = len(winning_days) / len(daily_returns[daily_returns != 0]) if len(daily_returns) > 0 else 0
    
    # Monthly returns
    monthly_returns = btc_df.set_index('date')['returns'].resample('M').apply(
        lambda x: (1 + x).prod() - 1
    ).fillna(0)
    
    # VaR calculation
    var_95 = np.percentile(btc_df['returns'], 5)
    
    # Sortino Ratio
    downside_returns = btc_df['returns'][btc_df['returns'] < 0]
    if len(downside_returns) > 0 and downside_returns.std() != 0:
        sortino = (btc_df['returns'].mean() * 252) / (downside_returns.std() * np.sqrt(252))
    else:
        sortino = 0
    
    return {
        'Trial Type': 'BTC Hold',
        'capital_multiple': f"{capital_multiple:.2f}x",
        'targets_hit': "N/A",
        'slot_utilization': "100.00%",  # Always fully invested
        'sharpe_ratio': f"{(np.mean(btc_df['returns']) * 252) / (np.std(btc_df['returns']) * np.sqrt(252)):.2f}",
        'sortino_ratio': f"{sortino:.2f}",
        'max_drawdown': f"{max_drawdown * 100:.2f}%",
        'win_rate': f"{win_rate * 100:.2f}%",
        'profit_factor': f"{abs(winning_days.sum() / losing_days.sum()):.2f}" if len(losing_days) > 0 else "inf",
        'monthly_win_rate': f"{len(monthly_returns[monthly_returns > 0]) / len(monthly_returns) * 100:.2f}%",
        'time_to_target': "N/A",
        'var_95': f"{var_95 * 100:.2f}%",
        'kelly_ratio': "N/A"  # Not applicable for buy & hold
    }

def print_analysis(all_strategy_results, btc_results, strategy_names):
    """Print comparative analysis between all strategy variants"""
    print("\nComparative Analysis: All Strategies")
    print("=" * 150)
    
    # Define metrics that are neutral (no coloring)
    neutral_metrics = {'Time to Target'}
    
    # Define which metrics are better when higher
    higher_better = {
        'Trial Type': True,
        'Capital Multiple': True,
        'Targets Hit': True,
        'Slot Utilization': True,
        'Sharpe Ratio': True,
        'Sortino Ratio': True,
        'Max Drawdown': True,
        'Win Rate': True,
        'Profit Factor': True,
        'Monthly Win Rate': True,
        'Time to Target': False,
        'VaR (95%)': True,
        'Kelly Ratio': True
    }
    
    # Group results by trial type
    trial_types = {result['Trial Type'] for results in all_strategy_results for result in results}
    
    # Color codes
    GREEN = '\033[92m'
    RESET = '\033[0m'
    
    for trial_type in sorted(trial_types):
        print(f"\n{trial_type}")
        print("-" * 150)
        
        # Prepare table headers with all strategy variants
        headers = ['Metric']
        for strategy in strategy_names:
            headers.extend([f"{strategy}-A", f"{strategy}-B"])
        headers.append('BTC Hold')
        metrics_table = [headers]
        
        def extract_number(value):
            if isinstance(value, str):
                return float(value.split('x')[0]) if 'x' in value else float(value.rstrip('%'))
            return float(value)
        
        metrics = [
            ("Capital Multiple", "capital_multiple"),
            ("Targets Hit", "targets_hit"),
            ("Slot Utilization", "slot_utilization"),
            ("Sharpe Ratio", "sharpe_ratio"),
            ("Sortino Ratio", "sortino_ratio"),
            ("Max Drawdown", "max_drawdown"),
            ("Win Rate", "win_rate"),
            ("Profit Factor", "profit_factor"),
            ("Monthly Win Rate", "monthly_win_rate"),
            ("Time to Target", "time_to_target"),
            ("VaR (95%)", "var_95")
        ]
        
        for metric_name, metric_key in metrics:
            row = [metric_name]
            best_value = None
            values = []
            
            # Get values for each strategy variant
            for strategy_results in all_strategy_results:
                strategy_trial = next((r for r in strategy_results if r['Trial Type'] == trial_type), None)
                if strategy_trial:
                    value = strategy_trial[metric_key]
                    values.append(value)
                    
                    if value != "N/A":
                        try:
                            num_value = extract_number(value)
                            if best_value is None:
                                best_value = num_value
                            elif higher_better[metric_name]:
                                best_value = max(best_value, num_value)
                            else:
                                best_value = min(best_value, num_value)
                        except:
                            pass
                else:
                    values.append("N/A")
            
            # Add values to row with highlighting
            for value in values:
                if value != "N/A":
                    try:
                        num_value = extract_number(value)
                        if num_value == best_value and metric_name not in neutral_metrics:
                            row.append(f"{GREEN}{value}{RESET}")
                        else:
                            row.append(value)
                    except:
                        row.append(value)
                else:
                    row.append(value)
            
            # Add BTC Hold value
            row.append(btc_results[metric_key])
            metrics_table.append(row)
        
        print(tabulate(metrics_table, headers='firstrow', tablefmt='pretty'))

def analyze_performance(conn, strategy_suffix):
    """Analyze performance for all blocks in a strategy"""
    cursor = conn.cursor()
    
    # Get database name from connection
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    db_name = Path(db_path).stem  # Gets 'S1', 'S2', or 'S3' from 'S1.db', 'S2.db', or 'S3.db'
    
    # Use strategy_suffix ('A' or 'B') to find tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?", 
                  (f'%s2{strategy_suffix.lower()}%account%regime%',))
    tables = [row[0] for row in cursor.fetchall()]
    
    results = []
    for table in tables:
        if 'regime_independent' in table:
            base_type = 'Independent'
        else:
            base_type = 'Sensitive'
            
        if 'continuous' in table:
            trial_type = f"{base_type} Continuous"
        else:
            trial_type = f"{base_type} Target Triggered"
            
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        block_metrics = []
        
        for block_id in df['block_id'].unique():
            block_df = df[df['block_id'] == block_id].copy()
            metrics = calculate_metrics_for_block(block_df, conn, table, block_id)
            block_metrics.append(metrics)
        
        # Calculate average metrics across blocks
        avg_time_to_target_values = [m['time_to_target'] for m in block_metrics if m['time_to_target'] is not None]
        avg_time_to_target = f"{np.mean(avg_time_to_target_values):.1f}" if avg_time_to_target_values else "N/A"
        
        avg_metrics = {
            'Trial Type': trial_type,
            'capital_multiple': f"{np.mean([m['capital_multiple'] for m in block_metrics]):.2f}x",
            'targets_hit': f"{np.mean([m['targets_hit'] for m in block_metrics]):.1f}",
            'slot_utilization': f"{np.mean([m['slot_utilization'] for m in block_metrics]):.2f}%",
            'sharpe_ratio': f"{np.mean([m['sharpe_ratio'] for m in block_metrics]):.2f}",
            'sortino_ratio': f"{np.mean([m['sortino_ratio'] for m in block_metrics]):.2f}",
            'max_drawdown': f"{np.mean([m['max_drawdown'] for m in block_metrics]):.2f}%",
            'win_rate': f"{np.mean([m['win_rate'] for m in block_metrics]):.2f}%",
            'profit_factor': f"{np.mean([m['profit_factor'] for m in block_metrics]):.2f}",
            'monthly_win_rate': f"{np.mean([m['monthly_win_rate'] for m in block_metrics]):.2f}%",
            'time_to_target': avg_time_to_target,
            'var_95': f"{np.mean([m['var_95'] for m in block_metrics]):.2f}%"
        }
        results.append(avg_metrics)
    
    return results

def print_summary_statistics(all_strategy_results, strategy_name, conn, best_trial_type, variant):
    """Print aggregate summary statistics for the best performing trial type"""
    print("\nAGGREGATE SUMMARY STATISTICS")
    print("=" * 100)
    
    # Print best performing trial type
    print(f"\nBEST PERFORMING TRIAL: {strategy_name}-{variant} {best_trial_type}")
    print("-" * 50)
    
    # Get detailed statistics from raw data
    stats = calculate_detailed_statistics(conn, variant, best_trial_type)
    
    # Print only the detailed statistics
    for category, metrics in stats.items():
        print(f"\n{category}:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

def calculate_detailed_statistics(conn, variant, best_trial_type):
    """Calculate detailed statistics from raw database data for the best performing case"""
    table_name = f"s2a_account_regime_{best_trial_type.lower().replace(' ', '_')}"
    if variant.endswith('-B'):
        table_name = table_name.replace('s2a', 's2b')
    
    # Get raw data including dates and target information
    query = f"""
    SELECT 
        block_id,
        date,
        capital,
        target_hits
    FROM {table_name}
    ORDER BY block_id, date
    """
    
    df = pd.read_sql_query(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    
    # Get simulation period
    start_date = df['date'].min().strftime('%Y-%m-%d')
    end_date = df['date'].max().strftime('%Y-%m-%d')
    total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    
    block_stats = []
    target_times = {}  # Store target timing information for each block
    initial_capital = 500  # Starting capital
    
    for block_id in df['block_id'].unique():
        block_data = df[df['block_id'] == block_id].copy()
        
        # Calculate returns
        final_capital = block_data['capital'].iloc[-1]
        return_multiple = final_capital / initial_capital
        
        # Calculate time to targets
        target_hits = block_data['target_hits'].diff().fillna(0)
        target_dates = block_data[target_hits > 0]['date']
        if len(target_dates) > 1:
            time_diffs = target_dates.diff().dropna()
            avg_time_to_target = time_diffs.mean().total_seconds() / (24 * 60 * 60)  # Convert to days
            target_times[block_id] = avg_time_to_target
        
        # Rest of calculations...
        block_data['peak'] = block_data['capital'].cummax()
        block_data['drawdown'] = (block_data['capital'] - block_data['peak']) / block_data['peak']
        max_drawdown = block_data['drawdown'].min()
        
        block_data['daily_returns'] = block_data['capital'].pct_change().fillna(0)
        returns_mean = block_data['daily_returns'].mean() * 252
        returns_std = block_data['daily_returns'].std() * np.sqrt(252)
        sharpe = returns_mean / returns_std if returns_std != 0 else 0
        
        downside_returns = block_data['daily_returns'][block_data['daily_returns'] < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = returns_mean / downside_std if downside_std != 0 else 0
        
        final_targets = block_data['target_hits'].iloc[-1]
        
        block_stats.append({
            'block_id': block_id,
            'return_multiple': return_multiple,
            'max_drawdown': max_drawdown,
            'targets': final_targets,
            'sharpe': sharpe,
            'sortino': sortino,
            'final_capital': final_capital
        })
    
    block_stats_df = pd.DataFrame(block_stats)
    
    # Calculate top 3 average time to target
    top_3_blocks = block_stats_df.nlargest(3, 'return_multiple')['block_id'].tolist()
    top_3_times = [target_times.get(block_id, 0) for block_id in top_3_blocks if block_id in target_times]
    avg_time_to_target = f"{np.mean(top_3_times):.1f} days" if top_3_times else "N/A"
    
    # Calculate average ending capital and ROI
    avg_ending_capital = block_stats_df['final_capital'].mean()
    avg_roi = ((avg_ending_capital - initial_capital) / initial_capital) * 100
    
    stats = {
        "Summary": {
            "Bot Activation Period": f"{start_date} - {end_date}",
            "Average Capital Change": f"${initial_capital} -> ${avg_ending_capital:.2f} in {total_days} days",
            "Average ROI": f"{avg_roi:.2f}%",
            "TOP 3 Average Time to Target": avg_time_to_target
        },
        "Returns Distribution": {
            "Best Return": f"{block_stats_df['return_multiple'].max():.2f}x",
            "Worst Return": f"{block_stats_df['return_multiple'].min():.2f}x",
            "Mean Return": f"{block_stats_df['return_multiple'].mean():.2f}x",
            "Median Return": f"{block_stats_df['return_multiple'].median():.2f}x",
            "Return Std Dev": f"{block_stats_df['return_multiple'].std():.2f}x"
        },
        "Drawdown Statistics": {
            "Worst Drawdown": f"{block_stats_df['max_drawdown'].min()*100:.2f}%",
            "Best Drawdown": f"{block_stats_df['max_drawdown'].max()*100:.2f}%",
            "Mean Drawdown": f"{block_stats_df['max_drawdown'].mean()*100:.2f}%",
            "Median Drawdown": f"{block_stats_df['max_drawdown'].median()*100:.2f}%"
        },
        "Target Statistics": {
            "Max Targets Hit": f"{block_stats_df['targets'].max():.1f}",
            "Min Targets Hit": f"{block_stats_df['targets'].min():.1f}",
            "Mean Targets Hit": f"{block_stats_df['targets'].mean():.1f}",
            "Total Targets Hit": f"{block_stats_df['targets'].sum():.0f}"
        },
        "Risk-Adjusted Returns": {
            "Best Sharpe": f"{block_stats_df['sharpe'].max():.2f}",
            "Mean Sharpe": f"{block_stats_df['sharpe'].mean():.2f}",
            "Best Sortino": f"{block_stats_df['sortino'].max():.2f}",
            "Mean Sortino": f"{block_stats_df['sortino'].mean():.2f}"
        }
    }
    
    return stats

if __name__ == "__main__":
    # List of database paths and their strategy names
    db_paths = ["results/S1.db", "results/S2.db", "results/S3.db"]
    strategy_names = ['S1', 'S2', 'S3']
    btc_path = 'data/raw/BTCUSDT_1d_2023-09-01_2025-01-01.csv'
    
    # Store results for all strategies
    all_strategy_results = []
    best_overall = {
        'performance': 0,
        'strategy': None,
        'variant': None,
        'trial_type': None,
        'conn': None
    }
    
    # Analyze each database
    for db_path, strategy_name in zip(db_paths, strategy_names):
        conn = sqlite3.connect(db_path)
        
        # Get results for both A and B variants (table names still start with s2a/s2b)
        a_results = analyze_performance(conn, 'A')
        b_results = analyze_performance(conn, 'B')
        all_strategy_results.extend([a_results, b_results])
        
        # Track best performing trial
        for results in [a_results, b_results]:
            for result in results:
                performance = float(result['capital_multiple'].rstrip('x'))
                if performance > best_overall['performance']:
                    best_overall.update({
                        'performance': performance,
                        'strategy': strategy_name,
                        'variant': 'A' if results == a_results else 'B',
                        'trial_type': result['Trial Type'],
                        'conn': conn
                    })
    
    # Calculate BTC metrics once
    btc_results = calculate_btc_metrics(btc_path)
    
    # Print comparative analysis for all strategies
    print_analysis(all_strategy_results, btc_results, strategy_names)
    
    # Print summary statistics for best overall trial
    print(f"\nBest Overall Strategy: {best_overall['strategy']}-{best_overall['variant']} {best_overall['trial_type']}")
    print_summary_statistics(
        all_strategy_results,
        best_overall['strategy'],
        best_overall['conn'],
        best_overall['trial_type'],
        best_overall['variant']
    )
    
    # Close all connections
    for db_path in db_paths:
        conn = sqlite3.connect(db_path)
        conn.close() 