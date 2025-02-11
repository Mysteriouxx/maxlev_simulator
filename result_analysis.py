import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

INITIAL_CAPITAL = 500  # Set your initial capital here once

def connect_to_db(db_path):
    """Establish connection to the SQLite database"""
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def get_account_history(conn, table_name):
    """Fetch average and top 3 candidates data"""
    # Query for daily averages
    if 'target_triggered' in table_name:
        avg_query = f"""
        SELECT date, AVG(capital + (target_hits * 9500)) as avg_capital,
               AVG(target_hits) as avg_target_hits
        FROM {table_name}
        GROUP BY date
        ORDER BY date
        """
        
        top_candidates_query = f"""
        WITH LastDayMetrics AS (
            SELECT block_id, capital + (target_hits * 9500) as final_metric
            FROM {table_name}
            WHERE date = (SELECT MAX(date) FROM {table_name})
        )
        SELECT block_id
        FROM LastDayMetrics
        ORDER BY final_metric DESC
        LIMIT 3
        """
    else:
        avg_query = f"""
        SELECT date, AVG(capital) as avg_capital,
               0 as avg_target_hits
        FROM {table_name}
        GROUP BY date
        ORDER BY date
        """
        
        top_candidates_query = f"""
        WITH LastDayMetrics AS (
            SELECT block_id, capital as final_metric
            FROM {table_name}
            WHERE date = (SELECT MAX(date) FROM {table_name})
        )
        SELECT block_id
        FROM LastDayMetrics
        ORDER BY final_metric DESC
        LIMIT 3
        """
    
    try:
        # Get average data
        avg_df = pd.read_sql_query(avg_query, conn)
        avg_df['date'] = pd.to_datetime(avg_df['date'])
        
        # Get top 3 candidates
        top_ids = pd.read_sql_query(top_candidates_query, conn)['block_id'].tolist()
        
        # Get history for top 3 candidates
        top_dfs = []
        for block_id in top_ids:
            candidate_query = f"""
            SELECT 
                date, 
                capital,
                target_hits,
                {block_id} as block_id,
                CASE 
                    WHEN '{table_name}' LIKE '%target_triggered%' THEN capital + (target_hits * 9500)
                    ELSE capital 
                END as adjusted_capital,
                CASE 
                    WHEN target_hits > 0 THEN MIN(CASE WHEN target_hits > 0 THEN date END) OVER ()
                    ELSE NULL
                END as first_target_hit
            FROM {table_name}
            WHERE block_id = {block_id}
            ORDER BY date
            """
            df = pd.read_sql_query(candidate_query, conn)
            df['date'] = pd.to_datetime(df['date'])
            top_dfs.append(df)
        
        return avg_df, top_dfs
    except pd.io.sql.DatabaseError as e:
        print(f"Error fetching data: {e}")
        return None, None

def calculate_time_to_targets(df, initial_capital=500, target_increment=9500):  # Changed from 2000 to 9500
    """Calculate average time to reach each target level"""
    df = df.copy()
    current_target = initial_capital + target_increment
    target_dates = []
    current_max = initial_capital
    
    for idx, row in df.iterrows():
        capital = row['adjusted_capital']
        if capital > current_max:
            current_max = capital
            while current_max >= current_target:
                if len(target_dates) == 0:
                    days = (row['date'] - df['date'].iloc[0]).days
                else:
                    days = (row['date'] - target_dates[-1]).days
                target_dates.append(row['date'])
                current_target += target_increment
    
    if len(target_dates) > 0:
        time_diffs = []
        for i in range(len(target_dates)):
            if i == 0:
                days = (target_dates[i] - df['date'].iloc[0]).days
            else:
                days = (target_dates[i] - target_dates[i-1]).days
            time_diffs.append(days)
        return np.mean(time_diffs), len(target_dates)
    return None, 0

def get_config_info(conn, table_name, block_id):
    """Fetch configuration details for a specific block"""
    config_table = table_name.replace('account', 'config')
    
    if 's2a' in table_name.lower():
        if 'sensitive' in table_name.lower():
            # For S2A sensitive: get all coins for the fixed combo_id, grouped by regime
            query = f"""
            SELECT combo_id, regime, GROUP_CONCAT(coin_name) as coins
            FROM {config_table}
            WHERE block_id = {block_id}
            GROUP BY combo_id, regime
            """
            try:
                config_df = pd.read_sql_query(query, conn)
                if not config_df.empty:
                    regimes = []
                    for regime in ['bull', 'flat', 'bear']:
                        regime_coins = config_df[config_df['regime'] == regime]
                        if not regime_coins.empty:
                            coins = regime_coins['coins'].iloc[0].split(',')
                            regimes.append(f"{regime}: {', '.join(coins)}")
                    return f"Combo {config_df['combo_id'].iloc[0]}\n{' | '.join(regimes)}"
            except Exception as e:
                print(f"Error in S2A sensitive: {e}")
                return "Config not found"
        else:
            # Regular S2A: get all coins for the same combo_id
            query = f"""
            SELECT combo_id, GROUP_CONCAT(DISTINCT coin_name) as coins
            FROM {config_table}
            WHERE block_id = {block_id}
            GROUP BY combo_id
            """
            try:
                config_df = pd.read_sql_query(query, conn)
                if not config_df.empty:
                    coins = config_df['coins'].iloc[0].split(',')
                    combo_id = config_df['combo_id'].iloc[0]
                    return f"Combo {combo_id}: {', '.join(coins)}"
                return "Config not found"  # Add explicit return for empty dataframe
            except Exception as e:
                print(f"Error in S2A regular: {e}")
                return "Config not found"
    else:
        if 'sensitive' in table_name.lower():
            # For S2B sensitive: get coin and combos grouped by regime
            query = f"""
            WITH BlockInfo AS (
                SELECT coin_name FROM {config_table} WHERE block_id = {block_id}
            )
            SELECT coin_name, 
                   GROUP_CONCAT(CASE WHEN regime = 'bull' THEN combo_id END) as bull_combos,
                   GROUP_CONCAT(CASE WHEN regime = 'flat' THEN combo_id END) as flat_combos,
                   GROUP_CONCAT(CASE WHEN regime = 'bear' THEN combo_id END) as bear_combos
            FROM {config_table}
            WHERE coin_name = (SELECT coin_name FROM BlockInfo)
            GROUP BY coin_name
            """
            try:
                config_df = pd.read_sql_query(query, conn)
                if not config_df.empty:
                    coin = config_df['coin_name'].iloc[0]
                    regimes = []
                    if config_df['bull_combos'].iloc[0]:
                        regimes.append(f"bull: {config_df['bull_combos'].iloc[0]}")
                    if config_df['flat_combos'].iloc[0]:
                        regimes.append(f"flat: {config_df['flat_combos'].iloc[0]}")
                    if config_df['bear_combos'].iloc[0]:
                        regimes.append(f"bear: {config_df['bear_combos'].iloc[0]}")
                    return f"{coin}\n{' | '.join(regimes)}"
            except:
                return "Config not found"
        else:
            # Regular S2B: get coin and all its combo_ids
            query = f"""
            WITH BlockInfo AS (
                SELECT coin_name FROM {config_table} WHERE block_id = {block_id}
            )
            SELECT coin_name, GROUP_CONCAT(DISTINCT combo_id) as combo_ids
            FROM {config_table}
            WHERE coin_name = (SELECT coin_name FROM BlockInfo)
            GROUP BY coin_name
            """
            try:
                config_df = pd.read_sql_query(query, conn)
                if not config_df.empty:
                    coin = config_df['coin_name'].iloc[0]
                    combo_ids = config_df['combo_ids'].iloc[0].split(',')
                    return f"{coin} (Combos: {', '.join(combo_ids)})"
            except:
                return "Config not found"
    
    return "Config not found"

def get_btc_performance(start_date, end_date):
    """Calculate BTC spot holding performance for the given period"""
    # Read BTC data
    btc_path = Path('data/raw/BTCUSDT_1d_2023-09-01_2025-01-01.csv')
    btc_df = pd.read_csv(btc_path, parse_dates=['timestamp'])
    
    # Filter date range
    btc_df = btc_df[(btc_df['timestamp'] >= start_date) & (btc_df['timestamp'] <= end_date)]
    
    if btc_df.empty:
        print(f"Warning: No BTC data found between {start_date} and {end_date}")
        return pd.DataFrame(columns=['date', 'btc_capital'])
    
    # Calculate BTC performance using INITIAL_CAPITAL
    initial_btc_price = float(btc_df['close'].iloc[0])
    btc_quantity = INITIAL_CAPITAL / initial_btc_price
    btc_df['btc_capital'] = btc_df['close'].astype(float) * btc_quantity
    
    # Rename timestamp to date for consistency
    btc_df = btc_df.rename(columns={'timestamp': 'date'})
    
    return btc_df[['date', 'btc_capital']]

def plot_capital_comparisons(dataframes, titles, strategy_name, conn, table_names):
    """Plot capital comparisons for all trial configurations"""
    # Get database name from path and format title
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    db_name = Path(db_path).stem  # Gets 'S1', 'S2', or 'S3' from 'S1.db', 'S2.db', or 'S3.db'
    plot_title = f"{db_name}-{strategy_name[-1]}"  # Added hyphen between strategy number and A/B
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(plot_title, fontsize=24, fontweight='bold', y=0.95)
    axes = axes.ravel()
    
    colors = ['red', 'blue', 'green']
    linestyles = ['--', ':', '-.']
    
    for idx, ((avg_df, top_dfs), title, table_name) in enumerate(zip(dataframes, titles, table_names)):
        # Skip empty plots
        if avg_df is None or avg_df.empty or not top_dfs:
            continue
            
        ax = axes[idx]
        
        # Define date_max at the start
        date_max = avg_df['date'].max()  # Move this line up
        
        # Plot BTC line - only if we have data
        try:
            btc_df = get_btc_performance(avg_df['date'].min(), avg_df['date'].max())
            if not btc_df.empty:
                btc_final_roi = ((btc_df['btc_capital'].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
                ax.plot(btc_df['date'], btc_df['btc_capital'], 
                       linewidth=2, color='gold', label='BTC Spot')
                
                # Add BTC annotation
                ax.annotate(f'ROI: {btc_final_roi:.1f}%',
                           xy=(date_max, btc_df['btc_capital'].iloc[-1]),
                           xytext=(10, 0),
                           textcoords='offset points',
                           fontsize=8,
                           color='gold',
                           va='center',
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        except Exception as e:
            print(f"Warning: Could not plot BTC data for {title}: {e}")
        
        # Plot average line
        final_avg_roi = ((avg_df['avg_capital'].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        avg_att, avg_targets = calculate_time_to_targets(
            pd.DataFrame({'date': avg_df['date'], 'adjusted_capital': avg_df['avg_capital']})
        )
        ax.plot(avg_df['date'], avg_df['avg_capital'], 
               linewidth=2, color='black', label='Average')
        
        # Plot top candidates
        for i, (df, color, style) in enumerate(zip(top_dfs, colors, linestyles)):
            config_info = get_config_info(conn, table_name, df['block_id'].iloc[0])
            final_roi = ((df['adjusted_capital'].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            att, num_targets = calculate_time_to_targets(df)
            
            ax.plot(df['date'], df['adjusted_capital'],
                   linewidth=2, color=color, linestyle=style,
                   label=f'#{i+1} {config_info}')
        
        # Calculate y-positions for annotations
        y_max = max([df['adjusted_capital'].max() for df in top_dfs] + [avg_df['avg_capital'].max()])
        y_min = min([df['adjusted_capital'].min() for df in top_dfs] + [avg_df['avg_capital'].min()])
        y_range = y_max - y_min
        
        # Position annotations with larger vertical spacing
        date_max = avg_df['date'].max()
        y_positions = [
            y_max * 0.95,  # Top position
            y_max * 0.75,  # Middle position
            y_max * 0.55   # Bottom position
        ]
        
        # Add annotations for all lines
        for i, (df, color, y_pos) in enumerate(zip(top_dfs, colors, y_positions)):
            final_roi = ((df['adjusted_capital'].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            att, num_targets = calculate_time_to_targets(df)
            block_id = df['block_id'].iloc[0]
            
            metrics = f'ROI: {final_roi:.1f}%\nTargets: {num_targets}'
            if att:
                metrics += f'\nATT: {int(att)}d'
            metrics += f'\n$\\mathbf{{B{block_id}}}$'  # Add bold block ID using math mode
                
            ax.annotate(metrics,
                       xy=(date_max, y_pos),
                       xytext=(10, 0),
                       textcoords='offset points',
                       fontsize=8,
                       color=color,
                       va='center',
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                       parse_math=True)  # Enable math mode parsing
        
        # Add BTC and average annotations
        ax.annotate(f'ROI: {final_avg_roi:.1f}%\nTargets: {avg_targets}',
                   xy=(date_max, avg_df['avg_capital'].iloc[-1]),
                   xytext=(10, 0),
                   textcoords='offset points',
                   fontsize=8,
                   color='black',
                   va='center',
                   bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Set labels and formatting
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Capital')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=30)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
        
        # Extend y-axis limits
        ax.set_ylim(y_min, y_max * 1.1)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Create output path with modified filename
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    output_filename = f'{db_name}_capital_comparisons_{strategy_name[-1].upper()}.png'  # Filename stays the same
    
    plt.savefig(output_dir / output_filename, 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot has been saved to output/{output_filename}")

def main():
    db_path = "results/S3.db"
    
    strategies = {
        'S2A': {
            'tables': [
                "s2a_account_regime_independent_continuous",
                "s2a_account_regime_independent_target_triggered",
                "s2a_account_regime_sensitive_continuous",
                "s2a_account_regime_sensitive_target_triggered"
            ],
            'titles': [
                "Independent Continuous",
                "Independent Target Triggered",
                "Sensitive Continuous",
                "Sensitive Target Triggered"
            ]
        },
        'S2B': {
            'tables': [
                "s2b_account_regime_independent_continuous",
                "s2b_account_regime_independent_target_triggered",
                "s2b_account_regime_sensitive_continuous",
                "s2b_account_regime_sensitive_target_triggered"
            ],
            'titles': [
                "Independent Continuous",
                "Independent Target Triggered",
                "Sensitive Continuous",
                "Sensitive Target Triggered"
            ]
        }
    }
    
    conn = connect_to_db(db_path)
    if conn is None:
        return
    
    try:
        for strategy_name, config in strategies.items():
            dataframes = []
            for table in config['tables']:
                avg_df, top_dfs = get_account_history(conn, table)
                if avg_df is not None and top_dfs is not None:
                    dataframes.append((avg_df, top_dfs))
                else:
                    print(f"No data found in table: {table}")
                    return
            
            plot_capital_comparisons(dataframes, config['titles'], strategy_name, conn, config['tables'])
            
    finally:
        conn.close()

if __name__ == "__main__":
    main()
