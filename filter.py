import sqlite3
from pathlib import Path
import time
import math
from tabulate import tabulate
import sqlite3

def get_latest_db():
    """Get the database file from the data/training directory"""
    db_path = 'data/training/Sim.db'
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file {db_path} not found")
    return db_path

def filter_results(db_path=None):
    """Filter combos based on step 1 and step 2 criteria"""
    start_time = time.time()
    
    if db_path is None:
        db_path = get_latest_db()
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Initialize result structures
        f3a_pool1 = {'bull': [], 'bear': [], 'flat': []}
        f3a_pool2 = {'combos': []}
        f3b_pool1 = {'bull': [], 'bear': [], 'flat': []}
        f3b_pool2 = {'coins': []}

        # Step 1: Initial filtering
        cursor.execute("""
            SELECT COUNT(DISTINCT w.combo_id)
            FROM window_results w
            JOIN regime_results r ON w.result_id = r.result_id
            WHERE r.regime_type = 'overall'
        """)
        total_combos = cursor.fetchone()[0]

        # Drop existing table and create new one for step 1
        cursor.execute("DROP TABLE IF EXISTS filtered_step1")
        cursor.execute("""
            CREATE TABLE filtered_step1 (
                combo_id TEXT PRIMARY KEY,
                combo_name TEXT,
                win_rate FLOAT,
                trade_count INTEGER,
                triples INTEGER,
                distance_mean FLOAT,
                distance_std FLOAT,
                num_coins INTEGER
            )
        """)
        
        # Insert filtered combo_ids with metrics for step 1
        cursor.execute("""
            WITH ValidResults AS (
                SELECT 
                            w.combo_id,
                    w.coin,
                            r.win_rate,
                            r.trades,
                            r.triple_win_count,
                    r.distance_mean,
                    r.distance_std
                        FROM window_results w
                        JOIN regime_results r ON w.result_id = r.result_id
                WHERE r.regime_type = 'overall'
                AND r.trades >= 75
                AND r.improvement >= 0.05
            )
            INSERT INTO filtered_step1 
            SELECT 
                v.combo_id,
                cm.indicators as combo_name,
                AVG(v.win_rate) as win_rate,
                AVG(v.trades) as trade_count,
                AVG(v.triple_win_count) as triples,
                AVG(v.distance_mean) as distance_mean,
                AVG(v.distance_std) as distance_std,
                COUNT(DISTINCT v.coin) as num_coins
            FROM ValidResults v
            JOIN combo_metadata cm ON v.combo_id = cm.combo_id
            GROUP BY v.combo_id, cm.indicators
        """)
        
        # Get count of filtered combos from step 1
        cursor.execute("SELECT COUNT(*) FROM filtered_step1")
        filtered_step1 = cursor.fetchone()[0]

        # Step 2: Calculate additional metrics and apply filters
        cursor.execute("""
            WITH MetricsStep2 AS (
                SELECT 
                    *,
                    -- Calculate TWE (Triple-win Efficiency)
                    CAST(triples AS FLOAT) / CAST(trade_count AS FLOAT) as twe,
                    -- Calculate Est. Duration
                    3 * distance_mean * 0.25 as est_duration,
                    -- Calculate Risk Penalty components
                    (distance_std / distance_mean) as volatility_ratio,
                    -- Calculate Triple Return Stats
                    -- For successful triples: 41.875 * number of triples
                    -- For failed attempts: -1 * (trade_count - triples)
                    (41.875 * triples - (trade_count - triples)) / trade_count as triple_return_mean,
                    SQRT(
                        (POWER(41.875 - ((41.875 * triples - (trade_count - triples)) / trade_count), 2) * triples + 
                         POWER(-1 - ((41.875 * triples - (trade_count - triples)) / trade_count), 2) * (trade_count - triples)) 
                        / trade_count
                    ) as triple_return_std
                FROM filtered_step1
            ),
            FilteredByTWE AS (
                SELECT 
                    *,
                    -- Calculate Risk Penalty
                    (volatility_ratio * est_duration * (1 + LOG(est_duration/24.0))) as risk_penalty,
                    -- Calculate IR (Information Ratio)
                    41.875 * (twe - 0.02332) / triple_return_std as ir,
                    -- Calculate WCC (Win Cluster Coefficient)
                    twe/POWER(win_rate, 3) as wcc
                FROM MetricsStep2
                WHERE twe > 0.02332
            ),
            FinalMetrics AS (
                SELECT 
                    *,
                    -- Calculate TQS (Trade Quality Score)
                    ir * wcc as tqs,
                    -- Calculate EVD (Expected Value per Day)
                    (41.875 * twe) / ((est_duration + risk_penalty) / 24.0) as evd
                FROM FilteredByTWE
            ),
            Percentiles AS (
                SELECT 
                    *,
                    NTILE(4) OVER (ORDER BY tqs) as tqs_percentile,
                    NTILE(4) OVER (ORDER BY evd) as evd_percentile
                FROM FinalMetrics
            )
            SELECT 
                (SELECT COUNT(*) FROM filtered_step1) as total_from_step1,
                (SELECT COUNT(*) FROM FilteredByTWE) as passed_twe,
                COUNT(CASE WHEN tqs_percentile > 3 AND evd_percentile > 3 THEN 1 END) as passed_all
            FROM Percentiles
        """)
        
        counts = cursor.fetchone()
        total_from_step1, passed_twe, passed_all = counts

        # F3A Pool 1 (Regime-sensitive) calculations
        for regime in ['bull', 'bear', 'flat']:
            cursor.execute("""
                WITH ComboTWE AS (
                    -- Calculate TWE for each result_id
                    SELECT DISTINCT  -- Added DISTINCT to ensure unique combinations
                        w.combo_id,
                        w.result_id,
                        w.coin,
                        cm.indicators as combo_name,
                        cm.window_size,
                        r.direction,
                        CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT) as twe,
                        r.trades,
                        r.win_rate,
                        r.distance_mean,
                        r.distance_std
                    FROM window_results w
                    JOIN regime_results r ON w.result_id = r.result_id
                    JOIN combo_metadata cm ON w.combo_id = cm.combo_id
                    WHERE r.regime_type = ?
                    AND w.combo_id IN (SELECT combo_id FROM filtered_step1)
                ),
                Top3PerCombo AS (
                    -- Get top 3 coins by TWE for each combo
                    SELECT 
                        *,
                        ROW_NUMBER() OVER (PARTITION BY combo_id ORDER BY twe DESC) as rank
                    FROM ComboTWE
                    WHERE trades >= 75  -- Apply minimum trades filter
                ),
                MetricsTop3 AS (
                    -- Calculate metrics for top 3
                    SELECT 
                        combo_id,
                        combo_name,
                        direction,
                        window_size,
                        AVG(twe) as avg_twe_top3,
                        MIN(twe) as min_twe_top3,
                        GROUP_CONCAT(DISTINCT coin) as top_3_coins,  -- Added DISTINCT
                        AVG(distance_mean) as avg_distance_mean,
                        AVG(distance_std) as avg_distance_std,
                        AVG(win_rate) as avg_win_rate,
                        COUNT(DISTINCT coin) as coin_count  -- Added to verify we have 3 distinct coins
                    FROM Top3PerCombo
                    WHERE rank <= 3
                    GROUP BY combo_id, combo_name, direction, window_size
                    HAVING coin_count = 3  -- Ensure we have exactly 3 distinct coins
                ),
                FinalScores AS (
                    -- Calculate final regime scores
                    SELECT 
                        m.*,
                        (0.7 * avg_twe_top3 + 0.3 * min_twe_top3) as cross_coin_score,
                        -- Recalculate TQS and EVD for top 3
                        41.875 * (avg_twe_top3 - 0.02332) / 
                        SQRT(POWER(41.875 - avg_twe_top3 * 41.875, 2) * avg_twe_top3 + 
                             POWER(-1 - avg_twe_top3 * (-1), 2) * (1 - avg_twe_top3)) as ir,
                        avg_twe_top3/POWER(avg_win_rate, 3) as wcc,
                        3 * avg_distance_mean * 0.25 as est_duration,
                        (avg_distance_std / avg_distance_mean) as volatility_ratio
                    FROM MetricsTop3 m
                ),
                RegimeScores AS (
                    SELECT 
                        *,
                        (volatility_ratio * est_duration * (1 + LOG(est_duration/24.0))) as risk_penalty,
                        (41.875 * avg_twe_top3) / ((est_duration + 
                            (volatility_ratio * est_duration * (1 + LOG(est_duration/24.0)))) / 24.0) as evd,
                        cross_coin_score * 
                        (ir * wcc) * -- TQS
                        ((41.875 * avg_twe_top3) / ((est_duration + 
                            (volatility_ratio * est_duration * (1 + LOG(est_duration/24.0)))) / 24.0)) -- EVD
                        as regime_score
                    FROM FinalScores
                )
                SELECT 
                    combo_id,
                    combo_name,
                    direction,
                    window_size,
                    regime_score,
                    top_3_coins,
                    avg_twe_top3
                FROM RegimeScores
                ORDER BY regime_score DESC
                LIMIT 14
            """, (regime,))
            
            results = cursor.fetchall()
            if not results:
                print(f"No results found for {regime} regime")
                continue  # Skip to next regime if no results found
            else:
                for row in results:
                    combo_id, name, direction, window_size, score, coins, avg_twe = row
                    combo_data = {
                        'combo_id': combo_id,
                        'combo_name': name,
                        'direction': direction,
                        'window_size': window_size * 24,
                        'signals': name.split('+'),
                        'top_3_coins': coins.split(','),
                        'combo_score': float(score),
                        'twe': float(avg_twe)
                    }
                    f3a_pool1[regime].append(combo_data)

        # F3A Pool 2 (Regime-independent) calculations
        cursor.execute("""
            WITH ComboTWE AS (
                -- Calculate TWE for each result_id in recency regime
                SELECT DISTINCT
                    w.combo_id,
                    w.result_id,
                    w.coin,
                    cm.indicators as combo_name,
                    cm.window_size,
                    r.direction,
                    CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT) as twe,
                    r.trades
                FROM window_results w
                JOIN regime_results r ON w.result_id = r.result_id
                JOIN combo_metadata cm ON w.combo_id = cm.combo_id
                WHERE r.regime_type = 'recency'
                AND w.combo_id IN (SELECT combo_id FROM filtered_step1)
                AND r.trades >= 75
            ),
            ComboStats AS (
                -- Calculate min and mean TWE for all coins
                SELECT 
                    combo_id,
                    MIN(twe) as min_twe_all,
                    AVG(twe) as mean_twe_all
                FROM ComboTWE
                GROUP BY combo_id
            ),
            Top3PerCombo AS (
                -- Get top 3 coins by TWE
                SELECT 
                    c.*,
                    cs.min_twe_all,
                    cs.mean_twe_all,
                    ROW_NUMBER() OVER (PARTITION BY c.combo_id ORDER BY c.twe DESC) as rank
                FROM ComboTWE c
                JOIN ComboStats cs ON c.combo_id = cs.combo_id
            ),
            FinalScores AS (
                -- Calculate final independence scores
                SELECT 
                    combo_id,
                    combo_name,
                    direction,
                    window_size,
                    GROUP_CONCAT(DISTINCT coin) as top_3_coins,
                    AVG(twe) as performance,
                    min_twe_all / mean_twe_all as consistency,
                    (AVG(twe) * (min_twe_all / mean_twe_all)) as independence_score,
                    COUNT(DISTINCT coin) as coin_count
                FROM Top3PerCombo
                WHERE rank <= 3
                GROUP BY combo_id, combo_name, direction, window_size, min_twe_all, mean_twe_all
                HAVING coin_count = 3
            )
            SELECT 
                combo_id,
                combo_name,
                direction,
                window_size,
                independence_score,
                top_3_coins,
                performance as avg_twe_top3
            FROM FinalScores
            ORDER BY independence_score DESC
            LIMIT 14
        """)
        
        results = cursor.fetchall()
        if not results:
            print(f"No results found for recency regime")
        else:
            for row in results:
                combo_id, name, direction, window_size, score, coins, avg_twe = row
                combo_data = {
                    'combo_id': combo_id,
                    'combo_name': name,
                    'direction': direction,
                    'window_size': window_size * 24,
                    'signals': name.split('+'),
                    'top_3_coins': coins.split(','),
                    'combo_score': float(score),
                    'twe': float(avg_twe)
                }
                f3a_pool2['combos'].append(combo_data)

        # F3B Pool 1 (Regime-sensitive, coin-focused) calculations
        for regime in ['bull', 'bear', 'flat']:
            cursor.execute("""
                WITH ComboScores AS (
                    -- Calculate combo scores for each coin-combo pair
                    SELECT DISTINCT  -- Ensure distinct coin-combo pairs
                        w.coin,
                        w.combo_id,
                        cm.indicators as combo_name,
                        cm.window_size,
                        r.direction,
                        CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT) as twe,
                        r.trades,
                        r.win_rate,
                        r.distance_mean,
                        r.distance_std,
                        -- Calculate IR for TQS
                        41.875 * (CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT) - 0.02332) / 
                        SQRT(POWER(41.875 - (CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT)) * 41.875, 2) * 
                             (CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT)) + 
                             POWER(-1 - (CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT)) * (-1), 2) * 
                             (1 - (CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT)))) as ir,
                        -- Calculate WCC
                        (CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT)) / POWER(r.win_rate, 3) as wcc,
                        -- Calculate EVD components
                        3 * r.distance_mean * 0.25 as est_duration,
                        (r.distance_std / r.distance_mean) * 
                        (3 * r.distance_mean * 0.25) * 
                        (1 + LOG((3 * r.distance_mean * 0.25)/24.0)) as risk_penalty
                    FROM window_results w
                    JOIN regime_results r ON w.result_id = r.result_id
                    JOIN combo_metadata cm ON w.combo_id = cm.combo_id
                    WHERE r.regime_type = ?
                    AND w.combo_id IN (SELECT combo_id FROM filtered_step1)
                    AND r.trades >= 75
                ),
                ComboScoresWithEVD AS (
                    SELECT 
                        *,
                        -- Calculate EVD
                        (41.875 * twe) / ((est_duration + risk_penalty) / 24.0) as evd,
                        -- Calculate TQS
                        ir * wcc as tqs,
                        -- Calculate final combo score
                        twe * (ir * wcc) * ((41.875 * twe) / ((est_duration + risk_penalty) / 24.0)) as combo_score
                    FROM ComboScores
                ),
                RankedCombos AS (
                    -- Get unique top combos for each coin with rank
                    SELECT 
                        *,
                        ROW_NUMBER() OVER (PARTITION BY coin ORDER BY combo_score DESC) as rank
                    FROM (
                        SELECT DISTINCT coin, combo_id, combo_name, direction, window_size, combo_score, twe
                        FROM ComboScoresWithEVD
                    )
                ),
                CoinStats AS (
                    -- Calculate weighted score and CV for each coin using distinct top 5
                    SELECT 
                        coin,
                        -- Weighted Score (WS)
                        SUM(CASE 
                            WHEN rank = 1 THEN 0.4 * combo_score
                            WHEN rank = 2 THEN 0.3 * combo_score
                            WHEN rank = 3 THEN 0.2 * combo_score
                            ELSE 0.05 * combo_score
                        END) as weighted_score,
                        -- Calculate CV (std/mean)
                        SQRT(AVG(combo_score * combo_score) - AVG(combo_score) * AVG(combo_score)) / 
                            NULLIF(AVG(combo_score), 0) as cv
                    FROM RankedCombos
                    WHERE rank <= 5
                    GROUP BY coin
                    HAVING COUNT(DISTINCT combo_id) = 5  -- Ensure 5 distinct combos
                ),
                CoinScores AS (
                    -- Calculate final coin scores
                    SELECT 
                        coin,
                        weighted_score * (1 - cv) as coin_score
                    FROM CoinStats
                ),
                Top3CombosPerCoin AS (
                    -- Get top 3 distinct combos for each coin
                    SELECT 
                        r.*,
                        cs.coin_score
                    FROM RankedCombos r
                    JOIN CoinScores cs ON r.coin = cs.coin
                    WHERE r.rank <= 3
                )
                SELECT 
                    t.coin,
                    t.coin_score,
                    GROUP_CONCAT(t.combo_id) as top_3_combos,
                    GROUP_CONCAT(t.combo_name) as top_3_names,
                    GROUP_CONCAT(t.direction) as directions,
                    GROUP_CONCAT(t.combo_score) as combo_scores,
                    GROUP_CONCAT(t.twe) as twes,
                    GROUP_CONCAT(t.window_size) as window_sizes
                FROM Top3CombosPerCoin t
                GROUP BY t.coin, t.coin_score
                ORDER BY t.coin_score DESC
                LIMIT 14
            """, (regime,))
            
            results = cursor.fetchall()
            if not results:
                print(f"No results found for {regime} regime")
                continue  # Skip to next regime if no results found
            else:
                for row in results:
                    coin, coin_score, combo_ids, combo_names, directions, scores, twes, window_sizes = row
                    # Add null checks before converting to float
                    if coin_score is None:
                        continue  # Skip this row if coin_score is None
                        
                    combo_ids = combo_ids.split(',') if combo_ids else []
                    combo_names = combo_names.split(',') if combo_names else []
                    directions = directions.split(',') if directions else []
                    scores = [float(s) for s in scores.split(',')] if scores else []
                    twes = [float(t) for t in twes.split(',')] if twes else []
                    window_sizes = [int(w) for w in window_sizes.split(',')] if window_sizes else []
                    
                    # Only process if we have valid data
                    if combo_ids and combo_names and directions and scores and twes and window_sizes:
                        top_3_combos = []
                        for i in range(len(combo_ids)):
                            combo_data = {
                                'combo_id': combo_ids[i],
                                'combo_name': combo_names[i],
                                'direction': directions[i],
                                'window_size': window_sizes[i] * 24,
                                'signals': combo_names[i].split('+'),
                                'combo_score': scores[i],
                                'twe': twes[i]
                            }
                            top_3_combos.append(combo_data)
                        
                        coin_data = {
                            'symbol': coin,
                            'coin_score': float(coin_score),
                            'top_3_combos': top_3_combos
                        }
                        f3b_pool1[regime].append(coin_data)

        # F3B Pool 2 (Regime-independent, coin-focused) calculations
        cursor.execute("""
            WITH RecencyScores AS (
                -- Calculate combo scores for each coin-combo pair in recency regime
                SELECT DISTINCT
                    w.coin,
                    w.combo_id,
                    cm.indicators as combo_name,
                    cm.window_size,
                    r.direction,
                    CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT) as twe,
                    r.trades,
                    r.win_rate,
                    r.distance_mean,
                    r.distance_std,
                    -- Calculate IR for TQS
                    41.875 * (CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT) - 0.02332) / 
                    SQRT(POWER(41.875 - (CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT)) * 41.875, 2) * 
                         (CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT)) + 
                         POWER(-1 - (CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT)) * (-1), 2) * 
                         (1 - (CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT)))) as ir,
                    -- Calculate WCC
                    (CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT)) / POWER(r.win_rate, 3) as wcc,
                    -- Calculate EVD components
                    3 * r.distance_mean * 0.25 as est_duration,
                    (r.distance_std / r.distance_mean) * 
                    (3 * r.distance_mean * 0.25) * 
                    (1 + LOG((3 * r.distance_mean * 0.25)/24.0)) as risk_penalty
                FROM window_results w
                JOIN regime_results r ON w.result_id = r.result_id
                JOIN combo_metadata cm ON w.combo_id = cm.combo_id
                WHERE r.regime_type = 'recency'
                AND w.combo_id IN (SELECT combo_id FROM filtered_step1)
                AND r.trades >= 75
            ),
            RegimeScores AS (
                -- Calculate combo scores for each regime for consistency calculation
                SELECT DISTINCT
                    w.coin,
                    w.combo_id,
                    r.regime_type,
                    CAST(r.triple_win_count AS FLOAT) / CAST(r.trades AS FLOAT) as twe,
                    r.trades,
                    r.win_rate,
                    r.distance_mean,
                    r.distance_std
                FROM window_results w
                JOIN regime_results r ON w.result_id = r.result_id
                WHERE r.regime_type IN ('bull', 'bear', 'flat')
                AND w.combo_id IN (SELECT combo_id FROM filtered_step1)
                AND r.trades >= 75
            ),
            RegimeConsistency AS (
                -- Calculate regime consistency (min/max) for each combo
                SELECT 
                    coin,
                    combo_id,
                    MIN(twe) / NULLIF(MAX(twe), 0) as regime_consistency
                FROM RegimeScores
                GROUP BY coin, combo_id
                HAVING COUNT(DISTINCT regime_type) = 3  -- Ensure we have all three regimes
            ),
            RecencyScoresWithEVD AS (
                SELECT 
                    r.*,
                    -- Calculate EVD
                    (41.875 * twe) / ((est_duration + risk_penalty) / 24.0) as evd,
                    -- Calculate TQS
                    ir * wcc as tqs,
                    -- Calculate final combo score
                    twe * (ir * wcc) * ((41.875 * twe) / ((est_duration + risk_penalty) / 24.0)) as combo_score,
                    -- Get regime consistency
                    COALESCE(rc.regime_consistency, 0) as regime_consistency
                FROM RecencyScores r
                LEFT JOIN RegimeConsistency rc ON r.coin = rc.coin AND r.combo_id = rc.combo_id
            ),
            RankedCombos AS (
                -- Get unique top combos for each coin with rank
                SELECT 
                    *,
                    ROW_NUMBER() OVER (PARTITION BY coin ORDER BY combo_score DESC) as rank
                FROM (
                    SELECT DISTINCT 
                        coin, combo_id, combo_name, direction, window_size, combo_score, twe,
                        regime_consistency
                    FROM RecencyScoresWithEVD
                )
            ),
            CoinStats AS (
                -- Calculate weighted score and CV for each coin using distinct top 5
                SELECT 
                    coin,
                    -- Weighted Score (WS)
                    SUM(CASE 
                        WHEN rank = 1 THEN 0.4 * combo_score
                        WHEN rank = 2 THEN 0.3 * combo_score
                        WHEN rank = 3 THEN 0.2 * combo_score
                        ELSE 0.05 * combo_score
                    END) as weighted_score,
                    -- Calculate CV (std/mean)
                    SQRT(AVG(combo_score * combo_score) - AVG(combo_score) * AVG(combo_score)) / 
                        NULLIF(AVG(combo_score), 0) as cv,
                    -- Average regime consistency for top 5 combos
                    AVG(regime_consistency) as avg_regime_consistency
                FROM RankedCombos
                WHERE rank <= 5
                GROUP BY coin
                HAVING COUNT(DISTINCT combo_id) = 5
            ),
            CoinScores AS (
                -- Calculate final coin scores with regime consistency factor
                SELECT 
                    coin,
                    weighted_score * (1 - cv) * (1 + avg_regime_consistency) as coin_score
                FROM CoinStats
            ),
            Top3CombosPerCoin AS (
                -- Get top 3 distinct combos for each coin
                SELECT 
                    r.*,
                    cs.coin_score
                FROM RankedCombos r
                JOIN CoinScores cs ON r.coin = cs.coin
                WHERE r.rank <= 3
            )
            SELECT 
                t.coin,
                t.coin_score,
                GROUP_CONCAT(t.combo_id) as top_3_combos,
                GROUP_CONCAT(t.combo_name) as top_3_names,
                GROUP_CONCAT(t.direction) as directions,
                GROUP_CONCAT(t.combo_score) as combo_scores,
                GROUP_CONCAT(t.twe) as twes,
                GROUP_CONCAT(t.window_size) as window_sizes
            FROM Top3CombosPerCoin t
            GROUP BY t.coin, t.coin_score
            ORDER BY t.coin_score DESC
            LIMIT 14
        """)
        
        results = cursor.fetchall()
        if not results:
            print(f"No results found for recency regime")
        else:
            for row in results:
                coin, coin_score, combo_ids, combo_names, directions, scores, twes, window_sizes = row
                combo_ids = combo_ids.split(',') if combo_ids else []
                combo_names = combo_names.split(',') if combo_names else []
                directions = directions.split(',') if directions else []
                scores = [float(s) for s in scores.split(',')] if scores else []
                twes = [float(t) for t in twes.split(',')] if twes else []
                window_sizes = [int(w) for w in window_sizes.split(',')] if window_sizes else []
                
                top_3_combos = []
                for i in range(len(combo_ids)):
                    combo_data = {
                        'combo_id': combo_ids[i],
                        'combo_name': combo_names[i],
                        'direction': directions[i],
                        'window_size': window_sizes[i] * 24,
                        'signals': combo_names[i].split('+'),
                        'combo_score': scores[i],
                        'twe': twes[i]
                    }
                    top_3_combos.append(combo_data)
                
                coin_data = {
                    'symbol': coin,
                    'coin_score': float(coin_score),
                    'top_3_combos': top_3_combos
                }
                f3b_pool2['coins'].append(coin_data)

        # Organize final results structure
        f3_results = {
            'F3A': {
                'pool_1': f3a_pool1,
                'pool_2': f3a_pool2
            },
            'F3B': {
                'pool_1': f3b_pool1,
                'pool_2': f3b_pool2
            }
        }

        conn.commit()
        return f3_results
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    f3 = filter_results()
    
    # Add null check before processing results
    if f3 is None:
        print("Error: No results returned from filter_results()")
        exit(1)
        
    # Print F3A Pool 1 (Regime-sensitive) Results
    print("\nF3A Pool 1 (Regime-sensitive) Results:")
    for regime in ['bull', 'bear', 'flat']:
        if regime in f3.get('F3A', {}).get('pool_1', {}) and f3['F3A']['pool_1'][regime]:
            print(f"\n{regime.upper()} Regime Top 18:")
            table_data = []
            for combo in f3['F3A']['pool_1'][regime]:
                table_data.append([
                    combo['combo_id'],
                    combo['combo_name'],
                    combo['direction'],
                    f"{combo['combo_score']:.4f}",
                    ', '.join(combo['top_3_coins']),
                    f"{combo['twe']:.4f}"
                ])
            print(tabulate(table_data, headers=['Combo ID', 'Combo Name', 'Direction', 'Score', 'Top 3 Coins', 'TWE'], 
                         tablefmt='grid'))

    # Print F3A Pool 2 (Regime-independent) Results
    print("\nF3A Pool 2 (Regime-independent) Results:")
    table_data = []
    for combo in f3['F3A']['pool_2']['combos']:
        table_data.append([
            combo['combo_id'],
            combo['combo_name'],
            combo['direction'],
            f"{combo['combo_score']:.4f}",
            ', '.join(combo['top_3_coins']),
            f"{combo['twe']:.4f}"
        ])
    print(tabulate(table_data, headers=['Combo ID', 'Combo Name', 'Direction', 'Score', 'Top 3 Coins', 'TWE'], 
                  tablefmt='grid'))

    # Print F3B Pool 1 (Regime-sensitive) Results
    print("\nF3B Pool 1 (Regime-sensitive) Results:")
    for regime in ['bull', 'bear', 'flat']:
        if regime in f3.get('F3B', {}).get('pool_1', {}) and f3['F3B']['pool_1'][regime]:
            print(f"\n{regime.upper()} Regime Top 18:")
            table_data = []
            for coin_data in f3['F3B']['pool_1'][regime]:
                # First row for coin info
                first_combo = coin_data['top_3_combos'][0]
                table_data.append([
                    coin_data['symbol'],
                    f"{coin_data['coin_score']:.4f}",
                    first_combo['combo_id'],
                    first_combo['combo_name'],
                    first_combo['direction'],
                    f"{first_combo['combo_score']:.4f}",
                    f"{first_combo['twe']:.4f}"
                ])
                # Additional rows for remaining combos
                for combo in coin_data['top_3_combos'][1:]:
                    table_data.append([
                        "",  # Empty symbol cell
                        "",  # Empty score cell
                        combo['combo_id'],
                        combo['combo_name'],
                        combo['direction'],
                        f"{combo['combo_score']:.4f}",
                        f"{combo['twe']:.4f}"
                    ])
                # Add separator row
                table_data.append(["-"*10] * 7)
            print(tabulate(table_data, headers=['Symbol', 'Coin Score', 'Combo ID', 'Combo Name', 'Direction', 'Combo Score', 'TWE'], 
                         tablefmt='grid'))

    # Print F3B Pool 2 (Regime-independent) Results
    print("\nF3B Pool 2 (Regime-independent) Results:")
    table_data = []
    for coin_data in f3['F3B']['pool_2']['coins']:
        # First row for coin info
        first_combo = coin_data['top_3_combos'][0]
        table_data.append([
            coin_data['symbol'],
            f"{coin_data['coin_score']:.4f}",
            first_combo['combo_id'],
            first_combo['combo_name'],
            first_combo['direction'],
            f"{first_combo['combo_score']:.4f}",
            f"{first_combo['twe']:.4f}"
        ])
        # Additional rows for remaining combos
        for combo in coin_data['top_3_combos'][1:]:
            table_data.append([
                "",  # Empty symbol cell
                "",  # Empty score cell
                combo['combo_id'],
                combo['combo_name'],
                combo['direction'],
                f"{combo['combo_score']:.4f}",
                f"{combo['twe']:.4f}"
            ])
        # Add separator row
        table_data.append(["-"*10] * 7)
    print(tabulate(table_data, headers=['Symbol', 'Coin Score', 'Combo ID', 'Combo Name', 'Direction', 'Combo Score', 'TWE'], 
                  tablefmt='grid'))