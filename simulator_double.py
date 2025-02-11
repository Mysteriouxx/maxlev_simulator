import numpy as np
from numba import cuda
import numba
from typing import Tuple, Dict
import math
import pandas as pd
from tabulate import tabulate

INITIAL_CAPITAL = 500.0
TARGET_CAPITAL = 10000.0

# CUDA Device Functions
@cuda.jit(device=True)
def check_conditions(block_idx: int, slot_idx: int, candle_idx: int, 
                    slot_configs: np.ndarray, indicator_values: np.ndarray, 
                    threshold_data: np.ndarray, coin_idx: int, 
                    counter_array: np.ndarray) -> bool:
    """
    Check if all indicators in a combo meet their conditions.
    
    Args:
        block_idx: Index of the current block
        slot_idx: Index of the slot to check
        candle_idx: Current candle index (in minutes)
        slot_configs: Array containing slot configurations [n_blocks, n_slots, 5 + max_signals]
                     [0: coin_idx, 1: combo_idx, 2: window_size, 3: direction, 4: regime_lock, 5+: signal_indices]
        indicator_values: All signal values [n_blocks, n_signals, n_coins, n_candles]
        threshold_data: Threshold values [n_blocks, n_coins, n_windows, n_signals, n_candles, 2]
        coin_idx: Index of the coin to check
        counter_array: Array to track signal checks and passes
    """
    # First check if slot is locked
    if slot_configs[block_idx, slot_idx, 4] == 1:  # regime_lock is 1
        return False
        
    # Map window size (24/72/120) to index (0/1/2)
    window_size = slot_configs[block_idx, slot_idx, 2]
    window_idx = -1  # Default invalid index
    
    # Direct mapping based on known window sizes [24, 72, 120]
    if window_size == 24:
        window_idx = 0
    elif window_size == 72:
        window_idx = 1
    elif window_size == 120:
        window_idx = 2
    else:
        # Invalid window size, exit early
        return False
    
    # Count valid signals (non -1 values)
    n_signals = 0
    for i in range(6):  # max signals
        if slot_configs[block_idx, slot_idx, 5 + i] == -1:
            break
        n_signals += 1
    
    all_conditions_met = True
    
    # Check each signal in the slot
    for sig_pos in range(n_signals):
        signal_idx = slot_configs[block_idx, slot_idx, 5 + sig_pos]
        
        if signal_idx == -1:  # End of valid signals
            break
            
        # Track total signals checked
        cuda.atomic.add(counter_array, 7, 1)  # total_signals_checked
        
        current_value = indicator_values[block_idx, signal_idx, coin_idx, candle_idx]
        
        # Track NaN values in indicators
        if math.isnan(current_value):
            cuda.atomic.add(counter_array, 20, 1)  # nan_indicators
            continue
            
        # Use mapped window_idx to access threshold data
        threshold = threshold_data[block_idx, coin_idx, window_idx, signal_idx, candle_idx, 0]
        condition = threshold_data[block_idx, coin_idx, window_idx, signal_idx, candle_idx, 1]
        
        # Track NaN values in thresholds
        if math.isnan(threshold):
            cuda.atomic.add(counter_array, 21, 1)  # nan_thresholds
            continue
            
        indicator_condition_met = False
        if condition == 0:  # Greater than
            indicator_condition_met = (current_value > threshold)
            cuda.atomic.add(counter_array, 22, 1)  # greater_than_checks
        elif condition == 1:  # Less than
            indicator_condition_met = (current_value < threshold)
            cuda.atomic.add(counter_array, 23, 1)  # less_than_checks
            
        if indicator_condition_met:
            cuda.atomic.add(counter_array, 8, 1)  # signals_passed
        else:
            cuda.atomic.add(counter_array, 24, 1)  # failed_conditions
            
        all_conditions_met = all_conditions_met and indicator_condition_met
        
        if not all_conditions_met:
            break
    
    return all_conditions_met

@cuda.jit(device=True)
def check_atr_conditions(atr_percentage: float) -> Tuple[float, float]:
    """Check ATR conditions and return target/retracement"""
    if atr_percentage < 1.8:  # 1.8%
        return 0.03, 0.018
    elif atr_percentage <= 3.8:  # 3.8%
        return 0.05, 0.038
    else:
        return 0.0, 0.0

@cuda.jit(device=True)
def process_trade(entry_price: float, direction: int, future_prices: np.ndarray, 
                 atr_percentage: float, stage_increment: float,
                 d_counter_array=None) -> Tuple[int, int, int]:
    """Process a single trade with proper stage management"""
    target, retracement = check_atr_conditions(atr_percentage)
    if target == 0.0:
        return -1, 0, -1
    
    trade_won = -1
    trade_resolved = 0
    resolution_offset = -1
    
    final_target = entry_price * (1 + (direction * target))
    initial_stop = entry_price * (1 - direction * retracement)
    current_stop = initial_stop
    
    n_stages = int(target / stage_increment)
    stage_levels = cuda.local.array(10, dtype=numba.float64)
    
    for stage_idx in range(n_stages):
        stage_levels[stage_idx] = entry_price * (1 + (direction * stage_increment * (stage_idx + 1)))
    
    for i in range(future_prices.shape[0]):
        high = future_prices[i, 0]  
        low = future_prices[i, 1]   
        
        if high == -1.0 or low == -1.0:
            trade_resolved = 0
            trade_won = -1
            resolution_offset = -1
            break
        
        if direction == 1 and low <= current_stop:
            trade_resolved = 1
            trade_won = 0
            resolution_offset = i + 1
            break
        elif direction == -1 and high >= current_stop:
            trade_resolved = 1
            trade_won = 0
            resolution_offset = i + 1
            break
        
        price_to_check = high if direction == 1 else low
        if (direction == 1 and price_to_check >= final_target) or \
           (direction == -1 and price_to_check <= final_target):
            trade_resolved = 1
            trade_won = 1
            resolution_offset = i + 1
            break
        
        highest_stage_reached = -1
        for stage in range(n_stages - 2, -1, -1):
            if (direction == 1 and price_to_check >= stage_levels[stage]) or \
               (direction == -1 and price_to_check <= stage_levels[stage]):
                highest_stage_reached = stage
                break
        
        if highest_stage_reached >= 0:
            new_entry = stage_levels[highest_stage_reached]
            potential_new_stop = new_entry * (1 - direction * retracement)
            if direction == 1:
                current_stop = max(current_stop, potential_new_stop)
            else:
                current_stop = min(current_stop, potential_new_stop)

    if trade_resolved == 0:
        trade_won = -1
        resolution_offset = -1
    
    return trade_won, trade_resolved, resolution_offset


@cuda.jit
def process_simulation_kernel(
    prices,               # [n_blocks, n_block_coins, n_candles, 5]
    indicator_values,     # [n_blocks, n_block_signals, n_block_coins, n_candles]
    threshold_data,       # [n_blocks, n_block_coins, n_block_windows, n_signals, n_candles, 2]
    atr_values,          # [n_blocks, n_block_coins, n_candles]
    timeframe_map,       # [n_candles]
    max_slots,           # Number of slots per block (3)
    stage_increment,     # Stage increment for trade size
    is_regime_sensitive, # Whether to use regime-specific configs
    account_type,        # 0 for S2A, 1 for S2B
    tp_mode,            # 0 for continuous, 1 for target_triggered
    account_states,      # [current_capital, targets_reached, margin_used, free_margin]
    slot_states,         # [active, trade_size, return_multiplier, slot_capital, entry_time, resolution_time, won, triple_progress, coin_idx, combo_idx, minutes_active, completed_trades, initial_margin]
    regime_map,          # [n_blocks, n_block_coins, n_candles]
    slot_configs,        # [n_blocks, n_slots, 5 + max_signals]
    test_day_global_start,
    counter_array,       # Debug counter array [23 elements]
    future_prices        # [n_blocks, n_block_coins, n_future_candles, 5]
):
    """Main simulation kernel for S2 strategy"""
    block_idx = cuda.blockIdx.x
    if block_idx >= prices.shape[0]:
        return
        
    # Constants
    MIN_CAPITAL_FOR_TRADING = 5.0
    MIN_TRADE_SIZE = 1.0
    MAX_TRADE_SIZE = 550.0
    CAPITAL_RISK_PERCENTAGE = 0.035  # 2% risk per trade
    TRADING_FEE_RATE = 0.0004  # 0.04%
    SLIPPAGE_RATE = 0.001  # 0.1%
    TARGET_CAPITAL = 10000.0
    
    # Get current account state
    current_capital = account_states[block_idx, 0]
    targets_reached = account_states[block_idx, 1]
    margin_used = account_states[block_idx, 2]
    free_margin = account_states[block_idx, 3]
    
    # Track initial state
    cuda.atomic.add(counter_array, 10, 1)
    
    # Main processing loop
    day_offset = test_day_global_start[block_idx]
    
    for local_minute in range(prices.shape[2]):
        global_minute = day_offset + local_minute
        prev_capital = current_capital
        
        # Check for bankruptcy
        any_active = False
        for slot_idx in range(max_slots):
            if abs(slot_states[block_idx, slot_idx, 0] - np.float32(1.0)) < 1e-6:
                any_active = True
                cuda.atomic.add(counter_array, 12, 1)
                break
        
        if free_margin <= MIN_CAPITAL_FOR_TRADING and not any_active:
            current_capital = 0.0
            margin_used = 0.0
            free_margin = 0.0
            cuda.atomic.add(counter_array, 11, 1)
            break
        

        # Track state changes
        if abs(current_capital - prev_capital) > 0.1:
            cuda.atomic.add(counter_array, 14, 1)
        if margin_used < 0:
            cuda.atomic.add(counter_array, 15, 1)
        
        # Process active trades first
        for slot_idx in range(max_slots):
            if abs(slot_states[block_idx, slot_idx, 0] - np.float32(1.0)) < 1e-6:  # Active trade
                slot_states[block_idx, slot_idx, 10] += 1  # Update minutes active
                
                resolution_offset = slot_states[block_idx, slot_idx, 5]
                if resolution_offset > 0:
                    minutes_passed = global_minute - slot_states[block_idx, slot_idx, 4]
                    if minutes_passed >= resolution_offset:
                        trade_size = slot_states[block_idx, slot_idx, 1]
                        return_multiplier = slot_states[block_idx, slot_idx, 2]
                        initial_margin = slot_states[block_idx, slot_idx, 12]
                        
                        # Calculate costs
                        leverage = 50 if return_multiplier == 3.5 else 25
                        n_trades = 4 if return_multiplier == 3.5 else 7
                        position_value = trade_size * leverage
                        total_fee = position_value * TRADING_FEE_RATE * n_trades
                        slippage_cost = trade_size * SLIPPAGE_RATE * n_trades
                        
                        if abs(slot_states[block_idx, slot_idx, 6] - 1.0) < 1e-6:  # Won
                            # Calculate returns
                            gross_profit = trade_size * return_multiplier
                            net_profit = gross_profit - total_fee - slippage_cost
                            
                            # Update slot capital
                            if slot_states[block_idx, slot_idx, 7] == 0:  # First win
                                slot_states[block_idx, slot_idx, 3] = net_profit
                            else:  # Compound previous wins with costs
                                slot_states[block_idx, slot_idx, 3] = net_profit  # Already includes costs
                            
                            slot_states[block_idx, slot_idx, 7] += 1  # triple_progress
                            
                            # Changed from 3.0 to 2.0 - Complete trade at double instead of triple
                            if abs(slot_states[block_idx, slot_idx, 7] - 2.0) < 1e-6:  # Double win completed
                                # Add final slot capital to mother account
                                current_capital += slot_states[block_idx, slot_idx, 3]
                                margin_used -= initial_margin  # Release initial margin
                                free_margin = current_capital - margin_used
                                
                                slot_states[block_idx, slot_idx, 11] += 1  # Increment completed trades
                                
                                # Reset slot completely
                                slot_states[block_idx, slot_idx, 0] = 0  # inactive
                                slot_states[block_idx, slot_idx, 1] = 0  # reset trade_size
                                slot_states[block_idx, slot_idx, 2] = 0  # reset return_multiplier
                                slot_states[block_idx, slot_idx, 3] = 0  # reset slot_capital
                                slot_states[block_idx, slot_idx, 4] = 0  # reset entry_time
                                slot_states[block_idx, slot_idx, 5] = 0  # reset resolution_time
                                slot_states[block_idx, slot_idx, 6] = 0  # reset won flag
                                slot_states[block_idx, slot_idx, 7] = 0  # reset triple_progress
                                slot_states[block_idx, slot_idx, 12] = 0  # reset initial margin
                            else:
                                # Prepare for next trade in sequence
                                slot_states[block_idx, slot_idx, 0] = 0  # inactive
                                slot_states[block_idx, slot_idx, 1] = 0  # reset trade_size
                                slot_states[block_idx, slot_idx, 2] = 0  # reset return_multiplier
                                slot_states[block_idx, slot_idx, 4] = 0  # reset entry_time
                                slot_states[block_idx, slot_idx, 5] = 0  # reset resolution_time
                                slot_states[block_idx, slot_idx, 6] = 0  # reset won flag
                        else:  # Lost
                            margin_used -= initial_margin  # Release initial margin
                            free_margin = current_capital - margin_used
                            
                            # Reset slot completely
                            slot_states[block_idx, slot_idx, 0] = 0  # inactive
                            slot_states[block_idx, slot_idx, 1] = 0  # reset trade_size
                            slot_states[block_idx, slot_idx, 2] = 0  # reset return_multiplier
                            slot_states[block_idx, slot_idx, 3] = 0  # reset slot_capital
                            slot_states[block_idx, slot_idx, 4] = 0  # reset entry_time
                            slot_states[block_idx, slot_idx, 5] = 0  # reset resolution_time
                            slot_states[block_idx, slot_idx, 6] = 0  # reset won flag
                            slot_states[block_idx, slot_idx, 7] = 0  # reset triple_progress
                            slot_states[block_idx, slot_idx, 12] = 0  # reset initial margin
        
        # Look for new trade entries
        for slot_idx in range(max_slots):
            if abs(slot_states[block_idx, slot_idx, 0]) < 1e-6:  # Inactive slot
                coin_idx = slot_configs[block_idx, slot_idx, 0]
                if coin_idx == -1:  # Skip invalid coin index
                    continue
                    
                # ATR check
                current_price = prices[block_idx, coin_idx, local_minute, 3]
                current_atr = atr_values[block_idx, coin_idx, local_minute]
                atr_percentage = (current_atr / current_price) * 100
                target, _ = check_atr_conditions(atr_percentage)
                
                cuda.atomic.add(counter_array, 0, 1)  # Track ATR check
                if target == 0.0:
                    continue
                cuda.atomic.add(counter_array, 2, 1)  # Track passed ATR
                
                # Determine trade size and handle margin
                using_slot_capital = slot_states[block_idx, slot_idx, 3] > 0.0

                if using_slot_capital:
                    # Use accumulated slot capital for ongoing sequence
                    trade_size = slot_states[block_idx, slot_idx, 3]
                else:
                    # Calculate new sequence trade size from mother account
                    if free_margin < 50.0:
                        trade_size = MIN_TRADE_SIZE if free_margin >= MIN_CAPITAL_FOR_TRADING else 0.0
                    else:
                        trade_size = min(free_margin * CAPITAL_RISK_PERCENTAGE, MAX_TRADE_SIZE)
                        
                    # Check free margin only for new sequences
                    if trade_size > free_margin:
                        cuda.atomic.add(counter_array, 9, 1)  # Track margin blocks
                        continue

                # Check entry conditions
                conditions_met = check_conditions(
                    block_idx, slot_idx, local_minute,
                    slot_configs, indicator_values,
                    threshold_data, coin_idx, counter_array
                )

                cuda.atomic.add(counter_array, 1, 1)  # Track condition check

                if conditions_met:
                    cuda.atomic.add(counter_array, 3, 1)  # Track conditions met

                    if trade_size > 0.0:
                        # Calculate entry costs
                        leverage = 50 if target == 0.03 else 25
                        n_trades = 4 if target == 0.03 else 7
                        position_value = trade_size * leverage
                        entry_fee = position_value * TRADING_FEE_RATE
                        entry_slippage = trade_size * SLIPPAGE_RATE
                        return_multiplier = 3.5 if target == 0.03 else 3.0

                        # Deduct entry costs
                        if using_slot_capital:
                            trade_size -= (entry_fee + entry_slippage)
                        else:
                            current_capital -= (trade_size + entry_fee + entry_slippage)
                            margin_used += trade_size
                            slot_states[block_idx, slot_idx, 12] = trade_size  # Store initial margin
                            free_margin = current_capital - margin_used

                        direction = slot_configs[block_idx, slot_idx, 3]
                        
                        # Process trade
                        won, _, resolution_offset = process_trade(
                            current_price, direction,
                            future_prices[block_idx, coin_idx, (local_minute + 1):, 1:3],
                            atr_percentage, stage_increment, None
                        )
                        
                        cuda.atomic.add(counter_array, 4, 1)  # Track executed trade
                        
                        # Update trade state
                        slot_states[block_idx, slot_idx, 0] = 1  # active
                        slot_states[block_idx, slot_idx, 1] = trade_size
                        slot_states[block_idx, slot_idx, 2] = return_multiplier
                        slot_states[block_idx, slot_idx, 4] = global_minute
                        slot_states[block_idx, slot_idx, 5] = resolution_offset
                        slot_states[block_idx, slot_idx, 6] = won
                        slot_states[block_idx, slot_idx, 8] = coin_idx
                        
                        # Track trade metrics
                        if trade_size < MIN_TRADE_SIZE:
                            cuda.atomic.add(counter_array, 16, 1)
                        elif trade_size > current_capital * 0.1:
                            cuda.atomic.add(counter_array, 17, 1)
                        
                        if margin_used < 0:
                            cuda.atomic.add(counter_array, 18, 1)
                        elif margin_used > current_capital * 0.8:
                            cuda.atomic.add(counter_array, 19, 1)
                        
                        if current_capital < prev_capital:
                            if prev_capital - current_capital > prev_capital * 0.1:
                                cuda.atomic.add(counter_array, 25, 1)
        
        # Handle target capital reached
        if current_capital >= TARGET_CAPITAL:
            if tp_mode == 0:  # continuous mode
                current_targets = int(current_capital / TARGET_CAPITAL)
                targets_reached = current_targets
                free_margin = current_capital - margin_used
            else:  # target triggered mode
                targets_reached += 1
                # Reset capital while preserving active trades' margins
                current_capital = 500.0 + margin_used  # Initial capital + active margins
                free_margin = 500.0  # Reset free margin portion only
    
    # Store final states
    account_states[block_idx, 0] = current_capital
    account_states[block_idx, 1] = targets_reached
    account_states[block_idx, 2] = margin_used
    account_states[block_idx, 3] = free_margin

class Simulator:
    def __init__(self, max_simultaneous_trades: int = 3, stage_increment: float = 0.0075):
        """Initialize simulator with configuration parameters"""
        self.max_simultaneous_trades = max_simultaneous_trades
        self.stage_increment = stage_increment
        self.candles_per_day = 1440  # 1-minute candles per day
        self.max_signals = 6
        # Memory structures
        self.historical_pools = None
        self.block_memory_pools = {}  # Dictionary to store memory pools per block
        self.block_metadata = {}
        self.mode = None
        self.current_trial = None
        self.n_units = 14
        
        # CUDA context
        self._initialize_cuda()
    
    def _initialize_cuda(self):
        """Initialize CUDA context"""
        try:
            if not cuda.is_available():
                raise RuntimeError("CUDA is not available")
            
            try:
                ctx = cuda.current_context()
            except:
                cuda.select_device(0)
                ctx = cuda.current_context()
            
            if ctx is None:
                raise RuntimeError("Failed to create CUDA context")
                
            return True
            
        except Exception as e:
            print(f"Error initializing CUDA: {str(e)}")
            return False

    def initialize_historical_pools(self, n_test_days: int):
        """Initialize historical pools that store results for ALL trials"""
        try:
            max_trades = self.max_simultaneous_trades
            n_units = self.n_units
            
            print("\nInitializing Historical Pools:")
            print(f"Number of Test Days: {n_test_days}")
            print(f"Max Trades per Unit: {max_trades}")
            print(f"Number of Units: {n_units}")
            
            self.historical_pools = {
                'S2A': {
                    'regime_sensitive': {
                        'continuous': self._create_history_structure(n_units, n_test_days, max_trades),
                        'target_triggered': self._create_history_structure(n_units, n_test_days, max_trades)
                    },

                    'regime_independent': {
                        'continuous': self._create_history_structure(n_units, n_test_days, max_trades),
                        'target_triggered': self._create_history_structure(n_units, n_test_days, max_trades)
                    }

                },
                'S2B': {
                    'regime_sensitive': {
                        'continuous': self._create_history_structure(n_units, n_test_days, max_trades),
                        'target_triggered': self._create_history_structure(n_units, n_test_days, max_trades)
                    },

                    'regime_independent': {
                        'continuous': self._create_history_structure(n_units, n_test_days, max_trades),
                        'target_triggered': self._create_history_structure(n_units, n_test_days, max_trades)
                    }

                }
            }
            
            print("Historical pools initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing historical pools: {str(e)}")
            return False

    def _create_history_structure(self, n_units: int, n_days: int, max_trades: int):
        """Create standardized history structure for a trial type"""
        return {
            f'unit_{unit}': {
                f'day_{day}': {
                    'mother_account': {
                        'current_capital': INITIAL_CAPITAL,  # Current total capital for this unit
                        'targets_reached': 0,                # Number of times account hit target capital (integer)
                        'margin_used': 0.0,                  # Total capital currently deployed in active trades
                        'free_margin': INITIAL_CAPITAL       # Available capital (current_capital - margin_used)
                    },
                    'slots': {
                        f'slot_{slot}': {
                            'active': False,             # Whether slot has an open trade
                            'minutes_active_today': 0,   # Number of candles slot was active today
                            'trade_size': 0.0,          # Capital invested in current trade
                            'return_multiplier': 0.0,    # Trade return multiplier (0, 3, or 3.5)
                            'slot_capital': 0.0,         # Capital accumulated in this slot
                            'entry_time': 0,            # Global candle index when trade opened
                            'resolution_time': 0,       # Global candle index when trade closed
                            'won': False,               # Whether trade was profitable
                            'triple_progress': 0,       # Progress toward triple win (0-3)
                            'coin_idx': 0,              # Index of coin being traded
                            'combo_idx': 0,            # Index of combo being traded
                            'completed_trades': 0,     # List of trades resolved in this slot
                            'initial_margin_used': 0    # Initial margin used for this slot
                        }
                        for slot in range(max_trades)
                    }
                }
                for day in range(n_days)
            }
            for unit in range(n_units)
        }

    def initialize_memory_pools(self, block_config: Dict, block_id: str):
        """Initialize memory pools specific to block configuration"""
        try:
            # Extract block configuration
            account_type = block_config['account_type']
            actual_n_coins = block_config['n_coins']
            actual_n_signals = block_config['n_signals']
            actual_n_windows = block_config['n_windows']
            n_candles = block_config['total_candles']
            future_n_candles = block_config['future_total_candles']  # Add future candles
            
            # Use provided maximum dimensions from block_data
            max_coins = block_config['max_coins']
            max_signals = block_config['max_signals']
            max_windows = block_config['max_windows']
            
            # Configure dimensions
            n_trade_slots = self.max_simultaneous_trades
            
            print(f"\nInitializing Block Memory Pools for block {block_id}:")
            print(f"Account Type: {account_type}")
            print(f"Actual Coins: {actual_n_coins} (Padded to {max_coins})")
            print(f"Actual Signals: {actual_n_signals} (Padded to {max_signals})")
            print(f"Actual Windows: {actual_n_windows} (Padded to {max_windows})")
            print(f"Current Day Candles: {n_candles}")
            print(f"Future Total Candles: {future_n_candles}")  # Add debug print
            
            # Create memory structure using padded dimensions
            self.block_memory_pools[block_id] = self._create_memory_structure(
                n_coins=max_coins,
                n_signals=max_signals,
                n_windows=max_windows,
                n_trade_slots=n_trade_slots,
                total_candles=n_candles,
                future_total_candles=future_n_candles  # Pass future candles
            )
            
            # Store complete metadata
            if 'metadata' in block_config:
                self.block_metadata[block_id] = block_config['metadata']
            else:
                # Fallback to basic metadata if not provided
                self.block_metadata[block_id] = {
                    'actual_n_coins': actual_n_coins,
                    'actual_n_signals': actual_n_signals,
                    'actual_n_windows': actual_n_windows
                }
            
            print(f"Block memory pools initialized successfully for block {block_id}")
            return True
            
        except Exception as e:
            print(f"Error initializing block memory pools for block {block_id}: {str(e)}")
            return False

    def _create_memory_structure(self, n_coins: int, n_signals: int, 
                               n_windows: int, n_trade_slots: int, total_candles: int, future_total_candles: int) -> Dict:
        """Create CUDA memory structure for a single block with window dimension"""
        return {
            # Block-specific price and signal data
            'price_data': cuda.device_array(shape=(n_coins, total_candles, 5), dtype=np.float32),
            'indicator_values': cuda.device_array(shape=(n_signals, n_coins, total_candles), dtype=np.float32),
            'atr_values': cuda.device_array(shape=(n_coins, total_candles), dtype=np.float32),
            'threshold_data': cuda.device_array(shape=(n_coins, n_windows, n_signals, total_candles, 2), dtype=np.float32),
            'future_prices': cuda.device_array(shape=(n_coins, future_total_candles, 5), dtype=np.float32),
            
            # Maps and regime data
            'regime_map': cuda.device_array(shape=(n_coins, total_candles), dtype=np.int32),
            'timeframe_map': cuda.device_array(shape=(total_candles,), dtype=np.int32),
            'test_day_global_start': cuda.device_array(shape=(1,), dtype=np.int32),
            
            # Slot-specific tracking
            'slot_states': cuda.device_array(shape=(n_trade_slots, 13), dtype=np.float32),  # Changed back to float32
                # [0: active, 1: trade_size, 2: return_multiplier, 3: slot_capital, 4: entry_time, 5: resolution_time,
                # 6: won, 7: triple_progress, 8: coin_idx, 9: combo_idx, 10: minutes_active, 11: completed_trades, 12: initial_margin_used]
            'slot_configs': cuda.device_array(shape=(n_trade_slots, 5 + self.max_signals), dtype=np.int32),
                # [0: coin_idx, 1: combo_idx, 2: window_size, 3: direction, 4: regime_lock, 5+: signal_indices]
                
            # Account state tracking (removed regime from account state)
            'account_state': cuda.device_array(shape=(4), dtype=np.float32),  
                # [current_capital, targets_reached, margin_used, free_margin]
            
            # Progress tracking
            'counter_array': cuda.device_array(shape=(23), dtype=np.int32),  
                # [0: total_condition_checks, 
                #  1: conditions_met_count, 
                #  2: atr_checks_passed,
                #  3: trades_attempted,
                #  4: trades_executed,
                #  5: trades_won,
                #  6: trades_lost,
                #  7: total_signals_checked,
                #  8: signals_passed,
                #  9: trades_blocked_by_margin,
                # 10: initial_state,
                # 11: bankruptcy,
                # 12: active_slots,
                # 13: capital_increase,
                # 14: capital_decrease,
                # 15: significant_capital_changes,
                # 16: too_small_trades,
                # 17: large_position_trades,
                # 18: negative_margin_count,
                # 19: high_margin_usage_count,
                # 20: large_loss_count,
                # 21: short_trade_count,
                # 22: long_trade_count]
            # Add future prices array
        }

    def _load_static_data(self, block_id: str, price_data, indicator_values, timeframe_map, 
                         threshold_data, test_day_global_start=0, atr_values=None, 
                         regime_map=None, reset_states=False, previous_day_state=None,
                         candidate=None, future_prices=None, account_type=None):
        try:
            if block_id not in self.block_memory_pools:
                raise ValueError(f"Block {block_id} memory pools not initialized")
            
            if account_type is None:
                raise ValueError("Account type must be provided")

            memory_pools = self.block_memory_pools[block_id]
            block_metadata = self.block_metadata[block_id]
            
            if atr_values is None:
                raise ValueError("ATR values must be provided")
            
            # Convert data types
            test_day_data = price_data.astype(np.float32)
            test_day_atr = atr_values.astype(np.float32)
            timeframe_map = timeframe_map.astype(np.int32)
            threshold_data = threshold_data.astype(np.float32)
            indicator_values = indicator_values.astype(np.float32)
            
            # Convert global start to array
            global_start = np.array([test_day_global_start], dtype=np.int32)
            
            # Validate shapes before copying
            shape_validations = {
                'price_data': (test_day_data.shape, memory_pools['price_data'].shape),
                'indicator_values': (indicator_values.shape, memory_pools['indicator_values'].shape),
                'atr_values': (test_day_atr.shape, memory_pools['atr_values'].shape),
                'timeframe_map': (timeframe_map.shape, memory_pools['timeframe_map'].shape),
                'threshold_data': (threshold_data.shape, memory_pools['threshold_data'].shape),
                'test_day_global_start': (global_start.shape, memory_pools['test_day_global_start'].shape),
                'future_prices': (future_prices.shape, memory_pools['future_prices'].shape)
            }
            
            for name, (input_shape, pool_shape) in shape_validations.items():
                if input_shape != pool_shape:
                    raise ValueError(f"{name} shape mismatch: {input_shape} vs {pool_shape}")
            
            # Copy data to GPU with synchronization
            data_copies = {
                'price_data': test_day_data,
                'indicator_values': indicator_values,
                'atr_values': test_day_atr,
                'timeframe_map': timeframe_map,
                'threshold_data': threshold_data,
                'test_day_global_start': global_start,
                'future_prices': future_prices
            }
            
            for name, data in data_copies.items():
                cuda.to_device(data, to=memory_pools[name])
                cuda.synchronize()
            
            # Load regime map
            if regime_map is None:
                raise ValueError("Regime map must be provided")
                    
            regime_map = regime_map.astype(np.int32)
            cuda.to_device(regime_map, to=memory_pools['regime_map'])
            cuda.synchronize()
            
            # Initialize slot states with proper coin and combo assignments
            if reset_states:  # Only initialize on first day
                slot_states = np.zeros((self.max_simultaneous_trades, 13), dtype=np.float32)
                slot_configs_array = np.full((self.max_simultaneous_trades, 5 + self.max_signals), -1, dtype=np.int32)
                
                metadata_slot_configs = block_metadata['slot_configs']
                
                for slot_idx in range(self.max_simultaneous_trades):
                    slot_config = metadata_slot_configs[slot_idx]
                    
                    # Convert direction string to numeric value
                    direction_value = -1 if slot_config['direction'].lower() == 'short' else 1
                    
                    # Update slot configs array
                    slot_configs_array[slot_idx, 0] = slot_config['coin_idx']
                    slot_configs_array[slot_idx, 1] = slot_config['combo_idx']
                    slot_configs_array[slot_idx, 2] = slot_config['window_size']
                    slot_configs_array[slot_idx, 3] = direction_value
                    slot_configs_array[slot_idx, 4] = slot_config['regime_lock']
                    
                    # Copy signal indices
                    for sig_idx, signal_idx in enumerate(slot_config['signal_indices']):
                        slot_configs_array[slot_idx, 5 + sig_idx] = signal_idx
                    
                    # Initialize slot states with default values
                    slot_states[slot_idx] = [
                        0,      # active
                        0.0,    # trade_size
                        0.0,    # return_multiplier
                        0.0,    # slot_capital
                        0,      # entry_time
                        0,      # resolution_time
                        0,      # won
                        0,      # triple_progress
                        slot_config['coin_idx'],  # coin_idx
                        slot_config['combo_idx'], # combo_idx
                        0,      # minutes_active
                        0,      # completed_trades
                        0.0     # initial_margin_used
                    ]
                
                # Copy both arrays to GPU
                cuda.to_device(slot_states, to=memory_pools['slot_states'])
                cuda.to_device(slot_configs_array, to=memory_pools['slot_configs'])
                cuda.synchronize()
                
                # Initialize account state array
                account_state = np.zeros(4, dtype=np.float32)
                # Initialize with starting capital
                account_state[0] = INITIAL_CAPITAL  # current_capital
                account_state[1] = 0                # targets_reached
                account_state[2] = 0.0              # margin_used
                account_state[3] = INITIAL_CAPITAL  # free_margin
                

                print(f"\nInitializing new account state for block {block_id}:")
                print(f"Current Capital: {account_state[0]}")
                print(f"Targets Reached: {account_state[1]}")
                print(f"Margin Used: {account_state[2]}")
                print(f"Free Margin: {account_state[3]}")
                
                # Copy account state to GPU
                cuda.to_device(account_state, to=memory_pools['account_state'])
                cuda.synchronize()
            
            elif previous_day_state is not None:
                # Load slot states
                slot_states = np.zeros((self.max_simultaneous_trades, 13), dtype=np.float32)
                
                for slot_idx in range(self.max_simultaneous_trades):
                    slot_data = previous_day_state['slots'][f'slot_{slot_idx}']
                    
                    # Preserve slot state exactly as it was
                    slot_states[slot_idx] = [
                        1 if slot_data['active'] else 0,
                        slot_data['trade_size'],
                        slot_data['return_multiplier'],
                        slot_data['slot_capital'],
                        slot_data['entry_time'],
                        slot_data['resolution_time'],
                        1 if slot_data['won'] else 0,
                        slot_data['triple_progress'],
                        slot_data['coin_idx'],
                        slot_data['combo_idx'],
                        slot_data['minutes_active'],
                        slot_data['completed_trades'],
                        slot_data['initial_margin_used']
                    ]
                
                # Copy states to GPU
                cuda.to_device(slot_states, to=memory_pools['slot_states'])
                cuda.synchronize()
                
                # Load previous day's account state
                prev_account = previous_day_state['mother_account']
                account_state = np.zeros(4, dtype=np.float32)
                account_state[0] = prev_account['current_capital']
                account_state[1] = prev_account['targets_reached']
                account_state[2] = prev_account['margin_used']
                account_state[3] = prev_account['free_margin']
                
                print(f"\nLoading previous account state for block {block_id}:")
                print(f"Current Capital: {account_state[0]}")
                print(f"Targets Reached: {account_state[1]}")
                print(f"Margin Used: {account_state[2]}")
                print(f"Free Margin: {account_state[3]}")
                
                # Copy account state to GPU
                cuda.to_device(account_state, to=memory_pools['account_state'])
                cuda.synchronize()
            
            return True
            
        except Exception as e:
            print(f"Error loading block {block_id} data: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def cleanup_block_memory_pools(self):
        """Clean up all block memory pools before starting a new trial"""
        try:
            # First synchronize CUDA
            cuda.synchronize()
            
            # Function to safely clear CUDA GPU memory allocated for arrays
            def safe_clear_array(array_name, array):
                try:
                    if isinstance(array, cuda.devicearray.DeviceNDArray):
                        # Ensure all operations on this array are complete
                        cuda.synchronize()
                        # Set array contents to None before deletion
                        array.copy_to_host(np.zeros(array.shape, dtype=array.dtype))
                        cuda.synchronize()
                        del array
                        return True
                except Exception as e:
                    print(f"Warning: Could not clear array {array_name}: {e}")
                    return False
            
            # Track cleanup success
            cleanup_success = True
            
            # Clear CUDA memory from memory pools
            if hasattr(self, 'block_memory_pools'):
                # First synchronize all CUDA operations
                cuda.synchronize()
                
                for block_id, memory_pool in self.block_memory_pools.items():
                    print(f"\nCleaning up block {block_id} memory pools...")
                    
                    # Block-specific price and signal data
                    cleanup_success &= safe_clear_array('price_data', memory_pool.get('price_data'))
                    cleanup_success &= safe_clear_array('indicator_values', memory_pool.get('indicator_values'))
                    cleanup_success &= safe_clear_array('atr_values', memory_pool.get('atr_values'))
                    cleanup_success &= safe_clear_array('future_prices', memory_pool.get('future_prices'))  # Add future prices to cleanup
                    
                    # Maps and regime data
                    cleanup_success &= safe_clear_array('regime_map', memory_pool.get('regime_map'))
                    cleanup_success &= safe_clear_array('timeframe_map', memory_pool.get('timeframe_map'))
                    cleanup_success &= safe_clear_array('threshold_data', memory_pool.get('threshold_data'))
                    cleanup_success &= safe_clear_array('test_day_global_start', memory_pool.get('test_day_global_start'))
                    
                    # Slot-specific tracking
                    cleanup_success &= safe_clear_array('slot_states', memory_pool.get('slot_states'))
                    cleanup_success &= safe_clear_array('slot_configs', memory_pool.get('slot_configs'))
                    
                    # Account state tracking
                    cleanup_success &= safe_clear_array('account_state', memory_pool.get('account_state'))
                                        
                    # Progress tracking
                    cleanup_success &= safe_clear_array('counter_array', memory_pool.get('counter_array'))
                    
                    # Clear the memory pool dictionary itself
                    memory_pool.clear()
            
            # Force context synchronization
            cuda.synchronize()
            
            # Reset all memory pools and tracking arrays
            self.block_memory_pools = {}
            if hasattr(self, 'block_trade_states'):
                self.block_trade_states = {}
            if hasattr(self, 'block_account_positions'):
                self.block_account_positions = {}
            if hasattr(self, 'block_slot_tracking'):
                self.block_slot_tracking = {}
            if hasattr(self, 'block_current_streaks'):
                self.block_current_streaks = {}
            if hasattr(self, 'block_distance_accumulators'):
                self.block_distance_accumulators = {}
            if hasattr(self, 'block_conflicts'):
                self.block_conflicts = {}
            if hasattr(self, 'block_results'):
                self.block_results = {}
            if hasattr(self, 'block_metadata'):
                self.block_metadata = {}
            if hasattr(self, 'block_configs'):
                self.block_configs = {}
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Final synchronization
            cuda.synchronize()
            
            print(f"\nCleaned up all block memory pools and CUDA resources. Success: {cleanup_success}")
            return cleanup_success
            
        except Exception as e:
            print(f"Error cleaning up block memory pools: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def simulate_combinations(self, simulation_input: Dict, debug: bool = False):
        try:
            # Clean up all block memory pools before starting
            if not self.cleanup_block_memory_pools():
                print("Failed to clean up block memory pools")
                return None
            
            print("\n" + "=" * 200)
            print("Starting Simulation")
            print("=" * 200)
            
            # 1. Extract trial configuration
            account_type = simulation_input['account_type']
            regime_type = simulation_input['regime_type']
            tp_mode = simulation_input['tp_mode']
            combo_data = simulation_input['combo_data']
            candidates = simulation_input['candidates']
            total_days = len(simulation_input['valid_test_days'])
            timeframe_map = combo_data['timeframe_map']

            # 2. Get dimensions from numpy arrays
            n_coins = combo_data['price_data'].shape[0]
            n_signals = combo_data['signal_values'].shape[0]
            n_candles = combo_data['timeframe_map'].shape[0]
            
            print(f"Dimensions: {n_coins} coins, {n_signals} signals, {n_candles} candles")
            
            print("\nArray Shapes:")
            print(f"Price Data: {combo_data['price_data'].shape}")
            print(f"Signal Values: {combo_data['signal_values'].shape}")
            print(f"ATR Values: {combo_data['atr_values'].shape}")
            print(f"Timeframe Map: {combo_data['timeframe_map'].shape}")
            print(f"Thresholds: {combo_data['thresholds'].shape}")
            print(f"Regime Map: {combo_data['regime_data'].shape}")
            
            print(f"\nTrial Configuration:")
            print(f"Account Type: {account_type}")
            print(f"Regime Type: {regime_type}")
            print(f"TP Mode: {tp_mode}")
            print(f"Number of Candidates: {len(candidates)}")
            
            # Configure CUDA grid
            threads_per_block = 1  # One thread per block to handle all slots
            n_blocks = len(candidates)  # One block per candidate           
            
            # Process each day sequentially
            for current_day in range(total_days):
                print(f"\nProcessing day {current_day + 1}/{total_days}")
                day_start = current_day * self.candles_per_day 
                day_end = min((current_day + 1) * self.candles_per_day, combo_data['timeframe_map'].shape[0])
                
                # Slice the data for this day
                day_data = {
                    'price_data': combo_data['price_data'][:, day_start:day_end],
                    'signal_values': combo_data['signal_values'][:, :, day_start:day_end],
                    'atr_values': combo_data['atr_values'][:, day_start:day_end],
                    'timeframe_map': combo_data['timeframe_map'][day_start:day_end],
                    'thresholds': combo_data['thresholds'][:, :, :, day_start:day_end],
                    'regime_data': combo_data['regime_data'][:, day_start:day_end],
                    'future_prices': combo_data['price_data'][:, day_start:]  # No day_end limit
                }
                
                print("\nDay Data Shapes:")
                for key, value in day_data.items():
                    print(f"{key}: {value.shape}")
                
                # Process all blocks for this day
                blocks = []
                for block_idx, candidate in enumerate(candidates):
                    print(f"\nPreparing block {block_idx} data")
                    
                    # Get previous day's state if not first day
                    previous_day_state = None
                    if current_day > 0:
                        previous_day_state = self.historical_pools[account_type][regime_type][tp_mode][f'unit_{block_idx}'][f'day_{current_day-1}']
                        print(f"DEBUG: Previous day state for block {block_idx}:", previous_day_state)
                    
                    # Prepare block data
                    block_data = self._prepare_block_data(
                        candidate=candidate,
                        combo_data={**combo_data, **day_data},
                        previous_day_state=previous_day_state
                    )
                    
                    if block_data is None:
                        raise RuntimeError(f"Failed to prepare data for block {block_idx}")
                    
                    # Initialize memory pools for this block
                    success = self.initialize_memory_pools({
                        'account_type': account_type,
                        'n_coins': block_data['metadata']['actual_n_coins'],
                        'n_signals': block_data['metadata']['actual_n_signals'],
                        'n_windows': block_data['metadata']['actual_n_windows'],
                        'total_candles': block_data['prices'].shape[1],  # Current day candles
                        'future_total_candles': day_data['future_prices'].shape[1],  # Add future candles parameter
                        'max_coins': block_data['prices'].shape[0],
                        'max_signals': block_data['signals'].shape[0],
                        'max_windows': block_data['thresholds'].shape[1],
                        'metadata': block_data['metadata']
                    }, block_idx)
                    
                    if not success:
                        raise RuntimeError(f"Failed to initialize memory pools for block {block_idx}")
                    
                    # Load data into memory pools with previous state
                    success = self._load_static_data(
                        block_id=block_idx,
                        price_data=block_data['prices'],
                        indicator_values=block_data['signals'],
                        timeframe_map=day_data['timeframe_map'],
                        threshold_data=block_data['thresholds'],
                        test_day_global_start=day_start,
                        atr_values=block_data['atr_values'],
                        regime_map=block_data['regime_data'],
                        reset_states=(current_day == 0),  # Only reset states on first day
                        previous_day_state=previous_day_state,  # Pass previous day's state
                        candidate=candidate,  # Pass the candidate data
                        future_prices=day_data['future_prices'],  # Add future prices to load
                        account_type=account_type  # Pass account_type
                    )
                    
                    if not success:
                        raise RuntimeError(f"Failed to load data for block {block_idx}")
                    
                    blocks.append({
                        'memory_pools': self.block_memory_pools[block_idx],
                        'data': block_data
                    })
                
                # Prepare stacked arrays for all blocks
                all_prices = np.stack([block['data']['prices'] for block in blocks])
                all_indicator_values = np.stack([block['data']['signals'] for block in blocks])
                
                all_threshold_data = np.stack([block['data']['thresholds'] for block in blocks])
                all_atr_values = np.stack([block['data']['atr_values'] for block in blocks])
                all_regime_map = np.stack([block['data']['regime_data'] for block in blocks])
                all_timeframe_map = day_data['timeframe_map']
                all_future_prices = np.stack([block['data']['future_prices'] for block in blocks])
                
                # Stack account and slot states
                all_account_states = np.stack([block['memory_pools']['account_state'] for block in blocks])
                all_slot_states = np.stack([block['memory_pools']['slot_states'] for block in blocks])
                
                # Use metadata slot configs instead of memory pool configs
                def map_direction(direction):
                    # Map string direction to integer
                    if isinstance(direction, str):
                        return 1 if direction.lower() == 'long' else -1
                    return direction  # Already an integer

                all_slot_configs = np.stack([
                    np.array([
                        [
                            config['coin_idx'],
                            config['combo_idx'],
                            config['window_size'],
                            map_direction(config['direction']),  # Convert direction string to int
                            config['regime_lock'],
                            *config['signal_indices']  # Unpack the 6 signal indices
                        ] for config in block['data']['metadata']['slot_configs'].values()
                    ], dtype=np.int32)
                    for block in blocks
                ])
                
                print("\nDebug - Before GPU Transfer:")
                print(f"all_slot_configs shape: {all_slot_configs.shape}")
                print(f"all_slot_configs content:")
                for block_idx, block_configs in enumerate(all_slot_configs):
                    print(f"\nBlock {block_idx} configs:")
                    for slot_idx, slot_config in enumerate(block_configs):
                        print(f"  Slot {slot_idx}: {slot_config}")
                
                # Transfer arrays to GPU
                d_prices = cuda.to_device(all_prices)
                cuda.synchronize()
                d_indicator_values = cuda.to_device(all_indicator_values)
                cuda.synchronize()
                d_threshold_data = cuda.to_device(all_threshold_data)
                cuda.synchronize()
                d_atr_values = cuda.to_device(all_atr_values)
                cuda.synchronize()
                d_timeframe_map = cuda.to_device(all_timeframe_map)
                cuda.synchronize()
                d_regime_map = cuda.to_device(all_regime_map)
                cuda.synchronize()
                d_test_day_global_start = cuda.to_device(np.array([day_start] * n_blocks, dtype=np.int32))
                cuda.synchronize()
                d_future_prices = cuda.to_device(all_future_prices)
                cuda.synchronize()
                d_account_states = cuda.to_device(all_account_states)
                cuda.synchronize()
                d_slot_states = cuda.to_device(all_slot_states)
                cuda.synchronize()
                d_slot_configs = cuda.to_device(all_slot_configs)
                cuda.synchronize()
                
                # Initialize counter array for this day
                d_counter_array = cuda.to_device(np.zeros(30, dtype=np.int32))
                cuda.synchronize()
                
                # Trading metrics (0-9)
                # [0: total_condition_checks
                #  1: conditions_met_count
                #  2: atr_checks_passed
                #  3: trades_attempted
                #  4: trades_executed
                #  5: trades_won
                #  6: trades_lost
                #  7: total_signals_checked
                #  8: signals_passed
                #  9: trades_blocked_by_margin]

                # State tracking (10-19)
                # [10: initial_state
                #  11: bankruptcy
                #  12: active_slots
                #  13: capital_increase
                #  14: capital_decrease
                #  15: significant_capital_changes
                #  16: too_small_trades
                #  17: large_position_trades
                #  18: negative_margin_count
                #  19: high_margin_usage_count]

                # Signal processing (20-29)
                # [20: nan_indicators
                #  21: nan_thresholds
                #  22: greater_than_checks
                #  23: less_than_checks
                #  24: failed_conditions
                #  25: large_loss_count
                #  26: short_trade_count
                #  27: long_trade_count
                #  28-29: reserved]

                # Before kernel launch
                print(f"\nDebug - Before kernel launch for day {current_day}:")
                print(f"test_day_global_start: {d_test_day_global_start}")
                print(f"prices shape: {d_prices.shape}")
                print(f"timeframe_map shape: {d_timeframe_map.shape}")
                
                print("\nArray shapes before kernel:")
                print(f"slot_configs: {all_slot_configs.shape}")
                print(f"indicator_values: {all_indicator_values.shape}")
                print(f"threshold_data: {all_threshold_data.shape}")

                print("\nSample slot configs:")
                for block_idx in range(min(2, all_slot_configs.shape[0])):
                    print(f"\nBlock {block_idx}:")
                    for slot_idx in range(min(3, all_slot_configs.shape[1])):
                        config = all_slot_configs[block_idx, slot_idx]
                        print(f"  Slot {slot_idx}: [coin:{config[0]}, combo:{config[1]}, window:{config[2]}, "
                              f"dir:{config[3]}, lock:{config[4]}, signals:{config[5:]}]")

                
                # Debug data ranges
                day_start = current_day * self.candles_per_day
                day_end = min((current_day + 1) * self.candles_per_day, combo_data['timeframe_map'].shape[0])
                print(f"Day range: {day_start} to {day_end}")
                
                # Launch kernel with modified parameters
                process_simulation_kernel[n_blocks, threads_per_block](
                    d_prices, d_indicator_values, d_threshold_data,
                    d_atr_values, d_timeframe_map, self.max_simultaneous_trades,
                    self.stage_increment, (regime_type == 'regime_sensitive'),
                    0 if account_type == 'S2A' else 1,
                    1 if tp_mode == 'target_triggered' else 0,
                    d_account_states,
                    d_slot_states,
                    d_regime_map, d_slot_configs,
                    d_test_day_global_start,
                    d_counter_array,
                    d_future_prices
                )
                
                # After kernel execution
                cuda.synchronize()
                test_day_start = d_test_day_global_start.copy_to_host()
                print(f"\nDebug - After kernel execution:")
                print(f"test_day_start: {test_day_start}")
                print(f"Account states:")
                final_account_states = d_account_states.copy_to_host()
                for block_idx in range(n_blocks):
                    print(f"  Block {block_idx}: Capital={final_account_states[block_idx, 0]:.2f}, "
                          f"Margin={final_account_states[block_idx, 2]:.2f}")
                
                # Get counter array results
                counter_results = d_counter_array.copy_to_host()
                print("\n=== Kernel Execution Statistics ===")

                print("\nTrading Metrics:")
                print(f"Total Condition Checks: {counter_results[0]}")
                print(f"Conditions Met Count: {counter_results[1]}")
                print(f"ATR Checks Passed: {counter_results[2]}")
                print(f"Trades Attempted: {counter_results[3]}")
                print(f"Trades Executed: {counter_results[4]}")
                print(f"Trades Won: {counter_results[5]}")
                print(f"Trades Lost: {counter_results[6]}")
                print(f"Total Signals Checked: {counter_results[7]}")
                print(f"Signals Passed: {counter_results[8]}")
                print(f"Trades Blocked by Margin: {counter_results[9]}")

                print("\nState Tracking:")
                print(f"Initial State: {counter_results[10]}")
                print(f"Bankruptcy: {counter_results[11]}")
                print(f"Active Slots: {counter_results[12]}")
                print(f"Capital Increase: {counter_results[13]}")
                print(f"Capital Decrease: {counter_results[14]}")
                print(f"Significant Capital Changes: {counter_results[15]}")
                print(f"Too Small Trades: {counter_results[16]}")
                print(f"Large Position Trades: {counter_results[17]}")
                print(f"Negative Margin Count: {counter_results[18]}")
                print(f"High Margin Usage Count: {counter_results[19]}")

                print("\nSignal Processing:")
                print(f"NaN Indicators: {counter_results[20]}")
                print(f"NaN Thresholds: {counter_results[21]}")
                print(f"Greater Than Checks: {counter_results[22]}")
                print(f"Less Than Checks: {counter_results[23]}")
                print(f"Failed Conditions: {counter_results[24]}")
                print(f"Large Loss Count: {counter_results[25]}")
                print(f"Short Trade Count: {counter_results[26]}")
                print(f"Long Trade Count: {counter_results[27]}")

                # Print percentages for key metrics
                print("\nSuccess Rates:")
                if counter_results[0] > 0:  # Total condition checks
                    print(f"Condition Check Pass Rate: {(counter_results[1] / counter_results[0]) * 100:.2f}%")
                if counter_results[3] > 0:  # Trades attempted
                    print(f"Trade Execution Rate: {(counter_results[4] / counter_results[3]) * 100:.2f}%")
                if counter_results[4] > 0:  # Trades executed
                    print(f"Trade Win Rate: {(counter_results[5] / counter_results[4]) * 100:.2f}%")
                if counter_results[7] > 0:  # Total signals checked
                    print(f"Signal Pass Rate: {(counter_results[8] / counter_results[7]) * 100:.2f}%")
                if counter_results[22] + counter_results[23] > 0:  # Total condition checks (greater + less)
                    total_checks = counter_results[22] + counter_results[23]
                    passed = counter_results[8]
                    print(f"Overall Condition Pass Rate: {(passed / total_checks) * 100:.2f}%")

                # Get final states from device memory
                final_account_states = d_account_states.copy_to_host()
                final_slot_states = d_slot_states.copy_to_host()

                print("\n=== Working Memory Pool Status After Kernel ===")
                for block_idx in range(n_blocks):
                    print(f"\nBlock {block_idx} Status:")
                    print("Account State:")
                    print(f"  Current Capital: {final_account_states[block_idx, 0]}")
                    print(f"  Targets Reached: {final_account_states[block_idx, 1]}")
                    print(f"  Margin Used: {final_account_states[block_idx, 2]}")
                    print(f"  Free Margin: {final_account_states[block_idx, 3]}")
                
                    # Save results for this day directly using final states
                    unit_data = self.historical_pools[account_type][regime_type][tp_mode][f'unit_{block_idx}'][f'day_{current_day}']
                    
                    # Update mother account state using final account states
                    unit_data['mother_account'] = {
                        'current_capital': float(final_account_states[block_idx, 0]),
                        'targets_reached': int(final_account_states[block_idx, 1]),
                        'margin_used': float(final_account_states[block_idx, 2]),
                        'free_margin': float(final_account_states[block_idx, 3])
                    }
                    print(f"\nDEBUG: After mother account update for day {current_day}:")
                    print(f"Historical pool state: {unit_data['mother_account']}")
                    
                    # Update slot states using final slot states
                    for slot in range(self.max_simultaneous_trades):
                        slot_state = final_slot_states[block_idx, slot]
                        slot_data = {
                            'active': bool(slot_state[0] == 1),
                            'trade_size': float(slot_state[1]),
                            'return_multiplier': float(slot_state[2]),
                            'slot_capital': float(slot_state[3]),
                            'entry_time': int(slot_state[4]),
                            'resolution_time': int(slot_state[5]),
                            'won': bool(slot_state[6] == 1),
                            'triple_progress': int(slot_state[7]),
                            'coin_idx': int(slot_state[8]),
                            'combo_idx': int(slot_state[9]),
                            'minutes_active': int(slot_state[10]),
                            'completed_trades': int(slot_state[11]),
                            'initial_margin_used': float(slot_state[12])  # Add new field
                        }
                        unit_data['slots'][f'slot_{slot}'] = slot_data
                        if slot_state[0] == 1:  # if active
                            print(f"\nDEBUG: After slot {slot} update for day {current_day}:")

                            print(f"Historical pool slot state: {slot_data}")
                
                # Clean up GPU memory for this day
                cuda.synchronize()
                
                # Free temporary GPU arrays
                del d_prices
                del d_indicator_values
                del d_threshold_data
                del d_atr_values
                del d_timeframe_map
                del d_regime_map
                del d_test_day_global_start
                del d_counter_array
                del d_future_prices  # Add cleanup for future prices
                
                print(f"Completed processing day {current_day + 1}")
            
            # Generate final summary
            print("\n=== Historical Pool Summary ===")
            account_type_str = 'S2A' if account_type == 'S2A' else 'S2B'
            regime_type_str = 'regime_sensitive' if regime_type == 'regime_sensitive' else 'regime_independent'
            tp_mode_str = 'continuous' if tp_mode == 'continuous' else 'target_triggered'
            
            pool = self.historical_pools[account_type_str][regime_type_str][tp_mode_str]
            
            # Create summary tables
            summary_data = []
            trade_data = []
            
            # Process data day by day
            for unit in range(n_blocks):
                prev_capital = None
                for day in range(total_days):
                    day_data = pool[f'unit_{unit}'][f'day_{day}']
                    
                    # Account summary
                    mother_account = day_data['mother_account']
                    current_capital = mother_account['current_capital']
                    
                    # Calculate percentage change with division by 0 check
                    pct_change = None
                    if prev_capital is not None:
                        if prev_capital == 0:
                            pct_change = 0
                        else:
                            pct_change = ((current_capital - prev_capital) / prev_capital) * 100
                    
                    summary_data.append({
                        'Unit': unit,
                        'Day': day,
                        'Capital': current_capital,
                        'Cap.Chg%': pct_change if pct_change is not None else 'N/A',
                        'Targets': mother_account['targets_reached'],
                        'Margin': mother_account['margin_used'],
                        'Free': mother_account['free_margin']
                    })
                    
                    prev_capital = current_capital
                    
                    # Active slots and trades
                    for slot in range(self.max_simultaneous_trades):
                        slot_data = day_data['slots'][f'slot_{slot}']
                        
                        # Always append data for each day to maintain chronological order
                        trade_data.append({
                            'Block': unit,
                            'Slot': slot,
                            'Day': day,
                            'Active': slot_data['active'],
                            'Size': f"{slot_data['trade_size']:.1f}",
                            'Mult': slot_data['return_multiplier'],
                            'Slot Capital': f"{slot_data['slot_capital']:.1f}",
                            'Entry': slot_data['entry_time'],
                            'Resolution': slot_data['resolution_time'],
                            'Won': slot_data['won'],
                            'Triple': slot_data['triple_progress'],
                            'Trades': slot_data['completed_trades'],
                            'Initial Margin': slot_data['initial_margin_used']
                        })
            
            # Convert to DataFrames
            summary_df = pd.DataFrame(summary_data)
            trade_df = pd.DataFrame(trade_data)
                        
            # Format numerical columns
            summary_df['Capital'] = summary_df['Capital'].map('{:,.1f}'.format)
            summary_df['Margin'] = summary_df['Margin'].map('{:,.1f}'.format)
            summary_df['Free'] = summary_df['Free'].map('{:,.1f}'.format)
            
            # Set display options for better formatting
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_rows', None)
            
            print("\nAccount Summary:")
            print(tabulate(summary_df, headers='keys', tablefmt='pretty', showindex=False))
            
            print("\nTrade Activity Summary:")
            # Sort trade_df by Block, Slot, then Day
            trade_df = trade_df.sort_values(['Block', 'Slot', 'Day'])
            # Add spacing between different blocks/slots
            trade_df_formatted = trade_df.copy()
            trade_df_formatted.loc[trade_df_formatted.groupby(['Block', 'Slot']).head(1).index, :] = \
                trade_df_formatted.loc[trade_df_formatted.groupby(['Block', 'Slot']).head(1).index, :].fillna('')
            
            print(tabulate(
                trade_df_formatted,
                headers={
                    'Block': 'Block',
                    'Slot': 'Slot',
                    'Day': 'Day',
                    'Active': 'Active',
                    'Size': 'Size',
                    'Mult': 'Mult',
                    'Slot Capital': 'Slot Capital',
                    'Entry': 'Entry',
                    'Resolution': 'Resolution',
                    'Won': 'Won',
                    'Triple': 'Triple',
                    'Trades': 'Trades',
                    'Initial Margin': 'Initial Margin'
                },
                tablefmt='pretty',

                showindex=False,
                numalign='right',
                stralign='center'
            ))
            
            # Print overall statistics in a more compact format
            print("\nOverall Statistics:")
            stats = {
                'Days Simulated': total_days,
                'Capital Range': f"${summary_df['Capital'].str.replace(',', '').astype(float).min():.1f} - ${summary_df['Capital'].str.replace(',', '').astype(float).max():.1f}",
                'Total Targets': summary_df['Targets'].sum(),
                'Total Trades': trade_df['Trades'].sum(),
                'Active Trades': trade_df['Active'].sum(),
                'Won Trades': len(trade_df[trade_df['Won'] == True])
            }
            
            # Format statistics in two columns
            stats_df = pd.DataFrame([stats]).T
            stats_df.columns = ['Value']
            print(tabulate(stats_df, headers='keys', tablefmt='pretty', showindex=True))
            
            # Final cleanup
            cuda.synchronize()
            self.cleanup_block_memory_pools()
            
            print("\nSimulation completed successfully")
            return True
            
        except Exception as e:
            print(f"Error in simulate_combinations: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def _prepare_block_data(self, candidate: Dict, combo_data: Dict, previous_day_state: Dict = None) -> Dict:
        """
        Prepare data for a specific block based on its candidate recipe with padding
        Args:
            candidate: Dictionary containing the candidate recipe
            combo_data: Dictionary containing combo data
            previous_day_state: Optional state from previous day to check for changes
        """
        try:
            print("\nPreparing block data:")
            block_signals = set()
            block_coins = set()
            block_windows = set()
            block_combos = {}
            combo_counter = 0
            slot_configs = {}  # New dictionary for slot configurations
            
            # Helper function to track regime changes (no longer blocks)
            def track_regime_change(slot_idx: int, new_coin_idx: int, new_combo_idx: int) -> bool:
                if previous_day_state is None:
                    return False
                    
                prev_slot = previous_day_state['slots'].get(f'slot_{slot_idx}')
                if prev_slot is None:
                    return False
                    
                # Track changes but don't use for blocking
                if prev_slot['coin_idx'] != new_coin_idx and prev_slot['coin_idx'] != -1:
                    print(f"Regime change in slot {slot_idx}: coin {prev_slot['coin_idx']} -> {new_coin_idx}")
                    return True
                    
                if prev_slot['combo_idx'] != new_combo_idx and prev_slot['combo_idx'] != -1:
                    print(f"Regime change in slot {slot_idx}: combo {prev_slot['combo_idx']} -> {new_combo_idx}")
                    return True
                    
                return False
            
            # Initialize slot configs with default values
            for slot_idx in range(self.max_simultaneous_trades):
                slot_configs[slot_idx] = {
                    'coin_idx': -1,
                    'combo_idx': -1,
                    'window_size': -1,
                    'direction': -1,
                    'regime_lock': 0,  # Default to locked
                    'signal_indices': [-1] * 6  # Pad with -1 for unused signal indices
                }         
            
            # Check if this is regime-independent case
            is_regime_independent = 'independent' in candidate['regime_data']
            
            if is_regime_independent:
                regime_info = candidate['regime_data']['independent']
                
                if 'top_3_combos' in regime_info:  # S2B case
                    print("\nProcessing S2B regime-independent case:")
                    if 'coin' in regime_info:
                        coin = regime_info['coin']
                        coin_idx = combo_data['coins'].index(coin)
                        
                        # Process all slots with same coin but different combos
                        for slot_idx, combo in enumerate(regime_info['top_3_combos'][:self.max_simultaneous_trades]):
                            combo_id = combo['combo_id']
                            
                            # Add combo to block tracking
                            if combo_id not in block_combos:
                                block_combos[combo_id] = {
                                    'signals': combo['signals'],
                                    'direction': combo['direction'],
                                    'window_size': combo['window_size'],
                                    'combo_idx': combo_counter,
                                    'regime': 'independent'
                                }
                                combo_counter += 1
                            
                            # Check if slot should be locked due to changes
                            should_lock = track_regime_change(slot_idx, coin_idx, block_combos[combo_id]['combo_idx'])
                            
                            # Update slot config
                            slot_configs[slot_idx].update({
                                'coin_idx': coin_idx,
                                'combo_idx': block_combos[combo_id]['combo_idx'],
                                'window_size': combo['window_size'],
                                'direction': combo['direction'],
                                'regime_lock': 1 if should_lock else 0,  # Lock if changes detected
                                'signal_indices': [-1] * 6  # Will be filled with actual signals
                            })
                            
                            # Add signals to tracking
                            for signal in combo['signals']:
                                block_signals.add(signal.strip())
                            block_windows.add(combo['window_size'])
                        
                        block_coins.add(coin)
                        
                elif 'window_size' in regime_info:  # S2A case
                    print("\nProcessing S2A regime-independent case:")
                    if 'top_3_coins' in regime_info:
                        combo_id = regime_info['combo_id']
                        
                        # Add combo to block tracking
                        if combo_id not in block_combos:
                            block_combos[combo_id] = {
                                'signals': regime_info['signals'],
                                'direction': regime_info['direction'],
                                'window_size': regime_info['window_size'],
                                'combo_idx': combo_counter,
                                'regime': 'independent'
                            }
                            combo_counter += 1
                        
                        # Process each slot with different coin but same combo
                        for slot_idx, coin in enumerate(regime_info['top_3_coins'][:self.max_simultaneous_trades]):
                            coin_idx = combo_data['coins'].index(coin)
                            
                            # Check if slot should be locked due to changes
                            should_lock = track_regime_change(slot_idx, coin_idx, block_combos[combo_id]['combo_idx'])
                            
                            # Update slot config
                            slot_configs[slot_idx].update({
                                'coin_idx': coin_idx,
                                'combo_idx': block_combos[combo_id]['combo_idx'],
                                'window_size': regime_info['window_size'],
                                'direction': regime_info['direction'],
                                'regime_lock': 1 if should_lock else 0,  # Lock if changes detected
                                'signal_indices': [-1] * 6  # Will be filled with actual signals
                            })
                            
                            block_coins.add(coin)
                        
                        # Add signals to tracking
                        for signal in regime_info['signals']:
                            block_signals.add(signal.strip())
                        block_windows.add(regime_info['window_size'])
            
            else:  # Regime-sensitive case
                print("\nProcessing regime-sensitive case:")
                regimes_to_check = ['bull', 'flat', 'bear']  # Order matters!
                
                if any('top_3_combos' in candidate['regime_data'].get(regime, {}) for regime in regimes_to_check):  # S2B
                    print("\nProcessing S2B regime-sensitive case:")
                    # Find first matching regime
                    matching_regime = None
                    matching_coin_idx = -1
                    
                    for regime in regimes_to_check:
                        if regime not in candidate['regime_data']:
                            continue
                            
                        regime_info = candidate['regime_data'][regime]
                        if 'coin' not in regime_info:
                            continue
                            
                        coin = regime_info['coin']
                        coin_idx = combo_data['coins'].index(coin)
                        coin_regime = int(combo_data['regime_data'][coin_idx, 0])
                        coin_regime_name = 'flat' if coin_regime == 3 else 'bull' if coin_regime == 1 else 'bear'
                        
                        if coin_regime_name == regime:
                            matching_regime = regime
                            matching_coin_idx = coin_idx
                            break
                    
                    if matching_regime:
                        regime_info = candidate['regime_data'][matching_regime]
                        
                        # Process all slots with same coin but different combos
                        for slot_idx, combo in enumerate(regime_info['top_3_combos'][:self.max_simultaneous_trades]):
                            combo_id = combo['combo_id']
                            
                            # Add combo to block tracking
                            if combo_id not in block_combos:
                                block_combos[combo_id] = {
                                    'signals': combo['signals'],
                                    'direction': combo['direction'],
                                    'window_size': combo['window_size'],
                                    'combo_idx': combo_counter,
                                    'regime': matching_regime
                                }
                                combo_counter += 1
                            
                            # Check if slot should be locked due to changes
                            should_lock = track_regime_change(slot_idx, matching_coin_idx, block_combos[combo_id]['combo_idx'])
                            
                            # Update slot config - lock if no regime match OR changes detected
                            slot_configs[slot_idx].update({
                                'coin_idx': matching_coin_idx,
                                'combo_idx': block_combos[combo_id]['combo_idx'],
                                'window_size': combo['window_size'],
                                'direction': combo['direction'],
                                'regime_lock': 1 if should_lock else 0,  # Lock if changes detected
                                'signal_indices': [-1] * 6  # Will be filled with actual signals
                            })
                            
                            # Add signals to tracking
                            for signal in combo['signals']:
                                block_signals.add(signal.strip())
                            block_windows.add(combo['window_size'])
                        
                        block_coins.add(regime_info['coin'])
                    else:
                        # No matching regime found - keep all slots locked
                        for slot_idx in range(self.max_simultaneous_trades):
                            slot_configs[slot_idx]['regime_lock'] = 1
                
                else:  # S2A case
                    print("\nProcessing S2A regime-sensitive case:")
                    # Process each slot independently
                    for slot_idx in range(self.max_simultaneous_trades):
                        slot_matched = False
                        
                        for regime in regimes_to_check:
                            if regime not in candidate['regime_data']:
                                continue
                                
                            regime_info = candidate['regime_data'][regime]
                            if 'top_3_coins' not in regime_info or slot_idx >= len(regime_info['top_3_coins']):
                                continue
                                
                            coin = regime_info['top_3_coins'][slot_idx]
                            coin_idx = combo_data['coins'].index(coin)
                            coin_regime = int(combo_data['regime_data'][coin_idx, 0])
                            coin_regime_name = 'flat' if coin_regime == 3 else 'bull' if coin_regime == 1 else 'bear'
                            
                            if coin_regime_name == regime:
                                combo_id = regime_info['combo_id']
                                
                                # Add combo to block tracking if not already added
                                if combo_id not in block_combos:
                                    block_combos[combo_id] = {
                                        'signals': regime_info['signals'],
                                        'direction': regime_info['direction'],
                                        'window_size': regime_info['window_size'],
                                        'combo_idx': combo_counter,
                                        'regime': regime
                                    }
                                    combo_counter += 1
                                
                                # Check if slot should be locked due to changes
                                should_lock = track_regime_change(slot_idx, coin_idx, block_combos[combo_id]['combo_idx'])
                                
                                # Update slot config
                                slot_configs[slot_idx].update({
                                    'coin_idx': coin_idx,
                                    'combo_idx': block_combos[combo_id]['combo_idx'],
                                    'window_size': regime_info['window_size'],
                                    'direction': regime_info['direction'],
                                    'regime_lock': 1 if should_lock else 0,  # Lock if changes detected
                                    'signal_indices': [-1] * 6  # Will be filled with actual signals
                                })
                                
                                # Add to tracking sets
                                block_coins.add(coin)
                                for signal in regime_info['signals']:
                                    block_signals.add(signal.strip())
                                block_windows.add(regime_info['window_size'])
                                
                                slot_matched = True
                                break
                        
                        if not slot_matched:
                            # Keep slot locked if no matching regime found
                            slot_configs[slot_idx]['regime_lock'] = 1
            
            # Fill signal indices for all slots
            for slot_idx in range(self.max_simultaneous_trades):
                if slot_configs[slot_idx]['combo_idx'] != -1:  # Only process active slots
                    combo_id = next(cid for cid, cinfo in block_combos.items() 
                                   if cinfo['combo_idx'] == slot_configs[slot_idx]['combo_idx'])
                    signals = block_combos[combo_id]['signals']
                    
                    # Map signals to their indices using the same signal_map as used for signal values/thresholds
                    signal_indices = []
                    for signal in signals:
                        signal_name = f"{signal.strip()}"  # The signal already includes direction (_long/_short)
                        if signal_name in combo_data['signal_map']:
                            signal_indices.append(combo_data['signal_map'][signal_name]['idx'])
                        else:
                            print(f"Warning: Signal {signal_name} not found in signal map")
                    
                    # Pad to length 6
                    signal_indices.extend([-1] * (6 - len(signal_indices)))
                    
                    # Update slot config
                    slot_configs[slot_idx]['signal_indices'] = signal_indices

            # Create mappings and padded arrays as before
            n_total_coins = len(combo_data['coins'])
            n_total_signals = len(combo_data['signal_map'])
            n_total_windows = len([24, 72, 120])
            
            coin_mapping = {coin: idx for idx, coin in enumerate(sorted(combo_data['coins']))}
            window_mapping = {w: i for i, w in enumerate([24, 72, 120])}
            
            # Create padded arrays
            padded_prices = np.full((n_total_coins, combo_data['timeframe_map'].shape[0], 5), 
                                  -1.0, dtype=np.float32)
            padded_signals = np.full((n_total_signals, n_total_coins, combo_data['timeframe_map'].shape[0]), 
                                   0.0, dtype=np.float32)
            padded_atr = np.full((n_total_coins, combo_data['timeframe_map'].shape[0]), 
                               -1.0, dtype=np.float32)
            padded_thresholds = np.full((n_total_coins, n_total_windows, n_total_signals, 
                                       combo_data['timeframe_map'].shape[0], 2), 0.0, dtype=np.float32)
            padded_regime = np.full((n_total_coins, combo_data['timeframe_map'].shape[0]), 
                                  -1, dtype=np.int32)
            padded_future_prices = np.full((n_total_coins, combo_data['future_prices'].shape[1], 5), 
                                         -1.0, dtype=np.float32)
            
            
            # Fill arrays with actual data
            required_coin_indices = [coin_mapping[coin] for coin in block_coins]
            required_signal_indices = [combo_data['signal_map'][signal.strip()]['idx'] for signal in block_signals]
            required_window_indices = [window_mapping[w] for w in block_windows]
            
            # Create a mapping between our signal indices and source data indices
            source_signal_mapping = {}
            for signal_name, signal_info in combo_data['signal_map'].items():
                source_signal_mapping[signal_info['idx']] = signal_name
            
            for coin_idx in required_coin_indices:
                padded_prices[coin_idx] = combo_data['price_data'][coin_idx]
                padded_atr[coin_idx] = combo_data['atr_values'][coin_idx]
                padded_regime[coin_idx] = combo_data['regime_data'][coin_idx]
                padded_future_prices[coin_idx] = combo_data['future_prices'][coin_idx]
                
                for sig_idx in required_signal_indices:
                    signal_values = combo_data['signal_values'][sig_idx, coin_idx]
                    
                    # Verify indices are within bounds
                    if sig_idx >= n_total_signals or coin_idx >= n_total_coins:
                        print(f"Warning: Index out of bounds - sig_idx: {sig_idx}/{n_total_signals}, coin_idx: {coin_idx}/{n_total_coins}")
                        continue
                    
                    # Copy signal values
                    padded_signals[sig_idx, coin_idx, :] = signal_values.astype(np.float32)
                
                for win_idx in required_window_indices:
                    for sig_idx in required_signal_indices:
                        # Ensure proper indexing for 5D threshold structure
                        threshold_values = combo_data['thresholds'][coin_idx, win_idx, sig_idx]
                        
                        # Add validation before assignment
                        if threshold_values.shape != (padded_thresholds.shape[-2], padded_thresholds.shape[-1]):
                            print(f"Threshold shape mismatch at coin={coin_idx}, window={win_idx}, signal={sig_idx}")
                            print(f"Expected: {(padded_thresholds.shape[-2], padded_thresholds.shape[-1])}")
                            print(f"Actual: {threshold_values.shape}")
                            continue
                            
                        if np.isnan(threshold_values).any():
                            print(f"NaN detected in thresholds at coin={coin_idx}, window={win_idx}, signal={sig_idx}")
                            
                        padded_thresholds[coin_idx, win_idx, sig_idx, :, :] = threshold_values
            
            print("Slot configurations:")
            for slot_idx in range(len(slot_configs)):
                _print_debug_section(f"Slot {slot_idx} Configuration", indent=1)
                config_data = {
                    "Coin Index": slot_configs[slot_idx]['coin_idx'],
                    "Combo Index": slot_configs[slot_idx]['combo_idx'],
                    "Window Size": slot_configs[slot_idx]['window_size'],
                    "Direction": slot_configs[slot_idx]['direction'],
                    "Regime Lock": "Locked" if slot_configs[slot_idx]['regime_lock'] == 1 else "Unlocked",
                    "Signal Indices": [idx for idx in slot_configs[slot_idx]['signal_indices'] if idx != -1]
                }
                _print_debug_section("Details", config_data, indent=2)

            # Return complete block data with slot configurations
            return {
                'prices': padded_prices,
                'future_prices': padded_future_prices,
                'signals': padded_signals,
                'atr_values': padded_atr,
                'thresholds': padded_thresholds,
                'regime_data': padded_regime,
                'metadata': {
                    'coins': sorted(block_coins),
                    'signals': sorted(block_signals),
                    'coin_indices': {coin: coin_mapping[coin] for coin in block_coins},
                    'signal_indices': {signal.strip(): combo_data['signal_map'][signal.strip()]['idx'] 
                                      for signal in block_signals},
                    'window_indices': {w: window_mapping[w] for w in block_windows},
                    'actual_n_coins': len(block_coins),
                    'actual_n_signals': len(block_signals),
                    'actual_n_windows': len(block_windows),
                    'combos': block_combos,
                    'slot_configs': slot_configs
                }
            }
                
        except Exception as e:
            print(f"Error preparing block data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# Create a helper function for formatted debug printing
def _print_debug_section(title, data=None, indent=0):
    """Helper function to print debug information in a formatted way"""
    indent_str = "  " * indent
    separator = "=" * (80 - len(indent_str))
    print(f"\n{indent_str}{separator}")
    print(f"{indent_str}=== {title} ===")
    print(f"{indent_str}{separator}")
    if data:
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"{indent_str}  {key}: {value}")
        elif isinstance(data, (list, tuple, set)):
            for item in data:
                print(f"{indent_str}  - {item}")
        else:
            print(f"{indent_str}  {data}")
