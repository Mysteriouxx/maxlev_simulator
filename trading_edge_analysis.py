import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

def calculate_triple_outcome(trade_size: float, slippage_rate: float = 0) -> Tuple[float, float]:
    """
    Calculate triple outcome with explicit slippage calculation at each step
    """
    position_value = trade_size * 50
    fee_rate = 0.0004  # 0.04%
    
    # For loss scenario
    entry_fees = position_value * fee_rate
    entry_slippage = position_value * slippage_rate
    exit_fees = position_value * fee_rate
    exit_slippage = position_value * slippage_rate
    
    loss = -(trade_size + entry_fees + entry_slippage + exit_fees + exit_slippage)
    
    # For win scenario (triple win)
    # First trade
    first_entry_fees = position_value * fee_rate
    first_entry_slippage = position_value * slippage_rate
    first_trade = trade_size * 3.5  # Successful first trade
    first_position_close = first_trade * 50
    first_exit_fees = first_position_close * fee_rate
    first_exit_slippage = first_position_close * slippage_rate
    first_net = first_trade - (first_entry_fees + first_entry_slippage + first_exit_fees + first_exit_slippage)
    
    # Second trade
    second_position_value = first_net * 50
    second_entry_fees = second_position_value * fee_rate
    second_entry_slippage = second_position_value * slippage_rate
    second_trade = first_net * 3.5  # Successful second trade
    second_position_close = second_trade * 50
    second_exit_fees = second_position_close * fee_rate
    second_exit_slippage = second_position_close * slippage_rate
    second_net = second_trade - (second_entry_fees + second_entry_slippage + second_exit_fees + second_exit_slippage)
    
    # Third trade
    third_position_value = second_net * 50
    third_entry_fees = third_position_value * fee_rate
    third_entry_slippage = third_position_value * slippage_rate
    third_trade = second_net * 3.5  # Successful third trade
    third_position_close = third_trade * 50
    third_exit_fees = third_position_close * fee_rate
    third_exit_slippage = third_position_close * slippage_rate
    final_net = third_trade - (third_entry_fees + third_entry_slippage + third_exit_fees + third_exit_slippage)
    
    win = final_net - trade_size
    
    total_costs = (entry_fees + entry_slippage + exit_fees + exit_slippage +  # Loss scenario costs
                  first_entry_fees + first_entry_slippage + first_exit_fees + first_exit_slippage +  # First trade
                  second_entry_fees + second_entry_slippage + second_exit_fees + second_exit_slippage +  # Second trade
                  third_entry_fees + third_entry_slippage + third_exit_fees + third_exit_slippage)  # Third trade
    
    # Print detailed breakdown for $10 trade
    if abs(trade_size - 10) < 0.01:
        print(f"\nDetailed Breakdown for ${trade_size} trade with {slippage_rate*100}% slippage:")
        print(f"First trade: ${first_net:.2f}")
        print(f"Second trade: ${second_net:.2f}")
        print(f"Final trade: ${final_net:.2f}")
        print(f"Total fees: ${total_costs:.2f}")
    
    return win, loss, total_costs

def analyze_slippage_impact():
    """
    Analyze and visualize the impact of different slippage rates
    """
    trade_size = 10
    slippage_rates = [0, 0.0005, 0.001, 0.003]  # 0%, 0.05%, 0.1%, 0.3%
    
    print("\nDetailed Analysis for $10 Trade Size:")
    for slip in slippage_rates:
        win, loss, costs = calculate_triple_outcome(trade_size, slip)
        print(f"\nSlippage Rate: {slip*100}%")
        print(f"Win Amount: ${win:.2f}")
        print(f"Loss Amount: ${loss:.2f}")
        print(f"Total Costs: ${costs:.2f}")
        print(f"Win/Loss Ratio: {abs(win/loss):.2f}")

def analyze_optimal_sizing():
    """
    Analyze optimal sizing with different slippage rates
    """
    win_rates = np.linspace(0.01, 0.15, 29)
    capital = 500
    slippage_rates = [0, 0.0005, 0.001, 0.003]  # 0%, 0.05%, 0.1%, 0.3%
    colors = ['blue', 'green', 'red', 'purple']
    
    plt.figure(figsize=(15, 15))
    
    for idx, slippage in enumerate(slippage_rates):
        optimal_sizes = []
        growth_rates = []
        expected_profits = []
        
        for wr in win_rates:
            sizes = np.linspace(1, 50, 100)
            best_growth = float('-inf')
            best_size = 0
            best_profit = 0
            
            for size in sizes:
                win, loss, _ = calculate_triple_outcome(size, slippage)
                
                # Calculate metrics
                win_capital = capital + win
                loss_capital = capital + loss
                
                if loss_capital <= 0:
                    continue
                
                growth_rate = wr * np.log(win_capital/capital) + \
                            (1 - wr) * np.log(loss_capital/capital)
                expected_profit = wr * win + (1 - wr) * loss
                
                if growth_rate > best_growth:
                    best_growth = growth_rate
                    best_size = size
                    best_profit = expected_profit
            
            optimal_sizes.append(best_size)
            growth_rates.append(best_growth)
            expected_profits.append(best_profit)
        
        label = f'{slippage*100}% slippage'
        
        plt.subplot(3, 1, 1)
        plt.plot(win_rates * 100, optimal_sizes, color=colors[idx], label=label)
        
        plt.subplot(3, 1, 2)
        plt.plot(win_rates * 100, growth_rates, color=colors[idx], label=label)
        
        plt.subplot(3, 1, 3)
        plt.plot(win_rates * 100, expected_profits, color=colors[idx], label=label)
    
    plt.subplot(3, 1, 1)
    plt.title('Optimal Position Size vs Win Rate ($500 Initial Capital)')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Optimal Position Size ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.title('Expected Growth Rate (%) vs Win Rate')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Expected Growth Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.title('Expected Profit ($) per Attempt vs Win Rate')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Expected Profit ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    stats_text = ("Parameters:\n"
                 "Initial Capital: $500\n"
                 "Max Trade Size: $50\n"
                 "Position Leverage: 50x\n"
                 "Base Fee Rate: 0.04% per entry/exit")
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('slippage_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    analyze_slippage_impact()
    analyze_optimal_sizing() 