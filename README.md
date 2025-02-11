# Maxlev Trading System Simulation Report

## Overview

Maxlev is a proprietary high-leverage pyramiding trading system designed to optimize returns in the cryptocurrency market through iterative position resizing. The system aims to amplify capital in a strategic way by utilizing high leverage at each stage of the trading cycle.

### Key Formula:
**Final Capital** = Initial Capital × (1 + Leverage × Stage_Return)^{Number_of_Stages}

Maxlev is designed to intelligently scale positions with leverage, maximizing profit potential while managing risk effectively.

![image](https://github.com/user-attachments/assets/ad8168fa-0ae6-47bc-aec3-19e4d18afd3f)

## Key Parameters for Trading
To trade using Maxlev, a trader must configure the following key parameters:
- **Trading Size**: The amount to be traded.
- **Price Change**: The desired price change to trigger trades.
- **Stage Increment**: The increment used at each stage of the pyramiding strategy.
- **Leverage**: The level of leverage applied.
- **Target Crypto**: The cryptocurrency chosen for trading.

## Optimizing Entry Points in Maxlev

The next step in the optimization process is identifying ideal entry points. Maxlev employs advanced backtesting and reverse-engineering techniques to determine the most profitable entry signals. The optimization is done in three key phases:

### Step 1: Brute-Force Test
Inspired by Richard Dennis' Turtle Trading experiment, we use indicator values as potential entry signals, which are tested through a brute-force methodology. A rolling window technique is employed to test multiple indicators on various timeframes, with each threshold tested to determine the best parameters for triggering trades. This results in an optimal "threshold lookup table" that identifies the most successful entry signals based on historical data.

### Step 2: Combo-Test
In this phase, we simulate live trading by combining up to three of the optimal signals identified in Step 1. All combinations of indicator thresholds are evaluated to measure performance in realistic trading setups. This step helps to select the most successful combinations, improving the win rate over the break-even point of Maxlev.

### Step 3: Simulation
The best-performing combinations from Step 2 are tested in a detailed simulation using real market data from February 2024 to January 2025. The simulation considers real-world constraints such as transaction fees, margin requirements, slippage, market regime changes, and various take-profit rhythms. It includes multiple trading strategies and is tested for both fixed-combo and fixed-coin scenarios to ensure adaptable performance under various market conditions.

## Simulated Results & Practical Application

The simulation focuses on a range of cryptocurrencies, from DeFi to NFTs, identifying trading opportunities that maximize returns while managing risk effectively. The results of the simulation are reported in terms of capital progression over the period and the performance of the top 3 portfolios under different trial setups. The performance is compared against the benchmark of holding BTC spot.

### Simulation Configuration:
- **Initial Capital**: $500
- **Target Capital**: $10,000
- **Training Period**: 2023-09-01 to 2024-02-01
- **Simulation Period**: 2024-02-01 to 2025-01-01
- **Tested Cryptocurrencies**: 18
- **Tested Strategies**:
  - **S1**: Single-Win Strategy
  - **S2**: Double-Win Strategy
  - **S3**: Triple-Win Strategy
- **Real-life Constraints**: Data acquired from Bitget Futures Exchange
- **Data Source**: Binance API using Python's `binance.py` and CCXT library
- **Trading Setup**: 2 Cases per Strategy, 4 Trials per Case, 14 Portfolio Variations per Trial, 3 Trading Slots per Portfolio = 24 Simulations in Total
- **Granularity**: 1-Minute Resolution (with special attention to edge cases)
- **Dynamic Market Regime Adjustment**: Considered
- **Dynamic Capital Management**: Considered
- **Tech Stack**: Python, CUDA, SQL

## Results

![S1_capital_comparisons_A](https://github.com/user-attachments/assets/26b82583-4311-4119-9f56-f9a3b8c25ed6)
![S1_capital_comparisons_B](https://github.com/user-attachments/assets/3fbf5a7b-bd73-457c-b478-3203babf6925)
![S2_capital_comparisons_A](https://github.com/user-attachments/assets/62060b32-f7e4-43c4-8aaa-154cac53a715)
![S2_capital_comparisons_B](https://github.com/user-attachments/assets/93f7d95f-9663-4213-9e81-ead899bb2277)
![S3_capital_comparisons_A](https://github.com/user-attachments/assets/3ad50130-826f-4d91-a8dd-6e9da4c61d2e)
![S3_capital_comparisons_B](https://github.com/user-attachments/assets/99be8aa7-9bd8-4f7b-9330-b19a8a452d81)

## Conclusion

Maxlev employs a dynamic and systematic approach to trading with high leverage. The combination of advanced backtesting, simulation, and real-world constraint considerations makes Maxlev an adaptable and robust tool for maximizing cryptocurrency trading profits. The simulation results demonstrate significant improvement over holding BTC spot, suggesting strong potential for further refinement and real-world application.

---

## Requirements

- Python 3.x
- CUDA (for GPU acceleration)
- SQL (for data management)
- CCXT library
- Binance API Key
- Bitget Futures Exchange Data
