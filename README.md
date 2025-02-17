# Maxlev Trading System Simulation Report

## Overview

Maxlev is a momentum strategy that uses a high-leverage pyramiding trading system designed to optimize returns in the cryptocurrency futures market through iterative position resizing. The following table shows the returns over a range of price change (regardless of direction): 

![image](https://github.com/user-attachments/assets/0bd273da-3b29-414a-a177-8d1f3797080e)

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

![image](https://github.com/user-attachments/assets/b2d58ff3-0aff-4e52-9470-a14e1b7397cf)

## Simulated Results & Practical Application

The simulation focuses on a range of cryptocurrencies, from DeFi to NFTs, identifying trading opportunities that maximize returns while managing risk effectively. The results of the simulation are reported in terms of capital progression over the period and the performance of the top 3 portfolios under different trial setups. The performance is compared against the benchmark of holding BTC spot. The portfolio (trial) construction uses a novel **Atomic Strategy Combination Framework (ASCF)** methodology that decomposes trading strategies into atomic, self-contained units known as "blocks." 

![image](https://github.com/user-attachments/assets/7752bd8a-de09-4dae-91de-3bc1f5b71cf9)

### Simulation Configuration:
- **Initial Capital**: $500
- **Target Capital**: $10,000
- **Training Period**: 1. 2021-07-01 to 2022-01-01; 2. 2022-07-01 to 2023-01-01; 3. 2023-07-01 to 2024-01-01
- **Simulation Period**: 1. 2022-01-01 to 2023-01-01; 2. 2023-01-01 to 2024-01-01; 3. 2024-01-01 to 2025-01-01
- **Tested Cryptocurrencies**: 50
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

## Results: Log-scaled Portfolio Capital Growth
![multi_year_performance_log](https://github.com/user-attachments/assets/7c711242-3b8b-4281-8413-7d6b2037765e)

![image](https://github.com/user-attachments/assets/e2d205e1-fe50-40c9-916a-4ad4febdafd4)

![image](https://github.com/user-attachments/assets/2d94906f-2d36-4070-bfbb-f5fb750948a4)


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
