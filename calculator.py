import pandas as pd
import numpy as np
import ta

class IndicatorCalculator:
    """
    Calculates technical indicators for pattern analysis using the 'ta' library.
    """
    
    def __init__(self, df: pd.DataFrame, timeframe:str, config_path='config.yaml'):
        """
        Initialize calculator with price data and configuration.
        
        Args:
            df (pd.DataFrame): OHLCV data with columns:
                - open, high, low, close, volume
            config_path (str): Path to configuration file
        """
        self.df = df.copy()
        self.timeframe = timeframe
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from yaml file"""
        import yaml
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
            
    def calculate_all_indicators(self) -> pd.DataFrame:
        """Calculate all technical indicators"""
        # Calculate base indicators first
        self._calculate_bf_indicators()
        
        # # Calculate rule-based indicators
        # self._calculate_rule_indicators()
        
        return self.df
        
    def _calculate_bf_indicators(self) -> None:
        """Calculate brute force indicators"""
        # MACD
        macd = ta.trend.MACD(
            close=self.df['close'],
            window_slow=self.config['indicators']['macd_slow'],
            window_fast=self.config['indicators']['macd_fast'],
            window_sign=self.config['indicators']['macd_signal']
        )
        
        # Calculate MACD Histogram
        self.df['bf_MACD_HG'] = macd.macd_diff()
        
        # RSI
        self.df['bf_RSI'] = ta.momentum.rsi(
            close=self.df['close'],
            window=self.config['indicators']['rsi_period']
        )
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            window=self.config['indicators']['stoch_period']
        )
        self.df['bf_STOCH_K'] = stoch.stoch()
        self.df['bf_STOCH_D'] = stoch.stoch_signal()
        
        # ADX and Directional Movement
        adx = ta.trend.ADXIndicator(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            window=self.config['indicators']['adx_period']
        )
        self.df['bf_ADX'] = adx.adx()
        self.df['DI_PLUS'] = adx.adx_pos()
        self.df['DI_MINUS'] = adx.adx_neg()
        
        # ATR for dynamic parameters
        self.df['ATR'] = ta.volatility.average_true_range(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            window=self.config['indicators']['atr_period']
        )
        
        # VZO
        self._calculate_vzo()
        
        # VPCI
        self.df['bf_VPCI'] = self._calculate_vpci()
        
        # OBV and variants
        self.df['bf_OBV'] = ta.volume.on_balance_volume(
            self.df['close'], 
            self.df['volume']
        )
        
        # ROC OBV calculation
        lookback = self.config['indicators']['rsi_period']  # typically 14
        
        # Initialize ROC OBV with NaN values
        self.df['bf_ROC_OBV'] = np.nan
        
        # Calculate ROC OBV only after the lookback period
        current_obv = self.df['bf_OBV'][lookback:]
        previous_obv = self.df['bf_OBV'].shift(lookback)[lookback:]
        
        # Compute ROC OBV starting from lookback period
        self.df.loc[self.df.index[lookback:], 'bf_ROC_OBV'] = (
            (current_obv - previous_obv) / previous_obv.abs() * 100
        ).replace([np.inf, -np.inf], np.nan)
        
        # Normalized OBV calculation remains the same
        rolling_min = self.df['bf_OBV'].rolling(window=lookback).min()
        rolling_max = self.df['bf_OBV'].rolling(window=lookback).max()
        denominator = (rolling_max - rolling_min).replace(0, np.nan)
        
        self.df['bf_N_OBV'] = (
            (self.df['bf_OBV'] - rolling_min) / 
            denominator * 100
        )
        
    def _calculate_rule_indicators(self) -> None:
        """Calculate rule-based indicators"""
        # Data validation
        required_columns = [
            'VPCI', 'bf_ADX', 'DI_PLUS', 'DI_MINUS', 
            'bf_STOCH_K', 'bf_STOCH_D'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Check for NaN values in required columns
        nan_counts = self.df[required_columns].isna().sum()
        if nan_counts.any():
            print("Warning: NaN values found in indicators:")
            for col, count in nan_counts.items():
                if count > 0:
                    print(f"{col}: {count} NaN values")
                    
        # VPCI Rule
        vpci_bb_period = self.config['indicators']['vpci_bb_period']
        vpci_bb_std = self.config['indicators']['vpci_bb_std']
        
        # Calculate moving average and standard deviation
        vpci_ma = self.df['VPCI'].rolling(window=vpci_bb_period).mean()
        vpci_std = self.df['VPCI'].rolling(window=vpci_bb_period).std()
        
        # Bollinger Bands
        self.df['VPCI_BB_upper'] = vpci_ma + (vpci_std * vpci_bb_std)
        self.df['VPCI_BB_lower'] = vpci_ma - (vpci_std * vpci_bb_std)
        
        # Initialize rule columns with 0
        self.df['rule_VPCI'] = 0
        self.df['rule_ADX'] = 0
        self.df['rule_STOCH'] = 0
        
        # Calculate lookback period
        lookback = max(
            vpci_bb_period, 
            self.config['indicators']['adx_period'],
            self.config['indicators']['stoch_period']
        )
        
        # Create valid data mask
        valid_mask = ~(
            self.df['VPCI'].isna() | 
            self.df['bf_ADX'].isna() | 
            self.df['bf_STOCH_K'].isna() | 
            self.df['bf_STOCH_D'].isna()
        )
        
        # VPCI signals
        vpci_bullish = (
            (self.df['VPCI'] > 0) & 
            (self.df['VPCI'] > self.df['VPCI_BB_lower']) & 
            (self.df['VPCI'].shift(1) <= self.df['VPCI_BB_lower'].shift(1))
        )
        
        vpci_bearish = (
            (self.df['VPCI'] < 0) & 
            (self.df['VPCI'] < self.df['VPCI_BB_upper']) & 
            (self.df['VPCI'].shift(1) >= self.df['VPCI_BB_upper'].shift(1))
        )
        
        # ADX signals
        adx_bullish = (
            (self.df['bf_ADX'] > 12) &
            (self.df['DI_PLUS'] > self.df['DI_MINUS'])
        )
        
        adx_bearish = (
            (self.df['bf_ADX'] > 12) &
            (self.df['DI_MINUS'] > self.df['DI_PLUS'])
        )
        
        # Stochastic signals
        stoch_bullish = (
            (self.df['bf_STOCH_K'] < 20) &
            (self.df['bf_STOCH_K'] > self.df['bf_STOCH_D']) &
            (self.df['bf_STOCH_K'].shift(1) <= self.df['bf_STOCH_D'].shift(1))
        )
        
        stoch_bearish = (
            (self.df['bf_STOCH_K'] > 80) &
            (self.df['bf_STOCH_K'] < self.df['bf_STOCH_D']) &
            (self.df['bf_STOCH_K'].shift(1) >= self.df['bf_STOCH_D'].shift(1))
        )
        
        # Apply signals only where data is valid
        self.df.loc[valid_mask & vpci_bullish, 'rule_VPCI'] = 1
        self.df.loc[valid_mask & vpci_bearish, 'rule_VPCI'] = -1
        
        self.df.loc[valid_mask & adx_bullish, 'rule_ADX'] = 1
        self.df.loc[valid_mask & adx_bearish, 'rule_ADX'] = -1
        
        self.df.loc[valid_mask & stoch_bullish, 'rule_STOCH'] = 1
        self.df.loc[valid_mask & stoch_bearish, 'rule_STOCH'] = -1

    def _calculate_vpci(self) -> pd.Series:
        """
        Calculate Volume Price Confirmation Indicator (VPCI)
        
        Returns:
            pd.Series: VPCI values
        """
        period = self.config['indicators']['vpci_period']
        
        # Price trend
        price_ma = self.df['close'].rolling(period).mean()
        price_ma_std = self.df['close'].rolling(period).std()
        price_trend = ((self.df['close'] - price_ma) / price_ma_std)
        
        # Volume trend
        vol_ma = self.df['volume'].rolling(period).mean()
        vol_ma_std = self.df['volume'].rolling(period).std()
        volume_trend = ((self.df['volume'] - vol_ma) / vol_ma_std)
        
        # Calculate VPCI
        vpci = price_trend * volume_trend
        
        # Store intermediate result
        self.df['VPCI'] = vpci
        
        return vpci  # Return the series for bf_VPCI assignment

    def _calculate_vzo(self) -> None:
        """
        Add Volume Zone Oscillator (VZO)
        """
        period = self.config['indicators']['vpci_period']  # Using same period as VPCI
        
        # Calculate volume changes
        self.df['vol_up'] = np.where(
            self.df['close'] >= self.df['close'].shift(1),
            self.df['volume'],
            0
        )
        self.df['vol_down'] = np.where(
            self.df['close'] < self.df['close'].shift(1),
            self.df['volume'],
            0
        )
        
        # Calculate VZO
        self.df['bf_VZO'] = (
            (self.df['vol_up'].rolling(period).sum() - 
             self.df['vol_down'].rolling(period).sum()
            ) / self.df['volume'].rolling(period).sum()
        ) * 100
        
        # Clean up temporary columns
        self.df.drop(['vol_up', 'vol_down'], axis=1, inplace=True)

    def calculate_market_regime(self) -> pd.DataFrame:
        """
        Calculate market regime status for each candle.
        """
        # Calculate EMAs
        self.df['ema20'] = self.df['close'].ewm(span=20, adjust=True).mean()
        self.df['ema60'] = self.df['close'].ewm(span=60, adjust=True).mean()
        self.df['ema120'] = self.df['close'].ewm(span=120, adjust=True).mean()

        # Volume metrics - modified to match TradingView exactly
        vol_sma20 = ta.trend.sma_indicator(self.df['volume'], window=20)
        vol_sma5 = ta.trend.sma_indicator(self.df['volume'], window=5)
        
        vol_strength = self.df['volume'] > vol_sma20 * 1.5
        vol_trend = vol_sma5 > vol_sma20  # Using SMAs directly for comparison
        
        # Momentum calculations
        self.df['mom_1d'] = (self.df['close'] - self.df['close'].shift(1)) / self.df['close'].shift(1) * 100
        self.df['mom_3d'] = (self.df['close'] - self.df['close'].shift(3)) / self.df['close'].shift(3) * 100
        self.df['mom_5d'] = (self.df['close'] - self.df['close'].shift(5)) / self.df['close'].shift(5) * 100
        self.df['mom_10d'] = (self.df['close'] - self.df['close'].shift(10)) / self.df['close'].shift(10) * 100

        # Momentum decay detection
        mom_weakening = (
            self.df['mom_1d'].abs() < self.df['mom_3d'].abs()
        ) & (
            self.df['mom_3d'].abs() < self.df['mom_5d'].abs()
        ) & (
            self.df['mom_5d'].abs() < self.df['mom_10d'].abs()
        )

        # Slope calculations
        self.df['ema20_slope'] = (self.df['ema20'] - self.df['ema20'].shift(3)) / self.df['ema20'].shift(3) * 100
        self.df['ema60_slope'] = (self.df['ema60'] - self.df['ema60'].shift(3)) / self.df['ema60'].shift(3) * 100

        # Initialize first row's counters
        first_idx = self.df.index[0]
        self.df.loc[first_idx, 'major_bull_days'] = 0.0
        self.df.loc[first_idx, 'major_bear_days'] = 0.0
        self.df.loc[first_idx, 'minor_bull_days'] = 0.0
        self.df.loc[first_idx, 'minor_bear_days'] = 0.0
        self.df.loc[first_idx, 'flat_days'] = 0.0
        self.df.loc[first_idx, 'regime'] = 0.0

        # Calculate regimes using index instead of integer position
        for current_idx, prev_idx in zip(self.df.index[1:], self.df.index[:-1]):
            if current_idx.strftime('%Y-%m-%d') in ['2024-03-01', '2024-03-02', '2024-03-03']:
                print(f"\nDebug for {current_idx.strftime('%Y-%m-%d')}:")
                print(f"EMA condition: {self.df['ema20'].loc[current_idx] > self.df['ema60'].loc[current_idx] > self.df['ema120'].loc[current_idx]}")
                print(f"Slope condition: {self.df['ema20_slope'].loc[current_idx] > 0.5}")
                print(f"Volume strength: {vol_strength.loc[current_idx]}")
                print(f"Volume trend: {vol_trend.loc[current_idx]}")
                print(f"Previous regime: {self.df.loc[prev_idx, 'regime']}")
                print(f"Previous major_bull_days: {self.df.loc[prev_idx, 'major_bull_days']}")

            # Get conditions for current bar
            major_bull = (
                self.df['ema20'].loc[current_idx] > self.df['ema60'].loc[current_idx] > self.df['ema120'].loc[current_idx]
                and self.df['ema20_slope'].loc[current_idx] > 0.5
                and vol_strength.loc[current_idx]
                and vol_trend.loc[current_idx]
            )
            major_bear = (
                self.df['ema20'].loc[current_idx] < self.df['ema60'].loc[current_idx] < self.df['ema120'].loc[current_idx]
                and self.df['ema20_slope'].loc[current_idx] < -0.5
                and vol_strength.loc[current_idx]
                and vol_trend.loc[current_idx]
            )
            minor_bull = (
                self.df['ema20'].loc[current_idx] > self.df['ema60'].loc[current_idx]
                and (self.df['ema60'].loc[current_idx] > self.df['ema120'].loc[current_idx] 
                     or self.df['ema20_slope'].loc[current_idx] > 0.3)
            )
            minor_bear = (
                self.df['ema20'].loc[current_idx] < self.df['ema60'].loc[current_idx]
                and (self.df['ema60'].loc[current_idx] < self.df['ema120'].loc[current_idx] 
                     or self.df['ema20_slope'].loc[current_idx] < -0.3)
            )
            
            ema_tight = abs(self.df['ema20'].loc[current_idx] - self.df['ema60'].loc[current_idx]) / self.df['ema60'].loc[current_idx] * 100 < 1.8
            vol_quiet = (self.df['volume'].loc[current_idx] < vol_sma20.loc[current_idx] * 1.2 
                        and self.df['volume'].loc[current_idx] > vol_sma20.loc[current_idx] * 0.8)
            flat_condition = ema_tight and mom_weakening.loc[current_idx] and vol_quiet

            # Copy previous counters first (this is key for proper state maintenance)
            self.df.loc[current_idx, 'major_bull_days'] = self.df.loc[prev_idx, 'major_bull_days']
            self.df.loc[current_idx, 'major_bear_days'] = self.df.loc[prev_idx, 'major_bear_days']
            self.df.loc[current_idx, 'minor_bull_days'] = self.df.loc[prev_idx, 'minor_bull_days']
            self.df.loc[current_idx, 'minor_bear_days'] = self.df.loc[prev_idx, 'minor_bear_days']
            self.df.loc[current_idx, 'flat_days'] = self.df.loc[prev_idx, 'flat_days']
            self.df.loc[current_idx, 'regime'] = self.df.loc[prev_idx, 'regime']  # Maintain previous regime

            # Then update based on current conditions
            if major_bull:
                self.df.loc[current_idx, 'major_bull_days'] += 1
                self.df.loc[current_idx, 'major_bear_days'] = 0.0
                self.df.loc[current_idx, 'minor_bull_days'] = 0.0
                self.df.loc[current_idx, 'minor_bear_days'] = 0.0
                self.df.loc[current_idx, 'flat_days'] = 0.0
            elif major_bear:
                self.df.loc[current_idx, 'major_bear_days'] += 1
                self.df.loc[current_idx, 'major_bull_days'] = 0.0
                self.df.loc[current_idx, 'minor_bull_days'] = 0.0
                self.df.loc[current_idx, 'minor_bear_days'] = 0.0
                self.df.loc[current_idx, 'flat_days'] = 0.0
            elif minor_bull and not flat_condition:
                self.df.loc[current_idx, 'minor_bull_days'] += 1
                self.df.loc[current_idx, 'minor_bear_days'] = 0.0
                if not major_bull:
                    self.df.loc[current_idx, 'major_bull_days'] = max(0.0, self.df.loc[prev_idx, 'major_bull_days'] - 1)
                if self.df.loc[prev_idx, 'flat_days'] > 0:
                    self.df.loc[current_idx, 'flat_days'] = max(0.0, self.df.loc[prev_idx, 'flat_days'] - 1)
            elif minor_bear and not flat_condition:
                self.df.loc[current_idx, 'minor_bear_days'] += 1
                self.df.loc[current_idx, 'minor_bull_days'] = 0.0
                if not major_bear:
                    self.df.loc[current_idx, 'major_bear_days'] = max(0.0, self.df.loc[prev_idx, 'major_bear_days'] - 1)
                if self.df.loc[prev_idx, 'flat_days'] > 0:
                    self.df.loc[current_idx, 'flat_days'] = max(0.0, self.df.loc[prev_idx, 'flat_days'] - 1)
            elif flat_condition:
                self.df.loc[current_idx, 'flat_days'] += 1
                self.df.loc[current_idx, 'major_bull_days'] = max(0.0, self.df.loc[prev_idx, 'major_bull_days'] - 0.5)
                self.df.loc[current_idx, 'major_bear_days'] = max(0.0, self.df.loc[prev_idx, 'major_bear_days'] - 0.5)
                self.df.loc[current_idx, 'minor_bull_days'] = max(0.0, self.df.loc[prev_idx, 'minor_bull_days'] - 0.5)
                self.df.loc[current_idx, 'minor_bear_days'] = max(0.0, self.df.loc[prev_idx, 'minor_bear_days'] - 0.5)

            # Determine regime (moved after counter updates)
            if self.df.loc[current_idx, 'major_bull_days'] >= 3:
                self.df.loc[current_idx, 'regime'] = 2.0
            elif self.df.loc[current_idx, 'major_bear_days'] >= 3:
                self.df.loc[current_idx, 'regime'] = -2.0
            elif self.df.loc[current_idx, 'flat_days'] >= 3 and flat_condition:
                self.df.loc[current_idx, 'regime'] = 0.0
            elif self.df.loc[current_idx, 'minor_bull_days'] >= 2:
                self.df.loc[current_idx, 'regime'] = 1.0
            elif self.df.loc[current_idx, 'minor_bear_days'] >= 2:
                self.df.loc[current_idx, 'regime'] = -1.0

        return self.df
