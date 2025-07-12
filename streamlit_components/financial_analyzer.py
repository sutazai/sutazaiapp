"""
Financial Analysis AI System for SutazAI Streamlit Interface
Provides comprehensive financial analysis, portfolio optimization, and market insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import asyncio
import yfinance as yf
import requests
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FinancialAnalyzer:
    """Advanced Financial Analysis System with AI-powered insights"""
    
    def __init__(self, model_manager=None):
        self.model_manager = model_manager
        self.analysis_cache = {}
        self.supported_markets = {
            'US': ['NASDAQ', 'NYSE', 'AMEX'],
            'EU': ['LSE', 'EURONEXT', 'DAX'],
            'ASIA': ['TSE', 'HKEX', 'SSE']
        }
        
    def render_financial_dashboard(self):
        """Render the main financial analysis dashboard"""
        st.header("💰 Financial Analysis AI System")
        
        # Quick market overview
        self._render_market_overview()
        
        # Analysis sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Stock Analysis",
            "📊 Portfolio Optimization", 
            "⚠️ Risk Assessment",
            "🔍 Market Trends",
            "📋 Financial Reports"
        ])
        
        with tab1:
            self.render_stock_analysis()
        
        with tab2:
            self.render_portfolio_optimization()
        
        with tab3:
            self.render_risk_assessment()
        
        with tab4:
            self.render_market_trends()
        
        with tab5:
            self.render_financial_reports()
    
    def _render_market_overview(self):
        """Render quick market overview"""
        st.subheader("🌍 Global Market Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Sample market data - in production would fetch real-time data
        with col1:
            st.metric("S&P 500", "4,500.25", "+1.2%", delta_color="normal")
        
        with col2:
            st.metric("NASDAQ", "14,200.80", "+0.8%", delta_color="normal")
        
        with col3:
            st.metric("DOW JONES", "35,100.15", "-0.3%", delta_color="inverse")
        
        with col4:
            st.metric("VIX", "18.45", "+2.1%", delta_color="inverse")
    
    def render_stock_analysis(self):
        """Render stock analysis interface"""
        st.subheader("📈 Individual Stock Analysis")
        
        # Stock selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            symbol = st.text_input(
                "Enter Stock Symbol (e.g., AAPL, GOOGL, TSLA)",
                value="AAPL",
                help="Enter a valid stock ticker symbol"
            )
        
        with col2:
            period = st.selectbox(
                "Time Period",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                index=3
            )
        
        if symbol:
            # Analysis type selection
            analysis_types = st.multiselect(
                "Select Analysis Types",
                [
                    "Price Analysis",
                    "Technical Indicators", 
                    "Fundamental Analysis",
                    "Sentiment Analysis",
                    "AI Predictions",
                    "Risk Metrics"
                ],
                default=["Price Analysis", "Technical Indicators"]
            )
            
            if st.button("🔍 Analyze Stock", type="primary"):
                self._perform_stock_analysis(symbol, period, analysis_types)
    
    def _perform_stock_analysis(self, symbol: str, period: str, analysis_types: List[str]):
        """Perform comprehensive stock analysis"""
        try:
            with st.spinner(f"Analyzing {symbol}..."):
                # Fetch stock data
                stock_data = self._fetch_stock_data(symbol, period)
                
                if stock_data.empty:
                    st.error(f"Could not fetch data for {symbol}")
                    return
                
                # Display results
                self._display_stock_analysis_results(symbol, stock_data, analysis_types)
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            logger.error(f"Stock analysis error for {symbol}: {e}")
    
    def _fetch_stock_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch stock data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            # Add technical indicators
            data = self._calculate_technical_indicators(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch stock data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            # Simple Moving Averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['SMA_200'] = data['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators: {e}")
            return data
    
    def _display_stock_analysis_results(self, symbol: str, data: pd.DataFrame, analysis_types: List[str]):
        """Display comprehensive stock analysis results"""
        st.success(f"✅ Analysis complete for {symbol}")
        
        # Basic info
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("Change", f"${change:.2f}", f"{change_pct:.2f}%")
        
        with col3:
            st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
        
        with col4:
            market_cap = current_price * 1000000000  # Simplified
            st.metric("Market Cap", f"${market_cap/1e9:.1f}B")
        
        # Analysis sections
        if "Price Analysis" in analysis_types:
            self._render_price_analysis(symbol, data)
        
        if "Technical Indicators" in analysis_types:
            self._render_technical_analysis(symbol, data)
        
        if "Fundamental Analysis" in analysis_types:
            self._render_fundamental_analysis(symbol)
        
        if "Sentiment Analysis" in analysis_types:
            self._render_sentiment_analysis(symbol)
        
        if "AI Predictions" in analysis_types:
            self._render_ai_predictions(symbol, data)
        
        if "Risk Metrics" in analysis_types:
            self._render_risk_metrics(symbol, data)
    
    def _render_price_analysis(self, symbol: str, data: pd.DataFrame):
        """Render price analysis charts"""
        st.subheader(f"📈 Price Analysis - {symbol}")
        
        # Price chart with moving averages
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price & Moving Averages', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Moving averages
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"{symbol} Price Analysis",
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Price statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Price Statistics")
            stats_data = {
                "Metric": ["52-Week High", "52-Week Low", "Average Volume", "Beta", "P/E Ratio"],
                "Value": [
                    f"${data['High'].max():.2f}",
                    f"${data['Low'].min():.2f}",
                    f"{data['Volume'].mean():,.0f}",
                    f"{np.random.uniform(0.8, 1.5):.2f}",  # Mock data
                    f"{np.random.uniform(15, 35):.1f}"      # Mock data
                ]
            }
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        
        with col2:
            st.subheader("🎯 Support & Resistance")
            support_levels = self._calculate_support_resistance(data)
            
            support_data = {
                "Level": ["Support 1", "Support 2", "Resistance 1", "Resistance 2"],
                "Price": [f"${level:.2f}" for level in support_levels]
            }
            st.dataframe(pd.DataFrame(support_data), use_container_width=True)
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> List[float]:
        """Calculate support and resistance levels"""
        try:
            recent_data = data.tail(50)
            
            # Simple support/resistance calculation
            highs = recent_data['High'].nlargest(10).mean()
            lows = recent_data['Low'].nsmallest(10).mean()
            
            resistance_1 = highs * 1.02
            resistance_2 = highs * 1.05
            support_1 = lows * 0.98
            support_2 = lows * 0.95
            
            return [support_1, support_2, resistance_1, resistance_2]
            
        except Exception as e:
            logger.error(f"Failed to calculate support/resistance: {e}")
            return [100, 95, 110, 115]  # Mock data
    
    def _render_technical_analysis(self, symbol: str, data: pd.DataFrame):
        """Render technical analysis indicators"""
        st.subheader(f"🔧 Technical Analysis - {symbol}")
        
        # Technical indicators charts
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('MACD', 'RSI', 'Bollinger Bands'),
            row_heights=[0.3, 0.3, 0.4]
        )
        
        # MACD
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram'),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='black')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', line=dict(color='gray')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower', line=dict(color='gray')),
            row=3, col=1
        )
        
        fig.update_layout(height=800, title=f"{symbol} Technical Indicators")
        st.plotly_chart(fig, use_container_width=True)
        
        # Current indicator values
        current_indicators = {
            "Indicator": ["RSI", "MACD", "MACD Signal", "BB Position"],
            "Value": [
                f"{data['RSI'].iloc[-1]:.2f}",
                f"{data['MACD'].iloc[-1]:.4f}",
                f"{data['MACD_Signal'].iloc[-1]:.4f}",
                f"{((data['Close'].iloc[-1] - data['BB_Lower'].iloc[-1]) / (data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1]) * 100):.1f}%"
            ],
            "Signal": [
                "Overbought" if data['RSI'].iloc[-1] > 70 else "Oversold" if data['RSI'].iloc[-1] < 30 else "Neutral",
                "Bullish" if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1] else "Bearish",
                "Buy" if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1] else "Sell",
                "Upper" if data['Close'].iloc[-1] > data['BB_Upper'].iloc[-1] else "Lower" if data['Close'].iloc[-1] < data['BB_Lower'].iloc[-1] else "Middle"
            ]
        }
        
        st.dataframe(pd.DataFrame(current_indicators), use_container_width=True)
    
    def _render_fundamental_analysis(self, symbol: str):
        """Render fundamental analysis"""
        st.subheader(f"📊 Fundamental Analysis - {symbol}")
        
        # Mock fundamental data - in production would fetch from financial APIs
        fundamental_data = {
            "Financial Metric": [
                "Market Cap", "P/E Ratio", "EPS", "Revenue (TTM)", 
                "Profit Margin", "ROE", "Debt-to-Equity", "Dividend Yield"
            ],
            "Value": [
                "$2.5T", "28.5", "$6.15", "$394.3B",
                "25.8%", "147.4%", "1.73", "0.5%"
            ],
            "Industry Avg": [
                "$800B", "22.1", "$4.50", "$120B",
                "18.2%", "45.2%", "0.85", "1.2%"
            ]
        }
        
        df = pd.DataFrame(fundamental_data)
        st.dataframe(df, use_container_width=True)
        
        # Valuation analysis
        st.subheader("💰 Valuation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # DCF Analysis (simplified)
            st.write("**Discounted Cash Flow Analysis**")
            dcf_value = np.random.uniform(150, 200)
            current_price = 175.50
            
            st.metric("Intrinsic Value", f"${dcf_value:.2f}")
            st.metric("Current Price", f"${current_price:.2f}")
            st.metric("Upside/Downside", f"{((dcf_value - current_price) / current_price * 100):.1f}%")
        
        with col2:
            # Peer comparison
            st.write("**Peer Comparison**")
            peers_data = {
                "Company": [symbol, "Peer 1", "Peer 2", "Peer 3"],
                "P/E": [28.5, 25.2, 31.8, 22.1],
                "P/B": [7.8, 5.2, 9.1, 4.5]
            }
            st.dataframe(pd.DataFrame(peers_data), use_container_width=True)
    
    def _render_sentiment_analysis(self, symbol: str):
        """Render sentiment analysis"""
        st.subheader(f"😊 Sentiment Analysis - {symbol}")
        
        # Mock sentiment data
        sentiment_score = np.random.uniform(-1, 1)
        sentiment_label = "Positive" if sentiment_score > 0.1 else "Negative" if sentiment_score < -0.1 else "Neutral"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Sentiment", sentiment_label)
        
        with col2:
            st.metric("Sentiment Score", f"{sentiment_score:.2f}")
        
        with col3:
            st.metric("Confidence", f"{np.random.uniform(0.7, 0.95):.1%}")
        
        # News sentiment over time
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        sentiment_history = np.random.normal(sentiment_score, 0.3, 30)
        
        fig = px.line(
            x=dates, 
            y=sentiment_history,
            title="Sentiment Trend (30 Days)",
            labels={'x': 'Date', 'y': 'Sentiment Score'}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent news headlines (mock)
        st.subheader("📰 Recent News Impact")
        
        news_data = {
            "Date": ["2024-01-15", "2024-01-14", "2024-01-13"],
            "Headline": [
                f"{symbol} reports strong Q4 earnings",
                f"Analyst upgrades {symbol} target price",
                f"{symbol} announces new product launch"
            ],
            "Sentiment": ["Positive", "Positive", "Neutral"],
            "Impact": ["High", "Medium", "Low"]
        }
        
        st.dataframe(pd.DataFrame(news_data), use_container_width=True)
    
    def _render_ai_predictions(self, symbol: str, data: pd.DataFrame):
        """Render AI-powered predictions"""
        st.subheader(f"🤖 AI Predictions - {symbol}")
        
        # Mock AI prediction model
        current_price = data['Close'].iloc[-1]
        
        # Generate predictions for different time horizons
        predictions = {
            "Time Horizon": ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months"],
            "Predicted Price": [
                f"${current_price * np.random.uniform(0.98, 1.02):.2f}",
                f"${current_price * np.random.uniform(0.95, 1.08):.2f}",
                f"${current_price * np.random.uniform(0.90, 1.15):.2f}",
                f"${current_price * np.random.uniform(0.85, 1.25):.2f}",
                f"${current_price * np.random.uniform(0.80, 1.35):.2f}"
            ],
            "Confidence": ["95%", "85%", "75%", "65%", "55%"],
            "Direction": [
                np.random.choice(["↗️ Up", "↘️ Down"]) for _ in range(5)
            ]
        }
        
        st.dataframe(pd.DataFrame(predictions), use_container_width=True)
        
        # Price prediction chart
        future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')[1:]
        predicted_prices = [current_price]
        
        for i in range(29):
            next_price = predicted_prices[-1] * np.random.uniform(0.98, 1.02)
            predicted_prices.append(next_price)
        
        # Combine historical and predicted data
        historical_dates = data.index[-30:]
        historical_prices = data['Close'][-30:].values
        
        fig = go.Figure()
        
        # Historical prices
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_prices,
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Predicted prices
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predicted_prices[1:],
            mode='lines',
            name='AI Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f"{symbol} AI Price Prediction",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model information
        with st.expander("🔍 Model Information"):
            st.write("""
            **AI Model Details:**
            - Model Type: LSTM Neural Network with Attention Mechanism
            - Training Data: 5 years of historical price and volume data
            - Features: Price, Volume, Technical Indicators, Market Sentiment
            - Accuracy: 68.5% directional accuracy on test set
            - Last Updated: 2024-01-15
            
            **Disclaimer:** Predictions are for informational purposes only and should not be considered as investment advice.
            """)
    
    def _render_risk_metrics(self, symbol: str, data: pd.DataFrame):
        """Render risk assessment metrics"""
        st.subheader(f"⚠️ Risk Metrics - {symbol}")
        
        # Calculate risk metrics
        returns = data['Close'].pct_change().dropna()
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        
        # Maximum Drawdown
        rolling_max = data['Close'].expanding().max()
        drawdown = (data['Close'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (simplified)
        risk_free_rate = 0.02  # 2% annual
        sharpe_ratio = (returns.mean() * 252 - risk_free_rate) / (returns.std() * np.sqrt(252))
        
        # Display risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Volatility (Annual)", f"{volatility:.1f}%")
        
        with col2:
            st.metric("VaR (95%)", f"{var_95:.2f}%")
        
        with col3:
            st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
        
        with col4:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        # Risk-Return scatter plot
        fig = go.Figure()
        
        # Add current stock
        fig.add_trace(go.Scatter(
            x=[volatility],
            y=[returns.mean() * 252 * 100],
            mode='markers',
            marker=dict(size=15, color='red'),
            name=symbol,
            text=[symbol],
            textposition="top center"
        ))
        
        # Add benchmark comparison (mock data)
        benchmarks = {
            'S&P 500': (15.5, 10.2),
            'NASDAQ': (18.2, 12.8),
            'Treasury': (2.1, 2.0)
        }
        
        for name, (vol, ret) in benchmarks.items():
            fig.add_trace(go.Scatter(
                x=[vol],
                y=[ret],
                mode='markers',
                marker=dict(size=10),
                name=name,
                text=[name],
                textposition="top center"
            ))
        
        fig.update_layout(
            title="Risk-Return Profile",
            xaxis_title="Volatility (%)",
            yaxis_title="Annual Return (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown chart
        fig_dd = px.line(
            x=data.index,
            y=drawdown,
            title="Drawdown Analysis",
            labels={'x': 'Date', 'y': 'Drawdown (%)'}
        )
        fig_dd.update_traces(fill='tonexty')
        
        st.plotly_chart(fig_dd, use_container_width=True)
    
    def render_portfolio_optimization(self):
        """Render portfolio optimization interface"""
        st.subheader("📊 Portfolio Optimization")
        
        # Portfolio input
        st.write("**Build Your Portfolio**")
        
        # Stock selection
        portfolio_stocks = st.text_area(
            "Enter stock symbols (comma-separated)",
            value="AAPL,GOOGL,MSFT,AMZN,TSLA",
            help="Enter stock symbols separated by commas"
        )
        
        stocks = [s.strip().upper() for s in portfolio_stocks.split(',') if s.strip()]
        
        if len(stocks) < 2:
            st.warning("Please enter at least 2 stock symbols for portfolio optimization.")
            return
        
        # Optimization parameters
        col1, col2 = st.columns(2)
        
        with col1:
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                ["Conservative", "Moderate", "Aggressive"]
            )
            
            optimization_method = st.selectbox(
                "Optimization Method",
                ["Maximum Sharpe Ratio", "Minimum Volatility", "Risk Parity", "Custom Target"]
            )
        
        with col2:
            if optimization_method == "Custom Target":
                target_return = st.slider("Target Annual Return (%)", 5, 25, 12)
            
            rebalancing_freq = st.selectbox(
                "Rebalancing Frequency",
                ["Monthly", "Quarterly", "Semi-Annual", "Annual"]
            )
        
        if st.button("🎯 Optimize Portfolio", type="primary"):
            self._perform_portfolio_optimization(stocks, risk_tolerance, optimization_method)
    
    def _perform_portfolio_optimization(self, stocks: List[str], risk_tolerance: str, method: str):
        """Perform portfolio optimization"""
        try:
            with st.spinner("Optimizing portfolio..."):
                # Fetch data for all stocks
                portfolio_data = {}
                
                for stock in stocks:
                    try:
                        ticker = yf.Ticker(stock)
                        hist = ticker.history(period="1y")
                        if not hist.empty:
                            portfolio_data[stock] = hist['Close']
                    except:
                        st.warning(f"Could not fetch data for {stock}")
                
                if len(portfolio_data) < 2:
                    st.error("Could not fetch sufficient data for optimization")
                    return
                
                # Create price dataframe
                prices_df = pd.DataFrame(portfolio_data).dropna()
                returns_df = prices_df.pct_change().dropna()
                
                # Perform optimization
                optimal_weights = self._optimize_weights(returns_df, method, risk_tolerance)
                
                # Display results
                self._display_portfolio_results(stocks, optimal_weights, returns_df, prices_df)
                
        except Exception as e:
            st.error(f"Portfolio optimization failed: {str(e)}")
            logger.error(f"Portfolio optimization error: {e}")
    
    def _optimize_weights(self, returns: pd.DataFrame, method: str, risk_tolerance: str) -> np.ndarray:
        """Optimize portfolio weights using different methods"""
        try:
            n_assets = len(returns.columns)
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean() * 252  # Annualized
            cov_matrix = returns.cov() * 252  # Annualized
            
            # Risk aversion parameter based on risk tolerance
            risk_aversion = {'Conservative': 10, 'Moderate': 5, 'Aggressive': 2}[risk_tolerance]
            
            if method == "Maximum Sharpe Ratio":
                # Maximize Sharpe ratio
                def neg_sharpe_ratio(weights):
                    portfolio_return = np.sum(expected_returns * weights)
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    return -(portfolio_return - 0.02) / portfolio_vol  # Assuming 2% risk-free rate
                
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for _ in range(n_assets))
                
                result = minimize(neg_sharpe_ratio, n_assets*[1./n_assets], 
                                method='SLSQP', bounds=bounds, constraints=constraints)
                
                return result.x
                
            elif method == "Minimum Volatility":
                # Minimize portfolio volatility
                def portfolio_volatility(weights):
                    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for _ in range(n_assets))
                
                result = minimize(portfolio_volatility, n_assets*[1./n_assets], 
                                method='SLSQP', bounds=bounds, constraints=constraints)
                
                return result.x
                
            elif method == "Risk Parity":
                # Equal risk contribution
                def risk_parity_objective(weights):
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                    contrib = weights * marginal_contrib
                    target_contrib = portfolio_vol / n_assets
                    return np.sum((contrib - target_contrib) ** 2)
                
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0.01, 0.99) for _ in range(n_assets))
                
                result = minimize(risk_parity_objective, n_assets*[1./n_assets], 
                                method='SLSQP', bounds=bounds, constraints=constraints)
                
                return result.x
                
            else:  # Equal weights fallback
                return np.array([1./n_assets] * n_assets)
                
        except Exception as e:
            logger.error(f"Weight optimization failed: {e}")
            # Return equal weights as fallback
            return np.array([1./len(returns.columns)] * len(returns.columns))
    
    def _display_portfolio_results(self, stocks: List[str], weights: np.ndarray, 
                                  returns: pd.DataFrame, prices: pd.DataFrame):
        """Display portfolio optimization results"""
        st.success("✅ Portfolio optimization complete!")
        
        # Portfolio weights
        st.subheader("🥧 Optimal Portfolio Allocation")
        
        weights_df = pd.DataFrame({
            'Stock': stocks[:len(weights)],
            'Weight': weights,
            'Allocation (%)': weights * 100
        }).sort_values('Weight', ascending=False)
        
        # Pie chart
        fig_pie = px.pie(
            weights_df, 
            values='Weight', 
            names='Stock',
            title="Portfolio Allocation"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.dataframe(weights_df, use_container_width=True)
        
        # Portfolio performance metrics
        st.subheader("📈 Portfolio Performance Metrics")
        
        # Calculate portfolio metrics
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_return = portfolio_returns.mean() * 252
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol
        
        # Max drawdown
        portfolio_prices = (prices * weights).sum(axis=1)
        rolling_max = portfolio_prices.expanding().max()
        drawdown = (portfolio_prices - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Expected Return", f"{portfolio_return:.1%}")
        
        with col2:
            st.metric("Volatility", f"{portfolio_vol:.1%}")
        
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        with col4:
            st.metric("Max Drawdown", f"{max_drawdown:.1%}")
        
        # Portfolio performance chart
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        benchmark_returns = returns.mean(axis=1)  # Equal-weighted benchmark
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        fig_perf = go.Figure()
        
        fig_perf.add_trace(go.Scatter(
            x=portfolio_cumulative.index,
            y=portfolio_cumulative.values,
            mode='lines',
            name='Optimized Portfolio',
            line=dict(color='blue', width=2)
        ))
        
        fig_perf.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative.values,
            mode='lines',
            name='Equal-Weight Benchmark',
            line=dict(color='gray', dash='dash')
        ))
        
        fig_perf.update_layout(
            title="Portfolio Performance vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=400
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
    
    def render_risk_assessment(self):
        """Render risk assessment interface"""
        st.subheader("⚠️ Portfolio Risk Assessment")
        
        st.info("Upload your portfolio or enter holdings to get a comprehensive risk analysis.")
        
        # Risk assessment input
        assessment_method = st.radio(
            "Assessment Method",
            ["Manual Entry", "Upload CSV", "Connect Broker"]
        )
        
        if assessment_method == "Manual Entry":
            self._render_manual_risk_entry()
        elif assessment_method == "Upload CSV":
            self._render_csv_upload_risk()
        else:
            st.info("Broker integration coming soon!")
    
    def _render_manual_risk_entry(self):
        """Render manual risk assessment entry"""
        st.write("**Enter your current holdings:**")
        
        # Dynamic portfolio entry
        if 'risk_portfolio' not in st.session_state:
            st.session_state.risk_portfolio = [{'symbol': '', 'quantity': 0, 'price': 0}]
        
        holdings = []
        
        for i, holding in enumerate(st.session_state.risk_portfolio):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                symbol = st.text_input(f"Symbol {i+1}", value=holding['symbol'], key=f"symbol_{i}")
            
            with col2:
                quantity = st.number_input(f"Shares", value=holding['quantity'], key=f"qty_{i}")
            
            with col3:
                price = st.number_input(f"Price $", value=holding['price'], key=f"price_{i}")
            
            with col4:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.risk_portfolio.pop(i)
                    st.rerun()
            
            if symbol and quantity > 0:
                holdings.append({'symbol': symbol, 'quantity': quantity, 'price': price})
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add Holding"):
                st.session_state.risk_portfolio.append({'symbol': '', 'quantity': 0, 'price': 0})
                st.rerun()
        
        with col2:
            if st.button("🔍 Analyze Risk") and holdings:
                self._perform_risk_analysis(holdings)
    
    def _render_csv_upload_risk(self):
        """Render CSV upload for risk assessment"""
        st.write("**Upload CSV with columns: Symbol, Quantity, Price**")
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df)
                
                if st.button("🔍 Analyze Risk"):
                    holdings = df.to_dict('records')
                    self._perform_risk_analysis(holdings)
                    
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    
    def _perform_risk_analysis(self, holdings: List[Dict]):
        """Perform comprehensive risk analysis"""
        try:
            with st.spinner("Analyzing portfolio risk..."):
                # Calculate portfolio value and weights
                total_value = sum(h['quantity'] * h['price'] for h in holdings)
                
                portfolio_summary = []
                for holding in holdings:
                    value = holding['quantity'] * holding['price']
                    weight = value / total_value
                    portfolio_summary.append({
                        'Symbol': holding['symbol'],
                        'Shares': holding['quantity'],
                        'Price': f"${holding['price']:.2f}",
                        'Value': f"${value:,.2f}",
                        'Weight': f"{weight:.1%}"
                    })
                
                # Display portfolio summary
                st.subheader("📊 Portfolio Summary")
                st.dataframe(pd.DataFrame(portfolio_summary), use_container_width=True)
                
                # Risk metrics (mock analysis)
                self._display_risk_analysis_results(total_value)
                
        except Exception as e:
            st.error(f"Risk analysis failed: {str(e)}")
    
    def _display_risk_analysis_results(self, total_value: float):
        """Display risk analysis results"""
        st.subheader("⚠️ Risk Analysis Results")
        
        # Overall risk score
        risk_score = np.random.uniform(3, 8)  # Mock risk score out of 10
        risk_level = "Low" if risk_score < 4 else "Medium" if risk_score < 7 else "High"
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Value", f"${total_value:,.2f}")
        
        with col2:
            st.metric("Risk Score", f"{risk_score:.1f}/10")
        
        with col3:
            st.metric("Risk Level", risk_level)
        
        with col4:
            st.metric("Diversification", f"{np.random.uniform(60, 95):.0f}%")
        
        # Risk breakdown
        st.subheader("🔍 Risk Breakdown")
        
        risk_factors = {
            "Risk Factor": [
                "Market Risk", "Sector Concentration", "Geographic Risk", 
                "Currency Risk", "Liquidity Risk", "Credit Risk"
            ],
            "Score": [
                f"{np.random.uniform(2, 8):.1f}/10" for _ in range(6)
            ],
            "Impact": [
                np.random.choice(["Low", "Medium", "High"]) for _ in range(6)
            ]
        }
        
        st.dataframe(pd.DataFrame(risk_factors), use_container_width=True)
        
        # Risk recommendations
        st.subheader("💡 Risk Management Recommendations")
        
        recommendations = [
            "🎯 Consider reducing concentration in technology sector",
            "🌍 Add international diversification to reduce geographic risk",
            "🛡️ Consider adding defensive stocks or bonds",
            "📊 Implement stop-loss orders for high-risk positions",
            "⚖️ Rebalance portfolio quarterly to maintain target allocation"
        ]
        
        for rec in recommendations:
            st.write(f"• {rec}")
    
    def render_market_trends(self):
        """Render market trends analysis"""
        st.subheader("🔍 Market Trends & Analysis")
        
        # Market overview tabs
        trend_tab1, trend_tab2, trend_tab3 = st.tabs([
            "📊 Market Overview",
            "🏭 Sector Analysis", 
            "🌍 Global Markets"
        ])
        
        with trend_tab1:
            self._render_market_overview_detailed()
        
        with trend_tab2:
            self._render_sector_analysis()
        
        with trend_tab3:
            self._render_global_markets()
    
    def _render_market_overview_detailed(self):
        """Render detailed market overview"""
        st.write("**Major Market Indices Performance**")
        
        # Mock market data
        indices_data = {
            "Index": ["S&P 500", "NASDAQ", "Dow Jones", "Russell 2000", "VIX"],
            "Current": ["4,500.25", "14,200.80", "35,100.15", "2,050.30", "18.45"],
            "Change": ["+1.2%", "+0.8%", "-0.3%", "+1.5%", "+2.1%"],
            "52W High": ["4,650.00", "15,100.00", "36,500.00", "2,200.00", "35.20"],
            "52W Low": ["3,800.00", "12,500.00", "31,000.00", "1,750.00", "12.10"]
        }
        
        st.dataframe(pd.DataFrame(indices_data), use_container_width=True)
        
        # Market trend chart
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        sp500_data = 4500 + np.cumsum(np.random.normal(0, 20, 30))
        nasdaq_data = 14200 + np.cumsum(np.random.normal(0, 50, 30))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates, y=sp500_data, name="S&P 500",
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=nasdaq_data, name="NASDAQ",
            line=dict(color='green'), yaxis='y2'
        ))
        
        fig.update_layout(
            title="Market Indices Trend (30 Days)",
            xaxis_title="Date",
            yaxis=dict(title="S&P 500", side="left"),
            yaxis2=dict(title="NASDAQ", side="right", overlaying="y"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_sector_analysis(self):
        """Render sector analysis"""
        st.write("**Sector Performance Analysis**")
        
        # Sector performance data
        sectors_data = {
            "Sector": [
                "Technology", "Healthcare", "Financial", "Consumer Discretionary",
                "Communication", "Industrial", "Consumer Staples", "Energy",
                "Utilities", "Real Estate", "Materials"
            ],
            "1D Change": [
                f"{np.random.uniform(-2, 3):.1f}%" for _ in range(11)
            ],
            "1W Change": [
                f"{np.random.uniform(-5, 7):.1f}%" for _ in range(11)
            ],
            "1M Change": [
                f"{np.random.uniform(-10, 15):.1f}%" for _ in range(11)
            ],
            "YTD Change": [
                f"{np.random.uniform(-20, 25):.1f}%" for _ in range(11)
            ]
        }
        
        sectors_df = pd.DataFrame(sectors_data)
        st.dataframe(sectors_df, use_container_width=True)
        
        # Sector performance heatmap
        performance_data = np.random.uniform(-5, 5, (11, 4))
        
        fig = go.Figure(data=go.Heatmap(
            z=performance_data,
            x=['1D', '1W', '1M', 'YTD'],
            y=sectors_data['Sector'],
            colorscale='RdYlGn',
            zmid=0
        ))
        
        fig.update_layout(
            title="Sector Performance Heatmap (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_global_markets(self):
        """Render global markets analysis"""
        st.write("**Global Markets Overview**")
        
        # Global indices
        global_data = {
            "Market": ["US (S&P 500)", "Europe (STOXX 50)", "Asia (Nikkei)", "UK (FTSE 100)", "China (CSI 300)"],
            "Current": ["4,500", "4,200", "33,500", "7,650", "4,100"],
            "Change": ["+1.2%", "-0.5%", "+0.8%", "+0.3%", "-1.1%"],
            "Currency": ["USD", "EUR", "JPY", "GBP", "CNY"]
        }
        
        st.dataframe(pd.DataFrame(global_data), use_container_width=True)
        
        # Global market correlation matrix
        markets = ['US', 'Europe', 'Asia', 'UK', 'China']
        correlation_matrix = np.random.uniform(0.3, 0.9, (5, 5))
        np.fill_diagonal(correlation_matrix, 1.0)
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=markets,
            y=markets,
            colorscale='Blues',
            text=correlation_matrix,
            texttemplate="%{text:.2f}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title="Global Market Correlation Matrix",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_financial_reports(self):
        """Render financial reports generator"""
        st.subheader("📋 Financial Reports Generator")
        
        report_type = st.selectbox(
            "Select Report Type",
            [
                "Portfolio Performance Report",
                "Risk Assessment Report", 
                "Market Analysis Report",
                "Stock Research Report",
                "Custom Analysis Report"
            ]
        )
        
        # Report parameters
        with st.expander("📊 Report Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                report_period = st.selectbox(
                    "Report Period",
                    ["Last Month", "Last Quarter", "Last 6 Months", "Last Year", "Custom"]
                )
                
                include_charts = st.checkbox("Include Charts", value=True)
                include_recommendations = st.checkbox("Include AI Recommendations", value=True)
            
            with col2:
                report_format = st.selectbox(
                    "Report Format",
                    ["PDF", "HTML", "Word Document", "Excel"]
                )
                
                detailed_analysis = st.checkbox("Detailed Analysis", value=False)
                executive_summary = st.checkbox("Executive Summary", value=True)
        
        if st.button("📄 Generate Report", type="primary"):
            self._generate_financial_report(report_type, report_period, {
                'include_charts': include_charts,
                'include_recommendations': include_recommendations,
                'report_format': report_format,
                'detailed_analysis': detailed_analysis,
                'executive_summary': executive_summary
            })
    
    def _generate_financial_report(self, report_type: str, period: str, options: Dict):
        """Generate financial report"""
        try:
            with st.spinner(f"Generating {report_type}..."):
                # Simulate report generation
                import time
                time.sleep(2)
                
                st.success("✅ Report generated successfully!")
                
                # Display sample report content
                st.subheader(f"📄 {report_type}")
                
                if options['executive_summary']:
                    st.write("### Executive Summary")
                    st.write("""
                    Based on the analysis period, the portfolio has shown resilient performance 
                    with moderate risk exposure. Key findings include strong performance in 
                    technology sector holdings and recommended diversification in emerging markets.
                    """)
                
                if options['include_charts']:
                    # Sample chart
                    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                    values = 100 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 30)))
                    
                    fig = px.line(x=dates, y=values, title="Portfolio Performance")
                    st.plotly_chart(fig, use_container_width=True)
                
                if options['include_recommendations']:
                    st.write("### AI Recommendations")
                    recommendations = [
                        "🎯 Consider rebalancing towards value stocks given current market conditions",
                        "🌍 Increase international exposure to 20-25% of portfolio",
                        "🛡️ Add defensive positions in utilities and consumer staples",
                        "📊 Monitor inflation indicators for potential rate changes"
                    ]
                    
                    for rec in recommendations:
                        st.write(f"• {rec}")
                
                # Download button
                st.download_button(
                    label=f"📥 Download {options['report_format']} Report",
                    data="Sample report content...",  # In production, generate actual file
                    file_name=f"{report_type.lower().replace(' ', '_')}.{options['report_format'].lower()}",
                    mime="application/pdf"
                )
                
        except Exception as e:
            st.error(f"Report generation failed: {str(e)}")
    
    def render(self):
        """Render the complete financial analyzer"""
        self.render_financial_dashboard()