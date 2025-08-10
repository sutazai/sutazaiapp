#!/usr/bin/env python3
"""
FinRobot Financial Analysis Service for SutazAI
Advanced financial data analysis and AI-powered insights
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Financial data providers
import yfinance as yf
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData

# Technical analysis
import ta
from ta.utils import dropna

# Machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI FinRobot Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:10011",  # Frontend Streamlit UI
        "http://localhost:10010",  # Backend API
        "http://127.0.0.1:10011",  # Alternative localhost
        "http://127.0.0.1:10010",  # Alternative localhost
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StockRequest(BaseModel):
    symbol: str
    period: str = "1y"  # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

class AnalysisRequest(BaseModel):
    symbol: str
    analysis_type: str  # technical, fundamental, sentiment, prediction
    parameters: Optional[Dict[str, Any]] = {}

class PortfolioRequest(BaseModel):
    symbols: List[str]
    weights: Optional[List[float]] = None
    investment_amount: float = 10000

class FinancialAnalyzer:
    def __init__(self):
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", "demo")  # Replace with actual API key
        self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        self.fd = FundamentalData(key=self.alpha_vantage_key, output_format='pandas')
        self.cache = {}

    async def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get stock data from Yahoo Finance"""
        try:
            cache_key = f"{symbol}_{period}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if datetime.now() - timestamp < timedelta(minutes=15):  # 15-minute cache
                    return cached_data

            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
            
            # Cache the data
            self.cache[cache_key] = (data, datetime.now())
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")

    async def technical_analysis(self, symbol: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform technical analysis on stock data"""
        try:
            data = await self.get_stock_data(symbol, parameters.get("period", "1y"))
            
            # Add technical indicators
            data = dropna(data)
            
            # Moving Averages
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['EMA_12'] = ta.trend.ema_indicator(data['Close'], window=12)
            data['EMA_26'] = ta.trend.ema_indicator(data['Close'], window=26)
            
            # MACD
            data['MACD'] = ta.trend.macd(data['Close'])
            data['MACD_signal'] = ta.trend.macd_signal(data['Close'])
            data['MACD_histogram'] = ta.trend.macd_diff(data['Close'])
            
            # RSI
            data['RSI'] = ta.momentum.rsi(data['Close'])
            
            # Bollinger Bands
            data['BB_upper'] = ta.volatility.bollinger_hband(data['Close'])
            data['BB_middle'] = ta.volatility.bollinger_mavg(data['Close'])
            data['BB_lower'] = ta.volatility.bollinger_lband(data['Close'])
            
            # Volume indicators
            data['Volume_SMA'] = ta.volume.volume_sma(data['Close'], data['Volume'])
            data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
            
            # Support and Resistance levels
            recent_data = data.tail(50)
            resistance = recent_data['High'].max()
            support = recent_data['Low'].min()
            
            current_price = data['Close'].iloc[-1]
            
            # Generate signals
            signals = self._generate_technical_signals(data)
            
            return {
                "symbol": symbol,
                "current_price": float(current_price),
                "support_level": float(support),
                "resistance_level": float(resistance),
                "rsi_current": float(data['RSI'].iloc[-1]),
                "macd_current": float(data['MACD'].iloc[-1]),
                "macd_signal": float(data['MACD_signal'].iloc[-1]),
                "sma_20": float(data['SMA_20'].iloc[-1]),
                "sma_50": float(data['SMA_50'].iloc[-1]),
                "signals": signals,
                "analysis_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Technical analysis failed: {str(e)}")

    def _generate_technical_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """Generate trading signals based on technical indicators"""
        signals = {}
        
        current_rsi = data['RSI'].iloc[-1]
        current_macd = data['MACD'].iloc[-1]
        current_macd_signal = data['MACD_signal'].iloc[-1]
        current_price = data['Close'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        
        # RSI signals
        if current_rsi > 70:
            signals['rsi'] = 'overbought'
        elif current_rsi < 30:
            signals['rsi'] = 'oversold'
        else:
            signals['rsi'] = 'neutral'
        
        # MACD signals
        if current_macd > current_macd_signal:
            signals['macd'] = 'bullish'
        else:
            signals['macd'] = 'bearish'
        
        # Moving Average signals
        if current_price > sma_20 > sma_50:
            signals['trend'] = 'bullish'
        elif current_price < sma_20 < sma_50:
            signals['trend'] = 'bearish'
        else:
            signals['trend'] = 'neutral'
        
        return signals

    async def fundamental_analysis(self, symbol: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform fundamental analysis"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Financial ratios
            pe_ratio = info.get('trailingPE', 'N/A')
            peg_ratio = info.get('pegRatio', 'N/A')
            price_to_book = info.get('priceToBook', 'N/A')
            debt_to_equity = info.get('debtToEquity', 'N/A')
            roe = info.get('returnOnEquity', 'N/A')
            roa = info.get('returnOnAssets', 'N/A')
            
            # Revenue and earnings
            revenue = info.get('totalRevenue', 'N/A')
            net_income = info.get('netIncomeToCommon', 'N/A')
            earnings_growth = info.get('earningsGrowth', 'N/A')
            revenue_growth = info.get('revenueGrowth', 'N/A')
            
            # Dividend information
            dividend_yield = info.get('dividendYield', 'N/A')
            dividend_rate = info.get('dividendRate', 'N/A')
            
            return {
                "symbol": symbol,
                "company_name": info.get('longName', 'N/A'),
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "market_cap": info.get('marketCap', 'N/A'),
                "enterprise_value": info.get('enterpriseValue', 'N/A'),
                "pe_ratio": pe_ratio,
                "peg_ratio": peg_ratio,
                "price_to_book": price_to_book,
                "debt_to_equity": debt_to_equity,
                "return_on_equity": roe,
                "return_on_assets": roa,
                "revenue": revenue,
                "net_income": net_income,
                "earnings_growth": earnings_growth,
                "revenue_growth": revenue_growth,
                "dividend_yield": dividend_yield,
                "dividend_rate": dividend_rate,
                "analysis_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fundamental analysis failed for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Fundamental analysis failed: {str(e)}")

    async def predict_price(self, symbol: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future price using machine learning"""
        try:
            period = parameters.get("period", "2y")
            forecast_days = parameters.get("forecast_days", 30)
            
            data = await self.get_stock_data(symbol, period)
            
            # Prepare features
            data['Returns'] = data['Close'].pct_change()
            data['SMA_5'] = ta.trend.sma_indicator(data['Close'], window=5)
            data['SMA_10'] = ta.trend.sma_indicator(data['Close'], window=10)
            data['RSI'] = ta.momentum.rsi(data['Close'])
            data['MACD'] = ta.trend.macd(data['Close'])
            
            # Create target variable (next day's price)
            data['Target'] = data['Close'].shift(-1)
            
            # Remove NaN values
            data = data.dropna()
            
            # Select features
            features = ['Open', 'High', 'Low', 'Volume', 'Returns', 'SMA_5', 'SMA_10', 'RSI', 'MACD']
            X = data[features]
            y = data['Target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            last_features = X.iloc[-1:].values
            predicted_price = model.predict(last_features)[0]
            
            # Calculate confidence based on model performance
            score = model.score(X_test, y_test)
            
            return {
                "symbol": symbol,
                "current_price": float(data['Close'].iloc[-1]),
                "predicted_price": float(predicted_price),
                "forecast_days": forecast_days,
                "confidence_score": float(score),
                "prediction_date": datetime.now().isoformat(),
                "model_type": "Random Forest"
            }
            
        except Exception as e:
            logger.error(f"Price prediction failed for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Price prediction failed: {str(e)}")

    async def portfolio_analysis(self, symbols: List[str], weights: Optional[List[float]] = None, 
                               investment_amount: float = 10000) -> Dict[str, Any]:
        """Analyze portfolio performance and optimization"""
        try:
            if weights is None:
                weights = [1/len(symbols)] * len(symbols)
            
            if len(weights) != len(symbols):
                raise HTTPException(status_code=400, detail="Weights must match number of symbols")
            
            # Get data for all symbols
            portfolio_data = {}
            for symbol in symbols:
                try:
                    data = await self.get_stock_data(symbol, "1y")
                    portfolio_data[symbol] = data['Close']
                except:
                    logger.warning(f"Could not fetch data for {symbol}")
                    continue
            
            if not portfolio_data:
                raise HTTPException(status_code=404, detail="No valid symbols found")
            
            # Create portfolio DataFrame
            portfolio_df = pd.DataFrame(portfolio_data)
            
            # Calculate returns
            returns = portfolio_df.pct_change().dropna()
            
            # Portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Portfolio metrics
            annual_return = portfolio_returns.mean() * 252
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Calculate portfolio value
            portfolio_value = investment_amount * (1 + portfolio_returns).cumprod().iloc[-1]
            total_return = (portfolio_value - investment_amount) / investment_amount
            
            # Individual stock allocation
            current_prices = {symbol: data.iloc[-1] for symbol, data in portfolio_data.items()}
            allocations = {}
            for i, symbol in enumerate(symbols[:len(weights)]):
                if symbol in current_prices:
                    allocations[symbol] = {
                        "weight": weights[i],
                        "value": investment_amount * weights[i],
                        "current_price": float(current_prices[symbol])
                    }
            
            return {
                "portfolio_symbols": symbols,
                "weights": weights,
                "investment_amount": investment_amount,
                "current_value": float(portfolio_value),
                "total_return": float(total_return),
                "annual_return": float(annual_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "allocations": allocations,
                "analysis_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Portfolio analysis failed: {str(e)}")

# Initialize financial analyzer
fin_analyzer = FinancialAnalyzer()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": ["stock_data", "technical_analysis", "fundamental_analysis", "prediction", "portfolio"],
        "data_providers": ["yahoo_finance", "alpha_vantage"]
    }

@app.get("/stock/{symbol}")
async def get_stock_data(symbol: str, period: str = "1y"):
    """Get basic stock data"""
    try:
        data = await fin_analyzer.get_stock_data(symbol.upper(), period)
        
        return {
            "symbol": symbol.upper(),
            "period": period,
            "current_price": float(data['Close'].iloc[-1]),
            "open": float(data['Open'].iloc[-1]),
            "high": float(data['High'].iloc[-1]),
            "low": float(data['Low'].iloc[-1]),
            "volume": int(data['Volume'].iloc[-1]),
            "data_points": len(data),
            "date_range": {
                "start": data.index[0].isoformat(),
                "end": data.index[-1].isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Stock data request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_stock(request: AnalysisRequest):
    """Perform stock analysis"""
    
    symbol = request.symbol.upper()
    analysis_type = request.analysis_type.lower()
    
    if analysis_type == "technical":
        return await fin_analyzer.technical_analysis(symbol, request.parameters)
    elif analysis_type == "fundamental":
        return await fin_analyzer.fundamental_analysis(symbol, request.parameters)
    elif analysis_type == "prediction":
        return await fin_analyzer.predict_price(symbol, request.parameters)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown analysis type: {analysis_type}")

@app.post("/portfolio")
async def analyze_portfolio(request: PortfolioRequest):
    """Analyze portfolio"""
    symbols = [s.upper() for s in request.symbols]
    return await fin_analyzer.portfolio_analysis(symbols, request.weights, request.investment_amount)

@app.get("/market/trending")
async def get_trending_stocks():
    """Get trending stocks"""
    # This would typically use a market data API
    # For demo purposes, return popular stocks
    trending = [
        {"symbol": "AAPL", "name": "Apple Inc.", "change": "+2.3%"},
        {"symbol": "GOOGL", "name": "Alphabet Inc.", "change": "+1.8%"},
        {"symbol": "MSFT", "name": "Microsoft Corp.", "change": "+1.2%"},
        {"symbol": "TSLA", "name": "Tesla Inc.", "change": "-0.5%"},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "change": "+0.8%"}
    ]
    
    return {
        "trending_stocks": trending,
        "last_updated": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)