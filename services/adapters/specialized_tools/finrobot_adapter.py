"""
FinRobot adapter for financial analysis and trading operations
"""
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime, timedelta
from ..base_adapter import ServiceAdapter
import logging

logger = logging.getLogger(__name__)


class FinRobotAdapter(ServiceAdapter):
    """Adapter for FinRobot financial analysis system"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("FinRobot", config)
        self.data_sources = config.get('data_sources', [])
        self.analysis_modules = config.get('analysis_modules', [])
        self.strategies = {}
        self.portfolios = {}
        
    async def initialize(self):
        """Initialize FinRobot connection"""
        try:
            await self.connect()
            
            # Initialize data sources
            for source in self.data_sources:
                await self._initialize_data_source(source)
                
            logger.info(f"FinRobot initialized with {len(self.data_sources)} data sources")
            
        except Exception as e:
            logger.error(f"Failed to initialize FinRobot: {str(e)}")
            raise
            
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get FinRobot capabilities"""
        return {
            'service': 'FinRobot',
            'type': 'specialized_tool',
            'features': [
                'market_data_analysis',
                'technical_indicators',
                'sentiment_analysis',
                'portfolio_optimization',
                'risk_management',
                'automated_trading',
                'backtesting',
                'real_time_alerts'
            ],
            'data_sources': self.data_sources,
            'analysis_modules': self.analysis_modules,
            'supported_markets': [
                'stocks',
                'forex',
                'crypto',
                'commodities',
                'options'
            ],
            'active_strategies': len(self.strategies)
        }
        
    async def get_market_data(self,
                            symbol: str,
                            interval: str = '1d',
                            period: str = '1mo',
                            source: Optional[str] = None) -> Dict[str, Any]:
        """Get market data for a symbol"""
        try:
            payload = {
                'symbol': symbol,
                'interval': interval,
                'period': period,
                'source': source or self.data_sources[0]
            }
            
            response = await self._make_request(
                'GET',
                '/api/v1/market/data',
                params=payload
            )
            
            if response:
                return {
                    'success': True,
                    'symbol': symbol,
                    'data': response.get('data', []),
                    'metadata': response.get('metadata', {})
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to get market data'
                }
                
        except Exception as e:
            logger.error(f"Failed to get market data: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def analyze_technical(self,
                              symbol: str,
                              indicators: List[str],
                              timeframe: str = '1d') -> Dict[str, Any]:
        """Perform technical analysis"""
        try:
            payload = {
                'symbol': symbol,
                'indicators': indicators,
                'timeframe': timeframe
            }
            
            response = await self._make_request(
                'POST',
                '/api/v1/analysis/technical',
                json=payload
            )
            
            if response:
                return {
                    'success': True,
                    'symbol': symbol,
                    'indicators': response.get('indicators', {}),
                    'signals': response.get('signals', []),
                    'recommendation': response.get('recommendation', 'neutral')
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to perform technical analysis'
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze technical: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def analyze_sentiment(self,
                              query: str,
                              sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze market sentiment"""
        try:
            payload = {
                'query': query,
                'sources': sources or ['news', 'social_media', 'forums']
            }
            
            response = await self._make_request(
                'POST',
                '/api/v1/analysis/sentiment',
                json=payload
            )
            
            if response:
                return {
                    'success': True,
                    'sentiment_score': response.get('sentiment_score', 0),
                    'sentiment_label': response.get('sentiment_label', 'neutral'),
                    'confidence': response.get('confidence', 0),
                    'sources_analyzed': response.get('sources_analyzed', []),
                    'key_topics': response.get('key_topics', [])
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to analyze sentiment'
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def create_portfolio(self,
                             name: str,
                             initial_capital: float,
                             positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a new portfolio"""
        try:
            portfolio_data = {
                'name': name,
                'initial_capital': initial_capital,
                'positions': positions,
                'created_at': datetime.utcnow().isoformat()
            }
            
            response = await self._make_request(
                'POST',
                '/api/v1/portfolios',
                json=portfolio_data
            )
            
            if response and 'id' in response:
                portfolio_id = response['id']
                self.portfolios[portfolio_id] = portfolio_data
                
                return {
                    'success': True,
                    'portfolio_id': portfolio_id,
                    'value': response.get('current_value', initial_capital)
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to create portfolio'
                }
                
        except Exception as e:
            logger.error(f"Failed to create portfolio: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def optimize_portfolio(self,
                               portfolio_id: str,
                               optimization_method: str = 'sharpe',
                               constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """Optimize portfolio allocation"""
        try:
            payload = {
                'portfolio_id': portfolio_id,
                'method': optimization_method,
                'constraints': constraints or {
                    'min_weight': 0.05,
                    'max_weight': 0.30
                }
            }
            
            response = await self._make_request(
                'POST',
                f'/api/v1/portfolios/{portfolio_id}/optimize',
                json=payload
            )
            
            if response:
                return {
                    'success': True,
                    'optimal_weights': response.get('optimal_weights', {}),
                    'expected_return': response.get('expected_return', 0),
                    'risk': response.get('risk', 0),
                    'sharpe_ratio': response.get('sharpe_ratio', 0)
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to optimize portfolio'
                }
                
        except Exception as e:
            logger.error(f"Failed to optimize portfolio: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def create_strategy(self,
                            name: str,
                            strategy_type: str,
                            parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a trading strategy"""
        try:
            strategy_data = {
                'name': name,
                'type': strategy_type,
                'parameters': parameters,
                'status': 'created'
            }
            
            response = await self._make_request(
                'POST',
                '/api/v1/strategies',
                json=strategy_data
            )
            
            if response and 'id' in response:
                strategy_id = response['id']
                self.strategies[strategy_id] = strategy_data
                
                return {
                    'success': True,
                    'strategy_id': strategy_id
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to create strategy'
                }
                
        except Exception as e:
            logger.error(f"Failed to create strategy: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def backtest_strategy(self,
                              strategy_id: str,
                              start_date: str,
                              end_date: str,
                              initial_capital: float = 10000) -> Dict[str, Any]:
        """Backtest a trading strategy"""
        try:
            payload = {
                'strategy_id': strategy_id,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital
            }
            
            response = await self._make_request(
                'POST',
                f'/api/v1/strategies/{strategy_id}/backtest',
                json=payload
            )
            
            if response:
                return {
                    'success': True,
                    'performance': response.get('performance', {}),
                    'trades': response.get('trades', []),
                    'metrics': response.get('metrics', {})
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to backtest strategy'
                }
                
        except Exception as e:
            logger.error(f"Failed to backtest strategy: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_risk_metrics(self, portfolio_id: str) -> Dict[str, Any]:
        """Calculate risk metrics for portfolio"""
        try:
            response = await self._make_request(
                'GET',
                f'/api/v1/portfolios/{portfolio_id}/risk'
            )
            
            if response:
                return {
                    'success': True,
                    'var': response.get('value_at_risk', {}),
                    'cvar': response.get('conditional_var', {}),
                    'beta': response.get('beta', 0),
                    'volatility': response.get('volatility', 0),
                    'max_drawdown': response.get('max_drawdown', 0)
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to get risk metrics'
                }
                
        except Exception as e:
            logger.error(f"Failed to get risk metrics: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def _initialize_data_source(self, source: str):
        """Initialize a data source connection"""
        try:
            await self._make_request(
                'POST',
                f'/api/v1/data-sources/{source}/connect'
            )
            logger.info(f"Data source {source} initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize data source {source}: {str(e)}")