#!/usr/bin/env python3
"""
FinRobot Wrapper - Financial Analysis Agent
"""

import os
import sys
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_agent_wrapper import BaseAgentWrapper, ChatRequest

class FinRobotLocal(BaseAgentWrapper):
    """FinRobot financial analysis wrapper"""
    
    def __init__(self):
        super().__init__(
            agent_name="FinRobot",
            agent_description="Financial analysis and market intelligence",
            port=8000
        )
        self.setup_finrobot_routes()
    
    def setup_finrobot_routes(self):
        """Setup FinRobot routes"""
        
        @self.app.post("/analyze/market")
        async def analyze_market(request: Dict[str, Any]):
            """Analyze market conditions"""
            try:
                market = request.get("market", "stocks")
                timeframe = request.get("timeframe", "1d")
                
                analysis_prompt = f"""Analyze {market} market conditions for {timeframe}:
                Provide insights on:
                1. Market trends
                2. Key indicators
                3. Risk factors
                4. Opportunities"""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are FinRobot, a financial analyst."},
                        {"role": "user", "content": analysis_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                analysis = response.choices[0]["message"]["content"]
                
                return {"success": True, "analysis": analysis}
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.post("/analyze/portfolio")
        async def analyze_portfolio(request: Dict[str, Any]):
            """Analyze investment portfolio"""
            try:
                portfolio = request.get("portfolio", [])
                risk_tolerance = request.get("risk_tolerance", "moderate")
                
                portfolio_prompt = f"""Analyze this portfolio:
                Holdings: {portfolio}
                Risk tolerance: {risk_tolerance}
                
                Provide recommendations for optimization."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are FinRobot analyzing portfolios."},
                        {"role": "user", "content": portfolio_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                recommendations = response.choices[0]["message"]["content"]
                
                return {"success": True, "recommendations": recommendations}
                
            except Exception as e:
                return {"success": False, "error": str(e)}

def main():
    agent = FinRobotLocal()
    agent.run()

if __name__ == "__main__":
    main()