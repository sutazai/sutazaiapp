"""
Sustainability Manager - Comprehensive sustainability tracking and carbon footprint management
"""

import time
import threading
import logging
import json
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import math

logger = logging.getLogger(__name__)

class CarbonIntensitySource(Enum):
    """Sources for carbon intensity data"""
    STATIC = "static"           # Static value
    ELECTRICITYMAP = "electricitymap"  # ElectricityMap API
    WATTTIME = "watttime"       # WattTime API
    CUSTOM_API = "custom_api"   # Custom API endpoint

class SustainabilityGoal(Enum):
    """Sustainability goals for the system"""
    CARBON_NEUTRAL = "carbon_neutral"
    NET_ZERO = "net_zero"
    FIFTY_PERCENT_REDUCTION = "fifty_percent_reduction"
    RENEWABLE_MATCHING = "renewable_matching"

@dataclass
class CarbonFootprint:
    """Carbon footprint metrics"""
    timestamp: datetime
    energy_consumed_kwh: float
    carbon_intensity_kg_per_kwh: float
    co2_emissions_kg: float
    renewable_percentage: float = 0.0
    grid_region: str = "unknown"

@dataclass
class SustainabilityMetrics:
    """Comprehensive sustainability metrics"""
    period_start: datetime
    period_end: datetime
    total_energy_kwh: float
    total_co2_kg: float
    avg_carbon_intensity: float
    peak_power_w: float
    avg_power_w: float
    renewable_energy_kwh: float
    efficiency_score: float
    sustainability_grade: str
    carbon_saved_kg: float = 0.0
    energy_saved_kwh: float = 0.0

@dataclass
class SustainabilityTarget:
    """Sustainability target definition"""
    name: str
    goal: SustainabilityGoal
    target_date: datetime
    baseline_co2_kg: float
    target_reduction_percent: float
    current_progress_percent: float = 0.0
    on_track: bool = False

class CarbonIntensityProvider:
    """Provides real-time carbon intensity data"""
    
    def __init__(self, source: CarbonIntensitySource = CarbonIntensitySource.STATIC, 
                 region: str = "US", api_key: Optional[str] = None):
        """
        Initialize carbon intensity provider
        
        Args:
            source: Source for carbon intensity data
            region: Grid region code
            api_key: API key for external services
        """
        self.source = source
        self.region = region
        self.api_key = api_key
        self._last_intensity = 0.4  # Default US grid average (kg CO2/kWh)
        self._last_update = datetime.now()
        self._cache_duration = timedelta(hours=1)  # Cache for 1 hour
    
    async def get_carbon_intensity(self) -> Tuple[float, float]:
        """
        Get current carbon intensity and renewable percentage
        
        Returns:
            Tuple of (carbon_intensity_kg_per_kwh, renewable_percentage)
        """
        try:
            if self.source == CarbonIntensitySource.STATIC:
                return self._get_static_intensity()
            elif self.source == CarbonIntensitySource.ELECTRICITYMAP:
                return await self._get_electricitymap_intensity()
            elif self.source == CarbonIntensitySource.WATTTIME:
                return await self._get_watttime_intensity()
            else:
                return self._get_static_intensity()
                
        except Exception as e:
            logger.error(f"Error getting carbon intensity: {e}")
            return self._last_intensity, 20.0  # Fallback values
    
    def _get_static_intensity(self) -> Tuple[float, float]:
        """Get static carbon intensity values"""
        # Time-based carbon intensity simulation (lower at night/weekends)
        now = datetime.now()
        hour = now.hour
        is_weekend = now.weekday() >= 5
        
        base_intensity = 0.4  # kg CO2/kWh
        renewable_base = 20.0  # 20% renewable
        
        # Lower intensity during off-peak hours
        if hour < 6 or hour > 22:
            base_intensity *= 0.8
            renewable_base += 10.0
        elif is_weekend:
            base_intensity *= 0.9
            renewable_base += 5.0
        
        # Add some seasonal variation
        month = now.month
        if month in [3, 4, 5, 9, 10]:  # Spring and fall - more renewables
            base_intensity *= 0.85
            renewable_base += 15.0
        
        return base_intensity, min(renewable_base, 60.0)
    
    async def _get_electricitymap_intensity(self) -> Tuple[float, float]:
        """Get carbon intensity from ElectricityMap API"""
        if datetime.now() - self._last_update < self._cache_duration:
            return self._last_intensity, 20.0
        
        try:
            url = f"https://api.electricitymap.org/v3/carbon-intensity/latest"
            params = {"zone": self.region}
            headers = {"auth-token": self.api_key} if self.api_key else {}
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                intensity = data.get("carbonIntensity", 400) / 1000  # Convert g to kg
                renewable_pct = data.get("fossilFreePercentage", 20.0)
                
                self._last_intensity = intensity
                self._last_update = datetime.now()
                
                return intensity, renewable_pct
            else:
                logger.warning(f"ElectricityMap API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching from ElectricityMap: {e}")
        
        return self._last_intensity, 20.0
    
    async def _get_watttime_intensity(self) -> Tuple[float, float]:
        """Get carbon intensity from WattTime API"""
        # WattTime API integration would go here
        # For now, return static values
        return self._get_static_intensity()

class SustainabilityManager:
    """Main sustainability management system"""
    
    def __init__(self, region: str = "US", carbon_intensity_source: CarbonIntensitySource = CarbonIntensitySource.STATIC):
        """
        Initialize sustainability manager
        
        Args:
            region: Grid region for carbon intensity
            carbon_intensity_source: Source for carbon intensity data
        """
        self.region = region
        self.carbon_provider = CarbonIntensityProvider(carbon_intensity_source, region)
        
        self._carbon_footprints: List[CarbonFootprint] = []
        self._sustainability_targets: List[SustainabilityTarget] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Baseline metrics for comparison
        self._baseline_metrics: Optional[SustainabilityMetrics] = None
        self._daily_budgets = {
            "energy_kwh": 24.0,      # 24 kWh daily budget
            "carbon_kg": 9.6,        # 9.6 kg CO2 daily budget (assuming 0.4 kg/kWh)
            "cost_usd": 3.60         # $3.60 daily cost budget (assuming $0.15/kWh)
        }
        
        # Current daily consumption
        self._daily_consumption = {
            "energy_kwh": 0.0,
            "carbon_kg": 0.0,
            "cost_usd": 0.0,
            "last_reset": datetime.now().date()
        }
        
        self._setup_default_targets()
    
    def _setup_default_targets(self) -> None:
        """Setup default sustainability targets"""
        current_date = datetime.now()
        
        # 50% reduction target by end of year
        self.add_target(SustainabilityTarget(
            name="Year-end 50% Reduction",
            goal=SustainabilityGoal.FIFTY_PERCENT_REDUCTION,
            target_date=datetime(current_date.year, 12, 31),
            baseline_co2_kg=3500.0,  # Estimated annual baseline
            target_reduction_percent=50.0
        ))
        
        # Carbon neutral target for next year
        self.add_target(SustainabilityTarget(
            name="Carbon Neutral 2026",
            goal=SustainabilityGoal.CARBON_NEUTRAL,
            target_date=datetime(current_date.year + 1, 12, 31),
            baseline_co2_kg=3500.0,
            target_reduction_percent=100.0
        ))
    
    def start_monitoring(self) -> None:
        """Start sustainability monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Sustainability monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop sustainability monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("Sustainability monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main sustainability monitoring loop"""
        while self._monitoring:
            try:
                self._update_carbon_footprint()
                self._update_daily_consumption()
                self._evaluate_targets()
                self._cleanup_old_data()
                
                time.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in sustainability monitoring: {e}")
                time.sleep(300)
    
    async def _update_carbon_footprint(self) -> None:
        """Update carbon footprint based on current energy consumption"""
        try:
            # Get current carbon intensity
            carbon_intensity, renewable_pct = await self.carbon_provider.get_carbon_intensity()
            
            # Get current energy consumption (this would integrate with energy profiler)
            # For now, we'll simulate based on system load
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Estimate power consumption based on system metrics
            estimated_power_w = 20.0 + (cpu_percent * 0.8) + (memory_percent * 0.3)
            
            # Calculate energy consumed in last 5 minutes
            energy_consumed_kwh = estimated_power_w * (5.0 / 60.0) / 1000.0  # 5 minutes in kWh
            
            # Calculate CO2 emissions
            co2_emissions_kg = energy_consumed_kwh * carbon_intensity
            
            # Create carbon footprint entry
            footprint = CarbonFootprint(
                timestamp=datetime.now(),
                energy_consumed_kwh=energy_consumed_kwh,
                carbon_intensity_kg_per_kwh=carbon_intensity,
                co2_emissions_kg=co2_emissions_kg,
                renewable_percentage=renewable_pct,
                grid_region=self.region
            )
            
            with self._lock:
                self._carbon_footprints.append(footprint)
                
                # Keep only last 24 hours of data
                cutoff_time = datetime.now() - timedelta(hours=24)
                self._carbon_footprints = [
                    fp for fp in self._carbon_footprints 
                    if fp.timestamp > cutoff_time
                ]
            
        except Exception as e:
            logger.error(f"Error updating carbon footprint: {e}")
    
    def _update_daily_consumption(self) -> None:
        """Update daily consumption tracking"""
        current_date = datetime.now().date()
        
        # Reset daily consumption if new day
        if current_date > self._daily_consumption["last_reset"]:
            self._daily_consumption.update({
                "energy_kwh": 0.0,
                "carbon_kg": 0.0,
                "cost_usd": 0.0,
                "last_reset": current_date
            })
        
        # Calculate today's consumption
        today_start = datetime.combine(current_date, datetime.min.time())
        today_footprints = [
            fp for fp in self._carbon_footprints 
            if fp.timestamp >= today_start
        ]
        
        if today_footprints:
            self._daily_consumption["energy_kwh"] = sum(fp.energy_consumed_kwh for fp in today_footprints)
            self._daily_consumption["carbon_kg"] = sum(fp.co2_emissions_kg for fp in today_footprints)
            self._daily_consumption["cost_usd"] = self._daily_consumption["energy_kwh"] * 0.15  # $0.15/kWh
    
    def _evaluate_targets(self) -> None:
        """Evaluate progress towards sustainability targets"""
        current_metrics = self.calculate_sustainability_metrics(hours_back=24 * 365)  # Annual data
        
        for target in self._sustainability_targets:
            if current_metrics.total_co2_kg > 0:
                # Calculate progress based on reduction from baseline
                reduction_achieved = max(0, target.baseline_co2_kg - current_metrics.total_co2_kg)
                target.current_progress_percent = (reduction_achieved / target.baseline_co2_kg) * 100
                
                # Check if on track
                days_remaining = (target.target_date - datetime.now()).days
                days_total = (target.target_date - datetime.now().replace(month=1, day=1)).days
                expected_progress = (1 - days_remaining / days_total) * target.target_reduction_percent
                
                target.on_track = target.current_progress_percent >= expected_progress * 0.9  # 90% of expected
    
    def _cleanup_old_data(self) -> None:
        """Clean up old carbon footprint data"""
        cutoff_time = datetime.now() - timedelta(days=30)  # Keep 30 days of data
        
        with self._lock:
            self._carbon_footprints = [
                fp for fp in self._carbon_footprints 
                if fp.timestamp > cutoff_time
            ]
    
    def calculate_sustainability_metrics(self, hours_back: float = 24.0) -> SustainabilityMetrics:
        """
        Calculate comprehensive sustainability metrics
        
        Args:
            hours_back: Hours of historical data to analyze
            
        Returns:
            SustainabilityMetrics: Calculated sustainability metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self._lock:
            relevant_footprints = [
                fp for fp in self._carbon_footprints 
                if fp.timestamp > cutoff_time
            ]
        
        if not relevant_footprints:
            return SustainabilityMetrics(
                period_start=cutoff_time,
                period_end=datetime.now(),
                total_energy_kwh=0.0,
                total_co2_kg=0.0,
                avg_carbon_intensity=0.4,
                peak_power_w=0.0,
                avg_power_w=0.0,
                renewable_energy_kwh=0.0,
                efficiency_score=0.0,
                sustainability_grade="F"
            )
        
        # Calculate metrics
        total_energy = sum(fp.energy_consumed_kwh for fp in relevant_footprints)
        total_co2 = sum(fp.co2_emissions_kg for fp in relevant_footprints)
        avg_carbon_intensity = sum(fp.carbon_intensity_kg_per_kwh for fp in relevant_footprints) / len(relevant_footprints)
        avg_renewable_pct = sum(fp.renewable_percentage for fp in relevant_footprints) / len(relevant_footprints)
        
        renewable_energy = total_energy * (avg_renewable_pct / 100.0)
        
        # Estimate power metrics (would be integrated with energy profiler in real implementation)
        power_estimates = [fp.energy_consumed_kwh * 12 * 1000 for fp in relevant_footprints]  # Rough power estimate
        peak_power = max(power_estimates) if power_estimates else 0.0
        avg_power = sum(power_estimates) / len(power_estimates) if power_estimates else 0.0
        
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(total_energy, total_co2, avg_renewable_pct)
        
        # Calculate sustainability grade
        sustainability_grade = self._calculate_sustainability_grade(efficiency_score, avg_renewable_pct, total_co2)
        
        # Calculate savings compared to baseline
        baseline_co2 = total_energy * 0.5  # Baseline of 0.5 kg CO2/kWh
        carbon_saved = max(0, baseline_co2 - total_co2)
        
        baseline_energy = total_energy * 1.2  # Assume 20% energy savings
        energy_saved = max(0, baseline_energy - total_energy)
        
        return SustainabilityMetrics(
            period_start=relevant_footprints[0].timestamp,
            period_end=relevant_footprints[-1].timestamp,
            total_energy_kwh=total_energy,
            total_co2_kg=total_co2,
            avg_carbon_intensity=avg_carbon_intensity,
            peak_power_w=peak_power,
            avg_power_w=avg_power,
            renewable_energy_kwh=renewable_energy,
            efficiency_score=efficiency_score,
            sustainability_grade=sustainability_grade,
            carbon_saved_kg=carbon_saved,
            energy_saved_kwh=energy_saved
        )
    
    def _calculate_efficiency_score(self, energy_kwh: float, co2_kg: float, renewable_pct: float) -> float:
        """Calculate efficiency score (0-100)"""
        if energy_kwh == 0:
            return 100.0
        
        # Base score on carbon intensity
        carbon_intensity = co2_kg / energy_kwh if energy_kwh > 0 else 0.5
        intensity_score = max(0, 100 - (carbon_intensity / 0.5) * 100)  # 0.5 kg/kWh as baseline
        
        # Bonus for renewable energy
        renewable_bonus = renewable_pct * 0.5  # Up to 50 points for 100% renewable
        
        # Combined score
        total_score = min(100, intensity_score + renewable_bonus)
        
        return total_score
    
    def _calculate_sustainability_grade(self, efficiency_score: float, renewable_pct: float, total_co2: float) -> str:
        """Calculate sustainability grade (A-F)"""
        # Weighted score considering multiple factors
        efficiency_weight = 0.4
        renewable_weight = 0.3
        absolute_emissions_weight = 0.3
        
        # Normalize absolute emissions (lower is better)
        emissions_score = max(0, 100 - (total_co2 * 10))  # Rough normalization
        
        composite_score = (
            efficiency_score * efficiency_weight +
            renewable_pct * renewable_weight +
            emissions_score * absolute_emissions_weight
        )
        
        if composite_score >= 90:
            return "A"
        elif composite_score >= 80:
            return "B"
        elif composite_score >= 70:
            return "C"
        elif composite_score >= 60:
            return "D"
        else:
            return "F"
    
    def add_target(self, target: SustainabilityTarget) -> None:
        """Add a sustainability target"""
        self._sustainability_targets.append(target)
        logger.info(f"Added sustainability target: {target.name}")
    
    def remove_target(self, target_name: str) -> bool:
        """Remove a sustainability target"""
        for i, target in enumerate(self._sustainability_targets):
            if target.name == target_name:
                del self._sustainability_targets[i]
                logger.info(f"Removed sustainability target: {target_name}")
                return True
        return False
    
    def get_daily_budget_status(self) -> Dict[str, Any]:
        """Get current daily budget status"""
        energy_utilization = self._daily_consumption["energy_kwh"] / self._daily_budgets["energy_kwh"]
        carbon_utilization = self._daily_consumption["carbon_kg"] / self._daily_budgets["carbon_kg"]
        cost_utilization = self._daily_consumption["cost_usd"] / self._daily_budgets["cost_usd"]
        
        return {
            "date": self._daily_consumption["last_reset"].isoformat(),
            "energy": {
                "consumed_kwh": self._daily_consumption["energy_kwh"],
                "budget_kwh": self._daily_budgets["energy_kwh"],
                "utilization_pct": energy_utilization * 100,
                "remaining_kwh": max(0, self._daily_budgets["energy_kwh"] - self._daily_consumption["energy_kwh"])
            },
            "carbon": {
                "emitted_kg": self._daily_consumption["carbon_kg"],
                "budget_kg": self._daily_budgets["carbon_kg"],
                "utilization_pct": carbon_utilization * 100,
                "remaining_kg": max(0, self._daily_budgets["carbon_kg"] - self._daily_consumption["carbon_kg"])
            },
            "cost": {
                "spent_usd": self._daily_consumption["cost_usd"],
                "budget_usd": self._daily_budgets["cost_usd"],
                "utilization_pct": cost_utilization * 100,
                "remaining_usd": max(0, self._daily_budgets["cost_usd"] - self._daily_consumption["cost_usd"])
            },
            "status": self._get_budget_status(max(energy_utilization, carbon_utilization, cost_utilization))
        }
    
    def _get_budget_status(self, max_utilization: float) -> str:
        """Get budget status based on utilization"""
        if max_utilization < 0.7:
            return "healthy"
        elif max_utilization < 0.9:
            return "warning"
        elif max_utilization < 1.0:
            return "critical"
        else:
            return "exceeded"
    
    def get_carbon_forecast(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Get carbon intensity forecast"""
        forecast = []
        current_time = datetime.now()
        
        for hour in range(hours_ahead):
            forecast_time = current_time + timedelta(hours=hour)
            
            # Simple forecast based on time patterns
            hour_of_day = forecast_time.hour
            is_weekend = forecast_time.weekday() >= 5
            
            base_intensity = 0.4
            if hour_of_day < 6 or hour_of_day > 22:
                base_intensity *= 0.8
            elif is_weekend:
                base_intensity *= 0.9
            
            forecast.append({
                "timestamp": forecast_time.isoformat(),
                "carbon_intensity_kg_per_kwh": base_intensity,
                "renewable_percentage": 20.0 + (10.0 if hour_of_day < 6 or hour_of_day > 22 else 0),
                "recommendation": "optimal" if base_intensity < 0.35 else "suboptimal"
            })
        
        return forecast
    
    def get_sustainability_recommendations(self) -> List[Dict[str, Any]]:
        """Get sustainability improvement recommendations"""
        recommendations = []
        
        # Analyze current metrics
        metrics = self.calculate_sustainability_metrics(24.0)
        budget_status = self.get_daily_budget_status()
        
        # Energy efficiency recommendations
        if metrics.efficiency_score < 70:
            recommendations.append({
                "category": "energy_efficiency",
                "priority": "high",
                "title": "Improve Energy Efficiency",
                "description": "Current efficiency score is below target. Consider enabling aggressive power optimization.",
                "estimated_savings_kwh": 2.0,
                "estimated_co2_reduction_kg": 0.8
            })
        
        # Carbon timing recommendations
        forecast = self.get_carbon_forecast(24)
        optimal_hours = [f for f in forecast if f["recommendation"] == "optimal"]
        if len(optimal_hours) > 0:
            recommendations.append({
                "category": "carbon_timing",
                "priority": "medium",
                "title": "Schedule Workloads During Low-Carbon Hours",
                "description": f"Schedule intensive tasks during {len(optimal_hours)} optimal hours today.",
                "estimated_savings_kwh": 0.5,
                "estimated_co2_reduction_kg": 0.3
            })
        
        # Budget recommendations
        if budget_status["status"] in ["warning", "critical", "exceeded"]:
            recommendations.append({
                "category": "budget_control",
                "priority": "high",
                "title": "Daily Budget Alert",
                "description": f"Daily budget utilization is {budget_status['status']}. Consider hibernating non-critical agents.",
                "estimated_savings_kwh": 1.0,
                "estimated_co2_reduction_kg": 0.4
            })
        
        # Renewable energy recommendations
        if metrics.renewable_energy_kwh / metrics.total_energy_kwh < 0.3:
            recommendations.append({
                "category": "renewable_energy",
                "priority": "medium",
                "title": "Increase Renewable Energy Usage",
                "description": "Current renewable energy usage is low. Consider timing-based optimizations.",
                "estimated_savings_kwh": 0.0,
                "estimated_co2_reduction_kg": 1.0
            })
        
        return recommendations
    
    def export_sustainability_report(self, filename: str, days_back: int = 30) -> None:
        """Export comprehensive sustainability report"""
        metrics = self.calculate_sustainability_metrics(days_back * 24.0)
        budget_status = self.get_daily_budget_status()
        recommendations = self.get_sustainability_recommendations()
        
        report_data = {
            "report_timestamp": datetime.now().isoformat(),
            "report_period_days": days_back,
            "sustainability_metrics": {
                "period_start": metrics.period_start.isoformat(),
                "period_end": metrics.period_end.isoformat(),
                "total_energy_kwh": metrics.total_energy_kwh,
                "total_co2_kg": metrics.total_co2_kg,
                "avg_carbon_intensity": metrics.avg_carbon_intensity,
                "renewable_energy_kwh": metrics.renewable_energy_kwh,
                "efficiency_score": metrics.efficiency_score,
                "sustainability_grade": metrics.sustainability_grade,
                "carbon_saved_kg": metrics.carbon_saved_kg,
                "energy_saved_kwh": metrics.energy_saved_kwh
            },
            "daily_budget_status": budget_status,
            "sustainability_targets": [
                {
                    "name": target.name,
                    "goal": target.goal.value,
                    "target_date": target.target_date.isoformat(),
                    "progress_percent": target.current_progress_percent,
                    "on_track": target.on_track
                }
                for target in self._sustainability_targets
            ],
            "recommendations": recommendations,
            "carbon_footprint_summary": {
                "total_measurements": len(self._carbon_footprints),
                "avg_carbon_intensity": sum(fp.carbon_intensity_kg_per_kwh for fp in self._carbon_footprints) / len(self._carbon_footprints) if self._carbon_footprints else 0,
                "avg_renewable_pct": sum(fp.renewable_percentage for fp in self._carbon_footprints) / len(self._carbon_footprints) if self._carbon_footprints else 0
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Sustainability report exported to {filename}")

# Global sustainability manager instance
_global_sustainability_manager: Optional[SustainabilityManager] = None

def get_global_sustainability_manager(
    region: str = "US", 
    carbon_source: CarbonIntensitySource = CarbonIntensitySource.STATIC
) -> SustainabilityManager:
    """Get or create global sustainability manager instance"""
    global _global_sustainability_manager
    if _global_sustainability_manager is None:
        _global_sustainability_manager = SustainabilityManager(region, carbon_source)
    return _global_sustainability_manager