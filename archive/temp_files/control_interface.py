"""
SutazAI Control Interface
Web interface for the chaos-to-value conversion system
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Configure Streamlit
st.set_page_config(
    page_title="SutazAI Control System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API base URL
API_BASE = "http://localhost:8000"

def make_api_request(endpoint, method="GET", data=None):
    """Make API request to backend"""
    try:
        url = f"{API_BASE}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("🎯 SutazAI Control System")
    st.markdown("**Chaos to Value Conversion • Silent Intelligence • Pattern Extraction**")
    
    # Sidebar controls
    st.sidebar.header("🛠️ System Controls")
    
    # System status
    status_response = make_api_request("/system/status")
    if "error" not in status_response:
        system_running = status_response.get("system_running", False)
        if system_running:
            st.sidebar.success("🟢 System Online")
        else:
            st.sidebar.warning("🟡 System Stopped")
    else:
        st.sidebar.error("🔴 Backend Offline")
        st.error("Cannot connect to backend. Make sure the control system is running.")
        st.code("./launch_control_system.sh")
        return
    
    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Start"):
            response = make_api_request("/system/start", "POST")
            if "error" not in response:
                st.sidebar.success("System started!")
            else:
                st.sidebar.error(f"Error: {response['error']}")
    
    with col2:
        if st.button("⏹️ Stop"):
            response = make_api_request("/system/stop", "POST")
            if "error" not in response:
                st.sidebar.info("System stopped!")
            else:
                st.sidebar.error(f"Error: {response['error']}")
    
    # Main dashboard
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🧠 Intelligence", 
        "💰 Value Extraction", 
        "🎯 Opportunities", 
        "⚙️ Configuration"
    ])
    
    with tab1:
        show_dashboard(status_response)
    
    with tab2:
        show_intelligence()
    
    with tab3:
        show_value_extraction()
    
    with tab4:
        show_opportunities()
    
    with tab5:
        show_configuration()

def show_dashboard(status_response):
    """Show main dashboard"""
    st.header("📊 System Dashboard")
    
    if "error" in status_response:
        st.error("Unable to load system status")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_packets = (
            status_response.get("chaos_engine", {}).get("total_packets", 0) +
            status_response.get("silent_operator", {}).get("recent_packets", 0)
        )
        st.metric("Intelligence Packets", total_packets)
    
    with col2:
        total_value = status_response.get("total_value_extracted", 0)
        st.metric("Total Value Extracted", f"{total_value:.2f}")
    
    with col3:
        intel_level = status_response.get("silent_operator", {}).get("intelligence_level", 1.0)
        st.metric("Intelligence Level", f"{intel_level:.1f}")
    
    with col4:
        stealth_level = status_response.get("silent_operator", {}).get("stealth_level", 0.0)
        stealth_status = "🔇 Silent" if stealth_level < 0.1 else "👁️ Visible"
        st.metric("Stealth Status", stealth_status)
    
    # System overview
    st.subheader("🔧 System Components")
    
    chaos_data = status_response.get("chaos_engine", {})
    silent_data = status_response.get("silent_operator", {})
    value_data = status_response.get("value_extractor", {})
    
    components_data = {
        "Component": ["Chaos Engine", "Silent Operator", "Value Extractor"],
        "Status": ["🟢 Active", "🔇 Silent", "💰 Processing"],
        "Packets": [chaos_data.get("total_packets", 0), silent_data.get("recent_packets", 0), value_data.get("total_extractions", 0)],
        "Value Score": [chaos_data.get("avg_value_score", 0), silent_data.get("total_value_accumulated", 0), value_data.get("total_value_score", 0)]
    }
    
    df = pd.DataFrame(components_data)
    st.dataframe(df, use_container_width=True)
    
    # Performance chart
    st.subheader("📈 Performance Metrics")
    
    # Create sample performance data
    hours = list(range(24))
    packets_processed = [max(0, 50 + 30 * np.sin(h/4) + np.random.normal(0, 10)) for h in hours]
    value_extracted = [max(0, 10 + 5 * np.sin(h/3) + np.random.normal(0, 2)) for h in hours]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=packets_processed, mode='lines+markers', name='Packets Processed', yaxis='y'))
    fig.add_trace(go.Scatter(x=hours, y=value_extracted, mode='lines+markers', name='Value Extracted', yaxis='y2'))
    
    fig.update_layout(
        title="24-Hour Performance",
        xaxis_title="Hour",
        yaxis=dict(title="Packets", side="left"),
        yaxis2=dict(title="Value", side="right", overlaying="y"),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_intelligence():
    """Show intelligence summary"""
    st.header("🧠 Intelligence Summary")
    
    intel_response = make_api_request("/intelligence/summary")
    
    if "error" in intel_response:
        st.error("Unable to load intelligence data")
        return
    
    # Intelligence sources
    st.subheader("📡 Intelligence Sources")
    
    sources = intel_response.get("intelligence_sources", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Chaos Engine**")
        chaos = sources.get("chaos_engine", {})
        st.metric("Packets", chaos.get("packets", 0))
        st.metric("Avg Value", f"{chaos.get('avg_value', 0):.3f}")
        st.metric("Avg Confidence", f"{chaos.get('avg_confidence', 0):.3f}")
    
    with col2:
        st.write("**Silent Operator**")
        silent = sources.get("silent_operator", {})
        st.metric("Recent Packets", silent.get("packets", 0))
        st.metric("Stealth Level", f"{silent.get('stealth_level', 0):.1f}")
        st.metric("Intelligence Level", f"{silent.get('intelligence_level', 1.0):.1f}")
    
    # Top insights
    st.subheader("💡 Top Insights")
    insights = intel_response.get("top_insights", [])
    
    if insights:
        for i, insight in enumerate(insights[:5], 1):
            st.write(f"{i}. {insight}")
    else:
        st.info("No insights available yet. Start the system to begin intelligence gathering.")
    
    # Opportunities and risks
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Opportunities")
        opp_count = intel_response.get("opportunity_count", 0)
        st.metric("Identified Opportunities", opp_count)
    
    with col2:
        st.subheader("⚠️ Risks")
        risk_count = intel_response.get("risk_count", 0)
        st.metric("Identified Risks", risk_count)

def show_value_extraction():
    """Show value extraction interface"""
    st.header("💰 Value Extraction")
    
    # Manual value extraction
    st.subheader("🔬 Manual Extraction")
    
    with st.expander("Extract Value from Data"):
        data_input = st.text_area("Enter data to analyze:", height=150)
        source_name = st.text_input("Source name:", value="manual_input")
        
        if st.button("🔍 Extract Value"):
            if data_input:
                try:
                    # Try to parse as JSON, fallback to string
                    try:
                        parsed_data = json.loads(data_input)
                    except:
                        parsed_data = data_input
                    
                    extraction_request = {
                        "data": parsed_data,
                        "source": source_name
                    }
                    
                    response = make_api_request("/extract/value", "POST", extraction_request)
                    
                    if "error" not in response:
                        st.success("✅ Value extraction completed!")
                        
                        # Display results
                        st.subheader("📊 Extraction Results")
                        
                        metrics = response.get("value_metrics", {})
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Value", f"{metrics.get('total_value', 0):.2f}")
                            st.metric("Monetary Value", f"{metrics.get('monetary_value', 0):.2f}")
                            st.metric("Strategic Value", f"{metrics.get('strategic_value', 0):.2f}")
                        
                        with col2:
                            st.metric("Confidence Score", f"{response.get('confidence_score', 0):.2f}")
                            st.metric("Pattern Count", response.get("pattern_count", 0))
                            st.metric("Competitive Advantage", f"{metrics.get('competitive_advantage', 0):.2f}")
                        
                        with col3:
                            st.metric("Opportunity Value", f"{metrics.get('opportunity_value', 0):.2f}")
                            st.metric("Risk Mitigation", f"{metrics.get('risk_mitigation', 0):.2f}")
                            st.metric("Time Value", f"{metrics.get('time_value', 0):.2f}")
                        
                        # Actionable items
                        st.subheader("🎯 Actionable Items")
                        actionable = response.get("actionable_items", [])
                        for item in actionable:
                            st.write(f"• {item}")
                        
                        # Hidden insights
                        st.subheader("🔍 Hidden Insights")
                        insights = response.get("hidden_insights", [])
                        for insight in insights[:5]:
                            st.write(f"• {insight}")
                        
                        # Next actions
                        st.subheader("➡️ Next Actions")
                        next_actions = response.get("next_actions", [])
                        for action in next_actions:
                            st.write(f"• {action}")
                    
                    else:
                        st.error(f"Extraction failed: {response['error']}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter data to analyze")
    
    # Value extraction history
    st.subheader("📈 Extraction Performance")
    
    performance_response = make_api_request("/analytics/performance")
    
    if "error" not in performance_response:
        intel_perf = performance_response.get("intelligence_performance", {})
        value_perf = performance_response.get("value_extraction_performance", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Intelligence Performance**")
            st.metric("Total Packets", intel_perf.get("total_packets_processed", 0))
            st.metric("Avg Value Score", f"{intel_perf.get('average_value_score', 0):.3f}")
            st.metric("High Value Ratio", f"{intel_perf.get('high_value_packet_ratio', 0):.2%}")
        
        with col2:
            st.write("**Value Extraction Performance**")
            st.metric("Total Extractions", value_perf.get("total_extractions", 0))
            st.metric("Cumulative Value", f"{value_perf.get('cumulative_value', 0):.2f}")
            st.metric("Avg Confidence", f"{value_perf.get('average_confidence', 0):.3f}")

def show_opportunities():
    """Show high-value opportunities"""
    st.header("🎯 High-Value Opportunities")
    
    opportunities_response = make_api_request("/opportunities/high-value")
    
    if "error" in opportunities_response:
        st.error("Unable to load opportunities")
        return
    
    opportunities = opportunities_response.get("high_value_opportunities", [])
    total_count = opportunities_response.get("total_count", 0)
    avg_value = opportunities_response.get("avg_value_score", 0)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Opportunities", total_count)
    
    with col2:
        st.metric("High-Value Count", len(opportunities))
    
    with col3:
        st.metric("Avg Value Score", f"{avg_value:.3f}")
    
    # Opportunities list
    if opportunities:
        st.subheader("📋 Opportunity Details")
        
        for i, opp in enumerate(opportunities[:10], 1):
            with st.expander(f"Opportunity #{i} - Value: {opp['value_score']:.3f}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Source:** {opp['source']}")
                    st.write(f"**Timestamp:** {opp['timestamp']}")
                    st.write(f"**Confidence:** {opp['confidence']:.3f}")
                
                with col2:
                    st.write("**Key Insights:**")
                    for insight in opp.get('insights', [])[:3]:
                        st.write(f"• {insight}")
                
                if opp.get('silent_patterns'):
                    st.write("**Silent Patterns Detected:**")
                    patterns = opp['silent_patterns']
                    for key, value in patterns.items():
                        if isinstance(value, list) and value:
                            st.write(f"• {key}: {len(value)} patterns")
    
    else:
        st.info("No high-value opportunities detected yet. Start the system and add data sources.")

def show_configuration():
    """Show system configuration"""
    st.header("⚙️ System Configuration")
    
    # Current configuration
    st.subheader("🔧 Current Settings")
    
    with st.form("system_config"):
        chaos_threshold = st.slider("Chaos Threshold", 0.0, 1.0, 0.7, 0.1)
        extraction_intensity = st.selectbox("Extraction Intensity", ["low", "medium", "high", "maximum"])
        stealth_mode = st.checkbox("Stealth Mode", value=True)
        value_priority = st.selectbox("Value Priority", ["monetary", "strategic", "competitive", "information"])
        
        if st.form_submit_button("💾 Update Configuration"):
            config_data = {
                "chaos_threshold": chaos_threshold,
                "extraction_intensity": extraction_intensity,
                "stealth_mode": stealth_mode,
                "value_priority": value_priority
            }
            
            response = make_api_request("/system/configure", "POST", config_data)
            
            if "error" not in response:
                st.success("✅ Configuration updated!")
            else:
                st.error(f"Configuration failed: {response['error']}")
    
    # Data sources management
    st.subheader("📡 Data Sources")
    
    # List current sources
    sources_response = make_api_request("/data-sources/list")
    
    if "error" not in sources_response:
        sources = sources_response.get("active_sources", [])
        
        if sources:
            st.write("**Active Data Sources:**")
            sources_df = pd.DataFrame(sources)
            st.dataframe(sources_df, use_container_width=True)
        else:
            st.info("No data sources configured")
    
    # Add new data source
    st.subheader("➕ Add Data Source")
    
    with st.form("add_source"):
        source_name = st.text_input("Source Name")
        source_url = st.text_input("URL/Endpoint")
        source_type = st.selectbox("Source Type", ["api", "scrape", "feed", "stream"])
        frequency = st.number_input("Frequency (seconds)", min_value=10, value=60)
        processor = st.text_input("Processor", value="default")
        
        if st.form_submit_button("📡 Add Source"):
            if source_name and source_url:
                source_data = {
                    "name": source_name,
                    "url": source_url,
                    "type": source_type,
                    "frequency": frequency,
                    "processor": processor
                }
                
                response = make_api_request("/data-sources/add", "POST", source_data)
                
                if "error" not in response:
                    st.success(f"✅ Data source '{source_name}' added!")
                else:
                    st.error(f"Failed to add source: {response['error']}")
            else:
                st.warning("Please provide name and URL")
    
    # Debug information
    st.subheader("🐛 Debug Information")
    
    if st.button("🔍 Show Debug Info"):
        debug_response = make_api_request("/debug/system-state")
        
        if "error" not in debug_response:
            st.json(debug_response)
        else:
            st.error(f"Debug info unavailable: {debug_response['error']}")

if __name__ == "__main__":
    main()