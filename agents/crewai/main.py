from crewai import Agent, Task, Crew
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="CrewAI Service")

# Define agents
researcher = Agent(
    role='Research Specialist',
    goal='Conduct thorough research on given topics',
    backstory='Expert researcher with access to comprehensive knowledge',
    verbose=True
)

analyst = Agent(
    role='Data Analyst', 
    goal='Analyze and interpret research findings',
    backstory='Skilled analyst who transforms data into insights',
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Create comprehensive reports from research and analysis',
    backstory='Professional writer who creates clear, actionable content',
    verbose=True
)

@app.post("/execute")
async def execute_crew(task_description: str):
    # Define task
    research_task = Task(
        description=f"Research and analyze: {task_description}",
        agent=researcher
    )
    
    analysis_task = Task(
        description="Analyze the research findings and identify key insights",
        agent=analyst
    )
    
    writing_task = Task(
        description="Create a comprehensive report based on research and analysis",
        agent=writer
    )
    
    # Create crew
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        verbose=True
    )
    
    # Execute
    result = crew.kickoff()
    
    return {"result": result, "status": "completed"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "crewai"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
