from crewai import Agent, Task, Crew
from fastapi import FastAPI
import uvicorn

app = FastAPI()

class CrewAIService:
    def __init__(self):
        self.researcher = Agent(
            role='Researcher',
            goal='Research and analyze information',
            backstory='Expert researcher with deep analytical skills',
            verbose=True,
            allow_delegation=False
        )
        
        self.writer = Agent(
            role='Writer',
            goal='Create compelling content',
            backstory='Professional writer with creative skills',
            verbose=True,
            allow_delegation=False
        )
    
    def create_crew(self, task_description):
        task = Task(
            description=task_description,
            agent=self.researcher
        )
        
        crew = Crew(
            agents=[self.researcher, self.writer],
            tasks=[task],
            verbose=True
        )
        
        return crew

service = CrewAIService()

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "crewai"}

@app.post("/execute")
async def execute(task: dict):
    crew = service.create_crew(task.get("description", ""))
    result = crew.kickoff()
    return {"result": str(result)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8502)
