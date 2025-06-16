import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Set your API key
os.environ["OPENAI_API_KEY"] = "api key"

# Define state
class BotState(TypedDict):
    messages: list

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

def classify_mood(state: BotState) -> BotState:
    return {"messages": state["messages"]}

def route_mood(state: BotState) -> str:
    user_msg = state['messages'][-1]['content']
    response = llm.predict(f"Classify the user's mood: sad, happy, or angry.\n\nUser: '{user_msg}'")
    mood = response.strip().lower()

    if "sad" in mood:
        return "motivate"
    elif "angry" in mood:
        return "calm"
    else:
        return "celebrate"

def motivate(state: BotState) -> BotState:
    msg = {"role": "assistant", "content": "You're doing great. Tough times don’t last."}
    return {"messages": state["messages"] + [msg]}

def calm(state: BotState) -> BotState:
    msg = {"role": "assistant", "content": "Take a deep breath. Relax and let go."}
    return {"messages": state["messages"] + [msg]}

def celebrate(state: BotState) -> BotState:
    msg = {"role": "assistant", "content": "That's awesome! Keep the good vibes going!"}
    return {"messages": state["messages"] + [msg]}

# LangGraph
graph = StateGraph(BotState)
graph.add_node("classify", classify_mood)
graph.add_node("motivate", motivate)
graph.add_node("calm", calm)
graph.add_node("celebrate", celebrate)
graph.add_conditional_edges("classify", route_mood, {
    "motivate": "motivate",
    "calm": "calm",
    "celebrate": "celebrate"
})
graph.add_edge("motivate", END)
graph.add_edge("calm", END)
graph.add_edge("celebrate", END)
graph.set_entry_point("classify")
mood_bot = graph.compile()

# FastAPI setup
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_msg = data.get("message", "")
    state = {"messages": [{"role": "user", "content": user_msg}]}
    result = mood_bot.invoke(state)
    reply = result["messages"][-1]["content"]
    return JSONResponse(content={"reply": reply})
