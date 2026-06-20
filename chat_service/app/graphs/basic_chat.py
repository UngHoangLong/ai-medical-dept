from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Annotated
from typing import TypedDict
from collections import deque

class BasicState(TypedDict):
    query: str
    response: str

class BasicChatGraph:
    def __init__(self, llm):
        self.llm = llm
        
        self.graph = self._build_sequential_graph()

    async def response(self, state: BasicState):
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=f"{state['query']}"),
        ])
        
        return {"response": response}

    def _build_sequential_graph(self):
        graph = StateGraph(BasicState)
        graph.add_node("response", self.response)
        graph.add_edge(START, "response")
        graph.add_edge("response", END)
        return graph.compile()
    
   
        