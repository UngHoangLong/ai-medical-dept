from typing import Annotated, Any, TypedDict, list

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class MainState(TypedDict):
    messages: Annotated[list, add_messages]
    rewrited_query: str
    response: str
    clinical_data: Any
    report: Any
    
class IsSafe(BaseModel):
    is_safe: bool
    
class QueryIntent(BaseModel):
    is_medical: bool = Field(description="True nếu câu hỏi liên quan đến sức khỏe, bệnh lý, thuốc men, dinh dưỡng, hoặc giải phẫu. False nếu là lời chào, cảm ơn, hỏi thăm thời tiết, giao tiếp thông thường.")
    
class MainGraph:
    def __init__(self, llm, mcp_tools, checkpointer=None):
        self.llm = llm
        self.mcp_tools = mcp_tools
        self.checkpointer = checkpointer
        
        self.rewrite_sys_prompt = "Dựa vào lịch sử trò chuyện, hãy viết lại câu hỏi cuối cùng của người dùng thành một câu hỏi đứng độc lập, rõ nghĩa. Chỉ trả về câu hỏi, không giải thích gì thêm."
        self.casual_sys_prompt = "Bạn là một trợ lý ảo y tế thân thiện tên là Bác Sĩ AI. Đối với các câu hỏi chào hỏi hoặc giao tiếp thông thường, hãy trả lời lịch sự, thân thiện và ngắn gọn. Tuyệt đối không tự bịa ra kiến thức y khoa."
        
        # 2. KHỞI TẠO PREBUILT MEDICAL AGENT (SUB-GRAPH)
        # Sử dụng LLM và MCP Tools của bạn. Cấp cho nó một system prompt chuyên môn hóa.
        medical_sys_prompt = "Bạn là một chuyên gia y tế. Hãy sử dụng các công cụ được cung cấp để tra cứu thông tin y khoa, tương tác thuốc, và hồ sơ an toàn trước khi trả lời bệnh nhân."
        self.medical_agent = create_agent(
            model=self.llm, 
            tools=self.mcp_tools,
            state_modifier=medical_sys_prompt # System prompt cho ReAct Agent
        )
        self.graph = self._build_sequential_graph()
        
    async def guardrail(self, state: MainState):
        latest_message = state["messages"][-1].content
        guardrail_llm = self.llm.with_structured_output(IsSafe)
        response = await guardrail_llm.ainvoke([
            SystemMessage(content="Check if the following user query is safe to answer without violating any content policies. Respond with only a boolean value.\n\n"),
            HumanMessage(content=f"Query: {latest_message}")
        ])
        return {"is_safe": response.is_safe}

    def route_guardrail(self, state: MainState) -> str:
        if state["is_safe"]:
            return "rewrite_query" 
        else:
            return "unsafe_response"

    async def unsafe_response(self, state: MainState):
        bot_reply = "Xin lỗi, tôi không thể trả lời câu hỏi này vì nó vi phạm chính sách hoặc không an toàn."
        return {
            "messages": [AIMessage(content=bot_reply)] 
        }
        
    async def rewrite_query(self, state: MainState):
        messages = state["messages"]
        latest_message = messages[-1].content
        
        if len(messages) <= 1:
            return {"rewrited_query": latest_message}
            
        response = await self.llm.ainvoke([
            SystemMessage(content=self.rewrite_sys_prompt),
            *messages
        ])
        return {"rewrited_query": response.content}

    async def classify_intent(self, state: MainState):
        """Kiểm tra xem câu hỏi có cần tra cứu Y khoa (RAG) không"""
        # FIX: Dùng câu hỏi đã được viết lại (có đầy đủ ngữ cảnh) để phân loại
        query_to_check = state.get("rewrited_query", state["messages"][-1].content)
        
        intent_llm = self.llm.with_structured_output(QueryIntent)
        response = await intent_llm.ainvoke([
            SystemMessage(content="Bạn là chuyên gia phân loại. Hãy xác định xem câu hỏi của người dùng có cần tra cứu kiến thức y khoa/sức khỏe hay không."),
            HumanMessage(content=f"Query: {query_to_check}")
        ])
        return {"is_medical": response.is_medical}
    
    
    def route_intent(self, state: MainState) -> str:
        if state["is_medical"]:
            return "medical_chat" 
        else:
            return "casual_chat"   

    async def casual_chat(self, state: MainState):
            # 1. Lấy toàn bộ lịch sử (trừ câu hỏi cuối cùng của user)
            history = state["messages"][:-1] 
            
            # 2. Tạo một message mới chứa câu hỏi đã được viết lại cho rõ nghĩa
            clarified_query = HumanMessage(content=state["rewrited_query"])
            
            # 3. Gộp System Prompt + Lịch sử + Câu hỏi rõ nghĩa
            prompt = [SystemMessage(content=self.casual_sys_prompt)] + history + [clarified_query]
            
            # 4. Gọi LLM
            response = await self.llm.ainvoke(prompt)
            
            return {
                "messages": [AIMessage(content=response.content)]
            }
        
    async def medical_chat(self, state: MainState):
            history = state["messages"][:-1]
            clarified_query = HumanMessage(content=state["rewrited_query"])
            
            # Prebuilt Agent của LangGraph yêu cầu input là một Dict có key "messages"
            result = await self.medical_agent.ainvoke({
                "messages": history + [clarified_query]
            })

            # Prebuilt Agent trả về một list messages, ta lấy nội dung của tin nhắn cuối cùng (là câu trả lời)
            final_report = result["messages"][-1].content

            return {
                "messages": [AIMessage(content=final_report)] 
            }

    async def trim_memory(self, state: MainState):
        messages = state["messages"]
        KEEP_COUNT = 11
        if len(messages) > KEEP_COUNT:
            messages_to_remove = messages[:-KEEP_COUNT]
            delete_commands = [RemoveMessage(id=msg.id) for msg in messages_to_remove if msg.id]
            return {"messages": delete_commands}
        return {}
        
    def _build_sequential_graph(self):
            builder = StateGraph(MainState) 
            
            # 1. Khai báo TẤT CẢ các Node
            builder.add_node("trim_memory", self.trim_memory)
            builder.add_node("guardrail", self.guardrail)
            builder.add_node("unsafe_response", self.unsafe_response)
            builder.add_node("rewrite_query", self.rewrite_query)
            builder.add_node("classify_intent", self.classify_intent) 
            builder.add_node("casual_chat", self.casual_chat)        
            builder.add_node("medical_chat", self.medical_chat)
            
            # 2. Vừa vào cửa là DỌN DẸP BỘ NHỚ ngay
            builder.add_edge(START, "trim_memory")
            
            # 3. Dọn xong thì đi qua trạm kiểm duyệt an toàn
            builder.add_edge("trim_memory", "guardrail")
            
            # 4. Chuyển hướng An toàn
            builder.add_conditional_edges(
                "guardrail",
                self.route_guardrail,
                ["rewrite_query", "unsafe_response"]
            )
            
            # 5. Luồng an toàn: Viết lại -> Phân loại
            builder.add_edge("rewrite_query", "classify_intent")
            
            # 6. Chuyển hướng Ý định
            builder.add_conditional_edges(
                "classify_intent",
                self.route_intent,
                ["medical_chat", "casual_chat"]
            )
            
            # 7. CÁC NHÁNH KẾT THÚC: Xong việc là đi thẳng ra cổng (END)
            builder.add_edge("medical_chat", END)
            builder.add_edge("casual_chat", END)
            builder.add_edge("unsafe_response", END)
            
            return builder.compile(checkpointer=self.checkpointer)