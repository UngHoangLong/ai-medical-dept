import json
from typing import Annotated, Any, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from ..utils import fetch_medical_context


class MainState(TypedDict):
    messages: Annotated[list, add_messages]
    report_id: str
    rewrited_query: str
    clinical_data: Any
    report_content: Any
    
    is_safe: bool
    is_medical: bool
    
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
        medical_sys_prompt = "Bạn là một chuyên gia y tế. Hãy sử dụng các công cụ được cung cấp để tra cứu thông tin y khoa, tương tác thuốc, và hồ sơ an toàn trước khi trả lời bệnh nhân."
        self.medical_agent = create_agent(
            model=self.llm, 
            tools=self.mcp_tools,
            system_prompt=medical_sys_prompt
        )
        self.graph = self._build_sequential_graph()

    # --- HÀM HELPER ĐỂ FORMAT JSON THÀNH TEXT MARKDOWN DỄ ĐỌC ---
    def _format_medical_json(self, data: Any) -> str:
        """Chuyển đổi dữ liệu JSON phức tạp thành văn bản Markdown để LLM dễ đọc hơn"""
        if isinstance(data, str):
            return data
            
        if not isinstance(data, dict):
            return str(data)

        formatted_lines = []
        
        # 1. Bóc tách phần Findings & Impression
        if "finding_impression" in data:
            fi = data["finding_impression"]
            formatted_lines.append("### 1. KẾT QUẢ KHÁM (FINDINGS)")
            formatted_lines.append(fi.get("findings", "Không có thông tin."))
            formatted_lines.append("\n### 2. CHẨN ĐOÁN/KẾT LUẬN (IMPRESSION)")
            formatted_lines.append(fi.get("impression", "Không có thông tin."))
            
        # 2. Bóc tách chi tiết Radiology
        if "radiology" in data:
            formatted_lines.append("\n### 3. CHI TIẾT HÌNH ẢNH (RADIOLOGY)")
            rad = data["radiology"]
            if "detail" in rad and "answer" in rad["detail"]:
                formatted_lines.append("**Thông số nốt mờ (Nodule Details):**")
                for k, v in rad["detail"]["answer"].items():
                    formatted_lines.append(f"- {k}: {v}")
            if "screening" in rad and "answer" in rad["screening"]:
                formatted_lines.append("**Tầm soát (Screening):**")
                for k, v in rad["screening"]["answer"].items():
                    formatted_lines.append(f"- {k}: {v}")

        # 3. Bóc tách Oncology & Cardiology
        if "oncology" in data or "cardiology" in data:
            formatted_lines.append("\n### 4. ĐÁNH GIÁ RỦI RO (RISK ASSESSMENT)")
            if "oncology" in data and "answer" in data["oncology"]:
                formatted_lines.append("**Ung bướu (Oncology):**")
                for k, v in data["oncology"]["answer"].items():
                    formatted_lines.append(f"- {k}: {v}")
            if "cardiology" in data and "answer" in data["cardiology"]:
                formatted_lines.append("**Tim mạch (Cardiology):**")
                for k, v in data["cardiology"]["answer"].items():
                    formatted_lines.append(f"- {k}: {v}")

        # 4. Bóc tách Verification/Analysis
        if "verification" in data and "analysis" in data["verification"]:
            formatted_lines.append("\n### 5. PHÂN TÍCH CHUYÊN SÂU (VERIFICATION ANALYSIS)")
            formatted_lines.append(data["verification"]["analysis"])

        # Nếu Dict không khớp schema trên, fallback về dạng JSON string cơ bản
        if not formatted_lines:
            return json.dumps(data, ensure_ascii=False, indent=2)

        return "\n".join(formatted_lines)

    async def guardrail(self, state: MainState):
        latest_message = state["messages"][-1].content
        guardrail_llm = self.llm.with_structured_output(IsSafe, method="json_mode")
        response = await guardrail_llm.ainvoke([
            SystemMessage(content="Check if the following user query is safe to answer without violating any content policies. You MUST respond in valid JSON format matching this schema: {\"is_safe\": boolean}.\n\n"),
            HumanMessage(content=f"Query: {latest_message}")
        ])
        return {"is_safe": response.is_safe}

    async def init_context(self, state: MainState, config: RunnableConfig):
            clinical_text = state.get("clinical_data") 
            report_content = state.get("report_content")
            
            # Mở rộng điều kiện: Chưa có data HOẶC đang mang giá trị rỗng/lỗi thì mới fetch
            need_fetch = (
                not clinical_text 
                or not report_content
                or clinical_text == "Không có thông tin lâm sàng."
                or report_content == "Không có dữ liệu báo cáo."
            )
            
            if need_fetch:
                report_id = state.get("report_id")
                pool = config["configurable"].get("db_pool")
                
                if not pool:
                    raise ValueError("Database pool is not provided in config.")

                db_data = await fetch_medical_context(pool, report_id)
                
                if db_data:
                    print("--- Đã đồng bộ dữ liệu từ DB vào LangGraph State ---")
                    print(f"Clinical Data: {db_data['clinical_text'][:200]}...")  # In một phần để kiểm tra
                    return {
                        "clinical_data": db_data["clinical_text"],
                        "report_content": db_data["report_content"]
                    }
                else:
                    return {
                        "clinical_data": "Không có thông tin lâm sàng.",
                        "report_content": "Không có dữ liệu báo cáo."
                    }
                    
            # Đã có dữ liệu hợp lệ từ các lượt chat trước, không cần làm gì cả
            return {}

    def route_guardrail(self, state: MainState) -> str:
        if state.get("is_safe", True):
            return "rewrite_query" 
        else:
            return "unsafe_response"

    async def unsafe_response(self, state: MainState):
        bot_reply = "Xin lỗi, tôi không thể trả lời câu hỏi này vì nó vi phạm chính sách hoặc không an toàn."
        return {"messages": [AIMessage(content=bot_reply)]}
        
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
        query_to_check = state.get("rewrited_query", state["messages"][-1].content)
        intent_llm = self.llm.with_structured_output(QueryIntent, method="json_mode")
        response = await intent_llm.ainvoke([
            SystemMessage(content="Bạn là chuyên gia phân loại. Hãy xác định xem câu hỏi của người dùng có cần tra cứu kiến thức y khoa/sức khỏe hay không. BẮT BUỘC phải trả về định dạng JSON với cấu trúc chính xác như sau: {\"is_medical\": boolean}."),
            HumanMessage(content=f"Query: {query_to_check}")
        ])
        return {"is_medical": response.is_medical}
    
    def route_intent(self, state: MainState) -> str:
        if state.get("is_medical", False):
            return "medical_chat" 
        else:
            return "casual_chat"   

    async def casual_chat(self, state: MainState):
        history = state["messages"][:-1] 
        clarified_query = HumanMessage(content=state["rewrited_query"])
        prompt = [SystemMessage(content=self.casual_sys_prompt)] + history + [clarified_query]
        response = await self.llm.ainvoke(prompt)
        return {"messages": [AIMessage(content=response.content)]}
        
    async def medical_chat(self, state: MainState):
        history = state["messages"][:-1]
        query = state["rewrited_query"]
        
        context_blocks = []
        
        # 1. Xử lý Clinical Data (Hồ sơ lâm sàng)
        clinical_data = state.get("clinical_data")
        if clinical_data:
            # Nếu là dict thì stringify cơ bản, nếu text thì giữ nguyên
            clinical_str = clinical_data if isinstance(clinical_data, str) else json.dumps(clinical_data, ensure_ascii=False, indent=2)
            context_blocks.append(f"[HỒ SƠ LÂM SÀNG BỆNH NHÂN]:\n{clinical_str}")
            
        # 2. Xử lý Report Content (Sử dụng hàm bóc tách JSON)
        report_content = state.get("report_content")
        if report_content:
            report_str = self._format_medical_json(report_content)
            context_blocks.append(f"[BÁO CÁO Y KHOA / CHẨN ĐOÁN]:\n{report_str}")
            
        medical_context_str = "\n\n".join(context_blocks)
        
        # 3. Ghép vào câu hỏi cuối cùng
        if medical_context_str:
            final_prompt_content = f"Dựa vào thông tin bệnh nhân dưới đây, hãy trả lời câu hỏi chuyên môn.\n\n{medical_context_str}\n\n[CÂU HỎI MỚI]: {query}"
        else:
            final_prompt_content = query
            
        clarified_query_with_context = HumanMessage(content=final_prompt_content)
        
        result = await self.medical_agent.ainvoke({
            "messages": history + [clarified_query_with_context]
        })

        final_report = result["messages"][-1].content
        return {"messages": [AIMessage(content=final_report)]}

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
        
        builder.add_node("init_context", self.init_context)
        builder.add_node("trim_memory", self.trim_memory)
        builder.add_node("guardrail", self.guardrail)
        builder.add_node("unsafe_response", self.unsafe_response)
        builder.add_node("rewrite_query", self.rewrite_query)
        builder.add_node("classify_intent", self.classify_intent) 
        builder.add_node("casual_chat", self.casual_chat)        
        builder.add_node("medical_chat", self.medical_chat)
        
        builder.add_edge(START, "init_context")
        builder.add_edge("init_context", "trim_memory")
        builder.add_edge("trim_memory", "guardrail")
        
        builder.add_conditional_edges("guardrail", self.route_guardrail, ["rewrite_query", "unsafe_response"])
        
        builder.add_edge("rewrite_query", "classify_intent")
        builder.add_conditional_edges("classify_intent", self.route_intent, ["medical_chat", "casual_chat"])
        
        builder.add_edge("medical_chat", END)
        builder.add_edge("casual_chat", END)
        builder.add_edge("unsafe_response", END)
        
        return builder.compile(checkpointer=self.checkpointer)