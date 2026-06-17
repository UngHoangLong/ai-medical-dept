import json
from typing import Annotated, Any, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from ..utils import fetch_medical_context

# ── i18n: Prompt templates cho 3 ngôn ngữ ──────────────────────────────
PROMPTS = {
    "vi": {
        "rewrite": "Dựa vào lịch sử trò chuyện, hãy viết lại câu hỏi cuối cùng của người dùng thành một câu hỏi đứng độc lập, rõ nghĩa. Chỉ trả về câu hỏi, không giải thích gì thêm.",
        "casual": "Bạn là một trợ lý ảo y tế thân thiện tên là Bác Sĩ AI. Đối với các câu hỏi chào hỏi hoặc giao tiếp thông thường, hãy trả lời lịch sự, thân thiện và ngắn gọn. Tuyệt đối không tự bịa ra kiến thức y khoa.",
        "medical_agent": "Bạn là một chuyên gia y tế. Hãy sử dụng các công cụ được cung cấp để tra cứu thông tin y khoa, tương tác thuốc, và hồ sơ an toàn trước khi trả lời bệnh nhân.",
        "guardrail": "Check if the following user query is safe to answer without violating any content policies. You MUST respond in valid JSON format matching this schema: {\"is_safe\": boolean}.\n\n",
        "unsafe": "Xin lỗi, tôi không thể trả lời câu hỏi này vì nó vi phạm chính sách hoặc không an toàn.",
        "classify": "Bạn là chuyên gia phân loại. Hãy xác định xem câu hỏi của người dùng có cần tra cứu kiến thức y khoa/sức khỏe hay không. BẮT BUỘC phải trả về định dạng JSON với cấu trúc chính xác như sau: {\"is_medical\": boolean}.",
        "medical_with_context": "Dựa vào thông tin bệnh nhân dưới đây, hãy trả lời câu hỏi chuyên môn.\n\n{context}\n\n[CÂU HỎI MỚI]: {query}",
        "fmt_findings": "### 1. KẾT QUẢ KHÁM (FINDINGS)",
        "fmt_impression": "\n### 2. CHẨN ĐOÁN/KẾT LUẬN (IMPRESSION)",
        "fmt_radiology": "\n### 3. CHI TIẾT HÌNH ẢNH (RADIOLOGY)",
        "fmt_nodule": "**Thông số nốt mờ (Nodule Details):**",
        "fmt_screening": "**Tầm soát (Screening):**",
        "fmt_risk": "\n### 4. ĐÁNH GIÁ RỦI RO (RISK ASSESSMENT)",
        "fmt_oncology": "**Ung bướu (Oncology):**",
        "fmt_cardiology": "**Tim mạch (Cardiology):**",
        "fmt_verification": "\n### 5. PHÂN TÍCH CHUYÊN SÂU (VERIFICATION ANALYSIS)",
        "fmt_clinical_label": "[HỒ SƠ LÂM SÀNG BỆNH NHÂN]",
        "fmt_report_label": "[BÁO CÁO Y KHOA / CHẨN ĐOÁN]",
        "fmt_no_info": "Không có thông tin.",
        "no_clinical": "Không có thông tin lâm sàng.",
        "no_report": "Không có dữ liệu báo cáo.",
    },
    "en": {
        "rewrite": "Based on the conversation history, rewrite the user's last question as a standalone, self-contained question. Return only the question, no explanation.",
        "casual": "You are a friendly medical AI assistant named Dr. AI. For greetings or casual conversation, respond politely, warmly and concisely. Never fabricate medical knowledge.",
        "medical_agent": "You are a medical expert. Use the provided tools to look up medical information, drug interactions, and safety profiles before answering the patient.",
        "guardrail": "Check if the following user query is safe to answer without violating any content policies. You MUST respond in valid JSON format matching this schema: {\"is_safe\": boolean}.\n\n",
        "unsafe": "Sorry, I cannot answer this question as it violates content policies or is unsafe.",
        "classify": "You are a classification expert. Determine whether the user's question requires medical/health knowledge lookup. You MUST respond in valid JSON format: {\"is_medical\": boolean}.",
        "medical_with_context": "Based on the patient information below, answer the clinical question.\n\n{context}\n\n[NEW QUESTION]: {query}",
        "fmt_findings": "### 1. EXAMINATION RESULTS (FINDINGS)",
        "fmt_impression": "\n### 2. DIAGNOSIS / CONCLUSION (IMPRESSION)",
        "fmt_radiology": "\n### 3. IMAGING DETAILS (RADIOLOGY)",
        "fmt_nodule": "**Nodule Details:**",
        "fmt_screening": "**Screening:**",
        "fmt_risk": "\n### 4. RISK ASSESSMENT",
        "fmt_oncology": "**Oncology:**",
        "fmt_cardiology": "**Cardiology:**",
        "fmt_verification": "\n### 5. IN-DEPTH ANALYSIS (VERIFICATION)",
        "fmt_clinical_label": "[PATIENT CLINICAL RECORD]",
        "fmt_report_label": "[MEDICAL REPORT / DIAGNOSIS]",
        "fmt_no_info": "No information available.",
        "no_clinical": "No clinical information available.",
        "no_report": "No report data available.",
    },
    "ja": {
        "rewrite": "会話履歴に基づいて、ユーザーの最後の質問を独立した明確な質問に書き直してください。質問のみを返し、説明は不要です。",
        "casual": "あなたはDr. AIという名前の親切な医療AIアシスタントです。挨拶や日常会話には、丁寧で親しみやすく簡潔に答えてください。医学知識を捏造しないでください。",
        "medical_agent": "あなたは医療の専門家です。患者に回答する前に、提供されたツールを使用して医学情報、薬物相互作用、安全性プロファイルを調べてください。",
        "guardrail": "Check if the following user query is safe to answer without violating any content policies. You MUST respond in valid JSON format matching this schema: {\"is_safe\": boolean}.\n\n",
        "unsafe": "申し訳ございませんが、この質問はコンテンツポリシーに違反するため、お答えすることができません。",
        "classify": "あなたは分類の専門家です。ユーザーの質問が医学・健康に関する知識の検索を必要とするかどうかを判断してください。必ずJSON形式で回答してください: {\"is_medical\": boolean}。",
        "medical_with_context": "以下の患者情報に基づいて、臨床的な質問にお答えください。\n\n{context}\n\n[新しい質問]: {query}",
        "fmt_findings": "### 1. 検査結果 (FINDINGS)",
        "fmt_impression": "\n### 2. 診断・結論 (IMPRESSION)",
        "fmt_radiology": "\n### 3. 画像詳細 (RADIOLOGY)",
        "fmt_nodule": "**結節の詳細:**",
        "fmt_screening": "**スクリーニング:**",
        "fmt_risk": "\n### 4. リスク評価",
        "fmt_oncology": "**腫瘍学:**",
        "fmt_cardiology": "**循環器学:**",
        "fmt_verification": "\n### 5. 詳細分析 (VERIFICATION)",
        "fmt_clinical_label": "[患者の臨床記録]",
        "fmt_report_label": "[医療レポート / 診断]",
        "fmt_no_info": "情報がありません。",
        "no_clinical": "臨床情報がありません。",
        "no_report": "レポートデータがありません。",
    },
}


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
    def __init__(self, llm, mcp_tools, checkpointer=None, language: str = "en"):
        self.llm = llm
        self.mcp_tools = mcp_tools
        self.checkpointer = checkpointer
        self.language = language
        self.p = PROMPTS.get(language, PROMPTS["en"])  # fallback to English
        
        self.rewrite_sys_prompt = self.p["rewrite"]
        self.casual_sys_prompt = self.p["casual"]
        
        # 2. KHỞI TẠO PREBUILT MEDICAL AGENT (SUB-GRAPH)
        self.medical_agent = create_agent(
            model=self.llm, 
            tools=self.mcp_tools,
            system_prompt=self.p["medical_agent"]
        )
        self.graph = self._build_sequential_graph()

    # --- HÀM HELPER ĐỂ FORMAT JSON THÀNH TEXT MARKDOWN DỄ ĐỌC ---
    def _format_medical_json(self, data: Any) -> str:
        """Chuyển đổi dữ liệu JSON phức tạp thành văn bản Markdown để LLM dễ đọc hơn"""
        p = self.p
        if isinstance(data, str):
            return data
            
        if not isinstance(data, dict):
            return str(data)

        formatted_lines = []
        
        # 1. Bóc tách phần Findings & Impression
        if "finding_impression" in data:
            fi = data["finding_impression"]
            formatted_lines.append(p["fmt_findings"])
            formatted_lines.append(fi.get("findings", p["fmt_no_info"]))
            formatted_lines.append(p["fmt_impression"])
            formatted_lines.append(fi.get("impression", p["fmt_no_info"]))
            
        # 2. Bóc tách chi tiết Radiology
        if "radiology" in data:
            formatted_lines.append(p["fmt_radiology"])
            rad = data["radiology"]
            if "detail" in rad and "answer" in rad["detail"]:
                formatted_lines.append(p["fmt_nodule"])
                for k, v in rad["detail"]["answer"].items():
                    formatted_lines.append(f"- {k}: {v}")
            if "screening" in rad and "answer" in rad["screening"]:
                formatted_lines.append(p["fmt_screening"])
                for k, v in rad["screening"]["answer"].items():
                    formatted_lines.append(f"- {k}: {v}")

        # 3. Bóc tách Oncology & Cardiology
        if "oncology" in data or "cardiology" in data:
            formatted_lines.append(p["fmt_risk"])
            if "oncology" in data and "answer" in data["oncology"]:
                formatted_lines.append(p["fmt_oncology"])
                for k, v in data["oncology"]["answer"].items():
                    formatted_lines.append(f"- {k}: {v}")
            if "cardiology" in data and "answer" in data["cardiology"]:
                formatted_lines.append(p["fmt_cardiology"])
                for k, v in data["cardiology"]["answer"].items():
                    formatted_lines.append(f"- {k}: {v}")

        # 4. Bóc tách Verification/Analysis
        if "verification" in data and "analysis" in data["verification"]:
            formatted_lines.append(p["fmt_verification"])
            formatted_lines.append(data["verification"]["analysis"])

        # Nếu Dict không khớp schema trên, fallback về dạng JSON string cơ bản
        if not formatted_lines:
            return json.dumps(data, ensure_ascii=False, indent=2)

        return "\n".join(formatted_lines)

    async def guardrail(self, state: MainState):
        latest_message = state["messages"][-1].content
        guardrail_llm = self.llm.with_structured_output(IsSafe, method="json_mode")
        response = await guardrail_llm.ainvoke([
            SystemMessage(content=self.p["guardrail"]),
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
                        "clinical_data": self.p["no_clinical"],
                        "report_content": self.p["no_report"]
                    }
                    
            # Đã có dữ liệu hợp lệ từ các lượt chat trước, không cần làm gì cả
            return {}

    def route_guardrail(self, state: MainState) -> str:
        if state.get("is_safe", True):
            return "rewrite_query" 
        else:
            return "unsafe_response"

    async def unsafe_response(self, state: MainState):
        bot_reply = self.p["unsafe"]
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
            SystemMessage(content=self.p["classify"]),
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
        
    async def medical_chat(self, state: MainState, config: RunnableConfig):
        history = state["messages"][:-1]
        query = state["rewrited_query"]
        
        context_blocks = []
        p = self.p
        
        # 1. Xử lý Clinical Data (Hồ sơ lâm sàng)
        clinical_data = state.get("clinical_data")
        if clinical_data:
            clinical_str = clinical_data if isinstance(clinical_data, str) else json.dumps(clinical_data, ensure_ascii=False, indent=2)
            context_blocks.append(f"{p['fmt_clinical_label']}:\n{clinical_str}")
            
        # 2. Xử lý Report Content (Sử dụng hàm bóc tách JSON)
        report_content = state.get("report_content")
        if report_content:
            report_str = self._format_medical_json(report_content)
            context_blocks.append(f"{p['fmt_report_label']}:\n{report_str}")
            
        medical_context_str = "\n\n".join(context_blocks)
        
        # 3. Ghép vào câu hỏi cuối cùng
        if medical_context_str:
            final_prompt_content = p["medical_with_context"].format(context=medical_context_str, query=query)
        else:
            final_prompt_content = query
            
        clarified_query_with_context = HumanMessage(content=final_prompt_content)
        
        result = await self.medical_agent.ainvoke(
            {"messages": history + [clarified_query_with_context]},
            config=config
        )

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