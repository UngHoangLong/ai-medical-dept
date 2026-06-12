# Import Postgres Saver và Pool
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# from ...graphs.basic_chat import BasicChatGraph

router = APIRouter(prefix="/api/v1")

class ChatContext(BaseModel):
    # 1. Dữ liệu cốt lõi (Bắt buộc)
    query: str = Field(..., description="Câu hỏi hoặc tin nhắn của người dùng")
    thread_id: str = Field(..., description="ID của phiên chat để LangGraph tra cứu State")
    
    # 2. Ngữ cảnh Y khoa (Chỉ cần gửi 1 lần ở tin nhắn đầu tiên)
    patient_id: str | None = Field(None, description="Mã bệnh nhân để kéo dữ liệu lâm sàng")
    ct_scan_id: str | None = Field(None, description="Mã mẫu CT Scan đang được phân tích")
    
    # 3. Thông tin Hệ thống / Audit
    user_id: str | None = Field(None, description="Mã bác sĩ/người dùng đang thao tác (phục vụ log hệ thống)")

# 1. Kế thừa BaseModel để FastAPI nhận diện request body
class BasicChatContext(BaseModel):
    query: str
    

@router.post("/chat")
async def chat(request: Request, context: ChatContext):
    
    input_state = {"messages": [("user", context.query)]}
    main_graph = request.state.main_graph
    # Cấu hình thread_id để LangGraph chọc đúng vào DB lấy lịch sử ra
    config = {"configurable": {"thread_id": context.thread_id}}
    
    async def event_generator():
        try:
            # Chạy graph với config chứa thread_id
            async for chunk in main_graph.graph.astream(
                input_state, 
                stream_mode=['updates', 'messages'],  
                version="v2",
                config=config,
                subgraphs=True
            ):
                
                # Bóc tách thủ công từ cấu trúc từ điển của v2
                stream_mode = chunk["type"]
                chunk_data = chunk["data"]
                
                # --- XỬ LÝ EVENT UPDATE ---
                if stream_mode == "updates":
                    for node_name, node_data in chunk_data.items():
                        payload = {
                            "type": "status", 
                            "node": node_name,
                            "message": f"Đang xử lý tại: {node_name}..."
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                        
                        # ---> THÊM ĐOẠN NÀY: Xử lý riêng cho node unsafe_response <---
                        if node_name == "unsafe_response":
                            # Bóc tách nội dung message từ state update của node này
                            messages = node_data.get("messages", [])
                            if messages:
                                last_msg = messages[-1]
                                # Xử lý cho cả trường hợp object hoặc dict
                                content = getattr(last_msg, "content", None) or (last_msg.get("content", "") if isinstance(last_msg, dict) else "")
                                
                                if content:
                                    # Ép nó thành dạng 'token' để Frontend in ra màn hình
                                    payload_token = {
                                        "type": "token",
                                        "content": content
                                    }
                                    yield f"data: {json.dumps(payload_token)}\n\n"

                # --- XỬ LÝ EVENT MESSAGES (TOKEN TỪ LLM) ---
                elif stream_mode == "messages":
                    msg, metadata = chunk_data
                    if msg.__class__.__name__ == "AIMessageChunk":
                        # Giữ nguyên logic cũ của bạn
                        if metadata.get("langgraph_node") in ["synthesizer", "casual_chat"]:
                            if msg.content:
                                payload = {
                                    "type": "token",
                                    "content": msg.content
                                }
                                yield f"data: {json.dumps(payload)}\n\n"
                    
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")



# @router.post("/chat-test")
# async def chat_test(context: BasicChatContext):
#     # Khởi tạo graph cho request này
#     chat_graph = BasicChatGraph(llm=llm)
    
#     # 2. Tạo một async generator để yield dữ liệu theo format chuẩn của SSE
#     async def event_generator():
#         try:
#             # Lặp trực tiếp qua async stream thay vì nhét vào queue
#             async for chunk in chat_graph.graph.astream({"query": context.query}, stream_mode=['updates', 'messages'],  version="v2", subgraphs=True):
                
#                 # SSE bắt buộc phải có prefix "data: " và kết thúc bằng "\n\n"
#                 # Dump chunk ra chuỗi JSON để frontend dễ parse
#                 if chunk['type'] == 'messages':
#                     msg, metadata = chunk["data"]
#                     yield f"data: {msg.content}\n\n"
                
#             # Tuỳ chọn: Gửi một sự kiện báo hiệu đã stream xong để frontend ngắt kết nối
#             yield "data: [DONE]\n\n"
            
#         except Exception as e:
#             # Xử lý lỗi nếu có trong quá trình stream
#             yield f"data: {json.dumps({'error': str(e)})}\n\n"

#     # 3. Trả về StreamingResponse với media_type chuyên dụng cho SSE
#     return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/history/{thread_id}")
async def get_chat_history(request: Request, thread_id: str):
    """
    API để Frontend gọi lấy lại lịch sử chat khi load lại trang
    """
    try:
        # Lấy đồ thị từ request state (giống như lúc chat)
        main_graph = request.state.main_graph
        
        # Cấu hình chứa thread_id cần lấy
        config = {"configurable": {"thread_id": thread_id}}
        
        # Vì chúng ta xài AsyncPostgresSaver, phải dùng aget_state thay vì get_state
        state_snapshot = await main_graph.graph.aget_state(config)
        
        # Nếu thread_id này chưa từng tồn tại (user mới tinh)
        if not state_snapshot or not state_snapshot.values:
            return {"messages": []}
            
        # Lấy mảng messages thô từ State của LangGraph
        raw_messages = state_snapshot.values.get("messages", [])
        
        formatted_messages = []
        for msg in raw_messages:
            # 1. Bỏ qua các message không phải là text thông thường (như tool gọi ngầm)
            if msg.type not in ["human", "ai"]:
                continue
                
            # 2. Bỏ qua các tin nhắn AI rỗng (thường là rác sinh ra lúc AI gọi Tool)
            if msg.type == "ai" and not msg.content:
                continue
                
            # 3. Chuyển đổi định dạng role cho Frontend dễ đọc
            role = "user" if msg.type == "human" else "assistant"
            
            formatted_messages.append({
                "role": role,
                "content": msg.content
            })
            
        return {"messages": formatted_messages}

    except Exception as e:
        # Trả về lỗi 500 nếu Database có vấn đề

        raise HTTPException(status_code=500, detail=str(e))

@router.get("/threads")
async def get_all_threads(request: Request):
    """
    API lấy danh sách tất cả các phiên chat (thread_id)
    Sắp xếp theo thời gian nhắn tin mới nhất.
    """
    try:
        # Lấy hồ chứa kết nối DB từ State
        pool = request.state.db_pool
        
        # Mở một kết nối (connection) từ trong hồ (pool)
        async with pool.connection() as conn:
            # Dùng cursor để thực thi câu lệnh SQL
            async with conn.cursor() as cur:
                # Câu lệnh SQL thần thánh:
                # 1. Chỉ lấy ở checkpoint_ns rỗng (đồ thị ngoài cùng) để tránh rác từ subgraph
                # 2. Gom nhóm theo thread_id (tránh trùng lặp)
                # 3. Lấy thời gian MAX (mới nhất) của mỗi thread
                # 4. Sắp xếp giảm dần (mới nhất lên đầu)
                query = """
                    SELECT thread_id
                    FROM checkpoints
                    WHERE checkpoint_ns = ''
                    GROUP BY thread_id
                """
                
                await cur.execute(query)
                records = await cur.fetchall()
                
        # Format lại kết quả trả về cho Frontend
        threads = []
        for row in records:
            threads.append({
                "thread_id": row[0]
            })
            
        return {"threads": threads}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))