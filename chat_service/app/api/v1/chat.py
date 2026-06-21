import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1")

# Payload từ FE gọn gàng hơn
class ChatContext(BaseModel):
    query: str = Field(..., description="Câu hỏi của người dùng")
    report_id: str = Field(..., description="Vừa dùng để query DB, vừa làm thread_id cho LangGraph")
    user_id: str | None = Field(None, description="Mã bác sĩ (phục vụ log audit)")

@router.post("/chat")
async def chat(request: Request, context: ChatContext):
    
    input_state = {
        "messages": [("user", context.query)],
        "report_id": context.report_id
    }
    
    main_graph = request.state.main_graph
    db_pool = request.state.db_pool
    config = {"configurable": {"thread_id": context.report_id, "db_pool": db_pool}}
    
    async def event_generator():
        try:
            # =========================================================
            # TRICK: Gửi 1KB "dữ liệu giả" ngay lập tức để ép Proxy mở van.
            # Dấu ":" ở đầu báo hiệu đây là SSE Comment, FE sẽ lờ nó đi.
            # =========================================================
            dummy_padding = ":" + " " * 8096 + "\n\n"
            yield dummy_padding

            # Bắt đầu chạy LangGraph
            async for chunk in main_graph.graph.astream(
                input_state, 
                stream_mode=['updates', 'messages'],  
                version="v2",
                config=config,
                subgraphs=True 
            ):
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
                        
                        # Xử lý riêng cho node unsafe_response
                        if node_name == "unsafe_response":
                            messages = node_data.get("messages", [])
                            if messages:
                                last_msg = messages[-1]
                                content = getattr(last_msg, "content", None) or (last_msg.get("content", "") if isinstance(last_msg, dict) else "")
                                
                                if content:
                                    payload_token = {
                                        "type": "token",
                                        "content": content
                                    }
                                    yield f"data: {json.dumps(payload_token)}\n\n"

                # --- XỬ LÝ EVENT MESSAGES (TOKEN TỪ LLM) ---
                elif stream_mode == "messages":
                    msg, metadata = chunk_data
                    
                    valid_nodes = ["casual_chat", "medical_chat", "model"] 
                    
                    if msg.__class__.__name__ == "AIMessageChunk":
                        is_tool_call = hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0
                        
                        if metadata.get("langgraph_node") in valid_nodes and not is_tool_call:
                            if msg.content:
                                payload = {
                                    "type": "token",
                                    "content": msg.content
                                }
                                yield f"data: {json.dumps(payload)}\n\n"
                
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            
    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Content-Type": "text/event-stream",
        "Content-Encoding": "identity",
        "Connection": "close"
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


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
                    FROM public.checkpoints
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

@router.delete("/history/{thread_id}")
async def delete_chat_history(request: Request, thread_id: str):
    """
    API xóa lịch sử chat của một phiên (thread_id) cụ thể
    """
    try:
        pool = request.state.db_pool
        
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                # Xoá dữ liệu tương ứng với thread_id trong các bảng LangGraph Checkpoint
                # (Không xoá checkpoint_migrations vì bảng đó dùng để quản lý schema database)
                await cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                await cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                await cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
                
        return {"status": "success", "message": f"Đã xóa thành công lịch sử chat của thread_id: {thread_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))