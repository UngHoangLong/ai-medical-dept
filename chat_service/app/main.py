from contextlib import asynccontextmanager

from app.api.v1.chat import router
from app.config import settings
from app.graphs.main_graph import MainGraph
from fastapi import FastAPI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool


# 1. QUẢN LÝ VÒNG ĐỜI SERVER (LIFESPAN)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, main_graph
    
    # Mở hồ chứa kết nối tới Azure Postgres
    pool = AsyncConnectionPool(
        conninfo=settings.DB_URI,
        max_size=20, # Phục vụ tối đa 20 lượt chat đồng thời
        kwargs={"autocommit": True, "prepare_threshold": 0}, # Cấu hình bắt buộc của LangGraph
    )
    await pool.open()
    
    # Khởi tạo Saver
    checkpointer = AsyncPostgresSaver(pool)
    
    # TỰ ĐỘNG TẠO BẢNG: Lệnh này sẽ tự tạo các bảng checkpoints, checkpoint_writes... vào DB nếu chưa có
    await checkpointer.setup() 
    
    # Khởi tạo các LLM
    llm = ChatOpenAI(model=settings.CHAT_MODEL, 
                     temperature=settings.LLM_TEMPERATURE_CHAT, 
                     stream=settings.ENABLE_STREAMING, 
                     timeout=settings.LLM_TIMEOUT, 
                     max_retries=settings.LLM_MAX_RETRIES, 
                     api_key=settings.DEEPSEEK_API_KEY)

    # Khởi tạo MCP Tools của bạn ở đây (ví dụ: search tool, database query tool...)
    mcp_client = MultiServerMCPClient({
            "medical_mcp": {
                "transport": "http", 
                "url": settings.MCP_SERVER_URL, # Đảm bảo FastMCP Docker đang chạy ở đây
            }
        })
    
    try:
        # Lấy danh sách tools từ Server (Không dùng async with theo chuẩn bản cập nhật mới)
        mcp_tools = await mcp_client.get_tools()
        print(f"✅ Đã tải thành công {len(mcp_tools)} tools từ hệ thống Y khoa (FDA, EHR...).")
    except Exception as e:
        print(f"❌ Lỗi kết nối MCP Server: {e}")
        mcp_tools = [] # Fallback nếu server sập
        
    # TRUYỀN CHECKPOINTER, LLM, MCP_TOOLS VÀO MAINGRAPH
    main_graph = MainGraph(llm=llm, mcp_tools=mcp_tools, checkpointer=checkpointer)
    
    print("✅ Database & LangGraph đã sẵn sàng!")
    
    yield {"main_graph": main_graph,
           "db_pool": pool}
    
    # Khi tắt Server thì đóng kết nối
    await pool.close()
    print("🛑 Đã đóng kết nối Database.")
    
# Đưa lifespan vào FastAPI
app = FastAPI(title="LangGraph Agent Backend", lifespan=lifespan)

# Gắn Router
app.include_router(router)