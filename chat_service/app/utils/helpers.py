import os

from psycopg_pool import AsyncConnectionPool


def get_system_prompt(agent_type: str) -> str:
    """
    Đọc file txt tương ứng với agent_type và trả về nội dung system prompt.
    agent_type: tên agent, ví dụ: 'orchestrator', 'worker_writer'
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(base_dir, '../prompts')
    file_map = {
        'planner': 'planner.txt',
        'worker': 'worker.txt',
        'evaluator_enough_context_agent': 'evaluator_enough_context_agent.txt',
        'generate_query_agent': 'generate_query_agent.txt',
        'synthesizer': 'synthesizer_agent.txt'
    }
    filename = file_map.get(agent_type)
    if not filename:
        raise ValueError(f"Unknown agent_type: {agent_type}")
    file_path = os.path.join(prompts_dir, filename)
    with open(file_path, encoding='utf-8') as f:
        return f.read().strip()


async def fetch_medical_context(pool: AsyncConnectionPool, report_id: str) -> dict | None:
    """
    Truy vấn DB để lấy nội dung khám lâm sàng và kết quả báo cáo CT.
    
    Args:
        pool: Hồ chứa kết nối database (AsyncConnectionPool)
        report_id: UUID của báo cáo cần lấy
        
    Returns:
        Dict chứa clinical_text và report_content, hoặc None nếu không tìm thấy.
    """
    
    # Câu lệnh SQL JOIN 2 bảng dựa trên record_id
    query = """
        SELECT 
            c.clinical_text, 
            r.report_content
        FROM 
            ai_demo.consultation_reports r
        JOIN 
            ai_demo.clinical_data c ON r.record_id = c.record_id
        WHERE 
            r.report_id = %s;
    """
    
    try:
        # Mở kết nối từ pool
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                # Thực thi query an toàn (chống SQL Injection)
                await cur.execute(query, (report_id,))
                
                # Lấy dòng đầu tiên (vì report_id là Unique/PK)
                result = await cur.fetchone()
                
                if result:
                    # result trả về dạng tuple (clinical_text, report_content)
                    return {
                        "clinical_text": result[0],
                        "report_content": result[1]
                    }
                else:
                    print(f"⚠️ Không tìm thấy dữ liệu cho report_id: {report_id}")
                    return None
                    
    except Exception as e:
        print(f"❌ Lỗi khi truy vấn Database: {e}")
        # Tùy logic hệ thống, bạn có thể raise lỗi hoặc return None
        return None