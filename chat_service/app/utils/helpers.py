import os


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
