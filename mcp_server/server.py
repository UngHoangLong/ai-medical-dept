from fastmcp import FastMCP
from custom_tools.fda_requests import get_fda_drug_interactions, get_fda_drug_safety_profile
mcp = FastMCP("AI-Medical-Dept")
    

@mcp.tool()
async def get_drug_interactions(drugs: list) -> str:
    """Get FDA drug interaction information."""
    return get_fda_drug_interactions(drugs)

@mcp.tool()
async def get_drug_safety_profile(drugs: list) -> dict | str:
    """Get drug safety profiles."""
    return get_fda_drug_safety_profile(drugs)

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)