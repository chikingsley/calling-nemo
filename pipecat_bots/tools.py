"""
Tool definitions for dual-agent bot.

Tools are defined here to keep the bot file clean and organized.
Each tool describes what the background Gemini agent can do.
"""

GEMINI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_information",
            "description": "Search for information on a given topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": "Analyze data and provide insights",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "The data to analyze",
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis: 'summary', 'statistics', 'trends'",
                        "enum": ["summary", "statistics", "trends"],
                    },
                },
                "required": ["data"],
            },
        },
    },
]


async def execute_tool(tool_name: str, **kwargs) -> str:
    """
    Execute a tool and return the result.

    Args:
        tool_name: Name of the tool to execute
        **kwargs: Tool arguments

    Returns:
        Tool result as string
    """
    if tool_name == "get_current_time":
        from datetime import datetime

        return datetime.now().isoformat()

    elif tool_name == "search_information":
        query = kwargs.get("query", "")
        # Placeholder - in real usage this would call a search API
        return f"Search results for '{query}': [Information would be fetched here]"

    elif tool_name == "analyze_data":
        data = kwargs.get("data", "")
        analysis_type = kwargs.get("analysis_type", "summary")
        return f"Analysis ({analysis_type}) of provided data: [Results would be generated here]"

    else:
        return f"Unknown tool: {tool_name}"
