import asyncio
import os
import shutil

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    uvx = shutil.which("uvx")

    server_params = StdioServerParameters(
        command=uvx,
        args=["tigergraph-mcp"],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            query = """
            INTERPRET QUERY () FOR GRAPH FraudInvestigationGraph {
                customers =
                    SELECT c
                    FROM customer:c
                    LIMIT 5;

                PRINT customers;
            }
            """

            result = await session.call_tool(
                "tigergraph__run_query",
                {
                    "graph_name": "FraudInvestigationGraph",
                    "query_text": query,
                },
            )

            print("\n===== MCP GRAPH RESULT =====\n")

            for content in result.content:
                if hasattr(content, "text"):
                    print(content.text)
                else:
                    print(content)


if __name__ == "__main__":
    asyncio.run(main())