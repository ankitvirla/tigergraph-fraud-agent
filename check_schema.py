import asyncio
import json
from tigergraph_client import TigerGraphMCPClient


async def main():
    client = TigerGraphMCPClient()

    query = """
    INTERPRET QUERY () FOR GRAPH FraudInvestigationGraph {

        customers =
            SELECT c
            FROM customer:c
            LIMIT 1;

        PRINT customers;
    }
    """

    result = await client.run_query(query)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())