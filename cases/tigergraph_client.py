import asyncio
import json
import os
import re
import shutil
from typing import Any


GRAPH_NAME = "FraudInvestigationGraph"


class TigerGraphMCPClient:

    def __init__(self):
        self.uvx = shutil.which("uvx")

        if not self.uvx:
            raise RuntimeError("uvx not found")

        self.server_params = {
            "command": self.uvx,
            "args": ["tigergraph-mcp"],
            "env": os.environ.copy(),
        }

    async def run_query(self, query_text: str) -> dict[str, Any]:

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        import re
        import json

        params = StdioServerParameters(
            command=self.server_params["command"],
            args=self.server_params["args"],
            env=self.server_params["env"],
        )

        async with stdio_client(params) as (read, write):

            async with ClientSession(read, write) as session:

                await session.initialize()

                result = await session.call_tool(
                    "tigergraph__run_query",
                    {
                        "graph_name": GRAPH_NAME,
                        "query_text": query_text,
                    },
                )

                output = []

                for content in result.content:

                    if hasattr(content, "text"):
                        output.append(content.text)

                    else:
                        output.append(str(content))

                # ---------------------------------------------------------
                # MCP returns a Markdown response containing one or more
                # fenced JSON blocks plus explanatory text.
                #
                # Example:
                #
                # ```json
                # {
                #   "success": true,
                #   ...
                # }
                # ```
                #
                # **Success: GSQL query executed successfully**
                #
                # ```json
                # {
                #   "query_type": "GSQL",
                #   ...
                # }
                # ```
                #
                # We only need the FIRST JSON block.
                # ---------------------------------------------------------

                for item in output:

                    if not isinstance(item, str):
                        continue

                    # Look specifically for ```json ... ```
                    match = re.search(
                        r"```json\s*(.*?)\s*```",
                        item,
                        flags=re.DOTALL | re.IGNORECASE,
                    )

                    if match:

                        json_text = match.group(1).strip()

                        try:

                            parsed = json.loads(json_text)

                            if isinstance(parsed, dict):
                                return parsed

                        except json.JSONDecodeError:
                            continue

                    # -----------------------------------------------------
                    # Fallback: maybe MCP returns plain JSON without fences
                    # -----------------------------------------------------

                    cleaned = item.strip()

                    try:

                        parsed = json.loads(cleaned)

                        if isinstance(parsed, dict):
                            return parsed

                    except json.JSONDecodeError:
                        continue

                return {
                    "success": False,
                    "error": "Could not parse MCP response as JSON",
                    "raw_output": output,
                }

    async def get_schema(self) -> dict[str, Any]:

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.server_params["command"],
            args=self.server_params["args"],
            env=self.server_params["env"],
        )

        async with stdio_client(params) as (read, write):

            async with ClientSession(read, write) as session:

                await session.initialize()

                result = await session.call_tool(
                    "tigergraph__get_graph_schema",
                    {
                        "graph_name": GRAPH_NAME,
                    },
                )

                output = []

                for content in result.content:
                    if hasattr(content, "text"):
                        output.append(content.text)
                    else:
                        output.append(str(content))

                for item in output:
                    try:
                        return json.loads(item)
                    except (json.JSONDecodeError, TypeError):
                        continue

                return {
                    "success": False,
                    "raw_output": output,
                }

    @staticmethod
    def _validate_id(value: str, name: str) -> str:

        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError(f"Invalid {name}: {value}")

        return value

    async def get_customer_profile(
        self,
        customer_id: str,
    ) -> dict[str, Any]:

        customer_id = self._validate_id(
            customer_id,
            "customer_id",
        )

        query = f"""
        INTERPRET QUERY () FOR GRAPH {GRAPH_NAME} {{

            customers =
                SELECT c
                FROM customer:c
                WHERE c.customer_id == "{customer_id}";

            cards =
                SELECT crd
                FROM customer:c -(USES_CARD)-> Card:crd
                WHERE c.customer_id == "{customer_id}";

            devices =
                SELECT dev
                FROM customer:c -(USES_DEVICE)-> Device:dev
                WHERE c.customer_id == "{customer_id}";

            cases =
                SELECT fc
                FROM customer:c -(HAS_CASE)-> FraudCase:fc
                WHERE c.customer_id == "{customer_id}";

            transactions =
                SELECT txn
                FROM customer:c -(HAS_TRANSACTION)-> Transaction:txn
                WHERE c.customer_id == "{customer_id}";

            PRINT customers;
            PRINT cards;
            PRINT devices;
            PRINT cases;
            PRINT transactions;
        }}
        """

        return await self.run_query(query)

    async def get_shared_device_customers(
        self,
        customer_id: str,
    ) -> dict[str, Any]:

        customer_id = self._validate_id(
            customer_id,
            "customer_id",
        )

        query = f"""
        INTERPRET QUERY () FOR GRAPH {GRAPH_NAME} {{

            target_devices =
                SELECT d
                FROM customer:c -(USES_DEVICE)-> Device:d
                WHERE c.customer_id == "{customer_id}";

            shared_customers =
                SELECT other
                FROM target_devices:d
                    -(reverse_USES_DEVICE)-> customer:other
                WHERE other.customer_id != "{customer_id}";

            PRINT target_devices;
            PRINT shared_customers;
        }}
        """

        return await self.run_query(query)

    async def get_transaction_context(
        self,
        transaction_id: str,
    ) -> dict[str, Any]:

        transaction_id = self._validate_id(
            transaction_id,
            "transaction_id",
        )

        query = f"""
        INTERPRET QUERY () FOR GRAPH {GRAPH_NAME} {{

            target_transaction =
                SELECT t
                FROM Transaction:t
                WHERE t.transaction_id == "{transaction_id}";

            related_customer =
                SELECT c
                FROM target_transaction:t
                    -(reverse_HAS_TRANSACTION)-> customer:c;

            related_card =
                SELECT crd
                FROM related_customer:c -(USES_CARD)-> Card:crd;

            related_device =
                SELECT d
                FROM related_customer:c -(USES_DEVICE)-> Device:d;

            historical_cases =
                SELECT fc
                FROM related_customer:c -(HAS_CASE)-> FraudCase:fc;

            containing_cases =
                SELECT fc
                FROM target_transaction:t
                    -(reverse_CONTAINS)-> FraudCase:fc;

            PRINT target_transaction;
            PRINT related_customer;
            PRINT related_card;
            PRINT related_device;
            PRINT historical_cases;
            PRINT containing_cases;
        }}
        """

        return await self.run_query(query)


async def main():

    client = TigerGraphMCPClient()

    print("\n" + "=" * 70)
    print("TEST 1: CUSTOMER PROFILE")
    print("=" * 70)

    result = await client.get_customer_profile("C00001")

    print(json.dumps(result, indent=2))

    print("\n" + "=" * 70)
    print("TEST 2: SHARED DEVICE CUSTOMERS")
    print("=" * 70)

    result = await client.get_shared_device_customers("C00001")

    print(json.dumps(result, indent=2))

    print("\n" + "=" * 70)
    print("TEST 3: TRANSACTION CONTEXT")
    print("=" * 70)

    result = await client.get_transaction_context("3082613")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())