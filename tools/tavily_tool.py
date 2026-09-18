from tavily import TavilyClient
import os
import time
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


def tavily_search(query):

    for attempt in range(3):

        try:
            response = client.search(
                query=query,
                max_results=5,
                search_depth="basic"
            )

            results = []

            for i, r in enumerate(response.get("results", []), 1):

                title = r.get("title", "Unknown")
                url = r.get("url", "")
                snippet = r.get("content", "").strip()

                if len(snippet) > 300:
                    snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

                results.append(
                    f"{i}. **{title}**\n"
                    f"{url}\n"
                    f"{snippet}"
                )

            return "\n\n".join(results)

        except Exception as e:

            print(
                f"Tavily attempt {attempt + 1} failed: {e}"
            )

            if attempt < 2:
                time.sleep(2)

    return "Unable to fetch hotel information from Tavily." 