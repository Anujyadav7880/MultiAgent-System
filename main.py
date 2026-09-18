import os
from typing import TypedDict, Annotated
import operator

import psycopg
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. LLM INITIALIZATION
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# 3. DATABASE
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")


# ============================================================
# 4. STATE DEFINITION
# ============================================================

class TravelState(TypedDict):

    # Original user message
    messages: Annotated[list[AnyMessage], operator.add]

    # Raw user query
    user_query: str

    # Structured travel information
    source_city: str
    destination_city: str
    duration_days: int
    budget: str
    preferences: str

    # Agent outputs
    flight_results: str
    hotel_results: str
    itinerary: str

    # Number of LLM calls
    llm_calls: int


# ============================================================
# 5. REQUEST PROCESSING AGENT
# ============================================================

def request_processing_agent(state: TravelState):

    user_query = state["user_query"]

    prompt = f"""
You are a travel request parser.

Read the following user request:

{user_query}

Extract the following information:

1. source_city
2. destination_city
3. duration_days
4. budget
5. preferences

IMPORTANT RULES:

- Do NOT invent information.
- If budget is not mentioned, return "Not specified".
- If preferences are not mentioned, return "Not specified".
- duration_days must be an integer.
- Extract the cities exactly from the user's request.
- Return ONLY the requested format.

Required format:

source_city: ...
destination_city: ...
duration_days: ...
budget: ...
preferences: ...
"""

    response = llm.invoke([
        SystemMessage(
            content=(
                "You are a precise travel request parser. "
                "Extract only information provided by the user."
            )
        ),
        HumanMessage(content=prompt)
    ])

    text = response.content

    # --------------------------------------------------------
    # Parse LLM response
    # --------------------------------------------------------

    data = {}

    for line in text.splitlines():

        if ":" in line:

            key, value = line.split(":", 1)

            key = key.strip().lower()
            value = value.strip()

            data[key] = value

    # --------------------------------------------------------
    # Duration conversion
    # --------------------------------------------------------

    try:
        duration_days = int(data.get("duration_days", "0"))
    except ValueError:
        duration_days = 0

    # --------------------------------------------------------
    # Return structured state
    # --------------------------------------------------------

    return {

        "source_city": data.get(
            "source_city",
            ""
        ),

        "destination_city": data.get(
            "destination_city",
            ""
        ),

        "duration_days": duration_days,

        "budget": data.get(
            "budget",
            "Not specified"
        ),

        "preferences": data.get(
            "preferences",
            "Not specified"
        ),

        "messages": [
            AIMessage(
                content="Travel request processed successfully."
            )
        ],

        "llm_calls": state.get(
            "llm_calls",
            0
        ) + 1
    }


# ============================================================
# 6. FLIGHT AGENT
# ============================================================

def flight_agent(state: TravelState):

    source = state["source_city"]
    destination = state["destination_city"]

    # Create a clean route query
    query = f"{source} to {destination}"

    print("\n[Flight Agent]")
    print("Searching:", query)

    flight_data = search_flights(query)

    return {

        "flight_results": flight_data,

        "messages": [
            AIMessage(
                content="Flight results fetched."
            )
        ],

        "llm_calls": state.get(
            "llm_calls",
            0
        ) + 1
    }


# ============================================================
# 7. HOTEL AGENT
# ============================================================

def hotel_agent(state: TravelState):

    destination = state["destination_city"]
    budget = state["budget"]

    # Search ONLY for hotels in destination
    query = f"hotels in {destination}"

    if budget != "Not specified":
        query += f" {budget}"

    print("\n[Hotel Agent]")
    print("Searching:", query)

    hotel_results = tavily_search(query)

    return {

        "hotel_results": hotel_results,

        "messages": [
            AIMessage(
                content="Hotel information fetched."
            )
        ],

        "llm_calls": state.get(
            "llm_calls",
            0
        ) + 1
    }


# ============================================================
# 8. ITINERARY AGENT
# ============================================================

def itinerary_agent(state: TravelState):

    source = state["source_city"]
    destination = state["destination_city"]
    duration = state["duration_days"]
    budget = state["budget"]
    preferences = state["preferences"]

    flight_results = state["flight_results"]
    hotel_results = state["hotel_results"]

    prompt = f"""
You are an expert travel planner.

Create a {duration}-day travel itinerary.

TRAVEL INFORMATION
------------------

Starting City:
{source}

Destination:
{destination}

Duration:
{duration} days

Budget:
{budget}

Preferences:
{preferences}


TRANSPORT INFORMATION
---------------------

{flight_results}


HOTEL INFORMATION
-----------------

{hotel_results}


IMPORTANT RULES
---------------

1. Create exactly {duration} days.

2. The trip starts from {source}.

3. The destination is {destination}.

4. Do NOT assume that the user is arriving in the starting city.

5. Do NOT invent a budget.

6. If budget is "Not specified", simply provide reasonable options
   without claiming a specific budget.

7. Do NOT say that a flight/train/hotel has been booked.

8. Do NOT claim "reservation confirmed".

9. Use phrases such as:
   - Suggested hotel
   - Recommended transport
   - Example option

10. Do not include unrelated flight or hotel information.

11. Do not recommend closed or obviously unavailable attractions
    if the provided information indicates that they are closed.

12. Keep the itinerary practical and easy to understand.

13. Include travel from the starting city to the destination
    at an appropriate point in the itinerary.

Return a clean day-by-day itinerary.
"""

    response = llm.invoke([

        SystemMessage(
            content="You are an expert and factual travel planner."
        ),

        HumanMessage(
            content=prompt
        )
    ])

    return {

        "itinerary": response.content,

        "messages": [
            response
        ],

        "llm_calls": state.get(
            "llm_calls",
            0
        ) + 1
    }


# ============================================================
# 9. FINAL AGENT
# ============================================================

def final_agent(state: TravelState):

    source = state["source_city"]
    destination = state["destination_city"]
    duration = state["duration_days"]
    budget = state["budget"]
    preferences = state["preferences"]

    flights = state["flight_results"]
    hotels = state["hotel_results"]
    itinerary = state["itinerary"]

    final_prompt = f"""
You are the final travel planning assistant.

Combine the information from the different agents and produce
ONE clean final travel plan.

TRIP DETAILS
============

Starting City:
{source}

Destination:
{destination}

Duration:
{duration} days

Budget:
{budget}

Preferences:
{preferences}


FLIGHT / TRANSPORT INFORMATION
==============================

{flights}


HOTEL INFORMATION
=================

{hotels}


ITINERARY
=========

{itinerary}


IMPORTANT RULES
===============

1. Give ONE final response.

2. Do NOT repeat the itinerary.

3. Do NOT repeat the same hotel or transport information.

4. Do NOT invent booking confirmations.

5. Never say:
   "Ticket booked"
   "Hotel booked"
   "Reservation confirmed"

   unless actual booking information is provided.

6. If budget is not specified, clearly write:
   "Budget: Not specified"

7. Only include information relevant to:
   {source} → {destination}

8. Remove unrelated flight information.

9. Remove unrelated hotel information.

10. Keep the final response organized.

Use this structure:

# {duration}-Day {source} to {destination} Trip

## Trip Details

Starting City:
Destination:
Duration:
Budget:
Preferences:

## Transportation

Give relevant options.

## Hotel Suggestions

Give relevant hotel suggestions.

## Day-by-Day Itinerary

Day 1:
Day 2:
...

## Important Notes

Mention that prices, availability and schedules should be
verified before booking.
"""

    response = llm.invoke([

        SystemMessage(
            content=(
                "You are the final travel response generator. "
                "Be accurate, concise and avoid unsupported claims."
            )
        ),

        HumanMessage(
            content=final_prompt
        )
    ])

    return {

        "messages": [
            response
        ],

        "llm_calls": state.get(
            "llm_calls",
            0
        ) + 1
    }


# ============================================================
# 10. CREATE GRAPH
# ============================================================

graph = StateGraph(TravelState)


# ============================================================
# 11. ADD NODES
# ============================================================

graph.add_node(
    "request_processing_agent",
    request_processing_agent
)

graph.add_node(
    "flight_agent",
    flight_agent
)

graph.add_node(
    "hotel_agent",
    hotel_agent
)

graph.add_node(
    "itinerary_agent",
    itinerary_agent
)

graph.add_node(
    "final_agent",
    final_agent
)


# ============================================================
# 12. GRAPH FLOW
# ============================================================

graph.add_edge(
    START,
    "request_processing_agent"
)

graph.add_edge(
    "request_processing_agent",
    "flight_agent"
)

graph.add_edge(
    "flight_agent",
    "hotel_agent"
)

graph.add_edge(
    "hotel_agent",
    "itinerary_agent"
)

graph.add_edge(
    "itinerary_agent",
    "final_agent"
)

graph.add_edge(
    "final_agent",
    END
)


# ============================================================
# 13. DATABASE CHECKPOINTER
# ============================================================

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True
)

checkpointer = PostgresSaver(
    _conn
)

checkpointer.setup()


# ============================================================
# 14. COMPILE GRAPH
# ============================================================

app = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# 15. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "user_aarohi"
        }
    }

    print("=" * 60)
    print("       MULTI-AGENT TRAVEL PLANNING SYSTEM")
    print("=" * 60)

    user_input = input(
        "\nEnter travel request: "
    )

    result = app.invoke(

        {
            "messages": [
                HumanMessage(
                    content=user_input
                )
            ],

            "user_query": user_input,

            "source_city": "",

            "destination_city": "",

            "duration_days": 0,

            "budget": "",

            "preferences": "",

            "flight_results": "",

            "hotel_results": "",

            "itinerary": "",

            "llm_calls": 0
        },

        config=config
    )


    # ========================================================
    # PRINT STRUCTURED REQUEST
    # ========================================================

    print("\n" + "=" * 60)
    print("PROCESSED TRAVEL REQUEST")
    print("=" * 60)

    print(
        "Starting City :",
        result["source_city"]
    )

    print(
        "Destination   :",
        result["destination_city"]
    )

    print(
        "Duration      :",
        result["duration_days"],
        "days"
    )

    print(
        "Budget        :",
        result["budget"]
    )

    print(
        "Preferences   :",
        result["preferences"]
    )


    # ========================================================
    # PRINT FINAL RESPONSE
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL RESPONSE")
    print("=" * 60)

    if result["messages"]:

        print(
            result["messages"][-1].content
        )