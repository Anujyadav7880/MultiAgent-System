import requests

def search_flights(query):
    url = "https://opensky-network.org/api/states/all"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        flights = []

        if "states" in data:
            for flight in data["states"][:5]:

                airline = flight[1] or "Unknown"
                departure = flight[2] or "Unknown"
                arrival = flight[3] or "Unknown"
                status = "In air"

                flights.append(
                    f"""
Airline: {airline}
Departure: {departure}
Arrival: {arrival}
Status: {status}
"""
                )

        return "\n".join(flights)

    except Exception as e:
        return f"Error: {str(e)}"