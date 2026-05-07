import httpx

WEATHER_TOOL_DEFINITION = {
    "name": "get_weather",
    "description": (
        "Get current weather for a city. Returns temperature, conditions, "
        "humidity, and wind speed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'Mumbai' or 'London'",
            }
        },
        "required": ["city"],
    },
}


async def get_weather(city: str) -> dict:
    """Fetch current weather from wttr.in (no API key required)."""
    url = f"https://wttr.in/{city}?format=j1"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers={"User-Agent": "ai-engineering-journey/1.0"})
        resp.raise_for_status()
        data = resp.json()

    current = data["current_condition"][0]
    area = data["nearest_area"][0]
    area_name = area["areaName"][0]["value"]
    country = area["country"][0]["value"]

    return {
        "city": f"{area_name}, {country}",
        "temperature_c": int(current["temp_C"]),
        "temperature_f": int(current["temp_F"]),
        "feels_like_c": int(current["FeelsLikeC"]),
        "condition": current["weatherDesc"][0]["value"],
        "humidity_pct": int(current["humidity"]),
        "wind_kmph": int(current["windspeedKmph"]),
        "visibility_km": int(current["visibility"]),
        "uv_index": int(current["uvIndex"]),
    }
