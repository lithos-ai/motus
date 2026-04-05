"""Structured output — Pydantic schema with tools.

Combines a Pydantic ``output_schema`` with function tools so the agent
gathers information via tools and returns a structured JSON response
conforming to the schema.

Deploy: motus serve start examples.google_adk.structured_output:root_agent
"""

from pydantic import BaseModel, Field

from motus.google_adk.agents.llm_agent import Agent


class CityInfo(BaseModel):
    """Structured information about a city."""

    name: str = Field(description="The city name")
    country: str = Field(description="The country the city is in")
    population: str = Field(description="Approximate population")
    weather: str = Field(description="Current weather conditions")
    fun_fact: str = Field(description="An interesting fact about the city")


def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: The city name.
    """
    # Mock weather data
    weather_data = {
        "Tokyo": {"temp": "22°C", "condition": "Partly cloudy"},
        "Paris": {"temp": "18°C", "condition": "Sunny"},
        "New York": {"temp": "15°C", "condition": "Overcast"},
    }
    data = weather_data.get(city, {"temp": "20°C", "condition": "Clear"})
    return {"city": city, **data}


def get_population(city: str) -> dict:
    """Get the approximate population of a city.

    Args:
        city: The city name.
    """
    populations = {
        "Tokyo": "14 million",
        "Paris": "2.2 million",
        "New York": "8.3 million",
    }
    return {"city": city, "population": populations.get(city, "unknown")}


root_agent = Agent(
    model="gemini-2.5-flash",
    name="city_info_agent",
    description="Gathers structured city information using tools.",
    instruction=(
        "You are a city information agent. When asked about a city, use the "
        "get_weather and get_population tools to gather data, then return "
        "a structured CityInfo response. Include a fun fact from your knowledge."
    ),
    tools=[get_weather, get_population],
    output_schema=CityInfo,
)
