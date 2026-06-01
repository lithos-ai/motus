# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fields output schema — list-typed structured output with output_key.

Returns a list of WeatherData objects using ``output_schema`` and stores
the result in session state via ``output_key``.

Deploy: motus serve start examples.google_adk.fields_output_schema:root_agent
"""

from pydantic import BaseModel, Field

from motus.google_adk.agents.llm_agent import Agent


class WeatherData(BaseModel):
    """Weather data for a single city."""

    city: str = Field(description="City name")
    temperature: str = Field(description="Temperature with unit")
    condition: str = Field(description="Weather condition")
    humidity: str = Field(description="Humidity percentage")


def get_weather(city: str) -> dict:
    """Get weather data for a city.

    Args:
        city: The city name.
    """
    data = {
        "Tokyo": {"temp": "22°C", "condition": "Partly cloudy", "humidity": "65%"},
        "Paris": {"temp": "18°C", "condition": "Sunny", "humidity": "45%"},
        "London": {"temp": "12°C", "condition": "Rainy", "humidity": "80%"},
        "New York": {"temp": "15°C", "condition": "Overcast", "humidity": "55%"},
    }
    info = data.get(city, {"temp": "20°C", "condition": "Clear", "humidity": "50%"})
    return {"city": city, **info}


root_agent = Agent(
    model="gemini-2.5-flash",
    name="weather_report_agent",
    description="Generates structured weather reports for multiple cities.",
    instruction=(
        "You are a weather report agent. When asked about weather in cities, "
        "use the get_weather tool for each city, then return a structured "
        "list of WeatherData objects."
    ),
    tools=[get_weather],
    output_schema=list[WeatherData],
    output_key="weather_report",
)
