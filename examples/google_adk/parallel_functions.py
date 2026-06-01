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

"""Parallel functions — multiple async tools invoked concurrently.

Demonstrates an agent with several async tools that the model can call
in parallel. Each tool simulates a network call with a delay. The ADK
runner executes parallel tool calls concurrently via asyncio.gather.

Deploy: motus serve start examples.google_adk.parallel_functions:root_agent
"""

import asyncio

from motus.google_adk.agents.llm_agent import Agent


async def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: The city name.
    """
    await asyncio.sleep(0.1)  # Simulate network call
    weather_data = {
        "Tokyo": {"temp": "22°C", "condition": "Partly cloudy"},
        "Paris": {"temp": "18°C", "condition": "Sunny"},
        "New York": {"temp": "15°C", "condition": "Overcast"},
        "London": {"temp": "12°C", "condition": "Rainy"},
    }
    data = weather_data.get(city, {"temp": "20°C", "condition": "Clear"})
    return {"city": city, **data}


async def get_population(city: str) -> dict:
    """Get the approximate population of a city.

    Args:
        city: The city name.
    """
    await asyncio.sleep(0.1)
    populations = {
        "Tokyo": "14 million",
        "Paris": "2.2 million",
        "New York": "8.3 million",
        "London": "9 million",
    }
    return {"city": city, "population": populations.get(city, "unknown")}


async def get_timezone(city: str) -> dict:
    """Get the timezone of a city.

    Args:
        city: The city name.
    """
    await asyncio.sleep(0.1)
    timezones = {
        "Tokyo": "JST (UTC+9)",
        "Paris": "CET (UTC+1)",
        "New York": "EST (UTC-5)",
        "London": "GMT (UTC+0)",
    }
    return {"city": city, "timezone": timezones.get(city, "unknown")}


async def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between currencies.

    Args:
        amount: The amount to convert.
        from_currency: Source currency code (e.g. USD, EUR, GBP, JPY).
        to_currency: Target currency code.
    """
    await asyncio.sleep(0.1)
    rates = {
        ("USD", "EUR"): 0.92,
        ("EUR", "USD"): 1.09,
        ("USD", "GBP"): 0.79,
        ("GBP", "USD"): 1.27,
        ("USD", "JPY"): 149.5,
        ("JPY", "USD"): 0.0067,
        ("EUR", "GBP"): 0.86,
        ("GBP", "EUR"): 1.16,
    }
    rate = rates.get((from_currency, to_currency), 1.0)
    return {
        "from": f"{amount} {from_currency}",
        "to": f"{round(amount * rate, 2)} {to_currency}",
        "rate": rate,
    }


root_agent = Agent(
    model="gemini-2.5-flash",
    name="city_explorer",
    description="Gathers information about cities using multiple tools in parallel.",
    instruction=(
        "You are a city information assistant with access to weather, population, "
        "timezone, and currency conversion tools. When asked about a city, use "
        "multiple tools at once to gather comprehensive information. When comparing "
        "cities, call tools for all cities in parallel."
    ),
    tools=[get_weather, get_population, get_timezone, convert_currency],
)
