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

"""Pydantic argument — tools that accept Pydantic models as args.

Demonstrates tools with complex typed arguments: Pydantic models with
Optional and nested fields. The ADK automatically converts JSON dicts
from the LLM into proper Pydantic instances.

Deploy: motus serve start examples.google_adk.pydantic_argument:root_agent
"""

from typing import Optional

from pydantic import BaseModel

from motus.google_adk.agents.llm_agent import Agent


class Address(BaseModel):
    """A street address."""

    street: str
    city: str
    country: str = "US"


class UserProfile(BaseModel):
    """A user profile with optional address."""

    name: str
    email: str
    age: Optional[int] = None
    address: Optional[Address] = None


def register_user(profile: UserProfile) -> dict:
    """Register a user in the system.

    Args:
        profile: The user profile to register.
    """
    return {
        "status": "registered",
        "type": "user",
        "name": profile.name,
        "email": profile.email,
        "has_address": profile.address is not None,
    }


def register_company(company_name: str, industry: str, employee_count: int) -> dict:
    """Register a company in the system.

    Args:
        company_name: The company name.
        industry: The industry sector.
        employee_count: Number of employees.
    """
    return {
        "status": "registered",
        "type": "company",
        "name": company_name,
        "industry": industry,
        "employees": employee_count,
    }


def lookup_entity(name: str) -> dict:
    """Look up a registered entity by name.

    Args:
        name: The name to search for.
    """
    return {"found": True, "name": name, "type": "user", "status": "active"}


root_agent = Agent(
    model="gemini-2.5-flash",
    name="registration_agent",
    description="Registers users and companies with structured data.",
    instruction=(
        "You are a registration assistant. Help users register people "
        "(register_user) or companies (register_company). Ask for required "
        "fields if not provided. Use lookup_entity to check existing records."
    ),
    tools=[register_user, register_company, lookup_entity],
)
