"""OAuth Device Authorization Flow for the Motus CLI."""

import logging
import os
import platform
import time
import webbrowser

import httpx

# Suppress httpx request logging during polling
logging.getLogger("httpx").setLevel(logging.WARNING)

# Motus cloud configuration (overridable via env vars for dev environments)
AUTH0_DOMAIN = os.getenv("MOTUS_AUTH0_DOMAIN", "auth.lithosai.cloud")
AUTH0_CLIENT_ID = os.getenv("MOTUS_AUTH0_CLIENT_ID", "gTZir02ty7fjpoYzIi1fSB0OtkVYfFel")
AUTH0_AUDIENCE = os.getenv("MOTUS_AUTH0_AUDIENCE", "https://api.lithosai.cloud")

# Maximum time to wait for the user to complete login
_LOGIN_TIMEOUT_SECONDS = 300


def login(api_url: str) -> dict:
    """Run device-flow login, create an API key, and return credentials.

    1. Request a device code from Auth0
    2. Open browser for user to confirm
    3. Poll Auth0 until user completes login
    4. Create an API key using the JWT
    """
    token_endpoint = f"https://{AUTH0_DOMAIN}/oauth/token"

    # Request device code
    resp = httpx.post(
        f"https://{AUTH0_DOMAIN}/oauth/device/code",
        data={
            "client_id": AUTH0_CLIENT_ID,
            "audience": AUTH0_AUDIENCE,
            "scope": "openid profile email",
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    device_code = data["device_code"]
    interval = data.get("interval", 5)

    # Open browser for user confirmation
    verification_url = data.get("verification_uri_complete") or data["verification_uri"]
    user_code = data.get("user_code", "")
    print()
    if user_code:
        print(f"  Your code: \033[1m{user_code}\033[0m")
        print()
    print(f"  {verification_url}")
    print()
    webbrowser.open(verification_url)
    print("Waiting for browser login...")

    # Poll for token
    deadline = time.monotonic() + _LOGIN_TIMEOUT_SECONDS
    access_token = None
    while time.monotonic() < deadline:
        time.sleep(interval)
        resp = httpx.post(
            token_endpoint,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": AUTH0_CLIENT_ID,
                "device_code": device_code,
            },
            timeout=10,
        )
        body = resp.json()
        if "access_token" in body:
            access_token = body["access_token"]
            break
        error = body.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        raise RuntimeError(body.get("error_description", "Login failed"))

    if not access_token:
        raise RuntimeError("Login timed out — no response within 5 minutes")

    # Create API key using the JWT
    key_name = f"cli-{platform.node()}"
    resp = httpx.post(
        f"{api_url}/api-keys",
        json={"name": key_name},
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    if resp.status_code == 403:
        raise RuntimeError(
            "You're on the waitlist. "
            "We're rolling out access gradually. You'll receive an email when your account is ready."
        )
    resp.raise_for_status()
    key_data = resp.json()

    return {
        "cloud_api_url": api_url,
        "api_key": key_data["key"],
        "key_id": key_data["key_id"],
    }
