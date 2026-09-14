"""
keep_awake.py
-------------
Visits a Streamlit Community Cloud app the way a real browser would, so
its activity timer resets and it doesn't go to sleep after 12 hours of
inactivity.

IMPORTANT: a plain HTTP GET (curl, requests, urllib) does NOT work for
this. Streamlit Cloud serves a static HTML/JS shell to any raw HTTP
request and returns 200 OK -- without ever starting the actual app
container -- so the inactivity timer never resets and monitoring tools
get a false "healthy" signal. Only a real browser session (JavaScript
executing, a WebSocket connection opening to the backend) counts as
genuine traffic. Hence Playwright (a real headless browser) instead of
a simple HTTP client.
"""

import os
import sys

from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("STREAMLIT_APP_URL", "").strip()

if not APP_URL:
    print("STREAMLIT_APP_URL is not set.", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        print(f"Visiting {APP_URL} ...")
        page.goto(APP_URL, wait_until="networkidle", timeout=60_000)

        # If the app was asleep, Streamlit shows a wake-up screen with a
        # button. Click it if present, then wait for the real app to load.
        try:
            wake_button = page.get_by_text("get this app back up", exact=False)
            if wake_button.is_visible(timeout=5_000):
                print("App was asleep -- clicking wake button ...")
                wake_button.click()
        except Exception:
            pass  # button not present -- app was likely already awake

        # Wait for the actual Streamlit app content to render, which
        # confirms the container is running (not just the static shell).
        try:
            page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=120_000)
            print("App is awake and rendered successfully.")
        except Exception as exc:
            print(f"Warning: app content did not confirm-load in time: {exc}", file=sys.stderr)

        browser.close()


if __name__ == "__main__":
    main()
