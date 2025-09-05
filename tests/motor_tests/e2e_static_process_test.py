# Copyright 2025 Intrinsic Innovation LLC

"""
No mouse movements but just clicking process without bounding boxes.
"""
import pytest
from playwright.sync_api import Page


def test_recording(page: Page):
    page.set_viewport_size({'width': 1567, 'height': 894})
    logs: list[str] = []
    page.on("console", lambda m: logs.append(m.text))
    page.on("console", lambda m: print("[console]", m.text))
    page.goto("http://localhost:5004/?e2e", wait_until="domcontentloaded")
    page.wait_for_timeout(3000)  # initial settle
    page.mouse.move(24.625, 3.756)
    page.mouse.down()
    page.mouse.move(24.625, 3.756)
    page.mouse.up()
    page.get_by_test_id("btn-run").click()
    if any("Hulls: 1" in t for t in logs):
        pass
    else:
        page.wait_for_event("console", lambda m: "Hulls: 1" in m.text, timeout=100_000)
    assert any("Hulls: 1" in t for t in logs)
    page.mouse.move(34.775, 10.486)
    page.mouse.down()
    page.mouse.move(34.775, 10.486)
    page.mouse.up()
