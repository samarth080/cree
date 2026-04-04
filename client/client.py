"""
CREEClient — HTTP wrapper around the environment server.

Mirrors the OpenEnv / OpenSpiel client style:
  client.reset()          → ObservableState
  client.step(action)     → StepResult
  client.list_actions()   → List[dict]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from typing import List, Dict
from models import ObservableState, StepResult


class CREEClient:

    def __init__(self, server_url: str = "http://localhost:8000", timeout: int = 10):
        self.server_url = server_url.rstrip('/')
        self.timeout    = timeout

    def reset(self) -> ObservableState:
        resp = requests.post(f"{self.server_url}/reset", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return ObservableState(**data["state"])

    def step(self, action: str) -> StepResult:
        resp = requests.post(
            f"{self.server_url}/step",
            json={"action": action},
            timeout=self.timeout,
        )
        if resp.status_code == 400:
            raise ValueError(resp.json().get("detail", "Bad request"))
        resp.raise_for_status()
        data = resp.json()
        return StepResult(
            state=ObservableState(**data["state"]),
            reward=data["reward"],
            done=data["done"],
            info=data["info"],
        )

    def get_state(self) -> ObservableState:
        resp = requests.get(f"{self.server_url}/state", timeout=self.timeout)
        resp.raise_for_status()
        return ObservableState(**resp.json()["state"])

    def list_actions(self) -> List[Dict]:
        resp = requests.get(f"{self.server_url}/actions", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["actions"]

    def health(self) -> dict:
        resp = requests.get(f"{self.server_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
