from __future__ import annotations
from datetime import datetime
import redis


class OnlineState:
    def __init__(self, redis_url: str):
        self.r = redis.Redis.from_url(redis_url, decode_responses=True)

    def get_last_ts(self, user_id: str) -> datetime | None:
        v = self.r.get(f"last_ts:{user_id}")
        return datetime.fromisoformat(v) if v else None

    def set_last_ts(self, user_id: str, ts: datetime) -> None:
        # keep for 30 days
        self.r.setex(f"last_ts:{user_id}", 30 * 24 * 3600, ts.isoformat())
