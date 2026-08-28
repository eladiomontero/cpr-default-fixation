import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from threading import Lock
from typing import Optional

import litellm


### caching logic
class SQLiteCache:
    """
    Persistent key-value store backed by SQLite.
    Thread-safe via a per-connection mutex.
    """

    _CREATE = """
        CREATE TABLE IF NOT EXISTS cache (
            key       TEXT PRIMARY KEY,
            value     TEXT NOT NULL,
            created   REAL NOT NULL,
            hits      INTEGER DEFAULT 0
        )
    """

    def __init__(self, db_path: str = "llm_cache.db", ttl: Optional[float] = None):
        """
        Parameters
        ----------
        db_path : str
            Path to the SQLite database file.
        ttl : float | None
            Time-to-live in seconds. ``None`` means entries never expire.
        """
        self.db_path = db_path
        self.ttl = ttl
        self._lock = Lock()
        with self._connect() as conn:
            conn.execute(self._CREATE)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def get(self, key: str) -> Optional[dict]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value, created FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            value_json, created = row
            if self.ttl is not None and (time.time() - created) > self.ttl:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                return None
            conn.execute("UPDATE cache SET hits = hits + 1 WHERE key = ?", (key,))
            return json.loads(value_json)

    def set(self, key: str, value: dict) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, created) VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time()),
            )

    def delete(self, key: str) -> bool:
        with self._lock, self._connect() as conn:
            affected = conn.execute("DELETE FROM cache WHERE key = ?", (key,)).rowcount
            return affected > 0

    def clear(self) -> int:
        """Delete all entries. Returns the number of rows removed."""
        with self._lock, self._connect() as conn:
            return conn.execute("DELETE FROM cache").rowcount

    def stats(self) -> dict:
        """Return aggregate cache statistics."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*), SUM(hits), MIN(created), MAX(created) FROM cache"
            ).fetchone()
            count, total_hits, oldest, newest = row
            return {
                "entries": count or 0,
                "total_hits": total_hits or 0,
                "oldest_entry": oldest,
                "newest_entry": newest,
                "db_path": self.db_path,
                "ttl_seconds": self.ttl,
            }


### LiteLLM wrapper to add caching

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.1%})"
        )



class CachedLLM:
    """
    Thin wrapper around ``litellm.completion`` / ``litellm.acompletion``
    that transparently caches responses.

    Cache key
    ---------
    SHA-256 of the JSON-serialised tuple:
        (model, messages, temperature, max_tokens, extra_kwargs)
    so two calls with identical inputs always hit the same entry.

    Parameters
    ----------
    model : str
        Any model string accepted by LiteLLM (e.g. ``"gpt-4o"``,
        ``"claude-3-5-sonnet-20241022"``, ``"ollama/llama3"``).
    db_path : str
        SQLite file path for the persistent cache.
    ttl : float | None
        Seconds before a cached entry expires. ``None`` = never.
    memory_cache : bool
        If ``True``, keep an in-process dict as an L1 cache in front of
        SQLite (faster for repeated calls within the same session).
    cache_key_extras : list[str]
        Additional ``completion()`` kwarg names to include in the cache
        key (e.g. ``["top_p", "stop"]``).
    """
    _DEFAULT_KEY_PARAMS = {"temperature",
                           "max_tokens",
                           "logprobs",
                           "top_logprobs",
                           "extra_body",
                           "sample_idx",
                           "seed",
                           "reasoning_effort",
                           "chat_template_kwargs",
                           "drop_params",
                           "allowed_openai_params",
                           "n",
                           }
    _DEFAULT_IGNORED_PARAMS = {"stream",
                               "timeout",
                               "api_key",
                               "api_base",
                               "service_tier",
                               }

    def __init__(
            self,
            model: str = "gpt-4o-mini",
            db_path: str = "llm_cache.db",
            ttl: Optional[float] = None,
            memory_cache: bool = True,
            cache_key_extras: Optional[list[str]] = None,
            ignored_params: Optional[list[str]] = None,
    ):
        self.model = model
        self._backend = SQLiteCache(db_path=db_path, ttl=ttl)
        self._memory: dict[str, dict] = {} if memory_cache else None
        self._key_params = self._DEFAULT_KEY_PARAMS | set(cache_key_extras or [])
        self._ignored_params = self._DEFAULT_IGNORED_PARAMS | set(ignored_params or [])
        self.stats = CacheStats()

    def _make_key(self, messages: list[dict], **kwargs) -> str:
        for k in kwargs:
            if k not in self._key_params and k not in self._ignored_params:
                print(
                    f"[CachedLLM] WARNING: param '{k}' is neither keyed nor ignored — it will not affect the cache key. Add it to cache_key_extras or ignored_params.")

        payload = {
            "model": self.model,
            "messages": messages,
            **{k: kwargs[k] for k in self._key_params if k in kwargs},
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Internal read / write
    # ------------------------------------------------------------------

    def _cache_get(self, key: str) -> Optional[dict]:
        if self._memory is not None and key in self._memory:
            return self._memory[key]
        value = self._backend.get(key)
        if value is not None and self._memory is not None:
            self._memory[key] = value  # promote to L1
        return value

    def _cache_set(self, key: str, value: dict) -> None:
        self._backend.set(key, value)
        if self._memory is not None:
            self._memory[key] = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def complete(
            self,
            messages: list[dict], # Axel's version
            # messages: list[dict[str, str]] | list[list[dict[str, str]]],
            sample_idx: int = 0,
            **kwargs,
    ) -> litellm.ModelResponse:
        key = self._make_key(messages, sample_idx=sample_idx, **kwargs)
        cached = self._cache_get(key)

        if cached is not None:
            self.stats.hits += 1
            response = litellm.ModelResponse(**cached)
            response._hidden_params["cache_hit"] = True
            return response

        self.stats.misses += 1
        # if all(isinstance(message, dict) for message in messages):
        #     response = litellm.completion(model=self.model, messages=messages, **kwargs)
        # if all(isinstance(message, list) for message in messages) and all(
        #         isinstance(item, dict)
        #         for message in messages
        #         for item in message
        # ): # messages is a list of lists of dicts
        #     response = litellm.batch_completion(model=self.model, messages=messages, **kwargs)

        response = litellm.completion(model=self.model, messages=messages, **kwargs)
        self._cache_set(key, response.model_dump())
        response._hidden_params["cache_hit"] = False
        return response

    async def acomplete(
            self,
            messages: list[dict],
            sample_idx: int = 0,
            **kwargs,
    ) -> litellm.ModelResponse:
        key = self._make_key(messages, sample_idx=sample_idx, **kwargs)
        cached = self._cache_get(key)

        if cached is not None:
            self.stats.hits += 1
            response = litellm.ModelResponse(**cached)
            response._hidden_params["cache_hit"] = True
            return response

        self.stats.misses += 1
        response = await litellm.acompletion(model=self.model, messages=messages, **kwargs)
        response._hidden_params["cache_hit"] = False
        self._cache_set(key, response.model_dump())
        return response

    def estimate_cost(
            self,
            messages: list[dict],
            max_tokens: int = 1000,
            service_tier: Optional[str] = None,
            **kwargs,
    ) -> dict:
        input_tokens = litellm.token_counter(model=self.model, messages=messages)

        model = self.model
        if service_tier == "flex":
            model = f"{self.model}-flex"  # litellm's naming for flex tier pricing

        try:
            input_cost, output_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=input_tokens,
                completion_tokens=max_tokens,
            )
        except Exception:
            # fall back to standard pricing if flex model string isn't recognised
            input_cost, output_cost = litellm.cost_per_token(
                model=self.model,
                prompt_tokens=input_tokens,
                completion_tokens=max_tokens,
            )
            print(
                f"[CachedLLM] WARNING: could not find pricing for service_tier='{service_tier}', falling back to standard pricing.")

        return {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": max_tokens,
            "input_cost": f"${input_cost:.6f}",
            "output_cost": f"${output_cost:.6f}",
            "total_cost": f"${input_cost + output_cost:.6f}",
            "service_tier": service_tier or "default",
        }

    def invalidate(self, messages: list[dict], **kwargs) -> bool:
        """Remove a specific entry from the cache. Returns ``True`` if found."""
        key = self._make_key(messages, **kwargs)
        if self._memory is not None:
            self._memory.pop(key, None)
        return self._backend.delete(key)

    def clear_cache(self) -> int:
        """Wipe the entire cache. Returns rows deleted."""
        if self._memory is not None:
            self._memory.clear()
        return self._backend.clear()

    def cache_info(self) -> dict:
        """Return combined runtime + backend stats."""
        backend = self._backend.stats()
        return {
            "runtime": {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "hit_rate": f"{self.stats.hit_rate:.1%}",
                "memory_entries": len(self._memory) if self._memory is not None else "disabled",
            },
            "backend": backend,
        }

    # Convenience: let the wrapper feel like a function
    def __call__(self, messages: list[dict], **kwargs) -> litellm.ModelResponse:
        return self.complete(messages, **kwargs)



