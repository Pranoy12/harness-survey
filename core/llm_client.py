import time
from datetime import datetime, timezone
import httpx
import litellm
from config import CONFIG


def is_deepseek_peak_utc() -> bool:
    """Checks if current UTC time falls in DeepSeek peak hours (01:00-04:00 & 06:00-10:00 UTC, Mon-Fri)."""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:  # Saturday (5) or Sunday (6) are always off-peak
        return False
    hour = now.hour
    return (1 <= hour < 4) or (6 <= hour < 10)


class UsageTracker:
    def __init__(self):
        # Session-wide cumulative counters (never reset across tasks)
        self.session_calls = 0
        self.session_cost_usd = 0.0
        self.session_cost_usd_deepseek = 0.0

        # Initialize per-task counters
        self.reset_task()

    def reset_task(self):
        """Resets per-task counters at the start of each new SWE-bench instance."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.call_count = 0

        self.cache_hit_tokens = 0
        self.cache_miss_tokens = 0
        self.completion_tokens = 0
        self.total_cost_usd_deepseek = 0.0

    @property
    def active_session_spend(self) -> float:
        """Returns the most accurate cumulative session spend for the budget circuit breaker."""
        return max(self.session_cost_usd_deepseek, self.session_cost_usd)

    def record(self, response):
        self.call_count += 1
        self.session_calls += 1
        usage = getattr(response, "usage", None)

        if not usage:
            return

        hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        miss = getattr(usage, "prompt_cache_miss_tokens", 0) or 0
        # Fallback if provider returns standard prompt_tokens without split
        if hit == 0 and miss == 0:
            miss = getattr(usage, "prompt_tokens", 0) or 0
        comp = getattr(usage, "completion_tokens", 0) or 0

        self.cache_hit_tokens += hit
        self.cache_miss_tokens += miss
        self.completion_tokens += comp

        # DeepSeek V4.1-Flash exact pricing calculation
        multiplier = 2.0 if is_deepseek_peak_utc() else 1.0
        turn_cost = (
            (hit * 0.003 * multiplier)
            + (miss * 0.15 * multiplier)
            + (comp * 0.60 * multiplier)
        ) / 1_000_000.0
        self.total_cost_usd_deepseek += turn_cost
        self.session_cost_usd_deepseek += turn_cost

        self.total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or (hit + miss)
        self.total_completion_tokens += comp
        self.total_tokens += getattr(usage, "total_tokens", 0) or (hit + miss + comp)

        # Attempt automated cost tracking via litellm's pricing database
        try:
            cost = litellm.completion_cost(completion_response=response)
            if cost:
                self.total_cost_usd += cost
                self.session_cost_usd += cost
        except Exception:
            pass

    def to_dict(self) -> dict:
        """Structured per-task metrics for clean JSON serialization in _traj.json."""
        total_in = self.cache_hit_tokens + self.cache_miss_tokens
        hit_rate = (self.cache_hit_tokens / total_in * 100) if total_in > 0 else 0.0
        return {
            "calls": self.call_count,
            "prompt_tokens": self.total_prompt_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "cache_hit_rate_pct": round(hit_rate, 2),
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "task_cost_usd": round(self.total_cost_usd_deepseek, 6),
            "session_cost_usd": round(self.active_session_spend, 6),
        }

    def summary(self) -> str:
        total_in = self.cache_hit_tokens + self.cache_miss_tokens
        hit_rate = (self.cache_hit_tokens / total_in * 100) if total_in > 0 else 0.0
        window = "PEAK (2x)" if is_deepseek_peak_utc() else "OFF-PEAK (1x)"
        return (
            f"[Quota Tracker] Calls: {self.call_count} | "
            f"Prompt Tokens: {self.total_prompt_tokens:,} | "
            f"Completion Tokens: {self.total_completion_tokens:,} | "
            f"Total Tokens: {self.total_tokens:,} | "
            f"Est. Cost: ${self.total_cost_usd:.4f}"
            f"[DEEPSEEK]"
            f"[DeepSeek {window}] Calls: {self.call_count} | "
            f"Cache Hit: {hit_rate:.1f}% ({self.cache_hit_tokens:,}/{total_in:,}) | "
            f"Out: {self.completion_tokens:,} | "
            f"Task Spend: ${self.total_cost_usd_deepseek:.5f} | "
            f"Total Session Spend: ${self.active_session_spend:.5f}"
        )


class BudgetExceededError(RuntimeError):
    pass


class LLMClient:
    def __init__(self):
        litellm.suppress_debug_info = True
        self.tracker = UsageTracker()
        self.check_initial_balance()

    def check_initial_balance(self):
        """Queries DeepSeek's live balance endpoint prior to starting a batch."""
        if "deepseek" not in CONFIG.MODEL_NAME.lower() or not CONFIG.API_KEY:
            return
        try:
            resp = httpx.get(
                "https://api.deepseek.com/user/balance",
                headers={"Authorization": f"Bearer {CONFIG.API_KEY}", "Accept": "application/json"},
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                infos = data.get("balance_infos", [])
                for info in infos:
                    print(f"[DeepSeek Account] Currency: {info.get('currency')} | Total Balance: {info.get('total_balance')}")
        except Exception as e:
            print(f"[DeepSeek Account] Could not fetch initial balance: {e}")

    def complete(self, messages, tools=None, tool_choice="auto", max_retries=6):
        if self.tracker.active_session_spend >= CONFIG.MAX_BUDGET_USD:
            raise BudgetExceededError(
                f"Halting execution: cumulative session spend (${self.tracker.active_session_spend:.4f}) "
                f"reached configured safety cap (${CONFIG.MAX_BUDGET_USD:.2f})."
            )

        kwargs = {
            "model": CONFIG.MODEL_NAME,
            "messages": messages,
            "api_key": CONFIG.API_KEY,
            "max_tokens": 2048,
            "timeout": 180,
        }

        if CONFIG.BASE_URL:
            kwargs["api_base"] = CONFIG.BASE_URL

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        base_delay = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = litellm.completion(**kwargs)
                self.tracker.record(response)
                print(self.tracker.summary())
                return response
            except litellm.exceptions.BadRequestError as e:
                err_msg = str(e)
                if "tool_use_failed" in err_msg or "attempted to call tool" in err_msg:
                    print("\n[DeepSeek Tool Warning] Model attempted invalid tool call. Injecting correction...")
                    messages.append({
                        "role": "user",
                        "content": "Error: Invalid tool name. Only call `run_shell_command` or `finish_task`."
                    })
                    continue
                raise e
            except (
                litellm.exceptions.ServiceUnavailableError,
                litellm.exceptions.InternalServerError,
                litellm.exceptions.APIConnectionError,
                litellm.exceptions.RateLimitError,
                litellm.exceptions.Timeout,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                httpx.RemoteProtocolError,
                ConnectionResetError,
                Exception,
            ) as e:
                err_str = str(e).lower()
                is_transient = (
                    any(
                        term in err_str
                        for term in [
                            "500", "502", "503", "504",
                            "demand", "rate", "unavailable", "timeout", "timed out", "429",
                            "connection", "reset", "104", "peer", "eof", "broken pipe",
                            "internalservererror", "server error",
                        ]
                    )
                    or isinstance(
                        e,
                        (
                            litellm.exceptions.InternalServerError,
                            litellm.exceptions.ServiceUnavailableError,
                            litellm.exceptions.APIConnectionError,
                            litellm.exceptions.RateLimitError,
                            litellm.exceptions.Timeout,
                            httpx.ReadTimeout,
                            httpx.ConnectTimeout,
                            httpx.RemoteProtocolError,
                            ConnectionResetError,
                        ),
                    )
                )

                if is_transient:
                    if attempt == max_retries:
                        raise e
                    sleep_time = base_delay * (2 ** (attempt - 1))
                    print(f"\n[{CONFIG.PROVIDER} Transient Error: {type(e).__name__}] Waiting {sleep_time}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(sleep_time)
                else:
                    raise e