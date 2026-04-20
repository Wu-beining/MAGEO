try:
    from tool.web_search import web_search
except ModuleNotFoundError as exc:
    def web_search(*args, **kwargs):
        raise ModuleNotFoundError(
            "web_search requires optional dependencies and credentials. "
            "Install `zai-sdk` and set `WEB_SEARCH_API_KEY` before using it."
        ) from exc

__all__ = ["web_search"]
