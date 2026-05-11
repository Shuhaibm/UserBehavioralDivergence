def raw_conversation(entry: dict) -> str:
    return "\n".join(f"[{m['role']}]: {m['content']}" for m in entry["messages"])


def raw_user_utterances(entry: dict) -> str:
    return "\n".join(
        f"[user]: {m['content']}" for m in entry["messages"] if m["role"] == "user"
    )


def raw_intent(entry: dict) -> str:
    return entry.get("intent", "")


def requests_facet(entry: dict) -> str:
    return entry.get("requests_facet", "")


def responses_facet(entry: dict) -> str:
    return entry.get("responses_facet", "")


def context_facet(entry: dict) -> str:
    return entry.get("context_facet", "")


def communication_style_facet(entry: dict) -> str:
    return entry.get("communication_style_facet", "")


def DAMSL_facet(entry: dict) -> str:
    results = entry.get("DAMSL_facet", [])
    if not results:
        return ""
    return "\n".join(r["content"] for r in results if r.get("content"))


def SGD_facet(entry: dict) -> str:
    results = entry.get("SGD_facet", [])
    if not results:
        return ""
    return "\n".join(r["content"] for r in results if r.get("content"))


def combined(entry: dict) -> str:
    return entry.get("combined", "")


BUILTIN_REPRESENTATIONS = {
    "raw_conversation": raw_conversation,
    "raw_user_utterances": raw_user_utterances,
    "raw_intent": raw_intent,
}

GENERATED_REPRESENTATIONS = {
    "requests_facet": requests_facet,
    "responses_facet": responses_facet,
    "context_facet": context_facet,
    "communication_style_facet": communication_style_facet,
    "DAMSL_facet": DAMSL_facet,
    "SGD_facet": SGD_facet,
    "combined": combined,
}

ALL_REPRESENTATIONS = {**BUILTIN_REPRESENTATIONS, **GENERATED_REPRESENTATIONS}