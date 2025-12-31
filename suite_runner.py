import argparse
import json
import logging
import os
import time
from collections import Counter
from typing import Any, Dict, List, Literal, Optional, TypedDict, Tuple

from dotenv import load_dotenv
import chess
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langgraph.graph import StateGraph, END

from chess_core import legal_moves_uci, validate_and_apply_move


Method = Literal["direct", "cot_hidden", "sc_cot"]


class AgentState(TypedDict, total=False):
    run_id: str
    fen: str
    method: Method

    legal_moves: List[str]

    sc_samples: int
    sc_votes: List[str]
    sc_raw: List[str]
    sc_i: int

    move_uci: str
    result: Dict[str, Any]
    error: str

    key_index: int
    raw_model_output: str
    vote_distribution: Dict[str, int]


CTX: Dict[str, Any] = {}


def set_ctx(cfg: Dict[str, Any]) -> None:
    global CTX
    CTX = cfg


def get_ctx() -> Dict[str, Any]:
    if not CTX:
        raise RuntimeError("CTX is not initialized.")
    return CTX


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def atomic_write_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def setup_logger(run_dir: str, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(run_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def parse_move_from_raw(raw: str) -> Optional[str]:
    obj = extract_json_object(raw)
    if not obj or "move_uci" not in obj:
        return None
    return str(obj["move_uci"]).strip()


class RateLimiter:
    def __init__(self, min_interval_sec: float):
        self.min_interval_sec = float(min_interval_sec)
        self.last_call = 0.0

    def wait(self) -> None:
        now = time.time()
        delta = now - self.last_call
        if delta < self.min_interval_sec:
            time.sleep(self.min_interval_sec - delta)
        self.last_call = time.time()


class DiskCheckpointer:
    def __init__(self, run_dir: str, logger: logging.Logger):
        self.run_dir = run_dir
        self.logger = logger
        ensure_dir(run_dir)
        self.state_path = os.path.join(run_dir, "state.json")
        self.events_path = os.path.join(run_dir, "events.jsonl")
        self.result_path = os.path.join(run_dir, "result.json")

    def load(self) -> Optional[AgentState]:
        data = load_json(self.state_path)
        return data  # type: ignore[return-value]

    def save(self, state: AgentState) -> None:
        atomic_write_json(self.state_path, state)

    def event(self, evt: Dict[str, Any]) -> None:
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")

    def save_result(self, result: Dict[str, Any]) -> None:
        atomic_write_json(self.result_path, result)


def load_keys_from_env() -> List[str]:
    load_dotenv(override=False)
    raw = os.getenv("GOOGLE_API_KEYS", "").strip()
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def set_active_api_key(key: str) -> None:
    os.environ["GOOGLE_API_KEY"] = key


def is_429_error(msg: str) -> bool:
    return ("RESOURCE_EXHAUSTED" in msg) or ("429" in msg)


def global_key_state_path(project_root: str) -> str:
    return os.path.join(project_root, ".key_state.json")


def load_global_key_state(project_root: str) -> Dict[str, Any]:
    p = global_key_state_path(project_root)
    data = load_json(p)
    if data is None:
        return {"key_index": 0}
    if "key_index" not in data:
        data["key_index"] = 0
    return data


def save_global_key_state(project_root: str, data: Dict[str, Any]) -> None:
    p = global_key_state_path(project_root)
    atomic_write_json(p, data)


def suite_state_path(results_dir: str, suite_id: str) -> str:
    return os.path.join(results_dir, suite_id, "suite_state.json")


def load_suite_state(results_dir: str, suite_id: str) -> Dict[str, Any]:
    p = suite_state_path(results_dir, suite_id)
    data = load_json(p)
    if data is None:
        return {"key_index": 0}
    if "key_index" not in data:
        data["key_index"] = 0
    return data


def save_suite_state(results_dir: str, suite_id: str, data: Dict[str, Any]) -> None:
    p = suite_state_path(results_dir, suite_id)
    atomic_write_json(p, data)


def persist_last_good_key_index(idx: int) -> None:
    cfg = get_ctx()
    results_dir = cfg["results_dir"]
    suite_id = cfg["suite_id"]
    project_root = cfg["project_root"]

    suite_state = load_suite_state(results_dir, suite_id)
    suite_state["key_index"] = int(idx)
    save_suite_state(results_dir, suite_id, suite_state)

    global_state = load_global_key_state(project_root)
    global_state["key_index"] = int(idx)
    save_global_key_state(project_root, global_state)


class RoundRobinKeyManager:
    def __init__(self, keys: List[str], start_index: int, logger: logging.Logger):
        if not keys:
            raise RuntimeError("No API keys found. Set GOOGLE_API_KEYS in .env.")
        self.keys = keys
        self.logger = logger
        self.index = int(start_index) % len(keys)
        self.activate(self.index)

    def activate(self, idx: int) -> None:
        self.index = idx % len(self.keys)
        set_active_api_key(self.keys[self.index])
        self.logger.info("Active API key index: %d/%d", self.index + 1, len(self.keys))

    def next_key(self) -> None:
        self.activate((self.index + 1) % len(self.keys))


def build_prompt(method: Method, fen: str, moves: List[str], feedback: str = "") -> str:
    moves_str = " ".join(moves)
    if method in ("cot_hidden", "sc_cot"):
        header = (
            "You are a chess move selector.\n"
            "Analyze the position internally step-by-step to choose the best move.\n"
            "Do NOT reveal your reasoning.\n"
        )
    else:
        header = "You are a chess move selector.\n"

    base = (
        header
        + "You MUST select exactly one move from the provided legal moves list.\n"
        + "Return ONLY valid JSON with this schema: {\"move_uci\": \"...\"}\n"
        + "No extra keys. No markdown. No prose.\n\n"
        + f"FEN: {fen}\n"
        + f"LEGAL_MOVES_UCI: {moves_str}\n"
    )
    return base + ("\n" + feedback + "\n" if feedback else "")


def llm_call_round_robin(prompt: str) -> str:
    cfg = get_ctx()
    logger: logging.Logger = cfg["logger"]
    limiter: RateLimiter = cfg["limiter"]
    km: RoundRobinKeyManager = cfg["keyman"]

    model_name: str = cfg["model"]
    temperature: float = float(cfg["temperature"])

    retry_on_429: bool = bool(cfg["retry_on_429"])
    rounds_limit: int = int(cfg["rounds_limit"])
    cooloff_sec: float = float(cfg.get("cooloff_sec", 60.0))

    keys_count = len(km.keys)

    if not retry_on_429:
        limiter.wait()
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        resp = llm.invoke(prompt)
        persist_last_good_key_index(km.index)
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    rounds_done = 0
    tried_in_round = 0

    while True:
        limiter.wait()
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

        try:
            resp = llm.invoke(prompt)
            persist_last_good_key_index(km.index)
            return resp.content if isinstance(resp.content, str) else str(resp.content)
        except ChatGoogleGenerativeAIError as e:
            msg = str(e)
            logger.warning("LLM error (key=%d): %s", km.index + 1, msg)

            if not is_429_error(msg):
                raise

            tried_in_round += 1
            logger.warning(
                "429 -> switching key immediately (tried_in_round=%d/%d round=%d/%d)",
                tried_in_round,
                keys_count,
                rounds_done + 1,
                rounds_limit,
            )

            km.next_key()

            if tried_in_round >= keys_count:
                rounds_done += 1
                tried_in_round = 0
                logger.warning("Completed a full key round. rounds_done=%d/%d", rounds_done, rounds_limit)

                if rounds_done >= rounds_limit:
                    sleep_for = float(min(max(cooloff_sec, 1.0), 600.0))
                    logger.warning("All rounds exhausted. Cooling off for %.1fs then restarting rounds.", sleep_for)
                    time.sleep(sleep_for)
                    rounds_done = 0


def node_get_legal_moves(state: AgentState) -> AgentState:
    cfg = get_ctx()
    logger: logging.Logger = cfg["logger"]
    cp: DiskCheckpointer = cfg["cp"]

    logger.info("NODE get_legal_moves")
    cp.event({"node": "get_legal_moves", "ts": time.time()})

    state["legal_moves"] = legal_moves_uci(state["fen"])
    cp.save(state)
    return state


def node_init_sc(state: AgentState) -> AgentState:
    cfg = get_ctx()
    logger: logging.Logger = cfg["logger"]
    cp: DiskCheckpointer = cfg["cp"]

    logger.info("NODE init_sc")
    cp.event({"node": "init_sc", "ts": time.time()})

    if state.get("method") == "sc_cot":
        state.setdefault("sc_votes", [])
        state.setdefault("sc_raw", [])
        state.setdefault("sc_i", 0)

    cp.save(state)
    return state


def node_sc_step(state: AgentState) -> AgentState:
    cfg = get_ctx()
    logger: logging.Logger = cfg["logger"]
    cp: DiskCheckpointer = cfg["cp"]

    logger.info("NODE sc_step")
    target = int(state.get("sc_samples", 7))
    current = int(state.get("sc_i", 0))
    logger.info("SC progress: %d/%d | valid_votes=%d", current, target, len(state.get("sc_votes", [])))

    cp.event(
        {
            "node": "sc_step",
            "ts": time.time(),
            "sc_i": state.get("sc_i"),
            "key_index": cfg["keyman"].index,
            "target": target,
            "valid_votes": len(state.get("sc_votes", [])),
        }
    )

    fen = state["fen"]
    moves = state["legal_moves"]
    method: Method = state["method"]

    raw = llm_call_round_robin(build_prompt(method, fen, moves))
    state["sc_raw"].append(raw)

    mv = parse_move_from_raw(raw)
    if mv is not None and mv in moves:
        state["sc_votes"].append(mv)

    state["sc_i"] = int(state.get("sc_i", 0)) + 1
    state["key_index"] = cfg["keyman"].index
    cp.save(state)
    return state


def compute_vote_distribution(votes: List[str]) -> Dict[str, int]:
    return dict(Counter(votes))


def node_choose_move(state: AgentState) -> AgentState:
    cfg = get_ctx()
    logger: logging.Logger = cfg["logger"]
    cp: DiskCheckpointer = cfg["cp"]

    logger.info("NODE choose_move")
    cp.event({"node": "choose_move", "ts": time.time(), "key_index": cfg["keyman"].index})

    fen = state["fen"]
    moves = state["legal_moves"]
    method: Method = state.get("method", "direct")

    if method == "sc_cot":
        votes = state.get("sc_votes", [])
        if not votes:
            state["error"] = "No valid samples produced."
            state["key_index"] = cfg["keyman"].index
            cp.save(state)
            return state

        dist = compute_vote_distribution(votes)
        best = max(dist.items(), key=lambda x: (x[1], x[0]))[0]
        state["move_uci"] = best
        state["vote_distribution"] = dist
        state["key_index"] = cfg["keyman"].index
        cp.save(state)
        return state

    max_attempts = int(get_ctx().get("max_attempts", 3))
    feedback = ""
    last_raw = ""

    for _ in range(max_attempts):
        prompt = build_prompt(method, fen, moves, feedback=feedback)
        raw = llm_call_round_robin(prompt)
        last_raw = raw
        mv = parse_move_from_raw(raw)

        if mv is None or mv not in moves:
            feedback = "Your previous answer was invalid. You must output a move_uci that appears exactly in LEGAL_MOVES_UCI."
            continue

        state["move_uci"] = mv
        state["key_index"] = cfg["keyman"].index
        cp.save(state)
        return state

    state["error"] = "Model did not return a valid move from legal moves."
    state["raw_model_output"] = last_raw
    state["key_index"] = cfg["keyman"].index
    cp.save(state)
    return state


def node_validate_apply(state: AgentState) -> AgentState:
    cfg = get_ctx()
    logger: logging.Logger = cfg["logger"]
    cp: DiskCheckpointer = cfg["cp"]

    logger.info("NODE validate_apply")
    cp.event({"node": "validate_apply", "ts": time.time()})

    key_index = cfg["keyman"].index
    state["key_index"] = key_index

    if "error" in state:
        result = {
            "input_fen": state.get("fen", ""),
            "method": state.get("method", "direct"),
            "valid": False,
            "error": state["error"],
            "meta": {"key_index": key_index},
        }
        state["result"] = result
        cp.save(state)
        cp.save_result(result)
        return state

    applied = validate_and_apply_move(state["fen"], state["move_uci"], method=state.get("method", "direct"))

    sc_info = None
    if state.get("method") == "sc_cot":
        votes = state.get("sc_votes", [])
        sc_info = {
            "samples_requested": state.get("sc_samples"),
            "samples_taken": state.get("sc_i"),
            "valid_samples": len(votes),
            "vote_distribution": state.get("vote_distribution"),
        }

    result = {
        "input_fen": applied.input_fen,
        "move_uci": applied.move_uci,
        "move_san": applied.move_san,
        "fen_after": applied.fen_after,
        "method": applied.method,
        "valid": applied.valid,
        "error": applied.error,
        "meta": {"key_index": key_index},
        "sc_info": sc_info,
    }

    state["result"] = result
    cp.save(state)
    cp.save_result(result)
    return state


def route_after_init(state: AgentState) -> str:
    if state.get("method") == "sc_cot":
        return "sc_step"
    return "choose_move"


def route_sc_loop(state: AgentState) -> str:
    i = int(state.get("sc_i", 0))
    target = int(state.get("sc_samples", 7))
    if i < target:
        return "sc_step"
    return "choose_move"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("get_legal_moves", node_get_legal_moves)
    g.add_node("init_sc", node_init_sc)
    g.add_node("sc_step", node_sc_step)
    g.add_node("choose_move", node_choose_move)
    g.add_node("validate_apply", node_validate_apply)

    g.set_entry_point("get_legal_moves")
    g.add_edge("get_legal_moves", "init_sc")
    g.add_conditional_edges("init_sc", route_after_init, {"sc_step": "sc_step", "choose_move": "choose_move"})
    g.add_conditional_edges("sc_step", route_sc_loop, {"sc_step": "sc_step", "choose_move": "choose_move"})
    g.add_edge("choose_move", "validate_apply")
    g.add_edge("validate_apply", END)

    return g.compile()


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if v is not None:
            out[k] = v
    return out


def require_fields(case: Dict[str, Any], fields: List[str]) -> None:
    missing = [f for f in fields if f not in case]
    if missing:
        raise RuntimeError(f"Missing required fields in case: {missing}")


def validate_suite(suite: Dict[str, Any]) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    suite_id = str(suite.get("suite_id", "suite"))
    defaults = suite.get("defaults", {})
    cases = suite.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("No cases found in testcases file.")

    seen = set()
    for c in cases:
        require_fields(c, ["case_id", "fen", "method"])
        cid = str(c["case_id"])
        if cid in seen:
            raise RuntimeError(f"Duplicate case_id: {cid}")
        seen.add(cid)

        m = c["method"]
        if m not in ("direct", "cot_hidden", "sc_cot"):
            raise RuntimeError(f"Invalid method in case {cid}: {m}")

        if c.get("method") == "sc_cot":
            if "sc_samples" in c and int(c["sc_samples"]) <= 0:
                raise RuntimeError(f"Invalid sc_samples in case {cid}")

    return suite_id, defaults, cases


def run_case(
    suite_id: str,
    case: Dict[str, Any],
    defaults: Dict[str, Any],
    keys: List[str],
    results_root: str,
    suite_key_index: int,
    results_dir: str,
    project_root: str,
) -> Tuple[Dict[str, Any], int]:
    case_id = str(case["case_id"])
    run_dir = os.path.join(results_root, suite_id, case_id)
    ensure_dir(run_dir)

    logger = setup_logger(run_dir, name=f"suite_{suite_id}_{case_id}")
    cp = DiskCheckpointer(run_dir, logger)

    logger.info("Suite=%s Case=%s", suite_id, case_id)

    loaded = cp.load()
    if loaded is not None:
        state: AgentState = loaded
        logger.info("Loaded existing state. method=%s sc_i=%s key_index=%s", state.get("method"), state.get("sc_i"), state.get("key_index"))
        start_key_index = int(state.get("key_index", suite_key_index))
    else:
        start_key_index = suite_key_index
        state = {
            "run_id": case_id,
            "fen": case.get("fen", chess.STARTING_FEN),
            "method": case.get("method", defaults.get("method", "direct")),
            "sc_samples": int(case.get("sc_samples", defaults.get("sc_samples", 7))),
            "key_index": start_key_index,
        }
        cp.save(state)
        logger.info("Initialized new state. method=%s sc_samples=%s key_index=%s", state.get("method"), state.get("sc_samples"), start_key_index)

    params = merge_dicts(defaults, case)

    limiter = RateLimiter(min_interval_sec=float(params.get("min_interval_sec", 0.0)))
    km = RoundRobinKeyManager(keys=keys, start_index=start_key_index, logger=logger)

    cfg = {
        "logger": logger,
        "cp": cp,
        "limiter": limiter,
        "keyman": km,
        "retry_on_429": bool(params.get("retry_on_429", True)),
        "rounds_limit": int(params.get("max_retries_per_key", 2)),
        "cooloff_sec": float(params.get("cooloff_sec", 60.0)),
        "max_attempts": int(params.get("max_attempts", 3)),
        "model": str(params.get("model", "gemini-2.5-flash-lite")),
        "temperature": float(params.get("temperature", 0.3)),
        "results_dir": results_dir,
        "suite_id": suite_id,
        "project_root": project_root,
    }

    set_ctx(cfg)
    app = build_graph()

    started_at = now_iso()
    try:
        out = app.invoke(state)
        result = out.get("result")
        finished_at = now_iso()
        if result is None:
            result = {"valid": False, "error": "No result produced."}

        meta = {
            "suite_id": suite_id,
            "case_id": case_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "run_dir": run_dir,
            "params": params,
            "result": result,
        }
        cp.save_result(result)
        return meta, km.index

    except Exception as e:
        finished_at = now_iso()
        err = repr(e)
        logger.error("Case failed: %s", err)

        state = cp.load() or state
        state["error"] = err
        state["key_index"] = km.index
        cp.save(state)

        meta = {
            "suite_id": suite_id,
            "case_id": case_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "run_dir": run_dir,
            "params": params,
            "result": {"valid": False, "error": err},
        }
        cp.save_result(meta["result"])
        return meta, km.index


def summarize_suite(summary_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(summary_items)
    ok = sum(1 for it in summary_items if it.get("result", {}).get("valid") is True)
    fail = total - ok
    by_method: Dict[str, int] = {}
    for it in summary_items:
        m = str(it.get("params", {}).get("method", it.get("result", {}).get("method", "unknown")))
        by_method[m] = by_method.get(m, 0) + 1

    return {
        "total": total,
        "success": ok,
        "failure": fail,
        "by_method": by_method,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testcases", required=True)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    project_root = os.path.abspath(os.getcwd())

    keys = load_keys_from_env()
    if not keys:
        raise RuntimeError("GOOGLE_API_KEYS is missing or empty in .env")

    with open(args.testcases, "r", encoding="utf-8") as f:
        suite = json.load(f)

    suite_id, defaults, cases = validate_suite(suite)

    suite_dir = os.path.join(args.results_dir, suite_id)
    ensure_dir(suite_dir)
    summary_jsonl = os.path.join(suite_dir, "summary.jsonl")
    summary_json = os.path.join(suite_dir, "summary.json")

    global_state = load_global_key_state(project_root)
    suite_key_index = int(global_state.get("key_index", 0)) % len(keys)

    summary_items: List[Dict[str, Any]] = []
    started_at = now_iso()

    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] Running case_id={case['case_id']} (suite_key_index={suite_key_index})")
        meta, new_key_index = run_case(
            suite_id=suite_id,
            case=case,
            defaults=defaults,
            keys=keys,
            results_root=args.results_dir,
            suite_key_index=suite_key_index,
            results_dir=args.results_dir,
            project_root=project_root,
        )
        append_jsonl(summary_jsonl, meta)
        summary_items.append(meta)

        suite_key_index = int(new_key_index) % len(keys)

        res = meta.get("result", {})
        ok = res.get("valid")
        err = res.get("error")
        print(f" -> done valid={ok} error={err} next_key_index={suite_key_index}")

    finished_at = now_iso()
    suite_summary = {
        "suite_id": suite_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "stats": summarize_suite(summary_items),
        "global_key_state": load_global_key_state(project_root),
        "items": summary_items,
    }
    atomic_write_json(summary_json, suite_summary)

    print(f"Summary written: {summary_jsonl}")
    print(f"Suite report written: {summary_json}")


if __name__ == "__main__":
    main()
