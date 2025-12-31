import re
from dataclasses import dataclass
from typing import Optional, List

import chess

UCI_MOVE_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


@dataclass
class AgentResult:
    input_fen: str
    move_uci: str
    move_san: str
    fen_after: str
    method: str
    valid: bool
    error: Optional[str] = None


def is_uci_format(s: str) -> bool:
    return bool(UCI_MOVE_RE.match(s.strip()))


def legal_moves_uci(fen: str) -> List[str]:
    board = chess.Board(fen)
    return [mv.uci() for mv in board.legal_moves]


def validate_and_apply_move(input_fen: str, move_uci: str, method: str) -> AgentResult:
    move_uci = move_uci.strip()
    try:
        board = chess.Board(input_fen)
    except Exception as e:
        return AgentResult(
            input_fen=input_fen,
            move_uci=move_uci,
            move_san="",
            fen_after="",
            method=method,
            valid=False,
            error=f"Invalid FEN: {e}",
        )

    if not is_uci_format(move_uci):
        return AgentResult(
            input_fen=input_fen,
            move_uci=move_uci,
            move_san="",
            fen_after="",
            method=method,
            valid=False,
            error="Move is not valid UCI format.",
        )

    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return AgentResult(
            input_fen=input_fen,
            move_uci=move_uci,
            move_san="",
            fen_after="",
            method=method,
            valid=False,
            error="Move is illegal in this position.",
        )

    move_san = board.san(move)
    board.push(move)
    fen_after = board.fen()

    return AgentResult(
        input_fen=input_fen,
        move_uci=move_uci,
        move_san=move_san,
        fen_after=fen_after,
        method=method,
        valid=True,
        error=None,
    )
