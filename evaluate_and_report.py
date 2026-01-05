import argparse
import io
import json
import logging
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.svg
from PIL import Image, ImageDraw, ImageFont

try:
    import cairosvg
except Exception:
    cairosvg = None


# ----------------------------
# Logging
# ----------------------------
LOG = logging.getLogger("evaluate_and_report")


def setup_logging(level: str, log_file: str = "") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)

    # Reset handlers to avoid duplicates in reruns
    LOG.handlers[:] = []
    LOG.setLevel(lvl)

    fmt = logging.Formatter("[%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(fmt)
    LOG.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        LOG.addHandler(fh)


# ----------------------------
# Utility helpers
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_suite_ids(results_dir: str) -> List[str]:
    if not os.path.isdir(results_dir):
        return []
    out = []
    for name in os.listdir(results_dir):
        p = os.path.join(results_dir, name)
        if os.path.isdir(p):
            out.append(name)
    out.sort()
    return out


def list_case_ids(suite_dir: str) -> List[str]:
    out = []
    for name in os.listdir(suite_dir):
        p = os.path.join(suite_dir, name)
        if os.path.isdir(p):
            if os.path.exists(os.path.join(p, "methods_index.json")):
                out.append(name)
    out.sort()
    return out


def read_methods_index(results_dir: str, suite_id: str, case_id: str) -> Dict[str, Any]:
    p = os.path.join(results_dir, suite_id, case_id, "methods_index.json")
    return load_json(p)


def read_result_json(run_dir: str) -> Dict[str, Any]:
    p = os.path.join(run_dir, "result.json")
    return load_json(p)


def try_parse_move(move_uci: str) -> Optional[chess.Move]:
    if not move_uci:
        return None
    try:
        return chess.Move.from_uci(move_uci)
    except Exception:
        return None


def svg_to_png_bytes(svg_text: str) -> bytes:
    if cairosvg is None:
        raise RuntimeError("cairosvg is not installed. pip install cairosvg")
    return cairosvg.svg2png(bytestring=svg_text.encode("utf-8"))


def render_board_png(
    fen: str,
    lastmove_uci: Optional[str],
    size: int,
    orientation: chess.Color = chess.WHITE,
    coordinates: bool = True,
) -> Image.Image:
    board = chess.Board(fen)
    lm = try_parse_move(lastmove_uci) if lastmove_uci else None
    svg = chess.svg.board(
        board=board,
        orientation=orientation,
        lastmove=lm,
        coordinates=coordinates,
        size=size,
    )
    png_bytes = svg_to_png_bytes(svg)
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bb = draw.textbbox((0, 0), text, font=font)
    return int(bb[2] - bb[0])


def wrap_text_pixels(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width_px: int,
) -> List[str]:
    s = (text or "").strip()
    if not s:
        return []
    words = s.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        cand = (cur + " " + w).strip()
        if text_width(draw, cand, font) <= max_width_px:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_centered(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str, font: ImageFont.ImageFont) -> None:
    x0, y0, x1, y1 = box
    bb = draw.textbbox((0, 0), text, font=font)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    draw.text((x0 + (x1 - x0 - tw) // 2, y0 + (y1 - y0 - th) // 2), text, font=font, fill=(0, 0, 0, 255))


def paste_center(canvas: Image.Image, img: Image.Image, box: Tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    ix, iy = img.size
    ox = x0 + (w - ix) // 2
    oy = y0 + (h - iy) // 2
    canvas.alpha_composite(img, (ox, oy))


# ----------------------------
# FEN sanitize
# ----------------------------
def sanitize_fen(fen: str) -> Optional[str]:
    s = (fen or "").strip()
    if not s:
        return None
    parts = s.split()
    if len(parts) == 4:
        s = s + " 0 1"
    elif len(parts) == 5:
        s = s + " 1"
    try:
        b = chess.Board(s)
        return b.fen(en_passant="legal")
    except Exception as e:
        LOG.warning("sanitize_fen failed: %s | fen=%r", e, fen)
        return None


# ----------------------------
# Stockfish engine
# ----------------------------
class StockfishEngine:
    def __init__(self, path: str, threads: int = 2, hash_mb: int = 128) -> None:
        self.path = path
        self.threads = threads
        self.hash_mb = hash_mb
        self.p: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self.p = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._cmd("uci")
        self._wait("uciok")
        self._cmd(f"setoption name Threads value {self.threads}")
        self._cmd(f"setoption name Hash value {self.hash_mb}")
        self._cmd("isready")
        self._wait("readyok")
        LOG.info("stockfish started | path=%s threads=%d hash=%d", self.path, self.threads, self.hash_mb)

    def close(self) -> None:
        if self.p and self.p.stdin:
            try:
                self._cmd("quit")
            except Exception:
                pass
        if self.p:
            try:
                self.p.kill()
            except Exception:
                pass
        self.p = None

    def _cmd(self, s: str) -> None:
        assert self.p and self.p.stdin
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def _readline(self) -> str:
        assert self.p and self.p.stdout
        return self.p.stdout.readline().strip()

    def _wait(self, token: str, timeout_s: float = 4.0) -> None:
        t0 = time.time()
        while True:
            if time.time() - t0 > timeout_s:
                raise RuntimeError(f"Stockfish wait timeout for {token}")
            line = self._readline()
            if token in line:
                return

    def eval_fen(self, fen: str, depth: int = 15) -> Dict[str, Any]:
        """
        Stockfish UCI reports score from SIDE-TO-MOVE POV.
        Returns dict:
          cp_stm: score cp from side-to-move POV (or mate mapped to +/-10000)
          bestmove_uci
          depth
        """
        safe_fen = sanitize_fen(fen) or fen
        self._cmd(f"position fen {safe_fen}")
        self._cmd(f"go depth {depth}")

        bestmove = "-"
        last_cp_stm: Optional[float] = None
        last_mate_stm: Optional[int] = None

        while True:
            line = self._readline()
            if line.startswith("info "):
                if " score cp " in line:
                    try:
                        cp = int(line.split(" score cp ")[1].split()[0])
                        last_cp_stm = float(cp)
                    except Exception:
                        pass
                elif " score mate " in line:
                    try:
                        mate = int(line.split(" score mate ")[1].split()[0])
                        last_mate_stm = mate
                    except Exception:
                        pass
            elif line.startswith("bestmove "):
                parts = line.split()
                bestmove = parts[1] if len(parts) > 1 else "-"
                break

        if last_mate_stm is not None:
            cp_like_stm = 10000.0 if last_mate_stm > 0 else -10000.0
        else:
            cp_like_stm = last_cp_stm if last_cp_stm is not None else 0.0

        return {"cp_stm": float(cp_like_stm), "bestmove_uci": bestmove, "depth": depth}


def find_stockfish_path(cli_path: str = "") -> Optional[str]:
    if cli_path:
        return cli_path if os.path.exists(cli_path) else None
    return shutil.which("stockfish")


def safe_san(board: chess.Board, move_uci: str) -> str:
    try:
        mv = chess.Move.from_uci(move_uci)
        if mv in board.legal_moves:
            return board.san(mv)
    except Exception:
        pass
    return "-"


# ----------------------------
# Move quality computation
# ----------------------------
def compute_move_quality_stockfish_only(
    input_fen: str,
    move_uci: str,
    stockfish: StockfishEngine,
    sf_depth: int,
) -> Dict[str, Any]:
    safe_in = sanitize_fen(input_fen)
    if not safe_in:
        return {"error": "bad_fen"}

    board_before = chess.Board(safe_in)

    try:
        mv = chess.Move.from_uci(move_uci)
    except Exception:
        return {"error": "bad_move_uci"}
    if mv not in board_before.legal_moves:
        return {"error": "illegal_move"}

    sf_before = stockfish.eval_fen(safe_in, depth=sf_depth)
    cp_before_stm = float(sf_before["cp_stm"])
    best_uci = str(sf_before["bestmove_uci"])
    depth_before = sf_before.get("depth")

    before_cp_mover = cp_before_stm

    best_san = safe_san(board_before, best_uci) if best_uci else "-"

    board_after = chess.Board(safe_in)
    board_after.push(mv)
    fen_after = board_after.fen(en_passant="legal")

    sf_after = stockfish.eval_fen(fen_after, depth=sf_depth)
    cp_after_stm = float(sf_after["cp_stm"])
    depth_after = sf_after.get("depth")

    after_cp_mover = -cp_after_stm

    drop = max(0.0, before_cp_mover - after_cp_mover)

    if drop < 15:
        label = "Excellent"
    elif drop < 50:
        label = "Good"
    elif drop < 120:
        label = "Inaccuracy"
    elif drop < 250:
        label = "Mistake"
    else:
        label = "Blunder"

    return {
        "source": "stockfish",
        "label": label,
        "drop_cp": round(drop, 2),
        "before_cp_mover": round(before_cp_mover, 2),
        "after_cp_mover": round(after_cp_mover, 2),
        "depth_before": depth_before,
        "depth_after": depth_after,
        "bestmove_uci": best_uci or "-",
        "bestmove_san": best_san or "-",
        "fen_in": safe_in,
        "fen_after": fen_after,
    }


# ----------------------------
# UI helpers
# ----------------------------
def badge_color(label: str) -> Tuple[int, int, int]:
    if label == "Excellent":
        return (46, 204, 113)
    if label == "Good":
        return (39, 174, 96)
    if label == "Inaccuracy":
        return (241, 196, 15)
    if label == "Mistake":
        return (230, 126, 34)
    if label == "Blunder":
        return (231, 76, 60)
    return (127, 140, 141)


def draw_badge(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    pad_x, pad_y = 14, 8
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    w, h = tw + 2 * pad_x, th + 2 * pad_y
    col = badge_color(text)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=col, outline=(0, 0, 0, 40), width=2)
    draw.text((x + pad_x, y + pad_y), text, font=font, fill=(255, 255, 255, 255))
    return w, h


def draw_drop_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, drop_cp: float) -> None:
    mx = 500.0
    frac = max(0.0, min(1.0, float(drop_cp) / mx))
    fill_w = int(w * frac)

    draw.rounded_rectangle((x, y, x + w, y + h), radius=10, fill=(230, 230, 230, 255), outline=(0, 0, 0, 60), width=2)
    if fill_w > 0:
        if drop_cp < 50:
            col = (46, 204, 113)
        elif drop_cp < 120:
            col = (241, 196, 15)
        elif drop_cp < 250:
            col = (230, 126, 34)
        else:
            col = (231, 76, 60)
        draw.rounded_rectangle((x, y, x + fill_w, y + h), radius=10, fill=col)


# ----------------------------
# Report builder
# ----------------------------
def build_case_report_image(
    results_dir: str,
    suite_id: str,
    case_id: str,
    out_dir: str,
    board_size: int,
    include_coordinates: bool,
    dpi: int,
    eval_moves: bool,
    eval_sleep_ms: int,
    stockfish: StockfishEngine,
    sf_depth: int,
) -> str:
    idx = read_methods_index(results_dir, suite_id, case_id)
    methods = idx.get("methods", [])
    if not methods:
        raise RuntimeError(f"No methods in methods_index.json for case {case_id}")

    rows: List[Tuple[str, Dict[str, Any]]] = []
    for m in methods:
        run_dir = str(m["run_dir"])
        res = read_result_json(run_dir)
        rows.append((str(m.get("method")), res))

    cols = ["INPUT", "PROPOSED", "OUTPUT"]
    col_count = 3
    row_count = len(rows)

    font_title = get_font(50)
    font_hdr = get_font(40)
    font_kv = get_font(34)
    font_small = get_font(28)

    title_h = 110
    col_hdr_h = 80
    info_bar_h = 210
    cell_pad = 22

    cell_w = board_size + cell_pad * 2
    cell_h = board_size + cell_pad * 2

    width = col_count * cell_w
    row_h = info_bar_h + cell_h
    height = title_h + col_hdr_h + row_count * row_h

    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Title
    title = f"Suite: {suite_id}   |   Case: {case_id}"
    draw_centered(draw, (0, 0, width, title_h), title, font_title)

    # Column header band
    y_hdr0 = title_h
    draw.rectangle((0, y_hdr0, width, y_hdr0 + col_hdr_h), fill=(250, 250, 250, 255))
    for ci, col in enumerate(cols):
        x0 = ci * cell_w
        x1 = x0 + cell_w
        draw_centered(draw, (x0, y_hdr0, x1, y_hdr0 + col_hdr_h), col, font_hdr)

    grid = (0, 0, 0, 255)
    draw.line((0, title_h, width, title_h), fill=grid, width=2)
    draw.line((0, title_h + col_hdr_h, width, title_h + col_hdr_h), fill=grid, width=2)
    for ci in range(1, col_count):
        x = ci * cell_w
        draw.line((x, title_h, x, height), fill=grid, width=2)

    for ri, (method_name, res) in enumerate(rows):
        row_y0 = title_h + col_hdr_h + ri * row_h
        info_y0 = row_y0
        boards_y0 = row_y0 + info_bar_h
        boards_y1 = boards_y0 + cell_h

        draw.rectangle((0, info_y0, width, info_y0 + info_bar_h), fill=(245, 247, 250, 255))
        draw.line((0, info_y0, width, info_y0), fill=grid, width=2)
        draw.line((0, boards_y0, width, boards_y0), fill=grid, width=2)
        draw.line((0, boards_y1, width, boards_y1), fill=grid, width=2)

        ok = bool(res.get("valid"))
        status = "OK" if ok else "FAIL"
        move_uci = str(res.get("move_uci", "") or "")
        move_san = str(res.get("move_san", "") or "")
        err = str(res.get("error", "") or "")

        input_fen = str(res.get("input_fen", "") or "")
        out_fen = str(res.get("fen_after", "") or "")

        # eval
        q: Dict[str, Any] = {}
        if eval_moves and ok and input_fen and move_uci:
            LOG.info(
                "eval start | suite=%s case=%s method=%s engine=stockfish move=%s depth=%d fen_len=%d",
                suite_id, case_id, method_name, move_uci, sf_depth, len(input_fen),
            )
            LOG.debug("eval input fen: %s", input_fen)

            q = compute_move_quality_stockfish_only(
                input_fen=input_fen,
                move_uci=move_uci,
                stockfish=stockfish,
                sf_depth=sf_depth,
            )

            if "error" in q:
                LOG.warning("eval failed | method=%s | %s", method_name, q)
            else:
                LOG.info("eval ok | method=%s label=%s drop=%s best=%s",
                         method_name, q.get("label"), q.get("drop_cp"), q.get("bestmove_uci"))
                LOG.info(
                    "eval details | method=%s before(mover)=%s after(mover)=%s drop=%s best=%s best_san=%s depth=%s→%s",
                    method_name,
                    q.get("before_cp_mover"),
                    q.get("after_cp_mover"),
                    q.get("drop_cp"),
                    q.get("bestmove_uci"),
                    q.get("bestmove_san"),
                    q.get("depth_before"),
                    q.get("depth_after"),
                )
                LOG.debug("eval fen_after: %s", q.get("fen_after"))

            if eval_sleep_ms > 0:
                time.sleep(eval_sleep_ms / 1000.0)

        pad = 22
        left_x = pad
        top_y = info_y0 + 18

        draw.text((left_x, top_y), f"Method: {method_name}", font=font_kv, fill=(20, 20, 20, 255))
        draw.text((left_x, top_y + 44), f"Status: {status}", font=font_kv, fill=(20, 20, 20, 255))
        draw.text((left_x, top_y + 88), f"SAN: {move_san or '-'}   |   UCI: {move_uci or '-'}", font=font_small, fill=(40, 40, 40, 255))

        if err:
            err_lines = wrap_text_pixels(draw, f"Error: {err}", font_small, int(width * 0.60))
            yy = top_y + 130
            for ln in err_lines[:3]:
                draw.text((left_x, yy), ln, font=font_small, fill=(120, 0, 0, 255))
                yy += 34

        divider_x = int(width * 0.62)
        draw.line((divider_x, info_y0 + 12, divider_x, info_y0 + info_bar_h - 12), fill=(0, 0, 0, 60), width=2)

        right_x = divider_x + 18
        right_pad_r = pad
        right_w = width - right_x - right_pad_r

        if not eval_moves or not ok or not input_fen or not move_uci:
            lines = wrap_text_pixels(draw, "EVAL: (disabled)", font_kv, right_w)
            yy = top_y
            for ln in lines:
                draw.text((right_x, yy), ln, font=font_kv, fill=(40, 40, 40, 255))
                yy += 44
        elif "error" in q:
            msg = str(q.get("error"))
            lines = wrap_text_pixels(draw, f"EVAL ERROR: {msg}", font_small, right_w)
            yy = top_y
            for i, ln in enumerate(lines[:4]):
                draw.text((right_x, yy), ln, font=(font_kv if i == 0 else font_small), fill=(120, 0, 0, 255))
                yy += 44 if i == 0 else 34
        else:
            label = str(q.get("label", "-"))
            drop_cp = float(q.get("drop_cp", 0.0))
            bm_san = str(q.get("bestmove_san", "-"))
            bm_uci = str(q.get("bestmove_uci", "-"))
            d1 = q.get("depth_before")
            d2 = q.get("depth_after")

            bw, _ = draw_badge(draw, right_x, top_y, label, font=font_kv)

            meta = f"Engine: stockfish   |   Depth: {d1}→{d2}"
            meta_lines = wrap_text_pixels(draw, meta, font_small, max(10, right_w - bw - 18))
            mx = right_x + bw + 14
            my = top_y + 6
            for ln in meta_lines[:2]:
                draw.text((mx, my), ln, font=font_small, fill=(40, 40, 40, 255))
                my += 34

            draw.text((right_x, top_y + 64), f"Drop: {drop_cp} cp", font=font_small, fill=(40, 40, 40, 255))
            bar_w = min(520, right_w - 10)
            draw_drop_bar(draw, right_x, top_y + 100, bar_w, 26, drop_cp)

            bm_line = f"Best: {bm_san} ({bm_uci})"
            bm_lines = wrap_text_pixels(draw, bm_line, font_small, right_w)
            yy = top_y + 140
            for ln in bm_lines[:2]:
                draw.text((right_x, yy), ln, font=font_small, fill=(40, 40, 40, 255))
                yy += 34

        # boards
        safe_in = sanitize_fen(input_fen) if input_fen else None
        safe_out = sanitize_fen(out_fen) if out_fen else None

        if safe_in:
            best_lastmove = None
            if q and "error" not in q:
                bm = str(q.get("bestmove_uci") or "")
                if bm and bm != "-":
                    best_lastmove = bm

            img_in = render_board_png(safe_in, best_lastmove, board_size, coordinates=include_coordinates)
            paste_center(canvas, img_in, (0 * cell_w, boards_y0, 1 * cell_w, boards_y1))

            img_prop = render_board_png(safe_in, move_uci if ok else None, board_size, coordinates=include_coordinates)
            paste_center(canvas, img_prop, (1 * cell_w, boards_y0, 2 * cell_w, boards_y1))

        if safe_out and ok:
            img_out = render_board_png(safe_out, None, board_size, coordinates=include_coordinates)
            paste_center(canvas, img_out, (2 * cell_w, boards_y0, 3 * cell_w, boards_y1))
        else:
            draw_centered(draw, (2 * cell_w, boards_y0, 3 * cell_w, boards_y1), "N/A", font_hdr)

    ensure_dir(os.path.join(out_dir, suite_id, case_id))
    out_path = os.path.join(out_dir, suite_id, case_id, "comparison.png")
    rgb = canvas.convert("RGB")
    rgb.save(out_path, "PNG", dpi=(dpi, dpi))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--suite-id", default="")
    ap.add_argument("--board-size", type=int, default=520)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--coordinates", action="store_true")

    ap.add_argument("--eval", action="store_true", help="Enable Stockfish scoring (required for eval box)")
    ap.add_argument("--eval-sleep-ms", type=int, default=0, help="Optional sleep between eval calls")

    ap.add_argument("--stockfish-path", default="", help="Optional path to stockfish binary")
    ap.add_argument("--stockfish-depth", type=int, default=15, help="Depth for local stockfish")
    ap.add_argument("--stockfish-threads", type=int, default=4)
    ap.add_argument("--stockfish-hash-mb", type=int, default=128)

    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--log-file", default="")

    args = ap.parse_args()
    setup_logging(args.log_level, args.log_file)

    if cairosvg is None:
        raise SystemExit("cairosvg is required. Install: pip install cairosvg")

    if args.eval:
        sf_path = find_stockfish_path(args.stockfish_path)
        if not sf_path:
            raise SystemExit("Stockfish not found. Install it or pass --stockfish-path.")
        stockfish = StockfishEngine(sf_path, threads=args.stockfish_threads, hash_mb=args.stockfish_hash_mb)
        stockfish.start()
    else:
        stockfish = None  # type: ignore

    LOG.info(
        "config | eval=%s stockfish_depth=%d threads=%d hash=%d board_size=%d dpi=%d",
        args.eval, args.stockfish_depth, args.stockfish_threads, args.stockfish_hash_mb, args.board_size, args.dpi,
    )

    try:
        suite_ids = [args.suite_id] if args.suite_id else find_suite_ids(args.results_dir)
        if not suite_ids:
            raise SystemExit("No suites found in results-dir")

        for suite_id in suite_ids:
            suite_dir = os.path.join(args.results_dir, suite_id)
            if not os.path.isdir(suite_dir):
                continue
            case_ids = list_case_ids(suite_dir)
            for case_id in case_ids:
                LOG.info("build report | suite=%s case=%s", suite_id, case_id)
                out_path = build_case_report_image(
                    results_dir=args.results_dir,
                    suite_id=suite_id,
                    case_id=case_id,
                    out_dir=args.out_dir,
                    board_size=args.board_size,
                    include_coordinates=bool(args.coordinates),
                    dpi=args.dpi,
                    eval_moves=bool(args.eval),
                    eval_sleep_ms=int(args.eval_sleep_ms),
                    stockfish=stockfish if stockfish is not None else None,  # type: ignore
                    sf_depth=int(args.stockfish_depth),
                )
                LOG.info("wrote: %s", out_path)
    finally:
        if args.eval and stockfish:
            stockfish.close()
            LOG.info("stockfish closed")


if __name__ == "__main__":
    main()
