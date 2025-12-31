import argparse
import io
import json
import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.svg
from PIL import Image, ImageDraw, ImageFont

try:
    import cairosvg
except Exception:
    cairosvg = None


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


def wrap_text(s: str, width: int) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=True, break_on_hyphens=False))


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


def build_case_report_image(
    results_dir: str,
    suite_id: str,
    case_id: str,
    out_dir: str,
    board_size: int,
    include_coordinates: bool,
    row_label_w: int,
    dpi: int,
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

    font_title = get_font(32)
    font_hdr = get_font(24)
    font_row = get_font(22)
    font_small = get_font(18)

    title_h = 86
    col_hdr_h = 54
    cell_pad = 18
    gap = 0

    cell_w = board_size + cell_pad * 2
    cell_h = board_size + cell_pad * 2

    width = row_label_w + col_count * cell_w
    height = title_h + col_hdr_h + row_count * cell_h

    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title = f"Suite: {suite_id}   |   Case: {case_id}"
    draw_centered(draw, (0, 0, width, title_h), title, font_title)

    y_hdr0 = title_h
    for ci, col in enumerate(cols):
        x0 = row_label_w + ci * cell_w
        x1 = x0 + cell_w
        draw_centered(draw, (x0, y_hdr0, x1, y_hdr0 + col_hdr_h), col, font_hdr)

    grid_color = (0, 0, 0, 255)
    draw.line((0, title_h, width, title_h), fill=grid_color, width=2)
    draw.line((0, title_h + col_hdr_h, width, title_h + col_hdr_h), fill=grid_color, width=2)
    draw.line((row_label_w, title_h, row_label_w, height), fill=grid_color, width=2)

    for ci in range(col_count + 1):
        x = row_label_w + ci * cell_w
        draw.line((x, title_h, x, height), fill=grid_color, width=2)

    for ri in range(row_count + 1):
        y = title_h + col_hdr_h + ri * cell_h
        draw.line((0, y, width, y), fill=grid_color, width=2)

    for ri, (method_name, res) in enumerate(rows):
        y0 = title_h + col_hdr_h + ri * cell_h
        y1 = y0 + cell_h

        ok = bool(res.get("valid"))
        status = "OK" if ok else "FAIL"
        move_uci = str(res.get("move_uci", "") or "")
        move_san = str(res.get("move_san", "") or "")
        err = str(res.get("error", "") or "")

        label_lines = [
            method_name,
            status,
            f"SAN: {move_san or '-'}",
            f"UCI: {move_uci or '-'}",
        ]
        if err:
            label_lines.append(wrap_text(err, 28))

        label = "\n".join(label_lines)
        draw.multiline_text((14, y0 + 14), label, font=font_row, fill=(0, 0, 0, 255), spacing=6)

        input_fen = str(res.get("input_fen", "") or "")
        out_fen = str(res.get("fen_after", "") or "")

        if input_fen:
            img_in = render_board_png(input_fen, None, board_size, coordinates=include_coordinates)
            paste_center(canvas, img_in, (row_label_w + 0 * cell_w, y0, row_label_w + 1 * cell_w, y1))

            img_prop = render_board_png(input_fen, move_uci if ok else None, board_size, coordinates=include_coordinates)
            paste_center(canvas, img_prop, (row_label_w + 1 * cell_w, y0, row_label_w + 2 * cell_w, y1))

        if out_fen and ok:
            img_out = render_board_png(out_fen, None, board_size, coordinates=include_coordinates)
            paste_center(canvas, img_out, (row_label_w + 2 * cell_w, y0, row_label_w + 3 * cell_w, y1))
        else:
            draw_centered(
                draw,
                (row_label_w + 2 * cell_w, y0, row_label_w + 3 * cell_w, y1),
                "N/A",
                font_small,
            )

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
    ap.add_argument("--row-label-width", type=int, default=340)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--coordinates", action="store_true")
    args = ap.parse_args()

    if cairosvg is None:
        raise SystemExit("cairosvg is required. Install: pip install cairosvg")

    suite_ids = [args.suite_id] if args.suite_id else find_suite_ids(args.results_dir)
    if not suite_ids:
        raise SystemExit("No suites found in results-dir")

    for suite_id in suite_ids:
        suite_dir = os.path.join(args.results_dir, suite_id)
        if not os.path.isdir(suite_dir):
            continue
        case_ids = list_case_ids(suite_dir)
        for case_id in case_ids:
            print(f"[report] suite={suite_id} case={case_id}")
            out_path = build_case_report_image(
                results_dir=args.results_dir,
                suite_id=suite_id,
                case_id=case_id,
                out_dir=args.out_dir,
                board_size=args.board_size,
                include_coordinates=bool(args.coordinates),
                row_label_w=args.row_label_width,
                dpi=args.dpi,
            )
            print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
