"""Voice / text commands for LidSort (and CupBasket: ``--task cupbasket``, "紅杯子放A，藍碗放C" -> [[0, 0], [3, 2]]): speech -> text (faster-whisper) -> ordered (cube, box) commands.

    python voice_command.py --text "紅色放A，藍色放C，綠色放B"
    python voice_command.py --audio cmd.wav [--model small] [--device cuda]
    python voice_command.py --mic 5          # record 5 s from the default microphone (Jetson Thor / laptop)

Cubes: 0 red / 1 green / 2 blue (``lid_sort_env_cfg.CUBE_COLORS``). Boxes: 0 A / 1 B / 2 C, also by colour
(yellow / purple / cyan, ``BOX_COLORS``) or number (1/2/3, 一/二/三). Every clause names a cube colour and a box, in either
order: "把紅色方塊放進A箱", "red cube to box C", "藍色的放到黃色箱子". Output: JSON list of [cube, box] in spoken order.

Take-out (LidSort v2): "把紅色方塊從A拿出來", "take the red cube out of box A", "remove blue from C" -> [cube, box, 1].
Move: "move red from A to B", "把紅色從A移到B" -> take-out then put-in, [[0, 0, 1], [0, 1]]. "take red out of A and put
it in B" also works ("it" = the cube just taken out).
"""

from __future__ import annotations

import argparse
import json
import re

CUBE_WORDS = {
    0: ("紅", "红", "red"),
    1: ("綠", "绿", "green"),
    2: ("藍", "蓝", "blue"),
}
BOX_COLOR_WORDS = {
    0: ("黃", "黄", "yellow"),
    1: ("紫", "purple"),
    2: ("青", "cyan", "天藍", "天蓝"),
}
BOX_NUMBERS = {"1": 0, "2": 1, "3": 2, "一": 0, "二": 1, "三": 2}
# Whisper prompt: biases the decoder toward the command vocabulary.
PROMPT = ("把紅色方塊放進A箱，綠色放B箱，藍色放C箱，把紅色從A拿出來。"
          "Put the red cube in box A, green in box B, blue in box C, take the red cube out of box A.")
# Take-out words (the next box named is where the cube comes OUT of) and source words ("from A").
OUT_WORDS = (r"\bout\b", r"\bremove\b", r"\btake\s+away\b", "拿出", "取出", "拿走", "取走", "出來", "出来", "移出")
FROM_WORDS = (r"\bfrom\b", "從", "从")


def _find(words: tuple[str, ...], text: str) -> list[int]:
    out = []
    for w in words:
        out += [m.start() for m in re.finditer(re.escape(w), text)]
    return out


def parse_command(text: str) -> list[tuple[int, ...]]:
    """Ordered commands from the cube / box / take-out / "from" mentions in spoken order. Put-in: (cube, box); each cube
    pairs with the next box mentioned (or the box said just before it, "A箱放紅色"). A box named after a take-out or
    "from" word is the cube's SOURCE: with a take-out word -> (cube, box, 1); followed by another box ("from A to B")
    -> (cube, A, 1), (cube, B) (a move). Works without punctuation, as speech recognition often returns none."""
    t = text.lower().replace("盒", "箱")
    tokens = _cube_tokens(t) + _box_tokens(t) + _marker_tokens(t)
    result: list[tuple[int, ...]] = []
    cube = src = dst = None
    out = want_src = False
    last_out_cube = None  # "... out of A and put it in B": a lone box after a take-out puts that cube there

    def clear():
        nonlocal cube, src, dst, out, want_src
        cube = src = dst = None
        out = want_src = False

    for _, kind, idx in sorted(tokens):
        if kind == "cube":
            if cube is not None and src is not None:  # "red from A" then a new cube: red was taken out
                result.append((cube, src, 1))
                clear()
            cube = idx
        elif kind == "out":
            out = want_src = True
        elif kind == "from":
            want_src = True
        else:
            if want_src and src is None:
                src = idx
                want_src = False
            elif cube is None and src is None and dst is None and last_out_cube is not None:
                result.append((last_out_cube, idx))
                last_out_cube = None
                continue
            else:
                dst = idx
        if cube is not None and src is not None and dst is not None:
            result += [(cube, src, 1), (cube, dst)]
            last_out_cube = None
            clear()
        elif cube is not None and src is not None and out:
            result.append((cube, src, 1))
            last_out_cube = cube
            clear()
        elif cube is not None and dst is not None and src is None and not want_src:
            result.append((cube, dst))
            last_out_cube = None
            clear()
    if cube is not None and src is not None:  # "red from A" at the end: take it out
        result.append((cube, src, 1))
    return result


def _marker_tokens(t: str) -> list[tuple[int, str, int]]:
    """Take-out and "from" words (overlapping take-out words, "拿出" + "出來" in "拿出來", count once)."""
    toks = []
    for kind, words in (("out", OUT_WORDS), ("from", FROM_WORDS)):
        spans = sorted((m.start(), m.end()) for w in words for m in re.finditer(w, t))
        end = -1
        for a, b in spans:
            if a >= end:
                toks.append((a, kind, -1))
            end = max(end, b)
    return toks


def _box_colour_spans(c: str) -> set[int]:
    """Character positions of box-colour words, so "天藍" (cyan box) is not read as the blue cube."""
    spans = set()
    for words in BOX_COLOR_WORDS.values():
        for w in words:
            for m in re.finditer(re.escape(w), c):
                spans.update(range(m.start(), m.end()))
    return spans


def _cube_tokens(t: str) -> list[tuple[int, str, int]]:
    skip = _box_colour_spans(t)
    return [(p, "cube", i) for i, words in CUBE_WORDS.items() for p in _find(words, t) if p not in skip]


def _box_tokens(t: str) -> list[tuple[int, str, int]]:
    found: dict[int, int] = {}  # position -> box index
    # Letters: "box a", "a箱", "a 號箱", lone b / c, lone a unless it is the English article ("a red cube").
    for pattern in (r"(?:box|箱子?)\s*([abc])(?![a-z])", r"(?<![a-z])([abc])\s*(?:號|号)?\s*箱",
                    r"(?<![a-z])([bc])(?![a-z])", r"(?<![a-z])(a)(?![a-z])(?!\s+\w+\s+(?:cube|one|block))"):
        for m in re.finditer(pattern, t):
            found.setdefault(m.start(1), "abc".index(m.group(1)))
    # Numbers: "box 1", "1號箱", "第一個箱子".
    for pattern in (r"(?:box|箱子?)\s*([123一二三])", r"([123一二三])\s*(?:號|号)", r"第\s*([123一二三])"):
        for m in re.finditer(pattern, t):
            found.setdefault(m.start(1), BOX_NUMBERS[m.group(1)])
    # Box colours (yellow / purple / cyan boxes).
    for j, words in BOX_COLOR_WORDS.items():
        for p in _find(words, t):
            found.setdefault(p, j)
    return [(p, "box", j) for p, j in found.items()]


# ---------------------------------------------------------------- CupBasket (cups / bowls -> baskets A/B/C)

# Objects (``cup_basket_env_cfg.OBJ_NAMES``): 0 red cup, 1 green cup, 2 blue cup, 3 red bowl, 4 green bowl, 5 blue bowl.
# Colour AND type name an object ("紅碗", "green cup"); a colour without a type word is ambiguous and dropped.
CB_COLOR_WORDS = {0: ("紅", "红", "red"), 1: ("綠", "绿", "green"), 2: ("藍", "蓝", "blue")}
CB_TYPE_WORDS = {0: ("杯", "cup", "mug"), 1: ("碗", "bowl")}
CB_PROMPT = ("紅碗放B，綠杯放A。把藍色杯子放進C籃，紅杯子放A，綠碗放C。"
             "Put the red cup in basket A and the blue bowl in basket C.")


def _cb_basket_tokens(t: str) -> list[tuple[int, str, int]]:
    found: dict[int, int] = {}
    for pattern in (r"(?:box|basket|bin|籃子?|篮子?|筐|箱子?)\s*([abc])(?![a-z])",
                    r"(?<![a-z])([abc])\s*(?:號|号)?\s*(?:籃|篮|筐|箱)",
                    r"(?<![a-z])([bc])(?![a-z])",
                    r"(?<![a-z])(a)(?![a-z])(?!\s+\w+\s+(?:cup|bowl|mug|one))"):
        for m in re.finditer(pattern, t):
            found.setdefault(m.start(1), "abc".index(m.group(1)))
    for pattern in (r"(?:box|basket|bin|籃子?|篮子?|筐|箱子?)\s*([123一二三])", r"([123一二三])\s*(?:號|号)",
                    r"第\s*([123一二三])"):
        for m in re.finditer(pattern, t):
            found.setdefault(m.start(1), BOX_NUMBERS[m.group(1)])
    return [(p, "basket", j) for p, j in found.items()]


def parse_cup_basket(text: str) -> list[tuple[int, int]]:
    """Ordered (object, basket) pairs: "紅碗放B，綠杯放A", "把藍色的杯子放到C籃", "red cup to basket B", "A籃放綠碗".
    An object = colour + type word (either order within the clause: "杯子紅色的" also works); it pairs with the next
    basket named, or with a basket said just before it. A clause without both colour and type is dropped (warning)."""
    t = text.lower()
    toks = [(p, "color", i) for i, ws in CB_COLOR_WORDS.items() for p in _find(ws, t)]
    toks += [(p, "type", k) for k, ws in CB_TYPE_WORDS.items() for p in _find(ws, t)]
    toks += _cb_basket_tokens(t)
    toks.sort()
    out: list[tuple[int, int]] = []
    color = typ = basket = None
    for _, kind, idx in toks:
        if kind == "color":
            if color is not None and typ is None:
                print("[voice] object needs colour and type (e.g. 紅碗): clause dropped", flush=True)
            color = idx
        elif kind == "type":
            typ = idx
        else:
            basket = idx
        if color is not None and typ is not None and basket is not None:
            out.append((3 * typ + color, basket))
            color = typ = basket = None
    if color is not None or typ is not None:
        print("[voice] incomplete clause at the end (object needs colour + type + basket): dropped", flush=True)
    return out


def cup_basket_instruction(text: str) -> list[list[int]]:
    """One CupBasket step = up to two (object, basket) pairs carried out by both arms at once:
    "紅碗放B，綠杯放A" -> [[3, 1, 1], [2, 0, 1]]; a single pair -> [[k, j, 1], [0, 0, 0]] (second slot inactive = that arm
    stays home). Arm assignment (nearest arm, 2 x 2 min distance) is done by the env / robot from the object poses, so
    the order here is the spoken order. More than two pairs -> consecutive steps (chained)."""
    pairs = parse_cup_basket(text)
    steps = [pairs[i:i + 2] for i in range(0, len(pairs), 2)] or [[]]
    return [[[k, j, 1] for k, j in st] + [[0, 0, 0]] * (2 - len(st)) for st in steps]


def transcribe(audio, model: str = "small", device: str = "auto", language: str | None = None, prompt: str = PROMPT) -> str:
    """Speech to text with faster-whisper (CTranslate2; runs on the Jetson GPU or CPU)."""
    from faster_whisper import WhisperModel

    if isinstance(audio, str) and audio.lower().endswith(".wav"):
        audio = load_wav(audio)
    m = WhisperModel(model, device=device, compute_type="auto")
    segments, _ = m.transcribe(audio, language=language, initial_prompt=prompt, beam_size=5, vad_filter=True)
    return "".join(s.text for s in segments).strip()


def load_wav(path: str, rate: int = 16000):
    """16 kHz mono float32 from a PCM .wav (stdlib only; avoids PyAV version issues on the Jetson)."""
    import wave

    import numpy as np

    with wave.open(path) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype={1: np.uint8, 2: np.int16, 4: np.int32}[width]).astype(np.float32)
    x = (x - 128.0) / 128.0 if width == 1 else x / float(2 ** (8 * width - 1))
    x = x.reshape(-1, ch).mean(1)
    if sr != rate:
        t = np.arange(0, len(x) / sr, 1.0 / rate)
        x = np.interp(t, np.arange(len(x)) / sr, x).astype(np.float32)
    return x


def record(seconds: float, rate: int = 16000):
    import sounddevice as sd

    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--audio")
    g.add_argument("--mic", type=float, help="seconds to record")
    p.add_argument("--model", default="small")
    p.add_argument("--device", default="auto")
    p.add_argument("--language", default=None)
    p.add_argument("--task", default="lidsort", choices=("lidsort", "cupbasket"))
    a = p.parse_args()
    if a.text is not None:
        text = a.text
    else:
        src = a.audio if a.audio else record(a.mic)
        text = transcribe(src, a.model, a.device, a.language, CB_PROMPT if a.task == "cupbasket" else PROMPT)
    cmds = cup_basket_instruction(text) if a.task == "cupbasket" else parse_command(text)
    print(json.dumps({"text": text, "commands": cmds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
