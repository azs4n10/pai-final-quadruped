"""
日本語の指示 → 経由地リスト への変換（ローカル LLM / Ollama）

  python planner.py "赤い柱に寄ってから青い柱へ行って、最後はホームに戻って"
"""
import json
import re
import sys
import urllib.request

from goal_ant import LANDMARK_JA, LANDMARKS

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"

SYSTEM = """あなたは四足歩行配送ロボットの経路プランナーです。
ユーザーの日本語の指示を読み、ロボットが順に訪れる地点のリストを JSON で返します。

使える地点（キー: 説明, 座標[m]）:
{places}

ルール:
- route には上のキーだけを、訪れる順に並べる。
- 「戻る」「帰る」はロボットが出発した home を指す。
- 「全部」「一周」などは、指示に順番がなければ近い順に回る。
- ロボットは home から出発する。最初の地点に home は入れない。
- reply には、ロボットとして何をするかを日本語で短く（30字以内）書く。
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "array", "items": {"type": "string", "enum": list(LANDMARKS)}},
        "reply": {"type": "string"},
    },
    "required": ["route", "reply"],
}


def _places_text():
    lines = []
    for k, ((x, y), _) in LANDMARKS.items():
        desc = "出発地点（白い円）" if k == "home" else LANDMARK_JA[k]
        lines.append(f"- {k}: {desc}, ({x:+.0f}, {y:+.0f})")
    return "\n".join(lines)


def plan_with_llm(instruction, model=MODEL, timeout=300):
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "format": SCHEMA,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM.format(places=_places_text())},
            {"role": "user", "content": instruction},
        ],
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        content = json.loads(r.read())["message"]["content"]
    out = json.loads(content)
    route = [k for k in out.get("route", []) if k in LANDMARKS]
    return {"route": route, "reply": out.get("reply", ""), "source": model}


def plan_with_keywords(instruction):
    """LLM が使えないとき用：文中に出てくる色を出現順に拾うだけの簡易版。"""
    hits = []
    words = {"home": r"ホーム|戻|帰", "red": "赤", "blue": "青", "green": "緑", "yellow": "黄"}
    for k, pat in words.items():
        for m in re.finditer(pat, instruction):
            hits.append((m.start(), k))
    route = [k for _, k in sorted(hits)]
    return {"route": route, "reply": "", "source": "keyword"}


def plan(instruction):
    try:
        return plan_with_llm(instruction)
    except Exception as e:  # Ollama 未起動など
        print(f"[planner] LLM unavailable ({e}); fallback to keywords", file=sys.stderr)
        return plan_with_keywords(instruction)


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "赤い柱に寄ってから青い柱へ行って、最後はホームに戻って"
    print(json.dumps(plan(text), ensure_ascii=False))
