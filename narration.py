"""
動画のナレーション音声（VOICEVOX:ずんだもん）を作る。

VOICEVOX ENGINE を localhost:50021 で起動しておく（例：docker run -p 50021:50021 voicevox/voicevox_engine:cpu-latest）。
同じ文は作り直さず、runs/voice/ のキャッシュを使う。
"""
import hashlib
import io
import json
import os
import urllib.parse
import urllib.request
import wave

import numpy as np

URL = "http://localhost:50021"
SPEAKER = 3  # ずんだもん（ノーマル）
RATE = 24000
SPEED = 1.15
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "voice")


def synth(text):
    """text を読み上げた音声（float32, RATE Hz, モノラル）を返す。"""
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.sha1(f"{SPEAKER}:{SPEED}:{text}".encode()).hexdigest()[:16]
    path = os.path.join(CACHE, f"{key}.wav")
    if not os.path.exists(path):
        q = urllib.request.urlopen(urllib.request.Request(
            f"{URL}/audio_query?" + urllib.parse.urlencode({"text": text, "speaker": SPEAKER}), method="POST"))
        query = json.load(q)
        query.update(speedScale=SPEED, outputSamplingRate=RATE, outputStereo=False)
        wav = urllib.request.urlopen(urllib.request.Request(
            f"{URL}/synthesis?speaker={SPEAKER}", data=json.dumps(query).encode(),
            headers={"Content-Type": "application/json"})).read()
        with open(path, "wb") as f:
            f.write(wav)
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def seconds(text):
    return len(synth(text)) / RATE


def write_track(events, total, path):
    """events = [(開始秒, text)] を total 秒の音声トラックに並べて WAV に書く。"""
    track = np.zeros(int(total * RATE) + RATE, dtype=np.float32)
    for start, text in events:
        a = synth(text)
        i = int(start * RATE)
        track[i:i + len(a)] += a[: len(track) - i]
    track = np.clip(track[: int(total * RATE)], -1, 1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((track * 32767).astype(np.int16).tobytes())
    with open(path, "wb") as f:
        f.write(buf.getvalue())
