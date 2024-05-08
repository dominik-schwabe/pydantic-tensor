from __future__ import annotations

import math


def base64_num_bytes(b64: str):
    num_padding = 0
    while num_padding < len(b64) and b64[-num_padding - 1] == "=":
        num_padding += 1
    return math.floor(((len(b64) - num_padding) * 3) / 4)
