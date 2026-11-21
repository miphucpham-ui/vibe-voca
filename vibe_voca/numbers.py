"""Digits -> Vietnamese words (Northern reading), for Vietnamese sentences of the front end.

    2024 -> hai nghìn không trăm hai mươi tư; 21 -> hai mươi mốt; 15 -> mười lăm; 105 -> một trăm linh năm;
    1.250.000 -> một triệu hai trăm năm mươi nghìn; 12,5 -> mười hai phẩy năm; 50% -> năm mươi phần trăm;
    0912345678 (a leading zero, or more than 12 digits) -> read digit by digit.
"""

import re

DIGITS = "không một hai ba bốn năm sáu bảy tám chín".split()
UNITS = ["", "nghìn", "triệu", "tỷ"]


def read_hundreds(n, full):
    """0..999 -> words; ``full`` says "không trăm" / "linh" inside a longer number (2005 = hai nghìn không trăm linh năm)."""
    h, t, u = n // 100, n // 10 % 10, n % 10
    words = [DIGITS[h], "trăm"] if h or full else []
    if t == 0:
        if u and (h or full):
            words.append("linh")
    elif t == 1:
        words.append("mười")
    else:
        words += [DIGITS[t], "mươi"]
    if u:
        if u == 1 and t >= 2:
            words.append("mốt")
        elif u == 5 and t >= 1:
            words.append("lăm")
        elif u == 4 and t >= 2:
            words.append("tư")
        else:
            words.append(DIGITS[u])
    return words


def read_integer(digits):
    if len(digits) > 1 and digits.startswith("0") or len(digits) > 12:    # phone numbers, codes: digit by digit
        return " ".join(DIGITS[int(d)] for d in digits)
    n = int(digits)
    if n == 0:
        return "không"
    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    words = []
    for i in range(len(groups) - 1, -1, -1):
        if groups[i]:
            words += read_hundreds(groups[i], full=i < len(groups) - 1) + ([UNITS[i]] if UNITS[i] else [])
    return " ".join(words)


def expand_numbers(text):
    text = re.sub(r"\d{1,3}(?:\.\d{3})+(?!\d)", lambda m: m.group(0).replace(".", ""), text)   # 1.250.000
    text = re.sub(r"(\d+),(\d+)", lambda m: f"{read_integer(m.group(1))} phẩy {read_integer(m.group(2))}", text)
    text = re.sub(r"\d+", lambda m: f" {read_integer(m.group(0))} ", text)
    return re.sub(r"[ \t]+", " ", text.replace("%", " phần trăm")).strip()
