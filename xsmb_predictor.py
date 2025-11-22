#!/usr/bin/env python3
# xsmb_predictor.py
"""
Trợ lý dự đoán kết quả XS Miền Bắc (2 chữ số & 3 chữ số cuối của giải Đặc Biệt).
- Thu thập kết quả lịch sử ~30 ngày từ các site (configurable).
- Trích số giải đặc biệt (full number), lấy 2 và 3 cuối.
- Sinh 10 phương án cho 2-chữ số và 10 phương án cho 3-chữ số bằng 3 chiến lược:
    A. Top frequency (tần suất cao nhất)
    B. Markov-order1 (dự đoán 3-chữ bằng mô hình chuyển trạng thái trên chuỗi)
    C. Weighted random (cân nhắc cả hot & recent)
Outputs: CSV và in ra màn hình.
CAVEAT: Đây là công cụ thống kê/khảo sát — không đảm bảo trúng.
"""

import re
import time
import random
import argparse
from datetime import datetime, timedelta
from collections import Counter, defaultdict

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tqdm import tqdm

# -----------------------
# Configuration
# -----------------------
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
HEADERS = {"User-Agent": USER_AGENT}
SITES = [
    "https://www.minhngoc.net.vn/",
    "https://xskt.com.vn/xsmb",
    "https://xoso.com.vn/xo-so-mien-bac/xsmb-p1.html",
    "https://xsmn.mobi/xsmb-xo-so-mien-bac.html",
    "https://ketqua.com.vn/xsmb",
]
# You can extend or change SITES if needed.

# -----------------------
# Helpers: fetch & parse
# -----------------------
def fetch_url(url, timeout=10, max_retries=3, pause=1.0):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.text
            else:
                time.sleep(pause)
        except Exception:
            time.sleep(pause)
    return None

def parse_minhngoc(html):
    """Parse minhngoc.net.vn day result page to find special prize or list of results.
       Implementation uses regex to find numbers with 5 digits (VN special prize length).
       This parser is tolerant: extracts any 5-digit segment found near 'ĐB' or 'Đặc biệt' keywords.
    """
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    # Look for 'ĐB' or 'Đặc biệt' then 3-6 digits
    patterns = [r"(?:ĐB|Đặ[c|n] biệt|Đặc biệt)[^\d]{0,10}(\d{3,6})", r"Giải Đặc Biệt[^\d]{0,10}(\d{3,6})"]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    # fallback: find any 3-6 digit with context 'ĐB' or 'Đặc biệt' anywhere
    m = re.search(r"(\d{3,6})", text)
    if m:
        return m.group(1)
    return None

# For other sites we use a generic approach: find 'ĐB' or 'Giải Đặc Biệt' then nearby digits.
def extract_special_from_html(html):
    if not html:
        return None
    # try minhngoc style
    sp = parse_minhngoc(html)
    if sp:
        return sp
    # generic
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    # find occurrences of 'ĐB' or 'Đặc biệt' and take the nearest number token
    tokens = text.split()
    for i,t in enumerate(tokens):
        if re.search(r"ĐB|Đặc|Đặt|Giải", t, flags=re.IGNORECASE):
            # look ahead for a number token
            for j in range(i, min(i+8, len(tokens))):
                m = re.search(r"(\d{3,6})", tokens[j])
                if m:
                    return m.group(1)
    # fallback global search for 3-6 digit
    m = re.search(r"(\d{3,6})", text)
    if m:
        return m.group(1)
    return None

# -----------------------
# Build list of date-specific URLs to try
# -----------------------
def candidate_urls_for_date(dt: datetime):
    """Return candidate URLs (heuristic) for the given date to try scraping.
       Many sites accept date params or have pages like /ngay-xx-yy
       We'll attempt some common forms (day-month-year, year-month-day).
    """
    ymd = dt.strftime("%Y-%m-%d")
    dmy = dt.strftime("%d/%m/%Y")
    dmy_dash = dt.strftime("%d-%m-%Y")
    dmy_nospace = dt.strftime("%d%m%Y")
    # common patterns per listed sites (heuristic)
    candidates = [
        f"https://www.minhngoc.net.vn/xsmb-{dmy_dash}.html",
        f"https://www.minhngoc.net.vn/kqxsmb-{dmy_dash}.html",
        f"https://xskt.com.vn/xsmb/{ymd}",
        f"https://xoso.com.vn/xo-so-mien-bac/{dmy_dash}",
        f"https://xsmn.mobi/xsmb-{dmy_dash}.html",
        f"https://ketqua.com.vn/xsmb/{dmy_dash}",
        # base site pages as fallback
        "https://www.minhngoc.net.vn/",
        "https://xskt.com.vn/xsmb",
        "https://xoso.com.vn/xo-so-mien-bac/xsmb-p1.html",
        "https://xsmn.mobi/xsmb-xo-so-mien-bac.html",
        "https://ketqua.com.vn/xsmb",
    ]
    return list(dict.fromkeys(candidates))  # unique preserve order

# -----------------------
# Crawl last N days
# -----------------------
def crawl_special_last_n_days(n_days=30, end_date=None, verbose=True):
    """
    Crawl from (end_date - n_days) .. (end_date - 1)
    By default end_date = today (so we gather previous days). We want ~30 days before 'today'.
    Returns dict date_str -> special_number (string)
    """
    if end_date is None:
        end_date = datetime.now()
    results = {}
    for delta in range(1, n_days+1):
        dt = end_date - timedelta(days=delta)
        date_key = dt.strftime("%Y-%m-%d")
        found = None
        # try candidate urls
        for url in candidate_urls_for_date(dt):
            html = fetch_url(url)
            if not html:
                continue
            sp = extract_special_from_html(html)
            if sp:
                # normalize: keep digits only
                sp = re.sub(r"\D", "", sp)
                # ensure length reasonable (3-6). pad if necessary? We'll keep as found.
                if 3 <= len(sp) <= 6:
                    found = sp
                    break
        if not found and verbose:
            print(f"[WARN] Không tìm thấy ĐB cho {date_key} từ các nguồn đã thử.")
        results[date_key] = found
        # polite pause
        time.sleep(0.6)
    return results

# -----------------------
# Processing functions
# -----------------------
def last_n_digits(numstr, n):
    if numstr is None:
        return None
    numstr = re.sub(r"\D", "", str(numstr))
    if len(numstr) >= n:
        return numstr[-n:]
    # if shorter, left-pad with zeros? We'll right-justify with zeros to length n
    return numstr.zfill(n)[-n:]

def build_frequency_lists(specials_dict):
    """Return counters for 2-digit and 3-digit last values and also sequence list for markov"""
    two = []
    three = []
    seq_three = []  # sequence ordered by date ascending (oldest -> newest)
    # iterate by date ascending
    for d in sorted(specials_dict.keys()):
        sp = specials_dict[d]
        if sp:
            t2 = last_n_digits(sp, 2)
            t3 = last_n_digits(sp, 3)
            two.append(t2)
            three.append(t3)
            seq_three.append(t3)
    c2 = Counter(two)
    c3 = Counter(three)
    return c2, c3, seq_three

# Markov order-1 on 3-digit strings treating each as token
def build_markov_transition(seq_three):
    trans = defaultdict(Counter)
    # build transitions from previous 3-digit to current 3-digit
    for a, b in zip(seq_three[:-1], seq_three[1:]):
        trans[a][b] += 1
    # normalize to probabilities when sampling
    trans_prob = {}
    for k, cnt in trans.items():
        total = sum(cnt.values())
        probs = {s: v/total for s, v in cnt.items()}
        trans_prob[k] = probs
    return trans_prob

# Weighted random sampling combining frequency & recency
def weighted_candidates(counter_obj, top_k=50, n_results=10, decay_days=None, seed=None):
    # take top_k by raw counts, then sample n_results weighted by count
    items = counter_obj.most_common(top_k)
    labels = [it[0] for it in items]
    weights = np.array([it[1] for it in items], dtype=float)
    if weights.sum() == 0:
        # uniform fallback
        weights = np.ones(len(labels))
    probs = weights / weights.sum()
    rng = np.random.default_rng(seed)
    # sample without replacement
    try:
        picks = rng.choice(labels, size=min(n_results, len(labels)), replace=False, p=probs)
    except Exception:
        # fallback to simple selection
        picks = labels[:min(n_results, len(labels))]
    return list(picks)

# -----------------------
# Make predictions
# -----------------------
def generate_predictions(specials_dict, n=10, seed=12345):
    c2, c3, seq3 = build_frequency_lists(specials_dict)
    markov = build_markov_transition(seq3)

    preds_2 = []
    preds_3 = []

    # Strategy A: Top frequency (deterministic)
    top2 = [x for x,_ in c2.most_common(n)]
    top3 = [x for x,_ in c3.most_common(n)]
    # pad lists
    def pad_list(lst, size, pad_func):
        if len(lst) >= size:
            return lst[:size]
        out = list(lst)
        while len(out) < size:
            out.append(pad_func(len(out)))
        return out

    # pad with random two-digit/three-digit if missing
    pad2 = lambda i: f"{i:02d}"
    pad3 = lambda i: f"{i:03d}"

    top2 = pad_list(top2, n, pad2)
    top3 = pad_list(top3, n, pad3)

    preds_2.append(("TopFrequency", top2[:n]))
    preds_3.append(("TopFrequency", top3[:n]))

    # Strategy B: Markov-based for 3-digit, for 2-digit use frequent endings of markov outputs
    # If last observed 3-digit exists, sample transitions
    if seq3:
        last = seq3[-1]
        trans = markov.get(last, None)
        if trans:
            # sort by prob
            trans_sorted = sorted(trans.items(), key=lambda x: x[1], reverse=True)
            markov_picks = [k for k,_ in trans_sorted][:n]
            markov_picks = pad_list(markov_picks, n, pad3)
        else:
            markov_picks = top3[:n]
    else:
        markov_picks = top3[:n]
    preds_3.append(("Markov1", markov_picks))

    # For 2 digits from markov picks:
    markov_2_from_3 = [last_n_digits(x, 2) for x in markov_picks]
    preds_2.append(("FromMarkov3->2", pad_list(markov_2_from_3, n, pad2)))

    # Strategy C: Weighted random combining freq & recency
    rng = np.random.default_rng(seed)
    random3 = weighted_candidates(c3, top_k=100, n_results=n, seed=seed)
    random2 = weighted_candidates(c2, top_k=100, n_results=n, seed=seed+1)
    preds_3.append(("WeightedRandom", pad_list(random3, n, pad3)))
    preds_2.append(("WeightedRandom", pad_list(random2, n, pad2)))

    # Merge strategies into final lists of up to n unique candidates preserving order of strategies
    def merge_candidates(list_of_lists, n):
        out = []
        for name, lst in list_of_lists:
            for it in lst:
                if it not in out:
                    out.append(it)
                if len(out) >= n:
                    return out
        # final pad with randoms if still short
        i = 0
        while len(out) < n:
            candidate = f"{i:0{len(out[0])}d}" if out else "0"
            out.append(candidate)
            i += 1
        return out[:n]

    final_2 = merge_candidates(preds_2, n)
    final_3 = merge_candidates(preds_3, n)

    # Also return breakdown details
    return {
        "final_2": final_2,
        "final_3": final_3,
        "c2": c2,
        "c3": c3,
        "seq3": seq3,
        "markov": markov,
        "raw_preds": {"by_strategy_2": preds_2, "by_strategy_3": preds_3}
    }

# -----------------------
# CLI main
# -----------------------
def main(args):
    # determine end_date (today)
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now()

    print(f"[INFO] Thu thập dữ liệu {args.days} ngày trước (tính từ {end_date.strftime('%Y-%m-%d')}).")
    specials = crawl_special_last_n_days(n_days=args.days, end_date=end_date, verbose=not args.quiet)

    df = pd.DataFrame([
        {"date": d, "special": specials[d] if specials[d] else ""}
        for d in sorted(specials.keys(), reverse=True)
    ])
    out_csv = args.output or f"xsmb_specials_{end_date.strftime('%Y%m%d')}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Lưu lịch sử đã thu thập vào {out_csv} (may có giá trị rỗng nếu không tìm thấy).")

    # Generate predictions
    preds = generate_predictions(specials, n=args.num, seed=args.seed)
    # Save predictions
    pred_df = pd.DataFrame({
        "rank": list(range(1, len(preds['final_2'])+1)),
        "pred_2": preds['final_2'],
        "pred_3": preds['final_3'],
    })
    pred_csv = f"xsmb_predictions_{end_date.strftime('%Y%m%d')}.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"[RESULT] Dự đoán ngày {end_date.strftime('%Y-%m-%d')}:")
    print("- 2 chữ số (top {n}):", preds['final_2'])
    print("- 3 chữ số (top {n}):", preds['final_3'])
    print(f"[INFO] Lưu dự đoán vào {pred_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XSMB Predictor")
    parser.add_argument("--days", type=int, default=30, help="Số ngày lịch sử lấy (mặc định 30)")
    parser.add_argument("--end-date", type=str, default=None, help="Ngày kết thúc (YYYY-MM-DD), mặc định today")
    parser.add_argument("--num", type=int, default=10, help="Số phương án muốn xuất (mặc định 10)")
    parser.add_argument("--output", type=str, default=None, help="Tên file CSV lưu lịch sử")
    parser.add_argument("--quiet", action="store_true", help="Giảm log")
    parser.add_argument("--seed", type=int, default=12345, help="Seed cho random")
    args = parser.parse_args()
    # expose n for print formatting inside main
    n = args.num
    main(args)
