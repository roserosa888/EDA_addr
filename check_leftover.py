"""
ตรวจคำที่เหลือใน addr_input ที่ไม่ได้อยู่ใน 11 key
- อ่านไฟล์จาก group/ (ผลลัพธ์ของ clean_address.py)
- ลบค่าของ 11 key + คำนำหน้า (ต., อำเภอ, จ. ...) ออกจาก addr_input
- คำที่เหลือ = ข้อมูลที่ไม่ถูกแยกเข้า key ใด เช่น
  "421 ถ.ประจักษ์ ต.ในเมือง อำเภอ..." แต่ tambol ว่าง → เหลือ "ต.ในเมือง"
- ส่งออกไปที่ group/leftover/
"""

import re
import pandas as pd
from pathlib import Path
from collections import Counter
from functools import lru_cache

from clean_address import FILL_KEYWORDS, build_keyword_re

GROUP_DIR = Path(r"D:\Thai_Adress\fill_data\group")
OUT_DIR = GROUP_DIR / "leftover"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = ["no", "moo", "mooban", "building", "floor", "room",
          "soi", "road", "tambol", "amphoe", "province"]

# ชุด (tambol, amphoe, province) ที่ถูกต้องจาก master ("*" ท้ายชื่อตำบลใน master ตัดทิ้งก่อนเทียบ)
PATC_MASTER = Path(r"D:\Thai_Adress\fill_data\PATC_Master1.parquet")


def load_patc_set():
    m = pd.read_parquet(PATC_MASTER, columns=["tambol_TH", "amphoe_TH", "province_TH"])
    m = m.fillna("").astype(str).apply(lambda s: s.str.replace("*", "", regex=False).str.strip())
    return set(zip(m["tambol_TH"], m["amphoe_TH"], m["province_TH"]))


def patc_key(rec):
    return tuple("" if is_blank(rec[f]) else str(rec[f]).strip() for f in ("tambol", "amphoe", "province"))

# คำนำหน้าของแต่ละ key ใช้ชุดเดียวกับ clean_address.py (ใช้เดาว่าคำที่เหลือควรอยู่ key ไหน)
FIELD_KEYWORDS = {f: [] for f in FIELDS}
for _kw, _f, _ in FILL_KEYWORDS:
    if _f in FIELD_KEYWORDS:
        FIELD_KEYWORDS[_f].append(_kw)
FIELD_KEYWORDS["mooban"].append("ม.")  # ม.+ตัวอักษร = mooban
KEYWORD_TO_FIELD = sorted(
    ((kw, f) for f, kws in FIELD_KEYWORDS.items() for kw in kws),
    key=lambda x: -len(x[0]),
)
KEYWORD_RE = re.compile("|".join(re.escape(kw) for kw, _ in KEYWORD_TO_FIELD))

PROVINCE_ALIASES ={"กรุงเทพมหานคร": ["กรุงเทพมหานคร", "กรุงเทพฯ", "กรุงเทพ", "กทม."]}

THAI_DIGIT_TRANS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
REPEATED_MARK_RE = re.compile(r"([\u0E31\u0E34-\u0E3A\u0E47-\u0E4E])\1+")
DISALLOWED_CHAR_RE = re.compile(r"[^a-zA-Z0-9\u0E00-\u0E7F\s\-/\(\)\[\]\{\}\|&,\+'\"\.\u2019\\\\@]")
WORD_CHAR_RE = re.compile(r"[a-zA-Z0-9\u0E00-\u0E7F]")
EDGE_PUNCT_RE = re.compile(r"^[^a-zA-Z0-9\u0E00-\u0E7F]+|[^a-zA-Z0-9\u0E00-\u0E7F.]+$")
SEP_RE = re.compile(r"[\s.\-]")
COMBINING_RE = re.compile(r"[\u0E31\u0E34-\u0E3A\u0E47-\u0E4E]")
NON_DIGIT_RE = re.compile(r"\D")

# leftover ที่มีแค่คำเดียวและเป็นคำเหล่านี้ ไม่ต้องส่งออก (หลายคำ เช่น "83 | และ | โฉนดเลขที่ | 8179" ยังส่งออก)
IGNORE_SINGLE_LEFTOVER = {"โฉนด", "โฉนดที่ดิน"}

# addr_input มีคำว่า น.ส. หรือ โฉนด แต่ building ไม่ว่าง → ส่งออกด้วย แม้ไม่มี leftover
DEED_RE = re.compile(r"น\.\s*ส\.|โฉนด")

# fuzzy match: \u0E04\u0E48\u0E32 key \u0E2A\u0E30\u0E01\u0E14\u0E15\u0E48\u0E32\u0E07\u0E08\u0E32\u0E01 addr_input \u0E40\u0E25\u0E47\u0E01\u0E19\u0E49\u0E2D\u0E22 (\u0E2D\u0E2D\u0E1F\u0E1F\u0E34\u0E28\u0E40\u0E28\u0E2A / \u0E2D\u0E2D\u0E1F\u0E1F\u0E34\u0E28\u0E40\u0E0B\u0E2A, \u0E40\u0E2E\u0E49\u0E32\u0E2A\u0E4C / \u0E40\u0E2E\u0E32\u0E2A\u0E4C)
FUZZY_MIN_LEN = 4           # \u0E04\u0E48\u0E32\u0E2A\u0E31\u0E49\u0E19\u0E01\u0E27\u0E48\u0E32\u0E19\u0E35\u0E49\u0E44\u0E21\u0E48\u0E17\u0E33 fuzzy (\u0E01\u0E31\u0E19\u0E44\u0E1B\u0E25\u0E1A\u0E21\u0E31\u0E48\u0E27)
FUZZY_MAX_DIST_RATIO = 0.3  # edit distance \u0E44\u0E14\u0E49\u0E44\u0E21\u0E48\u0E40\u0E01\u0E34\u0E19 30% \u0E02\u0E2D\u0E07\u0E04\u0E27\u0E32\u0E21\u0E22\u0E32\u0E27\u0E04\u0E48\u0E32
MIN_KEEP_GAP = 2            # \u0E15\u0E31\u0E27\u0E2D\u0E31\u0E01\u0E29\u0E23\u0E17\u0E35\u0E48\u0E41\u0E17\u0E23\u0E01\u0E2D\u0E22\u0E39\u0E48\u0E01\u0E25\u0E32\u0E07\u0E04\u0E48\u0E32 >= 2 \u0E15\u0E31\u0E27 (\u0E44\u0E21\u0E48\u0E19\u0E31\u0E1A\u0E2A\u0E23\u0E30/\u0E27\u0E23\u0E23\u0E13\u0E22\u0E38\u0E01\u0E15\u0E4C) \u0E16\u0E37\u0E2D\u0E40\u0E1B\u0E47\u0E19\u0E04\u0E33\u0E17\u0E35\u0E48\u0E02\u0E32\u0E14 \u0E40\u0E0A\u0E48\u0E19 "\u0E1E\u0E25\u0E32\u0E0B\u0E48\u0E32"


def is_blank(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v.strip() == "")


def normalize(text):
    """ทำให้ addr_input อยู่ในรูปเดียวกับค่า key ที่ผ่านการ clean แล้ว"""
    text = text.translate(THAI_DIGIT_TRANS)
    text = REPEATED_MARK_RE.sub(r"\1", text)
    text = DISALLOWED_CHAR_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=500_000)
def value_pattern(value):
    """ไม่สนช่องว่าง/จุด/ขีด ระหว่างตัวอักษร (คิว.เฮ้าส์ = คิวเฮ้าส์), ตัวเลขล้วนต้องไม่ติดตัวเลขอื่น"""
    chars = re.sub(r"[\s.\-]", "", value) or value
    body = r"[\s.\-]*".join(re.escape(c) for c in chars)
    if value.isdigit():
        body = rf"(?<!\d){body}(?!\d)"
    return re.compile(body)


# คำนำหน้าใช้กฎเดียวกับ clean_address.py (เช่น "วัด" ต้องไม่ติดอักษรไทยข้างหน้า
# กันไปจับ "วัดเชียงราย" กลาง "จังหวัดเชียงราย" แล้วเหลือ "จังห")
FIELD_KEYWORD_SPECS = {f: [k for k in FILL_KEYWORDS if k[1] == f] for f in FIELDS}
FIELD_KEYWORD_SPECS["mooban"].append(("ม.", "mooban", True))


@lru_cache(maxsize=None)
def field_prefix_pattern(field):
    return build_keyword_re(FIELD_KEYWORD_SPECS[field]).pattern


@lru_cache(maxsize=500_000)
def prefixed_value_pattern(field, value):
    return re.compile(rf"(?:{field_prefix_pattern(field)})\s*{value_pattern(value).pattern}")


@lru_cache(maxsize=None)
def field_keyword_re(field):
    kws = "|".join(re.escape(kw) for kw in sorted(FIELD_KEYWORDS[field], key=len, reverse=True))
    return re.compile(kws)


@lru_cache(maxsize=500_000)
def word_pattern(word):
    """คำสั้น (< 3 ตัว เช่น "บี", "ดิ") ต้องเป็นคำเดี่ยว ไม่งั้นจะไปลบกลางคำอื่น"""
    p = value_pattern(word).pattern
    if len(SEP_RE.sub("", word)) < 3:
        p = rf"(?<!\S){p}(?!\S)"
    return re.compile(p)


def remove_fuzzy(text, value):
    """หาช่วงใน text ที่ใกล้เคียง value ที่สุด (semi-global edit distance, ไม่สนช่องว่าง/จุด/ขีด)
    ลบตัวอักษรที่ match/สะกดต่าง ออก แต่เก็บข้อความที่แทรกอยู่กลางช่วง (>= MIN_KEEP_GAP) ไว้เป็น leftover
    เช่น value="เซ็นทรัล แจ้งวัฒนะ", text="เซ็นทรัลพลาซ่า แจ้งวัฒนะ" → เหลือ "พลาซ่า"
    คืน (text ใหม่, True/False ว่า match ได้)"""
    pat = SEP_RE.sub("", value)
    m = len(pat)
    if m < FUZZY_MIN_LEN or pat.isdigit():
        return text, False
    idx = [i for i, c in enumerate(text) if not SEP_RE.match(c)]
    txt = "".join(text[i] for i in idx)
    n = len(txt)
    if not n:
        return text, False

    # D[i][j] = cost ของ pat[:i] ที่จบที่ txt[j-1], เริ่มที่ไหนใน txt ก็ได้ (D[0][j] = 0)
    D = [[0] * (n + 1)]
    for i in range(1, m + 1):
        prev, cur, pc = D[-1], [i] + [0] * n, pat[i - 1]
        for j in range(1, n + 1):
            cur[j] = min(prev[j - 1] + (pc != txt[j - 1]), prev[j] + 1, cur[j - 1] + 1)
        D.append(cur)
    end = min(range(1, n + 1), key=D[m].__getitem__)
    if D[m][end] > m * FUZZY_MAX_DIST_RATIO:
        return text, False

    inserted = set()
    i, j = m, end
    while i > 0:
        if j > 0 and D[i][j] == D[i - 1][j - 1] + (pat[i - 1] != txt[j - 1]):
            i, j = i - 1, j - 1
        elif D[i][j] == D[i - 1][j] + 1:
            i -= 1
        else:
            inserted.add(j - 1)
            j -= 1
    start = j

    # ตัวเลขต้องตรงกันเป๊ะ (สุขุมวิท 13 ≠ สุขุมวิท 21)
    if NON_DIGIT_RE.sub("", txt[start:end]) != NON_DIGIT_RE.sub("", pat):
        return text, False

    keep, run = set(), []
    for k in range(start, end + 1):
        if k in inserted:
            run.append(k)
            continue
        if sum(1 for r in run if not COMBINING_RE.match(txt[r])) >= MIN_KEEP_GAP:
            keep.update(run)
        run = []

    chars = list(text)
    for k in range(start, end):
        if k not in keep:
            chars[idx[k]] = " "
    return "".join(chars), True


def find_leftover(row, fuzzy_fields=None):
    """fuzzy_fields: ถ้าส่ง list มา จะเติมชื่อ key ที่ต้องใช้ fuzzy/แยกคำ ถึงจะลบออกได้"""
    text = row["addr_input"]
    if is_blank(text):
        return []
    text = normalize(str(text))

    values = []
    for f in FIELDS:
        v = row[f]
        if is_blank(v):
            continue
        v = normalize(str(v))
        values.extend((f, a) for a in (PROVINCE_ALIASES.get(v, [v]) if f == "province" else [v]))

    # รอบแรก: ลบ "คำนำหน้า+ค่า" ของ key ตัวเอง (ถ.สุขุมวิท สำหรับ road)
    # กันไปลบค่าที่ซ้ำในส่วนอื่น เช่น "ซ.สุขุมวิท 101 ถ.สุขุมวิท"
    # ลบทุกตัวที่ซ้ำ: "ชั้นที่ 7 ชั้น 7" floor=7 → ลบทั้งสอง
    pending = []
    for f, v in sorted(values, key=lambda x: len(x[1]), reverse=True):
        text, n = prefixed_value_pattern(f, v).subn(" ", text)
        if not n:
            pending.append((f, v))

    # รอบสอง: ค่าที่ไม่มีคำนำหน้า ลบค่ายาวก่อน เพื่อไม่ให้ค่าสั้นไปตัดกลางค่ายาว
    # ไม่ dedupe: road=ลาดพร้าว กับ amphoe=ลาดพร้าว ต้องลบ 2 ครั้ง
    unmatched = []
    for f, v in pending:
        text, n = value_pattern(v).subn(" ", text, count=1)
        if not n:
            unmatched.append((f, v))

    # รอบสาม: ค่าที่สะกดไม่ตรง addr_input (มักเป็นค่าหลายคำ เช่น building)
    # ลอง fuzzy ทั้งค่าก่อน ถ้าไม่ได้ แยกเป็นทีละคำ (exact แล้วค่อย fuzzy)
    # เช่น building="อับดุลราฮิม เพลส" แต่ addr_input มีแค่ "อาคารอับดุลราฮิม"
    for f, v in unmatched:
        text, ok = remove_fuzzy(text, v)
        words = v.split()
        if not ok and len(words) > 1:
            for w in sorted(words, key=len, reverse=True):
                text, n = word_pattern(w).subn(" ", text, count=1)
                if not n:
                    text, n = remove_fuzzy(text, w)
                ok = ok or bool(n)
        if ok and fuzzy_fields is not None:
            fuzzy_fields.append(f)

    # รอบสุดท้าย: ช่วงที่ซ้ำของค่าหลายช่วง เช่น building="มหาวิทยาลัยเกษตรศาสตร์ อาคารศูนย์X"
    # addr_input มี "อาคารศูนย์X อาคารศูนย์X" → ลบตัวที่เหลือ
    for f, v in values:
        for part in field_keyword_re(f).split(v):
            part = part.strip()
            if len(SEP_RE.sub("", part)) >= 2:
                text = prefixed_value_pattern(f, part).sub(" ", text)

    leftovers = []
    for token in text.split():
        token = EDGE_PUNCT_RE.sub("", token)
        if not token:
            continue
        # token ที่เป็นแค่คำนำหน้า (เช่น "ถ.", "จังหวัด") ไม่นับ
        if WORD_CHAR_RE.search(KEYWORD_RE.sub("", token).replace(".", "")):
            leftovers.append(token)
    return leftovers


LETTER_RE = re.compile(r"[a-zA-Zก-ฮ]")


def find_duplicate_values(row):
    """ค่าที่ซ้ำกันหลาย key (เท่ากันหรืออยู่ในค่าของ key อื่น) แต่ addr_input มีคำนั้นน้อยกว่าจำนวน key
    เช่น soi=บางขุนพรหม, road=ตรอกบางขุนพรหม, tambol=บางขุนพรหม แต่ addr_input มี "บางขุนพรหม" แค่ 2 ที่
    road=ลาดพร้าว + amphoe=ลาดพร้าว และ addr_input มี "ถ.ลาดพร้าว เขตลาดพร้าว" → ไม่นับ
    คืน list ของ "ค่า: key1,key2 (addr_input N)" """
    text = row["addr_input"]
    if is_blank(text):
        return []
    text = normalize(str(text))
    vals = {f: normalize(str(row[f])) for f in FIELDS if not is_blank(row[f])}

    dups, seen = [], set()
    for f, v in vals.items():
        if v in seen or len(LETTER_RE.findall(v)) < 2:
            continue
        seen.add(v)
        pat = value_pattern(v)
        claimers = [g for g, w in vals.items()
                    if pat.search(w)
                    # อำเภอเมืองเชียงใหม่ + จังหวัดเชียงใหม่ ปกติ แม้ addr_input จะเขียนแค่ "อ.เมือง"
                    and not (f == "province" and g == "amphoe" and SEP_RE.sub("", w) == "เมือง" + SEP_RE.sub("", v))]
        if len(claimers) < 2:
            continue
        aliases = PROVINCE_ALIASES.get(v, [v]) if f == "province" else [v]
        n_found = sum(len(value_pattern(a).findall(text)) for a in aliases)
        if n_found < len(claimers):
            dups.append(f"{v}: {','.join(claimers)} (addr_input {n_found})")
    return dups


def guess_field(token):
    if token.startswith("ม."):
        return "moo" if token[2:3].isdigit() else "mooban"
    for kw, f in KEYWORD_TO_FIELD:
        if token.startswith(kw):
            return f
    return None


def main():
    files = sorted(GROUP_DIR.glob("*.parquet"))
    print("=" * 60)
    print("Leftover check: addr_input vs 11 keys")
    print("=" * 60)

    results = []
    total_rows = 0
    n_fuzzy = 0
    patc_set = load_patc_set()
    for fp in files:
        df = pd.read_parquet(fp)
        total_rows += len(df)
        cols = ["addr_input"] + FIELDS
        for rec in df[cols].to_dict("records"):
            fuzzy_fields = []
            tokens = find_leftover(rec, fuzzy_fields)
            n_fuzzy += bool(fuzzy_fields)
            if len(tokens) == 1 and tokens[0] in IGNORE_SINGLE_LEFTOVER:
                tokens = []
            # tambol/amphoe/province ไม่ตรงกับ PATC master → ส่งออกด้วย แม้ไม่มี leftover
            not_in_patc = patc_key(rec) not in patc_set
            deed_building = (not is_blank(rec["addr_input"]) and not is_blank(rec["building"])
                             and bool(DEED_RE.search(str(rec["addr_input"]))))
            dup_values = find_duplicate_values(rec)
            if not tokens and not not_in_patc and not deed_building and not dup_values:
                continue
            guesses = [guess_field(t) for t in tokens]
            blank_hit = sorted({g for g in guesses if g and is_blank(rec[g])})
            results.append({
                "source_file": fp.name,
                **rec,
                "leftover": " | ".join(tokens) or None,
                "leftover_field_guess": " | ".join(g or "-" for g in guesses) or None,
                "blank_key_with_leftover": ",".join(blank_hit) or None,
                "fuzzy_matched_fields": ",".join(fuzzy_fields) or None,
                "not_in_patc_master": not_in_patc,
                "deed_with_building": deed_building,
                "duplicate_values": " | ".join(dup_values) or None,
            })

    out = pd.DataFrame(results)
    n_out = len(out)
    n_left = int(out["leftover"].notna().sum()) if n_out else 0
    n_patc = int(out["not_in_patc_master"].sum()) if n_out else 0
    n_deed = int(out["deed_with_building"].sum()) if n_out else 0
    n_dup = int(out["duplicate_values"].notna().sum()) if n_out else 0
    blank_mask = out["blank_key_with_leftover"].notna() if n_out else pd.Series(dtype=bool)
    n_blank = int(blank_mask.sum())

    print(f"\nRows checked: {total_rows:,}  ({len(files)} files in {GROUP_DIR})")
    print(f"Rows with leftover words (addr_input เกินจาก 11 key): {n_left:,} "
          f"({n_left / max(total_rows, 1):.2%})")
    print(f"  └─ leftover มีคำนำหน้าของ key ที่ว่างอยู่ (เช่น ต.xxx แต่ tambol ว่าง): {n_blank:,} "
          f"({n_blank / max(total_rows, 1):.2%})")
    print(f"Rows ที่ tambol/amphoe/province ไม่มีใน PATC_Master1: {n_patc:,} "
          f"({n_patc / max(total_rows, 1):.2%})")
    print(f"Rows ที่ addr_input มี น.ส./โฉนด แต่ building ไม่ว่าง: {n_deed:,} "
          f"({n_deed / max(total_rows, 1):.2%})")
    print(f"Rows ที่ค่าซ้ำกันหลาย key เกินกว่าที่มีใน addr_input: {n_dup:,} "
          f"({n_dup / max(total_rows, 1):.2%})")
    print(f"Rows ที่ค่า key สะกดไม่ตรง addr_input (ลบออกด้วย fuzzy/แยกคำ): {n_fuzzy:,}")

    if n_out:
        print("\n--- จำนวน row ที่ key ว่างแต่มีคำใน addr_input ---")
        field_counts = Counter(f for s in out.loc[blank_mask, "blank_key_with_leftover"] for f in s.split(","))
        for f in FIELDS:
            if field_counts.get(f):
                print(f"  {f:<10} {field_counts[f]:>10,}")

        print("\n--- แยกตามไฟล์ใน group/ ---")
        by_file = out.groupby("source_file").agg(
            leftover_rows=("leftover", "count"),
            blank_key_rows=("blank_key_with_leftover", "count"),
            not_in_patc_rows=("not_in_patc_master", "sum"),
            deed_building_rows=("deed_with_building", "sum"),
            dup_rows=("duplicate_values", "count"),
        )
        for name, r in by_file.iterrows():
            print(f"  {name:<65} {r.leftover_rows:>9,}  (key ว่าง {r.blank_key_rows:,}, "
                  f"ไม่มีใน PATC {r.not_in_patc_rows:,}, น.ส./โฉนด+building {r.deed_building_rows:,}, "
                  f"ค่าซ้ำ {r.dup_rows:,})")

        word_counts = Counter(t for s in out["leftover"].dropna() for t in s.split(" | "))
        words = pd.DataFrame(word_counts.most_common(), columns=["word", "count"])
        words["field_guess"] = words["word"].map(guess_field)

        print("\n--- Top 30 leftover words ---")
        for w, c in word_counts.most_common(30):
            print(f"  {c:>8,}  {w}")

        out.to_parquet(OUT_DIR / "leftover_rows.parquet", index=False)
        out[blank_mask].to_parquet(OUT_DIR / "leftover_blank_key_rows.parquet", index=False)
        words.to_csv(OUT_DIR / "leftover_word_counts.csv", index=False, encoding="utf-8-sig")

        print(f"\nSaved:")
        print(f"  - {OUT_DIR / 'leftover_rows.parquet'}  ({n_out:,} rows, ไม่มีใน PATC {n_patc:,}, "
          f"น.ส./โฉนด+building {n_deed:,}, ค่าซ้ำ {n_dup:,})")
        print(f"  - {OUT_DIR / 'leftover_blank_key_rows.parquet'}  ({n_blank:,} rows)")
        print(f"  - {OUT_DIR / 'leftover_word_counts.csv'}  ({len(words):,} words)")
    print("=" * 60)


if __name__ == "__main__":
    main()
