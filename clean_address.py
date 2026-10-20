"""
Thai Address Cleaning Script
- ทำความสะอาดข้อมูลจาก GroupingAddress_20260709.parquet
- แบ่ง 90% train / 10% eval
- เก็บประเภทที่ clean ไว้ในโฟลเดอร์ group/
"""

import re
import difflib
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

# ==========================================
# CONFIG
# ==========================================
INPUT_PATH = r"D:\Thai_Adress\GroupingAddress_20260709.parquet"
OUTPUT_DIR = Path(r"D:\Thai_Adress\fill_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_DIR = OUTPUT_DIR / "group"
GROUP_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_COMBINED = OUTPUT_DIR / "combined_summary_result.parquet"
OUTPUT_EVAL = OUTPUT_DIR / "eval.parquet"

TRAIN_RATIO = 0.9
RANDOM_SEED = 42

# ==========================================
# REGEX PATTERNS
# ==========================================
THAI_RANGE = r"\u0E00-\u0E7F"
THAI_CHAR_RE = re.compile(f"[{THAI_RANGE}]")
DIGIT_RE = re.compile(r"[0-9]")
SPECIAL_CHARS_RE = re.compile(r"[!@#$%^&*_=+{}\[\]<>~`|\\\"\';]")

THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
ARABIC_DIGITS = "0123456789"
THAI_DIGIT_TRANS = str.maketrans(THAI_DIGITS, ARABIC_DIGITS)

FLOATING_VOWEL_RE = re.compile(r"(?<![\u0E01-\u0E4F])[\u0E31\u0E34-\u0E3A\u0E47-\u0E4E]")
REPEATED_MARK_RE = re.compile(r"([\u0E31\u0E34-\u0E3A\u0E47-\u0E4E])\1+")
DISALLOWED_CHAR_RE = re.compile(r"[^a-zA-Z0-9\u0E00-\u0E7F\s\-/\(\)\[\]\{\}\|&,\+'\"\.\u2019\\\\@]")
WORD_CHAR_RE = re.compile(r"[a-zA-Z0-9\u0E00-\u0E7F]")
LEADING_NON_WORD_RE = re.compile(r"^[^a-zA-Z0-9\u0E00-\u0E7F]+")
LEADING_NON_WORD_KEEP_AT_RE = re.compile(r"^[^a-zA-Z0-9\u0E00-\u0E7F@]+")
THAI_OR_ENG_CHAR_RE = re.compile(r"[a-zA-Z\u0E00-\u0E7F]")
DIGIT_CLASS = f"[{ARABIC_DIGITS}{THAI_DIGITS}]"
POSTCODE_TOKEN_RE = re.compile(rf'(?<!{DIGIT_CLASS})({DIGIT_CLASS}{{5}})(?!{DIGIT_CLASS})')
POSTCODE_PREFIX_RE = re.compile(
    rf'(?:เลขรหัสไปรษณีย์|รหัสไปรษณีย์|ปณ\.?)\s*[:\-]?\s*({DIGIT_CLASS}{{5}})(?!{DIGIT_CLASS})'
)

# เติม key ที่ว่างจาก addr_input ด้วยคำนำหน้า (rule base)
# (คำนำหน้า, key, ต้องไม่ติดอักษรไทยข้างหน้า) — คำย่อมีจุดและคำที่อาจอยู่กลางชื่อ
# (เช่น อาคาร ใน "หมู่บ้านอาคารสงเคราะห์") ต้องไม่ติดอักษรไทย กันไปจับกลางคำ
# key "_stop" ใช้ตัดช่วงอย่างเดียว ไม่เอาไปเติม
FILL_KEYWORDS = [
    ("บ้านเลขที่", "no", False), ("เลขที่ห้อง", "room", False), ("ห้องเลขที่", "room", False),
    ("เลขที่", "no", False),
    ("หมู่บ้าน", "mooban", True), ("หมู่ที่", "moo", True), ("หมู่", "moo", True),
    ("ม.", "moo", True),  # ม. ตามด้วยตัวอักษร = mooban (แยกใน split_by_keywords)
    ("อาคาร", "building", True), ("ตึก", "building", True),
    ("ชั้นที่", "floor", False), ("ชั้น", "floor", False),  # "อาคารชั้นที่ 2"
    ("ห้องที่", "room", True), ("ห้อง", "room", True),
    ("ซอย", "soi", True), ("ซ.", "soi", True),
    ("ถนน", "road", True), ("ถ.", "road", True),
    # ต./อ./จ. มักติดท้ายชื่อตำบล/อำเภอ เช่น "ต.ในเมืองอ.เมืองจ.ขอนแก่น"
    ("ตำบล", "tambol", False), ("แขวง", "tambol", False), ("ต.", "tambol", False),
    ("กิ่งอำเภอ", "amphoe", False), ("อำเภอ", "amphoe", False), ("เขต", "amphoe", True),
    ("อ.", "amphoe", False),
    ("จังหวัด", "province", False), ("จ.", "province", False),
    ("กรุงเทพมหานคร", "_stop", False), ("กรุงเทพฯ", "_stop", False), ("กทม", "_stop", False),
]
FILL_KEYWORD_FIELD = {kw: f for kw, f, _ in FILL_KEYWORDS}
FILL_KEYWORD_RE = re.compile("|".join(
    (rf"(?<![{THAI_RANGE}])" if no_thai_before else "") + re.escape(kw)
    for kw, _, no_thai_before in sorted(FILL_KEYWORDS, key=lambda x: -len(x[0]))
))
# key ที่ค่าเป็นคำเดียว (ตัดที่ช่องว่างแรก) — ที่เหลือเอาทั้งช่วงจนถึงคำนำหน้าถัดไป
SINGLE_TOKEN_FIELDS = {"no", "moo", "floor", "room", "tambol", "amphoe", "province"}
DB_POSTCODE_PATH = r"D:\Thai_Adress\qwen3_8b_lora\EDA\DataStandardList-Postcode_converted.csv"

# ไฟล์แก้ค่ามือ: แถวที่ addr_input ตรงกัน จะถูกแทน 11 key ด้วยค่าในไฟล์ (ช่องว่าง = ไม่มีค่า)
EDIT_DIR = OUTPUT_DIR / "edit"
EDIT_FILES = [EDIT_DIR / "edit_data.csv", EDIT_DIR / "bad_values.csv"]
ADDRESS_FIELDS = ['no', 'moo', 'mooban', 'building', 'floor', 'room', 'soi', 'road', 'tambol', 'amphoe', 'province']

KEEP_AT_COLUMNS = ["building"]

# คอลัมน์ที่ไม่ส่งออกในทุกไฟล์
EXCLUDE_OUTPUT_COLUMNS = ["source", "rgno", "branchnumber"]
# คอลัมน์ที่วางไว้ซ้ายสุด
LEAD_COLUMNS = ["addr", "addr_input"]

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def is_blank(v):
    if pd.isna(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False

def has_unbalanced_brackets(v):
    if not isinstance(v, str):
        return False
    stack = []
    pairs = {")": "(", "]": "["}
    opens = {"(", "["}
    for ch in v:
        if ch in opens:
            stack.append(ch)
        elif ch in pairs:
            if stack and stack[-1] == pairs[ch]:
                stack.pop()
            else:
                return True
    return len(stack) > 0

def prefix_any(words):
    pattern = re.compile("^(" + "|".join(re.escape(w) for w in words) + ")")
    return lambda v: bool(pattern.search(v.strip())) if isinstance(v, str) else False

def contains_any(words):
    pattern = re.compile("|".join(re.escape(w) for w in words))
    return lambda v: bool(pattern.search(v)) if isinstance(v, str) else False

def remove_prefix_with_reason(v, pattern, reason_text):
    if not isinstance(v, str) or is_blank(v):
        return v, []
    new_v = re.sub(pattern, '', v).strip()
    if new_v != v:
        return new_v, [reason_text]
    return v, []

def is_changed(before, after):
    if pd.isna(before) and pd.isna(after):
        return False
    return before != after

def is_addr_garbage_only(v):
    if not isinstance(v, str) or is_blank(v):
        return False
    return not bool(THAI_OR_ENG_CHAR_RE.search(v))

def as_text(v):
    return "" if is_blank(v) else str(v)

def describe_char_changes(before, after):
    """บอกว่าตัวอักษรไหนถูกลบ/เพิ่ม/แทนที่ ระหว่างค่าก่อนและหลัง clean"""
    before, after = as_text(before), as_text(after)
    parts = []
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            parts.append(f"ลบ '{before[i1:i2]}'")
        elif tag == "insert":
            parts.append(f"เพิ่ม '{after[j1:j2]}'")
        elif tag == "replace":
            parts.append(f"แทน '{before[i1:i2]}'→'{after[j1:j2]}'")
    return ", ".join(parts)

# ==========================================
# COLUMN RULES
# ==========================================
COLUMN_RULES = {
    "no": {
        "should_have": [
            ("มีตัวเลข", lambda v: isinstance(v, str) and bool(DIGIT_RE.search(v))),
        ],
        "should_not": [
            ("มีคำนำหน้า เลขที่/บ้านเลขที่", prefix_any(["บ้านเลขที่", "เลขที่"])),
        ],
    },
    "moo": {
        "should_have": [
            ("เป็นตัวเลขล้วน", lambda v: isinstance(v, str) and v.strip().isdigit()),
        ],
        "should_not": [
            ("มีคำนำหน้า หมู่/ม.", prefix_any(["หมู่","ม."])),
            ("มีอักขระพิเศษ", lambda v: isinstance(v, str) and bool(SPECIAL_CHARS_RE.search(v))),
        ],
    },
    "mooban": {
        "should_have": [
            ("มีอักษรไทย", lambda v: isinstance(v, str) and bool(THAI_CHAR_RE.search(v))),
        ],
        "should_not": [
            ("มีคำนำหน้า หมู่บ้าน", prefix_any(["หมู่บ้าน","ม."])),
            ("มีอักขระพิเศษ #$%^&*", lambda v: isinstance(v, str) and bool(re.search(r"[#$%^&*]", v))),
        ],
    },
    "building": {
        "should_have": [
            ("มีอักษรไทย", lambda v: isinstance(v, str) and bool(THAI_CHAR_RE.search(v))),
        ],
        "should_not": [
            ("มีคำนำหน้า อาคาร/ตึก", prefix_any(["อาคาร", "ตึก"])),
            ("มีอักขระพิเศษ", lambda v: isinstance(v, str) and bool(SPECIAL_CHARS_RE.search(v))),
        ],
    },
    "floor": {
        "should_have": [
            ("มีตัวเลข", lambda v: isinstance(v, str) and bool(DIGIT_RE.search(v))),
        ],
        "should_not": [
            ("มีคำนำหน้า ชั้น", prefix_any(["ชั้น"])),
            ("มีอักขระพิเศษ", lambda v: isinstance(v, str) and bool(SPECIAL_CHARS_RE.search(v))),
        ],
    },
    "room": {
        "should_have": [
            ("มีตัวเลข", lambda v: isinstance(v, str) and bool(DIGIT_RE.search(v))),
        ],
        "should_not": [
            ("มีคำนำหน้า ห้อง/เลขที่ห้อง", prefix_any(["เลขที่ห้อง", "ห้อง"])),
            ("มีอักขระพิเศษ", lambda v: isinstance(v, str)
                and bool(re.sub(r"[/\-]", "", v) and SPECIAL_CHARS_RE.search(re.sub(r"[/\-]", "", v)))),
        ],
    },
    "soi": {
        "should_have": [
            ("มีอักษรไทย", lambda v: isinstance(v, str) and bool(THAI_CHAR_RE.search(v))),
        ],
        "should_not": [
            ("มีคำนำหน้า ซอย/ซ.", prefix_any(["ซอย", "ซ."])),
            ("มีอักขระพิเศษ", lambda v: isinstance(v, str)
                and bool(SPECIAL_CHARS_RE.search(re.sub(r"[/\-]", "", v)))),
        ],
    },
    "road": {
        "should_have": [
            ("มีอักษรไทย", lambda v: isinstance(v, str) and bool(THAI_CHAR_RE.search(v))),
        ],
        "should_not": [
            ("มีคำนำหน้า ถนน/ถ.", prefix_any(["ถนน", "ถ."])),
            ("มีอักขระพิเศษ", lambda v: isinstance(v, str) and bool(SPECIAL_CHARS_RE.search(v))),
        ],
    },
    "tambol": {
        "should_have": [
            ("มีอักษรไทย", lambda v: isinstance(v, str) and bool(THAI_CHAR_RE.search(v))),
        ],
        "should_not": [
            ("มีคำนำหน้า ตำบล/ต.", prefix_any(["ตำบล", "ต."])),
        ],
    },
    "amphoe": {
        "should_have": [
            ("มีอักษรไทย", lambda v: isinstance(v, str) and bool(THAI_CHAR_RE.search(v))),
        ],
        "should_not": [
            ("มีคำนำหน้า อำเภอ/อ.", prefix_any(["อำเภอ", "อ."])),
        ],
    },
    "province": {
        "should_have": [
            ("มีอักษรไทย", lambda v: isinstance(v, str) and bool(THAI_CHAR_RE.search(v))),
        ],
        "should_not": [
            ("มีคำนำหน้า จังหวัด/จ.", prefix_any(["จังหวัด", "จ."])),
        ],
    },
}

ALL_KEYWORDS = [
    "ซอย", "ซ.", "หมู่บ้าน", "หมู่", "ม.",
    "ถนน", "ถ.", "ตำบล", "ต.", "อำเภอ", "อ.", "จังหวัด", "จ.",
    "ชั้น", "ห้อง", "อาคาร","เลขที่","บ้านเลขที่"
]

OWN_KEYWORDS = {
    "no":       ["เลขที่","บ้านเลขที่"],
    "moo":      ["หมู่","ม."],
    "mooban":   ["หมู่บ้าน", "ม."],
    "building": ["อาคาร"],
    "floor":    ["ชั้น"],
    "room":     ["ห้อง"],
    "soi":      ["ซอย", "ซ."],
    "road":     ["ถนน", "ถ."],
    "tambol":   ["ตำบล", "ต."],
    "amphoe":   ["อำเภอ", "อ."],
    "province": ["จังหวัด", "จ."],
}

# ==========================================
# MAIN CLEANING FUNCTION
# ==========================================
def clean_address_data():
    print("=" * 60)
    print("Thai Address Cleaning Script")
    print("=" * 60)

    # Load data
    print(f"\nLoading: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")

    original_rows = len(df)

    # ==========================================
    # Step 1: Remove blank addr FIRST (early optimization)
    # ==========================================
    blank_addr_mask = df['addr'].apply(is_blank)
    n_blank = blank_addr_mask.sum()
    df = df[~blank_addr_mask].copy()
    print(f"1. Remove blank addr: {n_blank:,} rows removed, {len(df):,} remaining")

    # ==========================================
    # Step 2: Remove duplicate addr (keep first) - early optimization
    # ==========================================
    n_before = len(df)
    df = df.drop_duplicates(subset=['addr'], keep='first').reset_index(drop=True)
    n_removed = n_before - len(df)
    print(f"2. Remove duplicate addr: {n_removed:,} rows removed, {len(df):,} remaining")

    # ==========================================
    # Step 3: Drop unused columns
    # ==========================================
    cols_to_drop = [c for c in ['postcode', 'addr1', 'addr2'] + EXCLUDE_OUTPUT_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"3. Dropped columns: {cols_to_drop}")

    # ==========================================
    # Step 4: Extract postcode
    # ==========================================
    def extract_postcode_and_input(addr_val):
        if not isinstance(addr_val, str) or addr_val.strip() == "":
            return None, addr_val
        m_prefix = POSTCODE_PREFIX_RE.search(addr_val)
        if m_prefix:
            postcode = m_prefix.group(1)
            addr_input = (addr_val[:m_prefix.start()] + addr_val[m_prefix.end():]).strip()
            return postcode, addr_input
        m_end = re.search(rf'(?<!{DIGIT_CLASS})({DIGIT_CLASS}{{5}})\s*$', addr_val)
        if m_end:
            postcode = m_end.group(1)
            addr_input = addr_val[:m_end.start()].rstrip()
            return postcode, addr_input
        return None, addr_val

    _extracted = df['addr'].apply(extract_postcode_and_input)
    df['POSTCODE'] = _extracted.apply(lambda x: x[0])
    df['addr_input'] = _extracted.apply(lambda x: x[1])
    print(f"4. Extracted POSTCODE from addr, {len(df):,} rows")

    # โหลดค่าแก้มือ {addr_input: {key: ค่า}} — ใช้ใน Step 5, 6 และ 7.6
    manual_edits = {}
    edit_source = {}  # addr_input → ชื่อไฟล์แก้มือ
    for path in EDIT_FILES:
        if not path.exists():
            print(f"   Edit file not found, skipping: {path}")
            continue
        df_edit = pd.read_csv(path, dtype=str, encoding="utf-8-sig").dropna(subset=["addr_input"])
        for _, r in df_edit.iterrows():
            manual_edits[r["addr_input"].strip()] = {
                c: (None if is_blank(r.get(c)) else str(r[c]).strip()) for c in ADDRESS_FIELDS
            }
            edit_source[r["addr_input"].strip()] = path.name
        print(f"   Loaded {len(df_edit):,} manual edits from {path.name}")

    def has_manual_edit(series):
        return series.apply(lambda v: isinstance(v, str) and v.strip() in manual_edits)

    # ==========================================
    # Step 5: Sample 700,000 rows with seed 7
    # ==========================================
    if len(df) > 700000:
        df_sampled = df.sample(n=700000, random_state=7)
        # เติมแถวที่มีค่าแก้มือแต่ไม่ติด sample กลับเข้ามา
        missing_edit = df[has_manual_edit(df['addr_input']) & ~df.index.isin(df_sampled.index)]
        df = pd.concat([df_sampled, missing_edit]).reset_index(drop=True)
        print(f"5. Sampled 700,000 rows (seed=7) + {len(missing_edit):,} rows with manual edits")
    else:
        print(f"5. Skipped sampling (only {len(df):,} rows)")
    edit_mask = has_manual_edit(df['addr_input'])

    # ==========================================
    # Step 6: Filter rows
    # ==========================================
    print("\n--- Filtering Steps ---")

    # 6.1: Remove garbage-only addr
    garbage_addr_mask = df['addr'].apply(is_addr_garbage_only) & ~edit_mask
    n_garbage = garbage_addr_mask.sum()
    df = df[~garbage_addr_mask].copy()
    edit_mask = edit_mask[~garbage_addr_mask]
    print(f"5.1 Remove garbage addr: {n_garbage:,} rows removed, {len(df):,} remaining")

    # 6.2: Remove specific bad values
    initial_rows = len(df)
    drop_mask = pd.Series(False, index=df.index)
    drop_reasons = pd.Series("", index=df.index)

    def mark_drop(mask, label):
        nonlocal drop_mask, drop_reasons
        n = int(mask.sum())
        drop_mask = drop_mask | mask
        drop_reasons[mask] = label

    if 'mooban' in df.columns:
        mark_drop(df['mooban'].apply(lambda v: isinstance(v, str) and v.strip() == "ตำแหน่งที่ PDP.F0G.BS061#00"),
                  "mooban = ตำแหน่งที่ PDP.F0G.BS061#00")
    if 'floor' in df.columns:
        mark_drop(df['floor'].apply(lambda v: isinstance(v, str) and "ชั้นใต้ดินอาคารจอดแล" in v),
                  "floor = ชั้นใต้ดินอาคารจอดแล...")
    if 'tambol' in df.columns:
        mark_drop(df['tambol'].apply(lambda v: isinstance(v, str) and v.strip() == "ทดสอบ  0101"),
                  "tambol = ทดสอบ  0101")

    # 6.3: Remove cross-column contamination
    for col, rules in COLUMN_RULES.items():
        if col not in df.columns:
            continue
        own_kw = OWN_KEYWORDS.get(col, [])
        for kw in ALL_KEYWORDS:
            if kw in own_kw:
                continue
            mask = df[col].apply(lambda v: isinstance(v, str) and kw in v)
            mark_drop(mask, f"{col} contains '{kw}' (should be in other column)")

    # แถวที่มีค่าแก้มือไม่ต้องลบ (จะถูกแทนค่าใน Step 7.6)
    drop_mask = drop_mask & ~edit_mask

    # Store filtered reasons (ต้องเก็บก่อนลบแถวออกจาก df)
    df_filtered_out = pd.DataFrame({
        'addr': df.loc[drop_mask, 'addr'],
        'filter_reason': drop_reasons[drop_mask],
    })

    # reset index เพื่อให้ index ตรงกับหลัง merge ใน Step 8 (ใช้ติด tag 'modified')
    df = df[~drop_mask].reset_index(drop=True)
    print(f"5.2-5.3 Remove bad values: {initial_rows - len(df):,} rows removed, {len(df):,} remaining")

    # ==========================================
    # Step 7: Clean column values
    # ==========================================
    print("\n--- Cleaning Column Values ---")

    modified_indices = set()
    modified_row_reasons = defaultdict(list)
    mod_logs_total = 0

    cols_to_clean = [col for col in df.columns if df[col].dtype == 'string' or df[col].dtype == 'object']

    for col in cols_to_clean:
        if col in ['addr', 'addr_input', 'POSTCODE']:
            continue

        new_col_data = []

        for row_idx, val in zip(df.index, df[col]):
            current_val = val
            reasons_list = []

            # Clear specific bad values
            target_clear_cols = ['moo', 'mooban', 'building', 'room', 'soi', 'road', 'province', 'floor', 'tambol', 'amphoe']
            if col in target_clear_cols:
                if isinstance(current_val, str) and current_val.strip() in ["-", "ไม่มี-", "ไม่ระบุ", "-ไม่มี-", "-ไม่มี"]:
                    current_val = ""
                    reasons_list.append("เคลียร์ค่า '-', 'ไม่มี-', หรือ 'ไม่ระบุ'")

            if col == 'no':
                if isinstance(current_val, str) and any(kw in current_val for kw in ["ไม่มีเลขที่", "ไม่ระบุ", "ไม่มี-", "-ไม่มี"]):
                    current_val = ""
                    reasons_list.append("เคลียร์ค่า 'ไม่มีเลขที่' หรือ 'ไม่ระบุ'")
                else:
                    current_val, r = remove_prefix_with_reason(current_val, r'^(\s*(บ้านเลขที่|เลขที่)\s*)', "ลบคำนำหน้า เลขที่/บ้านเลขที่")
                    reasons_list.extend(r)

            elif col == 'moo':
                current_val, r = remove_prefix_with_reason(current_val, r'^(\s*(หมู่\s*|ม\.\s*))+', "ลบคำนำหน้า หมู่/ม.")
                reasons_list.extend(r)

            elif col == 'mooban':
                current_val, r = remove_prefix_with_reason(current_val, r'^(\s*(หมู่บ้าน|ม\.)\s*)', "ลบคำนำหน้า หมู่บ้าน/ม.")
                reasons_list.extend(r)
                if isinstance(current_val, str) and '#' in current_val:
                    current_val = current_val.replace('#', ' ')
                    reasons_list.append("ลบ # ออกจาก mooban")

            elif col == 'building':
                current_val, r = remove_prefix_with_reason(current_val, r'^(\s*(อาคาร|ตึก)\s*)', "ลบคำนำหน้า อาคาร/ตึก")
                reasons_list.extend(r)

            elif col == 'floor':
                current_val, r = remove_prefix_with_reason(current_val, r'^(\s*ชั้น\s*)', "ลบคำนำหน้า ชั้น")
                reasons_list.extend(r)

            elif col == 'room':
                current_val, r = remove_prefix_with_reason(current_val, r'^(\s*(เลขที่ห้อง|ห้อง)\s*)', "ลบคำนำหน้า ห้อง/เลขที่ห้อง")
                reasons_list.extend(r)

            elif col == 'soi':
                current_val, r = remove_prefix_with_reason(current_val, r'^(\s*(ซอย|ซ\.)\s*)', "ลบคำนำหน้า ซอย/ซ.")
                reasons_list.extend(r)

            elif col == 'road':
                current_val, r = remove_prefix_with_reason(current_val, r'^(\s*(ถนน|ถ\.)\s*)', "ลบคำนำหน้า ถนน/ถ.")
                reasons_list.extend(r)

            elif col == 'tambol':
                current_val, r = remove_prefix_with_reason(current_val, r'^(\s*(ตำบล|ต\.)\s*)', "ลบคำนำหน้า ตำบล/ต.")
                reasons_list.extend(r)

            elif col == 'amphoe':
                current_val, r = remove_prefix_with_reason(current_val, r'^(\s*(อำเภอ|อ\.)\s*)', "ลบคำนำหน้า อำเภอ/อ.")
                reasons_list.extend(r)

            elif col == 'province':
                current_val, r = remove_prefix_with_reason(current_val, r'^(\s*(จังหวัด|จ\.)\s*)', "ลบคำนำหน้า จังหวัด/จ.")
                reasons_list.extend(r)

            # General cleaning for all string columns
            if isinstance(current_val, str):
                # Remove floating vowels
                new_val = FLOATING_VOWEL_RE.sub('', current_val)
                if new_val != current_val:
                    reasons_list.append("ลบสระ/วรรณยุกต์ลอย")
                    current_val = new_val

                # Remove repeated marks
                new_val = REPEATED_MARK_RE.sub(r'\1', current_val)
                if new_val != current_val:
                    reasons_list.append("ลบสระ/วรรณยุกต์ซ้ำ")
                    current_val = new_val

                # Convert Thai digits to Arabic
                new_val = current_val.translate(THAI_DIGIT_TRANS)
                if new_val != current_val:
                    reasons_list.append("แปลงตัวเลขไทยเป็นอารบิก")
                    current_val = new_val

                # Remove disallowed characters
                new_val = DISALLOWED_CHAR_RE.sub(' ', current_val)
                if new_val != current_val:
                    reasons_list.append("ลบอักขระพิเศษ")
                    current_val = new_val

                # Remove leading non-word characters
                if col in KEEP_AT_COLUMNS:
                    new_val = LEADING_NON_WORD_KEEP_AT_RE.sub('', current_val)
                else:
                    new_val = LEADING_NON_WORD_RE.sub('', current_val)
                if new_val != current_val:
                    reasons_list.append("ลบอักขระนำหน้า")
                    current_val = new_val

                # Check unbalanced brackets
                if has_unbalanced_brackets(current_val):
                    reasons_list.append("ลบวงเล็บไม่ครบคู่")
                    current_val = re.sub(r'[\(\)\[\]\{\}]', ' ', current_val)

                # Collapse multiple spaces
                new_val = re.sub(r'\s+', ' ', current_val).strip()
                if new_val != current_val:
                    reasons_list.append("ตัดช่องว่างเกิน")
                    current_val = new_val

            # Check if value changed
            if is_changed(val, current_val):
                modified_indices.add(row_idx)
                char_diff = describe_char_changes(val, current_val)
                modified_row_reasons[row_idx].append(
                    f"{col}: '{as_text(val)}' → '{as_text(current_val)}' "
                    f"[{char_diff}] ({', '.join(reasons_list)})"
                )
                mod_logs_total += 1

            new_col_data.append(current_val if not is_blank(current_val) else None)

        df[col] = new_col_data

    print(f"6. Modified {len(modified_indices):,} rows, {mod_logs_total:,} modifications")

    # ==========================================
    # Step 7.5: Fill blank keys from addr_input (rule base: คำนำหน้าของ 11 key)
    # ==========================================
    try:
        known_tambols = set(pd.read_csv(DB_POSTCODE_PATH, dtype=str)["tambol"].dropna().str.strip())
    except FileNotFoundError:
        known_tambols = set()

    def clean_filled(v):
        v = v.translate(THAI_DIGIT_TRANS)
        v = DISALLOWED_CHAR_RE.sub(' ', v)
        v = re.sub(r'\s+', ' ', v).strip()
        return LEADING_NON_WORD_RE.sub('', v).strip()

    def split_by_keywords(addr_input):
        """แยก addr_input เป็น {key: [ค่า]} ตามช่วงหลังคำนำหน้าแต่ละตัว"""
        found = defaultdict(list)
        if not isinstance(addr_input, str):
            return found
        text = addr_input.translate(THAI_DIGIT_TRANS)
        matches = list(FILL_KEYWORD_RE.finditer(text))
        for i, m in enumerate(matches):
            col = FILL_KEYWORD_FIELD[m.group(0)]
            if col == "_stop":
                continue
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            v = clean_filled(text[m.end():end].split(",")[0])
            if not v:
                continue
            if m.group(0) == "ม." and not v[0].isdigit():
                col = "mooban"
            if col in SINGLE_TOKEN_FIELDS:
                v = v.split(" ")[0]
            if col == "moo":
                v = re.match(r"\d*", v).group(0)
            # แขวงลาดยาวเขตจตุจักร → ลาดยาว (แต่ หลักเขต/คีรีเขต เป็นชื่อตำบลจริง)
            if col == "tambol" and "เขต" in v and v not in known_tambols:
                v = v[:v.rfind("เขต")]
            if col == "no" and not DIGIT_RE.search(v):
                continue
            if WORD_CHAR_RE.search(v) and v not in found[col]:
                found[col].append(v)
        return found

    fill_cols = [c for c in OWN_KEYWORDS if c in df.columns]
    n_filled = defaultdict(int)
    new_data = {c: list(df[c]) for c in fill_cols}
    for pos, (row_idx, addr_input) in enumerate(zip(df.index, df['addr_input'])):
        blank_cols = [c for c in fill_cols if is_blank(new_data[c][pos])]
        if not blank_cols:
            continue
        found = split_by_keywords(addr_input)
        for col in blank_cols:
            # เติมเฉพาะเมื่อเจอค่าเดียว ถ้าเจอหลายค่าไม่เดา
            if len(found.get(col, [])) != 1:
                continue
            filled = found[col][0]
            new_data[col][pos] = filled
            modified_indices.add(row_idx)
            modified_row_reasons[row_idx].append(
                f"{col}: '' → '{filled}' [เพิ่ม '{filled}'] (เติมจาก addr_input ด้วยคำนำหน้า)"
            )
            mod_logs_total += 1
            n_filled[col] += 1
    for col in fill_cols:
        df[col] = new_data[col]
    print(f"7. Filled blank keys from addr_input: "
          + ", ".join(f"{c}={n_filled[c]:,}" for c in fill_cols if n_filled[c]))

    # ==========================================
    # Step 7.6: Apply manual edits (แทน 11 key ตามไฟล์แก้มือ)
    # ==========================================
    print("7.6 Applying manual edits")
    df_keys = df['addr_input'].apply(lambda v: v.strip() if isinstance(v, str) else None)
    for path in EDIT_FILES:
        file_keys = {k for k, src in edit_source.items() if src == path.name}
        if not file_keys:
            continue
        done_keys = set()
        for row_idx in df.index[df_keys.isin(file_keys)]:
            key = df_keys[row_idx]
            done_keys.add(key)
            for col, new_val in manual_edits[key].items():
                old_val = df.at[row_idx, col] if col in df.columns else None
                if not is_changed(None if is_blank(old_val) else old_val, new_val):
                    continue
                df.at[row_idx, col] = new_val
                modified_indices.add(row_idx)
                modified_row_reasons[row_idx].append(
                    f"{col}: '{as_text(old_val)}' → '{as_text(new_val)}' "
                    f"[{describe_char_changes(old_val, new_val)}] (แก้มือจาก {path.name})"
                )
                mod_logs_total += 1
        # แจ้งเมื่อแก้ครบทั้งไฟล์แล้ว
        if len(done_keys) == len(file_keys):
            print(f"    แก้ตาม {path.name} เสร็จแล้ว: ครบ {len(file_keys):,}/{len(file_keys):,} แถว")
        else:
            print(f"    แก้ตาม {path.name} ยังไม่ครบ: {len(done_keys):,}/{len(file_keys):,} แถว "
                  f"(หา addr_input ไม่เจอ {len(file_keys) - len(done_keys):,} แถว)")

    # ==========================================
    # Step 8: Merge db_postcode (reference only)
    # ==========================================
    try:
        df_db = pd.read_csv(DB_POSTCODE_PATH, dtype=str)
        for c in ["CODE", "tambol", "amphoe", "province"]:
            if c in df_db.columns:
                df_db[c] = df_db[c].fillna("").astype(str).str.strip()

        df_db_clean = df_db.drop_duplicates(
            subset=["tambol", "amphoe", "province"]
        )[["tambol", "amphoe", "province", "CODE"]].rename(columns={"CODE": "db_postcode"})

        df["_tambol_match"] = df["tambol"].fillna("").astype(str).str.strip()
        df["_amphoe_match"] = df["amphoe"].fillna("").astype(str).str.strip()
        df["_province_match"] = df["province"].fillna("").astype(str).str.strip()

        df = df.merge(
            df_db_clean,
            left_on=["_tambol_match", "_amphoe_match", "_province_match"],
            right_on=["tambol", "amphoe", "province"],
            how="left",
            suffixes=("", "_ref_db"),
        )

        cols_to_drop = ["_tambol_match", "_amphoe_match", "_province_match",
                        "tambol_ref_db", "amphoe_ref_db", "province_ref_db"]
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

        print(f"8. Merged db_postcode (reference only)")
    except FileNotFoundError:
        df["db_postcode"] = None
        print(f"8. DB_POSTCODE not found, skipping")
    except Exception as e:
        print(f"8. Error merging db_postcode: {e}")

    # ==========================================
    # Step 9: Sort columns and reset index
    # ==========================================
    lead_cols = [c for c in LEAD_COLUMNS if c in df.columns]
    tail_cols = [c for c in ["POSTCODE", "db_postcode"] if c in df.columns]
    middle_cols = [c for c in df.columns if c not in lead_cols + tail_cols]
    df = df[lead_cols + middle_cols + tail_cols]
    df = df.reset_index(drop=True)

    print(f"\n--- Final Stats ---")
    print(f"Total rows after cleaning: {len(df):,}")

    # ==========================================
    # Step 10: Add sample_reason column
    # ==========================================
    def get_sample_reason(row):
        reasons = []

        # Check if postcode was extracted from addr
        postcode = row.get('POSTCODE')
        db_postcode = row.get('db_postcode')

        if not is_blank(postcode):
            reasons.append('postcode_extracted')

        # Check if db_postcode matches
        if not is_blank(db_postcode):
            postcode_str = '' if is_blank(postcode) else str(postcode)
            db_str = str(db_postcode)
            if postcode_str == db_str:
                reasons.append('postcode_match_db')
            elif postcode_str:
                reasons.append('postcode_mismatch_db')

        # Check if any field was modified
        row_idx = row.name if row.name is not None else None
        if row_idx in modified_indices:
            reasons.append('modified')

        # Check which fields are populated
        address_fields = ['no', 'moo', 'mooban', 'building', 'floor', 'room', 'soi', 'road', 'tambol', 'amphoe', 'province']
        populated = [f for f in address_fields if not is_blank(row.get(f))]

        if len(populated) >= 8:
            reasons.append('full_address')
        elif len(populated) >= 5:
            reasons.append('partial_address')
        else:
            reasons.append('minimal_address')

        return '|'.join(reasons) if reasons else 'unknown'

    df['sample_reason'] = df.apply(get_sample_reason, axis=1)

    # รายละเอียดการแก้: field, ค่าก่อน → หลัง, [ตัวอักษรที่ถูกแก้], (เหตุผล)
    df['modify_detail'] = [
        " | ".join(modified_row_reasons[idx]) if idx in modified_row_reasons else None
        for idx in df.index
    ]

    # ==========================================
    # Step 11: Save filtered out summary
    # ==========================================
    if len(df_filtered_out) > 0:
        df_filtered_out.to_csv(OUTPUT_DIR / "summary_filtered_out.csv", index=False, encoding="utf-8-sig")
        print(f"13. Saved summary_filtered_out.csv: {len(df_filtered_out):,} rows")

    # ==========================================
    # Step 12: Save modified rows summary
    # ==========================================
    if len(modified_indices) > 0:
        modified_index_list = sorted(modified_indices)
        cols_present = [c for c in LEAD_COLUMNS + ['no', 'moo', 'mooban', 'building',
                                      'floor', 'room', 'soi', 'road', 'tambol', 'amphoe', 'province',
                                      'POSTCODE', 'db_postcode', 'modify_detail'] if c in df.columns]

        df_modified = df.loc[df.index.isin(modified_index_list), cols_present].copy()
        df_modified.to_parquet(OUTPUT_DIR / "summary_modifications_full_columns.parquet", index=False)
        print(f"13. Saved summary_modifications_full_columns.parquet: {len(df_modified):,} rows")

    # ==========================================
    # Step 13: Save group files by sample_reason
    # ==========================================
    print(f"\n--- Saving Group Files ---")

    # ลบไฟล์กลุ่มจากรอบก่อน กันไฟล์ค้างของกลุ่มที่รอบนี้ไม่มีแล้ว
    for old_file in GROUP_DIR.glob("*.parquet"):
        old_file.unlink()

    for reason, group_df in df.groupby('sample_reason'):
        safe_name = reason.replace('|', '_').replace(' ', '_')
        group_file = GROUP_DIR / f"{safe_name}.parquet"
        group_df.to_parquet(group_file, index=False)
        print(f"12. {safe_name}.parquet: {len(group_df):,} rows")

    # ==========================================
    # Step 14: Split train/eval and save
    # ==========================================
    print(f"\n--- Splitting Train/Eval ---")

    df_shuffled = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * TRAIN_RATIO)

    df_train = df_shuffled.iloc[:split_idx].reset_index(drop=True)
    df_eval = df_shuffled.iloc[split_idx:].reset_index(drop=True)

    df_train.to_parquet(OUTPUT_COMBINED, index=False)
    print(f"13. Saved combined_summary_result.parquet (train 90%): {len(df_train):,} rows")

    df_eval.to_parquet(OUTPUT_EVAL, index=False)
    print(f"15. Saved eval.parquet (eval 10%): {len(df_eval):,} rows")

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input rows: {original_rows:,}")
    print(f"Output rows: {len(df):,}")
    print(f"Train (90%): {len(df_train):,}")
    print(f"Eval (10%): {len(df_eval):,}")
    print(f"Output files:")
    print(f"  - {OUTPUT_COMBINED}")
    print(f"  - {OUTPUT_EVAL}")
    print(f"  - {GROUP_DIR}/")
    print("=" * 60)
    print("Done!")

    return df, df_train, df_eval

if __name__ == "__main__":
    clean_address_data()
