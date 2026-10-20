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
    ("มหาวิทยาลัย", "building", True), ("มหาลัย", "building", True), ("อุทยาน", "building", True),
    ("โรงพยาบาล", "building", True), ("โรงเรียน", "building", True),
    ("ศาล", "building", True), ("วัด", "building", True),
    ("ศูนย์การค้า", "building", True), ("ท่าอากาศยาน", "building", True), ("โรงแรม", "building", True),
    ("ห้างสรรพสินค้า", "building", True), ("ห้าง", "building", True), ("คลังสินค้า", "building", True),
    ("การท่าเรือ", "building", True),
    ("นิคมอุตสาหกรรม", "mooban", True), ("โครงการ", "mooban", True),
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
# ชื่ออาคารเต็ม → building ทั้งชื่อ (ช่องว่างในชื่อจะมีหรือไม่มีก็ได้: "ทรูดิจิทัล พาร์ค")
BUILDING_NAMES = [
    "ทรู ดิจิทัล พาร์ค", "ทรู ดิจิทัล พาร์ค เวสต์", "แฟมมิลี่คอนโดมิเนียม", "ท่าอากาศยานดอนเมือง",
    "บ้านละไมกุล อพาร์ทเม้นท์", "ศูนย์ราชการแจ้งวัฒนะ", "ดิ เอ็กซ์เพลส",
    # คอนโด / แมนชั่น / อพาร์ทเม้นท์
    "นานาคอนโดมิเนียม", "นิรันดร์คอนโด", "เกษมคอนโดทาวน์", "เสนาทิพย์คอนโดทาวน์",
    "บางกะปิ แกรนด์คอนโด", "เดอะมาเจสติค จอมเทียนคอนโด", "เดอะนิมมานา คอนโดมิเนียม",
    "วิวทะเลจอมเทียน คอนโดมิเนียม", "มอนิ่งดิว คอนโดมิเนี่ยม", "วิภาวดีคอนโดวิว", "สวรรค์ท้ายบ้านคอนโดทาวน์",
    "เอื้อสถาปัตย์คอนโดมิเนียม", "คอนโดเดอะไพรเวซี่ รัชดา-สุทธิสาร", "แชมเบอร์ส อ่อนนุช สเตชั่น คอนโดมิเนียม",
    "คอนโด ลุมพินีวิลล์", "ป่าตอง ลอฟท์ คอนโดมิเนียม", "ลุมพินีเพลส สาทร",
    "ปณตแมนชั่น", "ชัชวลัย แมนชั่น", "ศรีวราแมนชั่น", "มิตรภาพแมนชั่น", "มารินทร์อพาร์ทเม้นท",
    # ทาวเวอร์ / พลาซ่า / มอลล์ / เซ็นเตอร์
    "รีเจ้นท์ศรีนครินทร์ทาวเวอร์", "เดอะไนน์ ทาวเวอร์", "มิตรทาวน์ ออฟฟิศ ทาวเวอร์", "ซี.อาร์.ซี ทาวเวอร์",
    "พูนผลไนท์พลาซ่า", "ดิโอลด์สยามพลาซ่า", "เดอะพลาซ่าสุรินทร์", "รอยัลการ์เด้นท์ พลาซ่า", "เกษรพลาซ่า",
    "สุขุมวิทพลาซ่า", "ซามาเนีย พลาซ่า",
    "ยูเนี่ยนมอลล์", "เดอะมอลล์ ไลฟ์สโตร์ บางกะปิ", "อีเคเอ็มซิกซ์ คอมมูนิตี้มอลล์",
    "คริสตัล ดีไซน์ เซ็นเตอร์", "เอฟวายไอ เซ็นเตอร์", "อาร์เคบิชเซนเตอร์", "ตลาดปทุมเซ็นเตอร์",
    "ซูเปอร์เซ็นเตอร์", "ซูปเปอร์เซ็นเตอร์", "ซูเปอร์เซนเตอร์", "ทาวน์อินทาวน์",
    # ออฟฟิศ / พาร์ค / คลังสินค้า / อื่นๆ
    "รังสิตบิซซิเนสพาร์ค", "บ้านกลางกรุงออฟฟิศปาร์ค", "เจ ดับบลิว พาร์ค โฮม ออฟฟิศ", "ฟาวด์ เซอร์วิส ออฟฟิศ",
    "พาร์ควิววิภาวดี",
    "วี-แวร์เฮ้าส์", "จีพีเฮ้าส์", "ไทยฟาร์เวอร์เฮ้าส์", "เคไอจี เฮ้าส์", "ดิเอ็กเซ็คคิวทีฟ เฮ้าส์",
    "วิวทะเลเรสซิเดนซ์", "ศูนย์วิจัยเรสซิเดนซ์", "ปาล์มฮิลล์ กอล์ฟคลับ แอนด์ เรสซิเดนซ์",
    "เดอะฟลอราวิลล์ คอมเพล็กซ์", "เซนสิริสปอร์ต ทาวน์", "พลีโน่ ทาวน์ เวสต์เกต", "พฤกษาทาวน์ เน็กซ์ล็อปท์",
]
# ชื่อเขตอุตสาหกรรมเต็ม → mooban ทั้งชื่อ (แบบเดียวกับ นิคมอุตสาหกรรม)
MOOBAN_NAMES = ["ทองโกรว์เมืองอุตสาหกรรม", "นวนครเขตส่งเสริมอุตสาหกรรม"]
# ชื่อยาวเจาะจง ติดอักษรไทยข้างหน้าได้ ("อาคารทรู ดิจิทัล พาร์ค")
FILL_KEYWORDS += [(name, "building", False) for name in BUILDING_NAMES]
FILL_KEYWORDS += [(name, "mooban", False) for name in MOOBAN_NAMES]
FILL_KEYWORD_FIELD = {kw: f for kw, f, _ in FILL_KEYWORDS}
# คำที่ match ได้ (ตัดช่องว่าง) → คำนำหน้าใน FILL_KEYWORDS
KEYWORD_CANON = {kw.replace(" ", ""): kw for kw, _, _ in FILL_KEYWORDS}


def canon_keyword(matched):
    """'ทรูดิจิทัล พาร์ค' → 'ทรู ดิจิทัล พาร์ค' (ใช้ค้น key; ค่าที่เก็บยังใช้ตามที่พิมพ์มา)"""
    return KEYWORD_CANON.get(re.sub(r"\s+", "", matched), matched)
# คำที่ห้ามตามด้วยตัวอักษรนี้ ("ศาล" ไม่ใช่ "ศาลา/ศาลายา")
KEYWORD_NOT_FOLLOWED = {"ศาล": "า"}
# คำที่ห้ามมีตัวอักษรนี้นำหน้า ("จ." ใน "บมจ.บิ๊กซี" ไม่ใช่จังหวัด)
KEYWORD_NOT_PRECEDED = {"จ.": "บม"}
# คำนำหน้าที่มักเป็นต้นชื่อ soi/road/tambol ด้วย (ซอยวัดใหม่, ตำบลศาลเจ้า) → ใช้แยกช่วงจาก
# addr_input เท่านั้น (Step 7) ไม่ใช้ตัดค่าเดิมของ key อื่น (Step 6.3) และไม่บังคับ key (OWNED_WORDS)
SOFT_KEYWORDS = {"โรงพยาบาล", "โรงเรียน", "ศาล", "วัด", "โครงการ",
                 "ศูนย์การค้า", "ท่าอากาศยาน", "โรงแรม", "ห้างสรรพสินค้า", "ห้าง", "คลังสินค้า", "การท่าเรือ",
                 "ทาวน์อินทาวน์"}  # ซอยทาวน์อินทาวน์


def build_keyword_re(keywords):
    return re.compile("|".join(
        (rf"(?<![{THAI_RANGE}])" if no_thai_before else "")
        + (f"(?<!{KEYWORD_NOT_PRECEDED[kw]})" if kw in KEYWORD_NOT_PRECEDED else "")
        + r"\s*".join(re.escape(w) for w in kw.split(" "))
        + (f"(?!{KEYWORD_NOT_FOLLOWED[kw]})" if kw in KEYWORD_NOT_FOLLOWED else "")
        for kw, _, no_thai_before in sorted(keywords, key=lambda x: -len(x[0]))
    ))


NUMERIC_FIELDS = {"no", "floor", "room", "moo"}


def glued_to_prev(prev_m, m, body):
    """คำนำหน้า 2 ตัวติดกันในคำเดียว → ยึดคำนำหน้าตัวแรก ตัวที่ 2 เป็นส่วนของชื่อ
    "ซอยตำบลใหม่" → soi = ตำบลใหม่, "ถ.ตึกดิน" → road = ตึกดิน
    ยกเว้น คำย่อมีจุด ("ต.อ.เมือง"), _stop, และ ชั้น/ห้อง/เลขที่ ตามด้วยเลข ("อาคารชั้นที่ 2")"""
    if prev_m is None or prev_m.end() != m.start():
        return False
    raw = m.group(0)
    field = FILL_KEYWORD_FIELD[canon_keyword(raw)]
    if raw.endswith(".") or field == "_stop":
        return False
    if field in NUMERIC_FIELDS and body.lstrip()[:1].isdigit():
        return False
    return True


FILL_KEYWORD_RE = build_keyword_re(FILL_KEYWORDS)
CUT_KEYWORD_RE = build_keyword_re([k for k in FILL_KEYWORDS if k[0] not in SOFT_KEYWORDS])
# คำนำหน้าที่เป็นส่วนหนึ่งของชื่อ เก็บไว้ในค่าด้วย (building = มหาวิทยาลัยเกษตรศาสตร์)
KEEP_KEYWORDS = {"มหาวิทยาลัย", "มหาลัย", "อุทยาน", "นิคมอุตสาหกรรม"} | SOFT_KEYWORDS | set(BUILDING_NAMES) | set(MOOBAN_NAMES)
# key ที่ค่าเป็นคำเดียว (ตัดที่ช่องว่างแรก) — ที่เหลือเอาทั้งช่วงจนถึงคำนำหน้าถัดไป
# tambol/amphoe/province ใช้หลายคำได้ถ้าตรงกับชื่อใน DB_POSTCODE
SINGLE_TOKEN_FIELDS = {"no", "moo"}
# คำที่ต้องอยู่ใน key ของมันเท่านั้น ถ้าไปอยู่ key อื่นให้ลบค่านั้นทิ้ง
OWNED_WORDS = {"มหาวิทยาลัย": "building", "มหาลัย": "building", "อุทยาน": "building",
               "นิคมอุตสาหกรรม": "mooban"}
DB_NAME_FIELDS = ("tambol", "amphoe", "province")
LEADING_NO_RE = re.compile(r"^\d[\d/\-]*$")
LETTER_NO_RE = re.compile(r"(?!ม)[ก-ฮ]\.?\s*\d[\d/\-]*(?![^\s,])")
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

def dedupe_prefixed(v, col):
    """ตัดช่วงซ้ำเฉพาะที่คำนำหน้าของ key ซ้ำ (คำซ้ำในชื่อ 'เอส เอส พี ทาวเวอร์' ไม่ตัด)
    soi: 'ซ. 1 ซอย 1' → (ลบคำนำหน้าแรกแล้ว) '1 ซอย 1' → '1'; 'ซอย 5 ซอย 7' → '5 ซอย 7'"""
    own_kw = OWN_KEYWORDS.get(col, [])  # mooban ใช้ "ม." ด้วย
    matches = [m for m in FILL_KEYWORD_RE.finditer(v)
               if FILL_KEYWORD_FIELD[canon_keyword(m.group(0))] == col or m.group(0) in own_kw]
    if not matches:
        return v
    head = v[:matches[0].start()]
    segs = [(head, head)]  # (ข้อความ, ค่าที่ใช้เทียบ)
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(v)
        body = v[m.end():end]
        keep = canon_keyword(m.group(0)) in KEEP_KEYWORDS
        segs.append((m.group(0) + body, m.group(0) + body if keep else body))
    out, seen = [], set()
    for text, key in segs:
        key = re.sub(r"\s+", "", key)
        if key in seen:
            continue
        if key:
            seen.add(key)
        out.append(text)
    return re.sub(r"\s+", " ", "".join(out)).strip()

def fmt_counts(counts):
    """{key: จำนวน} → 'building=596, soi=214' (ไม่มี = '0')"""
    return ", ".join(f"{c}={counts[c]:,}" for c in ADDRESS_FIELDS if counts.get(c)) or "0"

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
    # เก็บ postcode ไว้ส่งออก
    cols_to_drop = [c for c in ['addr1', 'addr2'] + EXCLUDE_OUTPUT_COLUMNS if c in df.columns]
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

    # แถวที่มีค่าแก้มือไม่ต้องลบ (จะถูกแทนค่าใน Step 7.6)
    drop_mask = drop_mask & ~edit_mask

    # Store filtered reasons (ต้องเก็บก่อนลบแถวออกจาก df)
    df_filtered_out = pd.DataFrame({
        'addr': df.loc[drop_mask, 'addr'],
        'filter_reason': drop_reasons[drop_mask],
    })

    # reset index เพื่อให้ index ตรงกับหลัง merge ใน Step 8 (ใช้ติด tag 'modified')
    df = df[~drop_mask].reset_index(drop=True)
    print(f"5.2 Remove bad values: {initial_rows - len(df):,} rows removed, {len(df):,} remaining")

    modified_indices = set()
    modified_row_reasons = defaultdict(list)
    mod_logs_total = 0

    # 6.3: ค่าที่มีคำนำหน้าของ key อื่นปน → ไม่ลบแถว แต่เก็บเฉพาะช่วงของ key ตัวเอง
    #      soi = "ซอย 5 ถนนสุขุมวิท" → "ซอย 5" (คำนำหน้า ซอย ถูกลบใน Step 7 → "5")
    #      ส่วนที่ตัดออกย้ายไป key ของมัน (เขียนทับค่าเดิม): tambol = "ในเมืองอ.เมือง" → amphoe = "เมือง"
    def keep_own_segments(col, v):
        """คืน (ค่าที่เก็บไว้, {key อื่น: [ค่าที่ตัดออก]})"""
        moved = defaultdict(list)
        matches = list(CUT_KEYWORD_RE.finditer(v))
        if not matches:
            return v, moved
        own_kw = OWN_KEYWORDS.get(col, [])
        out = [v[:matches[0].start()]]  # ข้อความก่อนคำนำหน้าแรก = ค่าของ key นี้
        prev_field = col
        for i, m in enumerate(matches):
            raw = m.group(0)  # ตามที่พิมพ์มา ใช้เก็บค่า
            kw = canon_keyword(raw)  # ใช้ค้น key
            end = matches[i + 1].start() if i + 1 < len(matches) else len(v)
            body = v[m.end():end]
            # "ถ.มหาวิทยาลัย" → มหาวิทยาลัยเป็นชื่อของช่วงก่อนหน้า
            # คำนำหน้าติดกันในคำเดียว ("ซอยตำบลใหม่") → ยึดคำนำหน้าตัวแรก
            prev_m = matches[i - 1] if i > 0 else None
            if i > 0 and ((kw in KEEP_KEYWORDS and not v[prev_m.end():m.start()].strip())
                          or glued_to_prev(prev_m, m, body)):
                if prev_field == col:
                    out.append(raw + body)
                elif prev_field != "_stop":
                    moved[prev_field][-1] += raw + body
                continue
            field = FILL_KEYWORD_FIELD[kw]
            if kw == "ม." and body.strip() and not body.strip()[0].isdigit():
                field = "mooban"
            if field in (col, "_stop") or kw in own_kw:
                prev_field = col
                out.append(raw + body)
            else:
                prev_field = field
                # คำนำหน้าตัดทิ้ง ยกเว้นคำที่เป็นส่วนของชื่อ (มหาวิทยาลัย/อุทยาน/นิคมอุตสาหกรรม)
                moved[field].append(raw + body if kw in KEEP_KEYWORDS else body)
        kept = re.sub(r"\s+", " ", "".join(out)).strip()
        moved = {f: [x for x in (re.sub(r"\s+", " ", s).strip() for s in segs) if x]
                 for f, segs in moved.items()}
        return kept, {f: segs for f, segs in moved.items() if segs}

    n_cut = defaultdict(int)
    n_moved = defaultdict(int)
    cut_cols = [c for c in COLUMN_RULES if c in df.columns]
    cut_data = {c: list(df[c]) for c in cut_cols}
    for col in cut_cols:
        for pos, (row_idx, val) in enumerate(zip(df.index, cut_data[col])):
            if not isinstance(val, str):
                continue
            new_val, moved = keep_own_segments(col, val)
            if new_val == re.sub(r"\s+", " ", val).strip():
                continue  # ไม่มีส่วนของ key อื่น ปล่อยให้ Step 7 จัดการช่องว่าง
            n_cut[col] += 1
            cut_data[col][pos] = new_val if new_val else None
            modified_indices.add(row_idx)
            modified_row_reasons[row_idx].append(
                f"{col}: '{val}' → '{new_val}' "
                f"[{describe_char_changes(val, new_val)}] (เก็บเฉพาะค่าของ key ตัดส่วนของ key อื่น)"
            )
            mod_logs_total += 1
            # ย้ายส่วนที่ตัดออกไป key ของมัน (เขียนทับค่าเดิม) เฉพาะเมื่อเจอค่าเดียว
            for f, segs in moved.items():
                if f not in cut_data or len(segs) != 1:
                    continue
                old_val = cut_data[f][pos]
                if old_val == segs[0]:
                    continue
                cut_data[f][pos] = segs[0]
                n_moved[f] += 1
                modified_row_reasons[row_idx].append(
                    f"{f}: '{as_text(old_val)}' → '{segs[0]}' "
                    f"[{describe_char_changes(old_val, segs[0])}] (ย้ายมาจาก {col} เขียนทับค่าเดิม)"
                )
                mod_logs_total += 1
    for col in cut_cols:
        df[col] = cut_data[col]
    print(f"5.3 [ใหม่] ตัดส่วนของ key อื่นออกจากค่า (ไม่ลบแถว): {fmt_counts(n_cut)}")
    print(f"5.4 [ใหม่] ย้ายส่วนที่ตัดออกไป key ของมัน (เขียนทับค่าเดิม): {fmt_counts(n_moved)}")

    # ==========================================
    # Step 7: Clean column values
    # ==========================================
    print("\n--- Cleaning Column Values ---")

    n_dedupe = defaultdict(int)

    cols_to_clean = [col for col in df.columns if df[col].dtype == 'string' or df[col].dtype == 'object']

    for col in cols_to_clean:
        if col in ['addr', 'addr_input', 'POSTCODE', 'postcode']:
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
                current_val, r = remove_prefix_with_reason(current_val, r'^(\s*(อาคาร|ตึก)\s*)+', "ลบคำนำหน้า อาคาร/ตึก")
                reasons_list.extend(r)

            elif col == 'floor':
                current_val, r = remove_prefix_with_reason(current_val, r'^(\s*ชั้น(ที่)?\s*)+', "ลบคำนำหน้า ชั้น")
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

                if col in ADDRESS_FIELDS:
                    new_val = dedupe_prefixed(current_val, col)
                    if new_val != current_val:
                        reasons_list.append("ตัดคำซ้ำ")
                        current_val = new_val
                        n_dedupe[col] += 1

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
    print(f"6.1 [ใหม่] ตัดช่วงซ้ำที่คำนำหน้าซ้ำ ('ซ. 1 ซอย 1' → '1') เสร็จแล้ว: จำนวน row ที่แก้ {fmt_counts(n_dedupe)}")

    # ==========================================
    # Step 7.5: Fill/overwrite keys from addr_input (rule base: คำนำหน้าของ 11 key)
    #           addr_input เป็นตัวตัดสิน: เจอค่า → เขียนทับ, เจอหลายค่า → รวมคั่นด้วย ", "
    # ==========================================
    try:
        df_names = pd.read_csv(DB_POSTCODE_PATH, dtype=str)
        known_names = {c: set(df_names[c].dropna().str.strip()) for c in DB_NAME_FIELDS}
    except FileNotFoundError:
        known_names = {c: set() for c in DB_NAME_FIELDS}

    def clean_filled(v):
        v = v.translate(THAI_DIGIT_TRANS)
        v = DISALLOWED_CHAR_RE.sub(' ', v)
        v = re.sub(r'\s+', ' ', v).strip()
        return LEADING_NON_WORD_RE.sub('', v).strip()

    def join_parts(parts):
        """รวมช่วงที่ติดกันของ key เดียวกัน ตัดช่วงที่ซ้ำ
        [("อาคาร","X "),("อาคาร","X ")] → "X"
        [("มหาวิทยาลัย","จุฬา และ "),("อาคาร"," เอ ")] → "มหาวิทยาลัยจุฬา และ อาคาร เอ" """
        out, seen = [], set()
        for kw, body in parts:
            body_clean = clean_filled(body)
            full = clean_filled(kw + body)
            # KEEP_KEYWORDS เป็นส่วนของชื่อ ("ทรู ดิจิทัล พาร์ค" + อาคาร... body ว่างก็ยังเป็นค่า)
            keep = canon_keyword(kw) in KEEP_KEYWORDS
            key = (full if keep else body_clean).replace(" ", "")
            if not key or key in seen:
                continue
            seen.add(key)
            # คำนำหน้าตัวแรกตัดทิ้ง (ยกเว้น KEEP_KEYWORDS) ตัวถัดไปเก็บตามที่พิมพ์มา
            out.append(full if out or keep else body_clean)
        return " ".join(out)

    def pick_known_name(col, v):
        """เอาคำยาวที่สุดที่ตรงกับชื่อใน DB ('เมือง ชลบุรี' → ไม่ตรง → 'เมือง')"""
        words = v.split(" ")
        for k in range(len(words), 1, -1):
            cand = " ".join(words[:k])
            if cand in known_names[col] or cand.replace(" ", "") in known_names[col]:
                return cand
        return words[0]

    def split_by_keywords(addr_input):
        """แยก addr_input เป็น {key: [ค่า]} ตามช่วงหลังคำนำหน้า จนถึงคำนำหน้าของ key อื่น"""
        found = defaultdict(list)
        if not isinstance(addr_input, str):
            return found
        text = addr_input.translate(THAI_DIGIT_TRANS)
        matches = list(FILL_KEYWORD_RE.finditer(text))

        # 1) แบ่งเป็นช่วง [key, คำนำหน้า, ข้อความหลังคำนำหน้า]
        segs = []
        for i, m in enumerate(matches):
            raw = m.group(0)  # ตามที่พิมพ์มา ใช้เก็บค่า
            kw = canon_keyword(raw)  # ใช้ค้น key
            col = FILL_KEYWORD_FIELD[kw]
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[m.end():end]
            # "ถ. มหาวิทยาลัย" → เป็นชื่อของ key ก่อนหน้า ไม่ใช่ building
            # คำนำหน้าติดกันในคำเดียว ("ซอยตำบลใหม่") → ยึดคำนำหน้าตัวแรก
            prev_m = matches[i - 1] if i > 0 else None
            if (segs and segs[-1][0] != "_stop" and not segs[-1][2].strip()
                    and (kw in KEEP_KEYWORDS or glued_to_prev(prev_m, m, body))):
                segs[-1][2] = raw + body
                continue
            if kw == "ม." and body.strip() and not body.strip()[0].isdigit():
                col = "mooban"
            segs.append([col, raw, body])

        # 2) ช่วงที่ติดกันและเป็น key เดียวกันรวมเป็นค่าเดียว ("ชั้นที่ 7 ชั้น 7" → 7)
        groups = []
        for col, kw, body in segs:
            if groups and groups[-1][0] == col:
                groups[-1][1].append((kw, body))
            else:
                groups.append([col, [(kw, body)]])

        for col, parts in groups:
            if col == "_stop":
                continue
            # ค่าจบที่ "," เหมือนเดิม
            cut = []
            for kw, body in parts:
                cut.append((kw, body.split(",")[0]))
                if "," in body:
                    break
            v = join_parts(cut)
            # "(จีพีเฮ้าส์) ถนน..." → ช่วงได้ "จีพีเฮ้าส์)" ลบวงเล็บไม่ครบคู่เหมือน Step 6
            if has_unbalanced_brackets(v):
                v = re.sub(r'\s+', ' ', re.sub(r'[\(\)\[\]\{\}]', ' ', v)).strip()
            if not v:
                continue
            if col in SINGLE_TOKEN_FIELDS:
                v = v.split(" ")[0]
            if col in DB_NAME_FIELDS:
                v = pick_known_name(col, v)
            if col == "moo":
                v = re.match(r"\d*", v).group(0)
            # แขวงลาดยาวเขตจตุจักร → ลาดยาว (แต่ หลักเขต/คีรีเขต เป็นชื่อตำบลจริง)
            if col == "tambol" and "เขต" in v and v not in known_names["tambol"]:
                v = v[:v.rfind("เขต")]
            if col == "no" and not DIGIT_RE.search(v):
                continue
            if WORD_CHAR_RE.search(v) and v not in found[col]:
                found[col].append(v)

        # 3) เลขบ้านขึ้นต้นโดยไม่มีคำนำหน้า: "191/66 ถ.รัชดาภิเษก ..." → no = 191/66
        #    อักษรไทย 1 ตัว (ไม่ใช่ ม) + เลข เก็บตามที่พิมพ์: "ข 49/3 ถ.วิไลเสนา" → no = "ข 49/3"
        if not found["no"]:
            head = text[:matches[0].start()] if matches else text
            first = head.split()[0] if head.split() else ""
            letter_no = LETTER_NO_RE.match(head.lstrip())
            if LEADING_NO_RE.match(first):
                found["no"].append(first)
                found["_leading_no"] = True
            elif letter_no:
                found["no"].append(letter_no.group(0))
                found["_leading_no"] = True
        return found

    fill_cols = [c for c in OWN_KEYWORDS if c in df.columns]
    n_filled = defaultdict(int)
    n_overwritten = defaultdict(int)
    n_leading_no = 0
    new_data = {c: list(df[c]) for c in fill_cols}
    for pos, (row_idx, addr_input) in enumerate(zip(df.index, df['addr_input'])):
        found = split_by_keywords(addr_input)
        for col in fill_cols:
            if not found.get(col):
                continue
            # เจอหลายค่า → รวมทุกค่าคั่นด้วย ", " ("อาคาร A ... อาคาร B" → "A, B")
            new_val = ", ".join(found[col])
            old_val = new_data[col][pos]
            if not is_blank(old_val) and str(old_val) == new_val:
                continue
            if is_blank(old_val):
                if col == "no" and found.get("_leading_no"):
                    reason = "เติมจากเลขบ้านต้น addr_input"
                    n_leading_no += 1
                else:
                    reason = "เติมจาก addr_input ด้วยคำนำหน้า"
                    n_filled[col] += 1
            else:
                # addr_input เป็นตัวตัดสิน: เขียนทับค่าเดิม (แม้สะกดต่างจากเดิม เช่น เมือง → เมอง)
                reason = "เขียนทับตาม addr_input"
                n_overwritten[col] += 1
            new_data[col][pos] = new_val
            modified_indices.add(row_idx)
            modified_row_reasons[row_idx].append(
                f"{col}: '{as_text(old_val)}' → '{new_val}' "
                f"[{describe_char_changes(old_val, new_val)}] ({reason})"
            )
            mod_logs_total += 1
    for col in fill_cols:
        df[col] = new_data[col]
    print(f"7. เติม key ว่างจาก addr_input ด้วยคำนำหน้า เสร็จแล้ว: {fmt_counts(n_filled)}")
    print(f"7.1 [ใหม่] เติม no จากเลขบ้านต้น addr_input (ไม่มีคำนำหน้า) เสร็จแล้ว: no={n_leading_no:,}")
    print(f"7.2 [ใหม่] เขียนทับค่าเดิมตาม addr_input เสร็จแล้ว: {fmt_counts(n_overwritten)}")

    # ==========================================
    # Step 7.55: มหาวิทยาลัย/อุทยาน ต้องอยู่ใน building, นิคมอุตสาหกรรม ต้องอยู่ใน mooban
    #            ถ้าไปอยู่ key อื่นให้ลบค่านั้นทิ้ง
    #            ยกเว้นคำนั้นตามหลังคำนำหน้าของ key นั้นใน addr_input (ยึดคำนำหน้าตัวแรก)
    #            "หมู่บ้านมหาวิทยาลัยขอนแก่น" → mooban = มหาวิทยาลัยขอนแก่น (ไม่ลบ)
    # ==========================================
    def first_prefix_owns(col, word, addr_input):
        if not isinstance(addr_input, str):
            return False
        prefixes = [kw for kw, f, _ in FILL_KEYWORDS if f == col and kw not in KEEP_KEYWORDS]
        prefixes += OWN_KEYWORDS.get(col, [])
        pattern = "(?:" + "|".join(re.escape(p) for p in sorted(set(prefixes), key=len, reverse=True)) + r")\s*" + re.escape(word)
        return bool(re.search(pattern, addr_input))

    n_cleared = defaultdict(int)
    for word, owner in OWNED_WORDS.items():
        for col in fill_cols:
            if col == owner:
                continue
            mask = df[col].apply(lambda v: isinstance(v, str) and word in v)
            for row_idx in df.index[mask]:
                if first_prefix_owns(col, word, df.at[row_idx, 'addr_input']):
                    continue
                old_val = df.at[row_idx, col]
                df.at[row_idx, col] = None
                modified_indices.add(row_idx)
                modified_row_reasons[row_idx].append(
                    f"{col}: '{old_val}' → '' [ลบ '{old_val}'] (มีคำ '{word}' ต้องอยู่ใน {owner})"
                )
                mod_logs_total += 1
                n_cleared[col] += 1
    print(f"7.4 [ใหม่] ลบค่าที่อยู่ผิด key (มหาวิทยาลัย/มหาลัย/อุทยาน→building, "
          f"นิคมอุตสาหกรรม→mooban) เสร็จแล้ว: {fmt_counts(n_cleared)}")

    # ==========================================
    # Step 7.58: ค่าที่ไม่ปรากฏใน addr_input ให้ลบทิ้ง (addr_input เป็นตัวตัดสิน)
    #            เช่น addr_input "... จ.ขอนแก่น" แต่ amphoe = "เมืองขอนแก่น" → ว่าง
    #            no/moo/floor/room ต้องเจอเป็นตัวทั้งตัว (moo "1" ไม่นับใน "101")
    # ==========================================
    WHOLE_TOKEN_FIELDS = {"no", "moo", "floor", "room"}

    def norm_text(v):
        """ทำให้อยู่ในรูปเดียวกับค่าที่ผ่าน Step 7 แล้ว"""
        v = FLOATING_VOWEL_RE.sub('', v)
        v = REPEATED_MARK_RE.sub(r'\1', v)
        v = v.translate(THAI_DIGIT_TRANS)
        v = DISALLOWED_CHAR_RE.sub(' ', v)
        v = re.sub(r'[\(\)\[\]\{\}]', ' ', v)
        return re.sub(r'\s+', ' ', v).strip()

    def appears_in(col, part, inp, inp_nospace):
        p = norm_text(part)
        if not p:
            return False
        if col in WHOLE_TOKEN_FIELDS:
            return bool(re.search(rf"(?<![0-9A-Za-z]){re.escape(p)}(?![0-9A-Za-z])", inp))
        return p.replace(" ", "") in inp_nospace

    n_not_in_input = defaultdict(int)
    for row_idx, addr_input in zip(df.index, df['addr_input']):
        if is_blank(addr_input):
            continue
        inp = norm_text(addr_input)
        inp_nospace = inp.replace(" ", "")
        for col in fill_cols:
            old_val = df.at[row_idx, col]
            if is_blank(old_val):
                continue
            # ค่าที่รวมหลายค่า ("A, B") เช็คทีละค่า
            parts = [x.strip() for x in str(old_val).split(",") if x.strip()]
            kept = [x for x in parts if appears_in(col, x, inp, inp_nospace)]
            if len(kept) == len(parts):
                continue
            new_val = ", ".join(kept) or None
            df.at[row_idx, col] = new_val
            modified_indices.add(row_idx)
            modified_row_reasons[row_idx].append(
                f"{col}: '{old_val}' → '{as_text(new_val)}' "
                f"[{describe_char_changes(old_val, new_val)}] (ไม่ปรากฏใน addr_input)"
            )
            mod_logs_total += 1
            n_not_in_input[col] += 1
    print(f"7.5 [ใหม่] ลบค่าที่ไม่ปรากฏใน addr_input เสร็จแล้ว: {fmt_counts(n_not_in_input)}")

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
        # addr_input ที่ไม่มีใน df → เพิ่มเป็นแถวใหม่ (addr = addr_input, index ต่อท้าย)
        new_keys = sorted(file_keys - done_keys)
        if new_keys:
            start = len(df)
            df_new = pd.DataFrame(
                [{"addr": k, "addr_input": k, **manual_edits[k]} for k in new_keys],
                index=range(start, start + len(new_keys)),
            )
            df = pd.concat([df, df_new])
            df_keys = pd.concat([df_keys, df_new["addr_input"]])
            for row_idx, key in zip(df_new.index, new_keys):
                modified_indices.add(row_idx)
                modified_row_reasons[row_idx].append(f"เพิ่มแถวใหม่จาก {path.name} (ไม่มี addr_input นี้ในข้อมูล)")
                mod_logs_total += 1
        print(f"    แก้ตาม {path.name} เสร็จแล้ว: แก้ {len(done_keys):,} แถว, "
              f"เพิ่มแถวใหม่ {len(new_keys):,} แถว (รวม {len(file_keys):,})")

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
    tail_cols = [c for c in ["postcode", "POSTCODE", "db_postcode"] if c in df.columns]
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
                                      'postcode', 'POSTCODE', 'db_postcode', 'modify_detail'] if c in df.columns]

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
