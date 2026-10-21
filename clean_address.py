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
SPECIAL_CHARS_RE = re.compile(r"[!@#$%^&*_=+{}\[\]<>~`|\\\"\';]")

# เลขไทยเก็บตามที่พิมพ์มา ไม่แปลงเป็นอารบิก (แปลงเฉพาะตอนเทียบค่าใน Step 7.6)
THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
ARABIC_DIGITS = "0123456789"
THAI_DIGIT_TRANS = str.maketrans(THAI_DIGITS, ARABIC_DIGITS)
DIGIT_RE = re.compile(f"[{ARABIC_DIGITS}{THAI_DIGITS}]")

# \u0E2D\u0E31\u0E01\u0E02\u0E23\u0E30\u0E1E\u0E34\u0E40\u0E28\u0E29\u0E17\u0E35\u0E48\u0E44\u0E21\u0E48\u0E04\u0E25\u0E35\u0E19 (\u0E40\u0E01\u0E47\u0E1A\u0E44\u0E27\u0E49\u0E43\u0E19\u0E04\u0E48\u0E32) \u0E19\u0E2D\u0E01\u0E19\u0E31\u0E49\u0E19\u0E41\u0E17\u0E19\u0E14\u0E49\u0E27\u0E22\u0E0A\u0E48\u0E2D\u0E07\u0E27\u0E48\u0E32\u0E07
KEEP_SPECIAL_CHARS = "-/&',_|+\".\u2019\\@"
# \u0e2d\u0e31\u0e01\u0e02\u0e23\u0e30\u0e1e\u0e34\u0e40\u0e28\u0e29\u0e17\u0e35\u0e48\u0e2b\u0e49\u0e32\u0e21\u0e04\u0e25\u0e35\u0e19: \u0e40\u0e01\u0e47\u0e1a\u0e44\u0e27\u0e49\u0e15\u0e32\u0e21\u0e17\u0e35\u0e48\u0e1e\u0e34\u0e21\u0e1e\u0e4c\u0e21\u0e32\u0e17\u0e38\u0e01\u0e02\u0e31\u0e49\u0e19\u0e15\u0e2d\u0e19 (\u0e44\u0e21\u0e48\u0e25\u0e1a \u0e44\u0e21\u0e48\u0e41\u0e17\u0e19\u0e14\u0e49\u0e27\u0e22\u0e0a\u0e48\u0e2d\u0e07\u0e27\u0e48\u0e32\u0e07 \u0e44\u0e21\u0e48\u0e15\u0e31\u0e14\u0e2d\u0e2d\u0e01\u0e08\u0e32\u0e01\u0e15\u0e49\u0e19\u0e04\u0e48\u0e32)
# "(\u0e08\u0e35\u0e1e\u0e35\u0e40\u0e2e\u0e49\u0e32\u0e2a\u0e4c)" \u2192 "(\u0e08\u0e35\u0e1e\u0e35\u0e40\u0e2e\u0e49\u0e32\u0e2a\u0e4c)", "#12" \u2192 "#12", "A:1" \u2192 "A:1"
PROTECTED_SPECIAL_CHARS = "()[]{}#$%*=;:!?<>~^`"
KEEP_SPECIAL_CHARS += PROTECTED_SPECIAL_CHARS
DISALLOWED_CHAR_RE = re.compile(rf"[^a-zA-Z0-9{THAI_RANGE}\s{re.escape(KEEP_SPECIAL_CHARS)}]")
# \u0E27\u0E07\u0E40\u0E25\u0E47\u0E1A\u0E17\u0E38\u0E01\u0E41\u0E1A\u0E1A\u0E43\u0E19 11 key \u0E41\u0E17\u0E19\u0E14\u0E49\u0E27\u0E22\u0E0A\u0E48\u0E2D\u0E07\u0E27\u0E48\u0E32\u0E07 (\u0E04\u0E23\u0E1A\u0E04\u0E39\u0E48\u0E2B\u0E23\u0E37\u0E2D\u0E44\u0E21\u0E48\u0E01\u0E47\u0E25\u0E1A)
BRACKET_RE = re.compile(r"[\(\)\[\]\{\}]")
WORD_CHAR_RE = re.compile(r"[a-zA-Z0-9\u0E00-\u0E7F]")
# \u0E25\u0E1A\u0E2D\u0E31\u0E01\u0E02\u0E23\u0E30\u0E19\u0E33\u0E2B\u0E19\u0E49\u0E32\u0E17\u0E35\u0E48\u0E44\u0E21\u0E48\u0E43\u0E0A\u0E48\u0E15\u0E31\u0E27\u0E2D\u0E31\u0E01\u0E29\u0E23/\u0E15\u0E31\u0E27\u0E40\u0E25\u0E02 \u0E22\u0E01\u0E40\u0E27\u0E49\u0E19 PROTECTED_SPECIAL_CHARS ("(\u0E0B\u0E2D\u0E22 5)" \u0E44\u0E21\u0E48\u0E16\u0E39\u0E01\u0E15\u0E31\u0E14 "(")
_PROTECTED_CLASS = re.escape(PROTECTED_SPECIAL_CHARS)
LEADING_NON_WORD_RE = re.compile(rf"^[^a-zA-Z0-9\u0E00-\u0E7F{_PROTECTED_CLASS}]+")
LEADING_NON_WORD_KEEP_AT_RE = re.compile(rf"^[^a-zA-Z0-9\u0E00-\u0E7F@{_PROTECTED_CLASS}]+")
THAI_OR_ENG_CHAR_RE = re.compile(r"[a-zA-Z\u0E00-\u0E7F]")
DIGIT_CLASS = f"[{ARABIC_DIGITS}{THAI_DIGITS}]"
POSTCODE_TOKEN_RE = re.compile(rf'(?<!{DIGIT_CLASS})({DIGIT_CLASS}{{5}})(?!{DIGIT_CLASS})')
POSTCODE_PREFIX_RE = re.compile(
    rf'(?:เลขรหัสไปรษณีย์|รหัสไปรษณีย์|ปณ\.?)\s*[:\-]?\s*({DIGIT_CLASS}{{5}})(?!{DIGIT_CLASS})'
)

# พื้นที่ร้านค้า/บูธ ในอาคาร เป็น room (ใช้ใน FILL_KEYWORDS และ PREFIX_RULES)
ROOM_UNIT_WORDS = ["บูธ", "ยูนิต", "แผงค้า", "แผง", "ล็อค", "ล็อก", "พื้นที่"]  # ต้องตามด้วย เลขที่
ROOM_UNIT_ALONE_WORDS = ["บูธ", "ยูนิต"]  # ไม่มี เลขที่ ก็เป็น room ("บูธ J12")

# เติม key ที่ว่างจาก addr_input ด้วยคำนำหน้า (rule base)
# (คำนำหน้า, key, ต้องไม่ติดอักษรไทยข้างหน้า) — คำย่อมีจุดและคำที่อาจอยู่กลางชื่อ
# (เช่น อาคาร ใน "หมู่บ้านอาคารสงเคราะห์") ต้องไม่ติดอักษรไทย กันไปจับกลางคำ
# key "_stop" ใช้ตัดช่วงอย่างเดียว ไม่เอาไปเติม
FILL_KEYWORDS = [
    # เลขที่ หลัง บ้าน → no, หลัง ห้อง → room (มีช่องว่างคั่นหรือไม่ก็ได้: "ห้อง เลขที่ 5" → room = 5)
    ("บ้าน เลขที่", "no", False), ("เลขที่ ห้อง", "room", False), ("ห้อง เลขที่", "room", False),
    ("เลขที่", "no", False),
    ("หมู่บ้าน", "mooban", True), ("หมู่ที่", "moo", True), ("หมู่", "moo", True),
    ("ม.", "moo", True),  # ม. ตามด้วยตัวอักษร = mooban (แยกใน split_by_keywords)
    ("อาคาร", "building", True), ("ตึก", "building", True),
    ("มหาวิทยาลัย", "building", True), ("มหาลัย", "building", True), ("อุทยาน", "building", True),
    ("โรงพยาบาล", "building", True), ("โรงเรียน", "building", True),
    # วัด/ศาล ไม่ใช่คำนำหน้า building (เป็นส่วนของชื่อ key อื่น)
    ("ศูนย์การค้า", "building", True), ("ท่าอากาศยาน", "building", True), ("โรงแรม", "building", True),
    ("ห้างสรรพสินค้า", "building", True), ("ห้าง", "building", True), ("คลังสินค้า", "building", True),
    ("การท่าเรือ", "building", True),
    ("นิคมอุตสาหกรรม", "mooban", True), ("โครงการ", "mooban", True),
    ("ชั้นที่", "floor", False), ("ชั้น", "floor", False),  # "อาคารชั้นที่ 2"
    ("ห้องที่", "room", True), ("ห้อง", "room", True),
    # บูธ/ยูนิต/แผงค้า... (+ เลขที่) → room ไม่ใช่ no ("919/1 บูธเลขที่ บีบี017" → no = 919/1, room = บีบี017)
    *[(w, "room", True) for w in ROOM_UNIT_ALONE_WORDS],
    *[(f"{w} เลขที่", "room", True) for w in ROOM_UNIT_WORDS],
    ("ซอย", "soi", True), ("ซ.", "soi", True),
    ("ถนน", "road", True), ("ถ.", "road", True),
    # ตำบล/อำเภอ/จังหวัด มักติดท้ายชื่อ เช่น "ตำบลในเมืองอำเภอเมือง"
    # คำย่อมีจุด (ต./อ./จ./ม./ซ./ถ.) ห้ามมีตัวอักษรนำหน้าเสมอ ("บจ." ไม่ใช่จังหวัด)
    ("ตำบล", "tambol", False), ("แขวง", "tambol", False), ("ต.", "tambol", True),
    ("กิ่งอำเภอ", "amphoe", False), ("อำเภอ", "amphoe", False), ("เขต", "amphoe", True),
    ("อ.", "amphoe", True),
    ("จังหวัด", "province", False), ("จ.", "province", True),
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
    "แฟลตการเคหะ"
]
# ชื่อเขตอุตสาหกรรมเต็ม → mooban ทั้งชื่อ (แบบเดียวกับ นิคมอุตสาหกรรม)
MOOBAN_NAMES = ["ทองโกรว์เมืองอุตสาหกรรม", "นวนครเขตส่งเสริมอุตสาหกรรม"]
# ชื่อยาวเจาะจง ติดอักษรไทยข้างหน้าได้ ("อาคารทรู ดิจิทัล พาร์ค")
FILL_KEYWORDS += [(name, "building", False) for name in BUILDING_NAMES]
FILL_KEYWORDS += [(name, "mooban", False) for name in MOOBAN_NAMES]
FILL_KEYWORD_FIELD = {kw: f for kw, f, _ in FILL_KEYWORDS}
# คำที่ match ได้ (ตัดช่องว่าง) → คำนำหน้าใน FILL_KEYWORDS
KEYWORD_CANON = {kw.replace(" ", ""): kw for kw, _, _ in FILL_KEYWORDS}


# คำไทย 1 คำระหว่างคำนำหน้า (ห้าม ตัวเลข/อังกฤษ/คำนำหน้า key อื่น) → ยังเป็นคำนำหน้าเดียวกัน
# "ห้อง บูธเลขที่ JB303", "ห้อง พื้นที่เลขที่ ทีเอ็นซี.เอฟ01.ซ", "ห้อง แผงค้าเลขที่ FMF15-006" → room
_INFIX_WORD = r"(?:(?:(?!อาคาร|ตึก|ชั้น|ถนน|ซอย|หมู่|บ้าน|เลขที่)[ก-๎])+\s*)?"
KEYWORD_INFIX = {"ห้อง เลขที่": _INFIX_WORD}
KEYWORD_INFIX_RE = {kw: re.compile(rf"{re.escape(kw.split(' ')[0])}\s*{infix}{re.escape(kw.split(' ')[1])}")
                    for kw, infix in KEYWORD_INFIX.items()}


def canon_keyword(matched):
    """'ทรูดิจิทัล พาร์ค' → 'ทรู ดิจิทัล พาร์ค' (ใช้ค้น key; ค่าที่เก็บยังใช้ตามที่พิมพ์มา)
    'ห้อง บูธเลขที่' → 'ห้อง เลขที่'"""
    kw = KEYWORD_CANON.get(re.sub(r"\s+", "", matched))
    if kw:
        return kw
    for kw, pattern in KEYWORD_INFIX_RE.items():
        if pattern.fullmatch(matched):
            return kw
    return matched
# คำที่ห้ามตามด้วยตัวอักษรนี้ เช่น {"ศาล": "า"} ("ศาล" ไม่ใช่ "ศาลา/ศาลายา")
KEYWORD_NOT_FOLLOWED = {}
# คำที่ห้ามมีข้อความนี้นำหน้า (regex ความยาวคงที่)
# คำนำหน้าที่มี "." ทุกตัว ห้ามมีตัวอักษรไทย/อังกฤษนำหน้า (ดู build_keyword_re)
#   "บจ." "บมจ." ไม่ใช่จังหวัด, "กม." "กทม." "km." ไม่ใช่ หมู่/หมู่บ้าน
# "ม." เพิ่ม: "ก.ม." ไม่ใช่ หมู่/หมู่บ้าน
KEYWORD_NOT_PRECEDED = {"ม.": [rf"[{THAI_RANGE}A-Za-z]\."]}
# คำนำหน้าที่มักเป็นต้นชื่อ soi/road/tambol ด้วย (ซอยวัดใหม่, ตำบลศาลเจ้า) → ใช้แยกช่วงจาก
# addr_input เท่านั้น (Step 7) ไม่ใช้ตัดค่าเดิมของ key อื่น (Step 6.3) และไม่บังคับ key (OWNED_WORDS)
SOFT_KEYWORDS = {"โรงพยาบาล", "โรงเรียน", "โครงการ",
                 "ศูนย์การค้า", "ท่าอากาศยาน", "โรงแรม", "ห้างสรรพสินค้า", "ห้าง", "คลังสินค้า", "การท่าเรือ",
                 "ทาวน์อินทาวน์"}  # ซอยทาวน์อินทาวน์


def build_keyword_re(keywords):
    # สระหน้า เ แ โ ใ ไ ต้องมีพยัญชนะตามเสมอ คำนำหน้าจึงขึ้นต้นหลังสระหน้าไม่ได้
    # ("อาคารเอ.เอช.เอช" → "อ." ใน "เอ." ไม่ใช่อำเภอ)
    return re.compile("|".join(
        r"(?<![เ-ไ])"
        + (rf"(?<![{THAI_RANGE}A-Za-z])" if kw.endswith(".")
           else rf"(?<![{THAI_RANGE}])" if no_thai_before else "")
        + "".join(f"(?<!{p})" for p in KEYWORD_NOT_PRECEDED.get(kw, []))
        + (KEYWORD_INFIX_RE[kw].pattern if kw in KEYWORD_INFIX_RE
           else r"\s*".join(re.escape(w) for w in kw.split(" ")))
        + (f"(?!{KEYWORD_NOT_FOLLOWED[kw]})" if kw in KEYWORD_NOT_FOLLOWED else "")
        for kw, _, no_thai_before in sorted(keywords, key=lambda x: -len(x[0]))
    ))


# ==========================================
# THAI CHARACTER ORDER — สระ/วรรณยุกต์ผิดตำแหน่ง (ใช้ใน fix_thai_chars)
# ==========================================
# อักษรไทยแบ่งตามตำแหน่งที่เขียน:
#   พยัญชนะ           ก-ฮ (รวม ฤ ฦ)                    U+0E01-0E2E
#   สระหน้า            เ แ โ ใ ไ                        U+0E40-0E44  ต้องมีพยัญชนะตามหลังทันที
#   สระหลัง            ะ า ำ ๅ                          U+0E30 0E32 0E33 0E45  ต้องตามหลังพยัญชนะ
#   สระบน/ล่าง         ั ิ ี ึ ื ็ / ุ ู                     U+0E31 0E34-0E37 0E47 / 0E38-0E39
#   วรรณยุกต์           ่ ้ ๊ ๋                             U+0E48-0E4B
#   การันต์/นิคหิต/ยามักการ ์ ํ ๎                          U+0E4C-0E4E
#   สระบน/ล่าง วรรณยุกต์ และ ์ ํ ๎ ต้องเกาะบนพยัญชนะ
#   พินทุ ฺ (U+0E3A) ใช้แต่ภาษาบาลี ไม่มีในที่อยู่ → ลบทิ้ง
TH_CONSONANT = {chr(c) for c in range(0x0E01, 0x0E2F)}
TH_LEAD_VOWEL = set("\u0E40\u0E41\u0E42\u0E43\u0E44")
TH_FOLLOW_VOWEL = set("\u0E30\u0E32\u0E33\u0E45")
TH_ABOVE_BELOW_VOWEL = set("\u0E31\u0E34\u0E35\u0E36\u0E37\u0E47\u0E38\u0E39")
TH_TONE = set("\u0E48\u0E49\u0E4A\u0E4B")
TH_OTHER_MARK = set("\u0E4C\u0E4D\u0E4E")
TH_COMBINING = TH_ABOVE_BELOW_VOWEL | TH_TONE | TH_OTHER_MARK
TH_WORD_KINDS = {"cons", "cons_vowel", "follow", "follow_aa", "pai"}  # ตัวที่จบคำไทยได้ (ให้ ฯ ๆ ตาม)
DOUBLE_E_RE = re.compile(r"\u0E40\u0E40")  # เ เ → แ (ทำก่อนทุกขั้นตอน)
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
PHINTHU_RE = re.compile(r"\u0E3A")
TH_UNUSED_RE = re.compile(r"[\u0E3B-\u0E3F\u0E4F\u0E5A-\u0E7F]")  # ฿ ๏ ๚ ๛ และ code point ว่าง
SARA_AM_SPLIT_RE = re.compile(r"\u0E4D([\u0E48-\u0E4B]?)\u0E32")  # ํ (+วรรณยุกต์) + า → ำ
SARA_AM_TONE_RE = re.compile(r"\u0E33([\u0E48-\u0E4B])")          # ำ + วรรณยุกต์ → วรรณยุกต์ + ำ


def fix_thai_chars(v):
    """แก้อักษรไทยที่ผิดตำแหน่ง/เกิน คืน (ค่าใหม่, [เหตุผล])
    หลักคือ สระและวรรณยุกต์ไทยต้องเกาะพยัญชนะไทยเสมอ (อังกฤษ/ตัวเลขห้ามมีสระ วรรณยุกต์ไทย)
      0. เ สองตัว → แ (ทำก่อนทุกขั้นตอน)                        "เเดง" → "แดง"
      1. ลบอักขระมองไม่เห็น (zero-width space/joiner, BOM)     "ดิน<ZWSP>แดง" → "ดินแดง"
      2. ลบพินทุ ฺ                                                "พฤกษาวิลลฺ" → "พฤกษาวิลล"
         ลบสัญลักษณ์ไทยที่ไม่ใช้ในที่อยู่ ฿ ๏ ๚ ๛ (แทนด้วยช่องว่าง)
      3. สระอำพิมพ์แยก: ํ + า → ำ, วรรณยุกต์ที่พิมพ์หลัง ำ ย้ายไว้หน้า   "นํ้า", "นำ้" → "น้ำ"
      4. สระหน้า เ แ โ ใ ไ
         - ตัวเดียวกันซ้ำติดกัน เหลือตัวเดียว                        "แแดง" → "แดง"
         - คนละตัวซ้อนกัน ไม่ลบ เก็บไว้ตามเดิม                       "โแนด" → "โแนด"
         - ไม่มีพยัญชนะไทยตามทันที → ลบ                            "ดงเ" → "ดง", "เA" → "A"
      5. สระบน/ล่าง วรรณยุกต์ ์ ํ ๎
         - ไม่ได้เกาะพยัญชนะไทย (ต้นค่า หลังช่องว่าง หลังอังกฤษ/ตัวเลข
           หลังสระหน้า/สระหลัง) → ลบ                              "ิดิน" → "ดิน", "Aิ" → "A", "กา่" → "กา"
         - ตัวเดียวกันซ้ำบนพยัญชนะเดียว เก็บตัวแรก (คนละตัวไม่ลบ)     "ลััลนา" → "ลัลนา", "ล่่ลนา" → "ล่ลนา"
         - ไม่สลับลำดับรหัสสระ/วรรณยุกต์ คงลำดับที่พิมพ์มา (ตาเห็นเหมือนเดิม)
      6. สระหลัง ะ า ำ ๅ
         - ต้องตามหลังพยัญชนะ (มีวรรณยุกต์ได้ แต่ห้ามมีสระบน/ล่าง หรือ ์)
           → ไม่ใช่ ลบ                                             "าาการ" → "การ", "Aา" → "A", "กิา" → "กิ"
         - ซ้อนกันได้แค่ "าะ" (เกาะ) ตัวเดียวกันซ้ำเหลือ 1           "กาา" → "กา"
         - ็ อยู่หน้าสระหลัง → ลบ ็ เก็บสระหลัง                      "กร็าฟฟิค" → "กราฟฟิค"
      7. ฯ และ ๆ ต้องมีคำไทยนำหน้า (เว้นวรรคได้ "ลาดพร้าว ฯ", "ต่าง ๆ") และไม่ซ้ำติดกัน
                                                                "ฯกรุงเทพ" → "กรุงเทพ", "5ๆ" → "5"
    ช่องว่างที่เหลือจากการลบ ให้ขั้นตอนถัดไปตัดเอง"""
    reasons = []
    if not isinstance(v, str):
        return v, reasons

    def note(reason):
        if reason not in reasons:
            reasons.append(reason)

    def sub(pattern, repl, s, reason):
        new = pattern.sub(repl, s)
        if new != s:
            note(reason)
        return new

    v = sub(DOUBLE_E_RE, "\u0E41", v, "แก้ เเ เป็น แ")
    v = sub(ZERO_WIDTH_RE, "", v, "ลบอักขระมองไม่เห็น")
    v = sub(PHINTHU_RE, "", v, "ลบพินทุ ฺ")
    v = sub(TH_UNUSED_RE, " ", v, "ลบสัญลักษณ์ไทยที่ไม่ใช้")
    v = sub(SARA_AM_SPLIT_RE, "\\1\u0E33", v, "รวม ํ+า เป็น ำ")
    v = sub(SARA_AM_TONE_RE, "\\1\u0E33", v, "ย้ายวรรณยุกต์ไว้หน้า ำ")

    out = []
    # ชนิดของตัวที่เพิ่งเก็บ: start/space/other/lead/cons/cons_vowel/follow/follow_aa/pai/yamok
    prev = "start"
    last_word = "start"  # ชนิดของตัวล่าสุดที่ไม่ใช่ช่องว่าง (ใช้กับ ๆ)
    i, n = 0, len(v)
    while i < n:
        ch = v[i]
        if ch in TH_LEAD_VOWEL:
            j = i + 1
            while j < n and v[j] in TH_LEAD_VOWEL:
                j += 1
            run = v[i:j]
            # ตัวเดียวกันซ้ำติดกัน เหลือตัวเดียว ("แแดง" → "แดง") คนละตัวเก็บไว้ทั้งชุด ("โแนด")
            deduped = "".join(c for k, c in enumerate(run) if k == 0 or c != run[k - 1])
            if deduped != run:
                note("ลบสระหน้าซ้ำ (เหลือ 1)")
                run = deduped
            if j < n and v[j] in TH_CONSONANT:
                out.append(run)
                prev = last_word = "lead"
            else:
                note("ลบสระหน้าที่ไม่มีพยัญชนะตาม")
            i = j
            continue

        if ch in TH_CONSONANT:
            j = i + 1
            while j < n and v[j] in TH_COMBINING:
                j += 1
            marks = v[i + 1:j]
            # ตัวเดียวกันซ้ำ เก็บตัวแรก ("ลััลนา" → "ลัลนา") คนละตัวเก็บไว้ทั้งหมด
            kept = "".join(m for k, m in enumerate(marks) if m not in marks[:k])
            for m in set(marks):
                if marks.count(m) > 1:
                    note("ลบสระซ้ำ (เหลือ 1)" if m in TH_ABOVE_BELOW_VOWEL
                         else "ลบวรรณยุกต์ซ้ำ (เหลือ 1)" if m in TH_TONE
                         else "ลบ ์ ํ ๎ ซ้ำ (เหลือ 1)")
            # ไม่สลับลำดับรหัสสระ/วรรณยุกต์ (ตาเห็นเหมือนเดิม) คงลำดับที่พิมพ์มา
            fixed = kept
            out.append(ch + fixed)
            has_mark = any(m not in TH_TONE for m in fixed)
            prev = last_word = "cons_vowel" if has_mark else "cons"
            i = j
            continue

        if ch in TH_FOLLOW_VOWEL:
            if prev == "cons" or (ch == "\u0E30" and prev == "follow_aa"):
                out.append(ch)
                prev = last_word = "follow_aa" if ch == "\u0E32" else "follow"
            elif out and out[-1] and out[-1][-1] == ch:
                note("ลบสระหลังซ้ำ (เหลือ 1)")
            elif (prev == "cons_vowel" and "็" in out[-1]
                  and all(m in TH_TONE for m in out[-1][1:].replace("็", ""))):
                # ็ อยู่หน้าสระหลังไม่ได้ ลบ ็ เก็บสระหลัง ("กร็าฟฟิค" → "กราฟฟิค")
                out[-1] = out[-1].replace("็", "")
                out.append(ch)
                prev = last_word = "follow_aa" if ch == "า" else "follow"
                note("ลบ ็ ที่อยู่หน้าสระหลัง")
            else:
                note("ลบสระหลังที่ไม่ได้ตามพยัญชนะ")
        elif ch in TH_COMBINING:
            note("ลบสระ/วรรณยุกต์ลอย")
        elif ch == "\u0E2F":  # ฯ
            if last_word in TH_WORD_KINDS:
                out.append(ch)
                prev = last_word = "pai"
            else:
                note("ลบ ฯ ที่ไม่ได้ตามคำไทย")
        elif ch == "\u0E46":  # ๆ
            if last_word in TH_WORD_KINDS:
                out.append(ch)
                prev = last_word = "yamok"
            else:
                note("ลบ ๆ ที่ไม่ได้ตามคำไทย")
        else:
            out.append(ch)
            if ch.isspace():
                prev = "space"
            else:
                prev = last_word = "other"
        i += 1
    return "".join(out), reasons


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


# moo เป็นเลขล้วน ส่วน mooban มีตัวอักษร (อาจมีเลขด้วย)
MOO_NUMBER_RE = re.compile(rf"\d+(?![\d{THAI_RANGE}A-Za-z])")


def moo_or_mooban(body):
    """key ของช่วงหลัง "ม.": "ม.1" → moo, "ม.ชัยพฤกษ์ 2" / "ม.88เพลส" → mooban"""
    body = body.strip()
    return "moo" if not body or MOO_NUMBER_RE.match(body) else "mooban"


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
# เลขบ้าน: เลข นำด้วยอักษรไทย 1 ตัวได้ (ไม่ใช่ ม = หมู่) ต่อกันด้วย / - , (มีช่องว่างได้)
# "191/66", "ข 49/3", "ค.2-ค.3", "1/ 1", "139,141" → เอาทั้งก้อน
# "และ" คั่นเลขเหมือน "," ("440 และ 442" → เอาทั้งก้อน)
_NO_UNIT = r"(?:(?!ม)[ก-ฮ]\.?\s?)?\d+"
NO_VALUE_RE = re.compile(rf"{_NO_UNIT}(?:\s*(?:[/\-,]|และ)\s*{_NO_UNIT})*[/\-]?(?![^\s,])")
# เอกสารสิทธิ์ที่ดิน (โฉนด, น.ส.3, น.ส.3ก, น.ส.3ข, น.ส.4, ส.ป.ก. 4-01) + เลข → เก็บทั้งก้อนเป็น no
# ตั้งแต่ชื่อเอกสารจนถึงตัวเลขตัวสุดท้าย (รวม บ้านเลขที่/เลขที่ ที่ตามมาติดกัน)
# "โฉนดเลขที่ 123" → no = "โฉนดเลขที่ 123"
# "น.ส.3ก เลขที่ 123 บ้านเลขที่ 456" → no = "น.ส.3ก เลขที่ 123 บ้านเลขที่ 456"
# รูปแบบอื่น: "โฉนดที่ดินเลขที่ 2595, 2596", "น.ส.3ก. 1441", "น.ส.3 ก.เลขที่ 701",
# "(น.ส.3) ก. 1014", "น.ส.3ข.เลขที่ 164", "นส.4 ที่ 3856", "ส.ป.ก.4-01ข เลขที่ 6898,7132"
# นส/สปก ไม่มีจุดเลย ("นส3ก", "สปก") ต้องไม่ติดอักษรไทยข้างหน้า (กันไปจับกลางคำ)
_DEED_NUM = rf"{DIGIT_CLASS}+(?:\s*(?:[,/\-]|และ)\s*{DIGIT_CLASS}+)*"
_DEED_SUFFIX = r"(?:[กขคจ](?![ก-๎])\.?)?"  # น.ส.3ก, น.ส.3ข, ส.ป.ก.4-01ข
_NOT_IN_WORD = r"(?<![ก-๎])"
_DEED_HEAD = (
    rf"(?:โฉนด(?:ที่ดิน)?"
    rf"|\(?(?:น\.?\s*ส\.|{_NOT_IN_WORD}นส)\s*[34๓๔]\s*\)?\s*{_DEED_SUFFIX}"
    rf"|(?:ส\.?\s*ป\.?\s*ก\.|{_NOT_IN_WORD}สปก)\s*(?:[4๔]\s*-\s*{DIGIT_CLASS}{{2}}\s*{_DEED_SUFFIX})?)"
    rf"\s*(?:เลขที่|เล่มที่|เล่ม|เลข|ที่)?\s*[:\-]?\s*"
)
# ไม่มีเลขตาม ("โฉนด ตำบล...") → ตัดทิ้ง ไม่เป็นค่าของ key ไหน
DEED_RE = re.compile(rf"{_DEED_HEAD}(?:{_DEED_NUM})?")
# มีเลขตาม → เก็บเป็น no จนถึงตัวเลขตัวสุดท้าย รวม "บ้านเลขที่ 456" ที่ตามมา
_HOUSE_NUM = rf"{_NO_UNIT}(?:\s*[/\-,]\s*{_NO_UNIT})*"
DEED_NO_RE = re.compile(
    rf"{_DEED_HEAD}{_DEED_NUM}(?:\s*,?\s*(?:บ้าน\s*)?เลขที่\s*[:\-]?\s*{_HOUSE_NUM})*"
)


def split_deed(v):
    """แยกเลขโฉนด/น.ส.3/น.ส.3ก/น.ส.3ข/น.ส.4/ส.ป.ก. ออกจากข้อความ คืน (ข้อความที่เหลือ, [ค่า no])
    "น.ส.3ก เลขที่ 123 บ้านเลขที่ 456 ตำบล..." → (" ตำบล...", ["น.ส.3ก เลขที่ 123 บ้านเลขที่ 456"])"""
    if not isinstance(v, str):
        return v, []
    deeds = [re.sub(r"\s+", " ", m.group(0)).strip() for m in DEED_NO_RE.finditer(v)]
    return DEED_RE.sub(" ", DEED_NO_RE.sub(" ", v)), deeds
DB_POSTCODE_PATH = r"D:\Thai_Adress\qwen3_8b_lora\EDA\DataStandardList-Postcode_converted.csv"

# ไฟล์แก้ค่ามือ: แถวที่ addr_input ตรงกัน จะถูกแทน 11 key ด้วยค่าในไฟล์ (ช่องว่าง = ไม่มีค่า)
EDIT_DIR = OUTPUT_DIR / "edit"
EDIT_FILES = [EDIT_DIR / "edit_data.csv", EDIT_DIR / "bad_values.csv", EDIT_DIR / "edit_web.csv"]
# addr_input ที่กด Pass ใน review_web.py → ไม่ส่งออกในทุกไฟล์
PASS_CSV = GROUP_DIR / "leftover" / "pass.csv"
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

def prefix_any(words):
    pattern = re.compile("^(" + "|".join(re.escape(w) for w in words) + ")")
    return lambda v: bool(pattern.search(v.strip())) if isinstance(v, str) else False

def contains_any(words):
    pattern = re.compile("|".join(re.escape(w) for w in words))
    return lambda v: bool(pattern.search(v)) if isinstance(v, str) else False

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

BRACKET_PAIRS = {")": "(", "]": "[", "}": "{"}


def strip_unmatched_brackets(v):
    """ลบ () [] {} ที่ไม่ครบคู่ (แทนด้วยช่องว่าง) วงเล็บที่ครบคู่เก็บไว้ตามที่พิมพ์มา
    "(ซอย 5" → " ซอย 5", "5)" → "5 ", "(จีพีเฮ้าส์)" → "(จีพีเฮ้าส์)", "(A]" → " A " """
    stack, unmatched = [], set()
    for i, ch in enumerate(v):
        if ch in "([{":
            stack.append(i)
        elif ch in BRACKET_PAIRS:
            if stack and v[stack[-1]] == BRACKET_PAIRS[ch]:
                stack.pop()
            else:
                unmatched.add(i)
    unmatched.update(stack)
    if not unmatched:
        return v
    return "".join(" " if i in unmatched else ch for i, ch in enumerate(v))

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

# คำนำหน้าที่ลบออกจากต้นค่า (Step 6): key → (regex, คำที่ลบ)
PREFIX_RULES = {
    "no":       (r'^(\s*(บ้าน\s*เลขที่|เลขที่)\s*)', "บ้านเลขที่/เลขที่"),
    "moo":      (r'^(\s*(หมู่\s*|ม\.\s*))+', "หมู่/ม."),
    "mooban":   (r'^(\s*(หมู่บ้าน|ม\.)\s*)', "หมู่บ้าน/ม."),
    "building": (r'^(\s*(อาคาร|ตึก)\s*)+', "อาคาร/ตึก"),
    "floor":    (r'^(\s*ชั้น(ที่)?\s*)+', "ชั้น/ชั้นที่"),
    "room":     (rf'^(\s*(เลขที่\s*ห้อง|ห้อง\s*{_INFIX_WORD}เลขที่|ห้อง'
                 rf'|(?:{"|".join(ROOM_UNIT_WORDS)})\s*เลขที่|{"|".join(ROOM_UNIT_ALONE_WORDS)})\s*)',
                 "เลขที่ห้อง/ห้องเลขที่/ห้อง บูธเลขที่/ห้อง/บูธเลขที่/บูธ/ยูนิต"),
    "soi":      (r'^(\s*(ซอย|ซ\.)\s*)', "ซอย/ซ."),
    "road":     (r'^(\s*(ถนน|ถ\.)\s*)', "ถนน/ถ."),
    "tambol":   (r'^(\s*(ตำบล|ต\.)\s*)', "ตำบล/ต."),
    "amphoe":   (r'^(\s*(อำเภอ|อ\.)\s*)', "อำเภอ/อ."),
    "province": (r'^(\s*(จังหวัด|จ\.)\s*)', "จังหวัด/จ."),
}
# ค่าที่แปลว่า "ไม่มี" → เคลียร์เป็นค่าว่าง (no: มีคำเหล่านี้อยู่ในค่า, key อื่น: ค่าตรงทั้งค่า)
CLEAR_VALUES = ["-", "ไม่มี-", "ไม่ระบุ", "-ไม่มี-", "-ไม่มี"]
NO_CLEAR_WORDS = ["ไม่มีเลขที่", "ไม่ระบุ", "ไม่มี-", "-ไม่มี"]
# ขั้นตอนคลีนใน Step 6 ตามลำดับ (ชื่อ, คำอธิบายที่พิมพ์ใน terminal)
# วงเล็บที่ครบคู่เก็บตามที่พิมพ์มา ลบเฉพาะที่ไม่ครบคู่ (strip_unmatched_brackets)
CLEAN_STEPS = [
    ("clear", "ค่าว่าง/ไม่ระบุ → ลบทิ้ง"),
    ("prefix", "ลบคำนำหน้า"),
    ("disallowed", "ลบ emoji/สัญลักษณ์แปลก"),
    ("bracket", "ลบ () [] {} ที่ไม่ครบคู่ ('(ซอย 5' → 'ซอย 5', '(จีพีเฮ้าส์)' เก็บไว้)"),
    ("space", "ตัดช่องว่างเกิน"),
    ("dedupe", "ตัดคำซ้ำ ('ซ. 1 ซอย 1' → '1')"),
]

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

    # ==========================================
    # Step 4.1: Remove addr_input ที่อยู่ใน pass.csv (ห้ามส่งออก)
    # ==========================================
    pass_inputs = set()
    n_pass_removed = 0
    if PASS_CSV.exists():
        df_pass = pd.read_csv(PASS_CSV, dtype=str, encoding="utf-8-sig").dropna(subset=["addr_input"])
        pass_inputs = {v.strip() for v in df_pass["addr_input"]}
        pass_mask = df['addr_input'].apply(lambda v: isinstance(v, str) and v.strip() in pass_inputs)
        n_pass_removed = int(pass_mask.sum())
        df = df[~pass_mask].reset_index(drop=True)
        print(f"4.1 [ใหม่] ลบแถวที่ addr_input อยู่ใน {PASS_CSV.name} เสร็จแล้ว: "
              f"{int(pass_mask.sum()):,} rows removed ({len(pass_inputs):,} ค่าใน pass), {len(df):,} remaining")
    else:
        print(f"4.1 Pass file not found, skipping: {PASS_CSV}")

    # โหลดค่าแก้มือ {addr_input: {key: ค่า}} — ใช้ใน Step 5, 6 และ 7.6
    manual_edits = {}
    edit_source = {}  # addr_input → ชื่อไฟล์แก้มือ
    for path in EDIT_FILES:
        if not path.exists():
            print(f"   Edit file not found, skipping: {path}")
            continue
        df_edit = pd.read_csv(path, dtype=str, encoding="utf-8-sig").dropna(subset=["addr_input"])
        # addr_input ที่อยู่ใน pass.csv ไม่ใช้ค่าแก้มือ (กันเพิ่มกลับเป็นแถวใหม่ใน Step 7.6)
        n_pass_skip = int(df_edit["addr_input"].str.strip().isin(pass_inputs).sum())
        df_edit = df_edit[~df_edit["addr_input"].str.strip().isin(pass_inputs)]
        for _, r in df_edit.iterrows():
            manual_edits[r["addr_input"].strip()] = {
                c: (None if is_blank(r.get(c)) else str(r[c]).strip()) for c in ADDRESS_FIELDS
            }
            edit_source[r["addr_input"].strip()] = path.name
        print(f"   Loaded {len(df_edit):,} manual edits from {path.name} เสร็จแล้ว"
              + (f" (ข้าม {n_pass_skip:,} แถวที่อยู่ใน {PASS_CSV.name})" if n_pass_skip else ""))

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

    # ค่า 11 key รวมกันมีอักษรมากกว่า addr_input (นับไม่รวมช่องว่างและ ",")
    def n_chars(v):
        return len(re.sub(r"[\s,]", "", as_text(v)))

    key_cols = [c for c in ADDRESS_FIELDS if c in df.columns]
    value_lens = [sum(n_chars(v) for v in vals) for vals in zip(*(df[c] for c in key_cols))]
    input_lens = [n_chars(v) for v in df['addr_input']]
    over_mask = pd.Series([a > b for a, b in zip(value_lens, input_lens)], index=df.index)
    mark_drop(over_mask, "ค่า 11 key รวมกันมีอักษรเกิน addr_input")
    print(f"5.2.1 [ใหม่] ค่า 11 key รวมกันมีอักษรเกิน addr_input: {int((over_mask & ~edit_mask).sum()):,} rows")

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
    # แยกประเภทการเปลี่ยนแปลง ใช้แบ่งไฟล์ใน group/
    edit_indices = set()   # แก้: ตัด/ย้าย/เขียนทับ/ลบค่าผิด key/แก้มือ
    fill_indices = set()   # เติม: key ว่างเติมจาก addr_input
    clean_indices = set()  # คลีน: ทำความสะอาดตัวอักษร (Step 7)
    thai_indices = set()   # แก้อักษรไทย: สระ/วรรณยุกต์ผิดตำแหน่ง (Step 6.0)
    modified_row_reasons = defaultdict(list)
    mod_logs_total = 0

    # 6.0: แก้อักษรไทยผิดตำแหน่งครั้งเดียว (ดู fix_thai_chars) ทุก Step หลังจากนี้ได้ค่าที่สะอาดแล้ว
    #      - 11 key: เขียนทับค่า
    #      - addr_input: ไม่เขียนทับ (เป็น input ของโมเดล ต้องคงความสกปรกแบบของจริง
    #        และใช้จับคู่ไฟล์แก้มือ) เก็บรูปที่แก้แล้วใน addr_input_clean ให้ Step 7.5, 7.4, 7.58 ใช้
    #      ทำหลัง reset index ของ 6.2 เพื่อให้ index ใน log ตรงกับแถวสุดท้าย
    n_thai_fix = defaultdict(int)   # {key: จำนวน row ที่แก้}
    n_thai_rule = defaultdict(int)  # {กฎใน fix_thai_chars: จำนวนค่า}
    thai_rule_example = {}          # {กฎ: (key, ก่อน, หลัง)} ตัวอย่างแรกที่เจอ
    thai_cols = [c for c in ADDRESS_FIELDS if c in df.columns]
    for col in thai_cols:
        new_col_data = []
        for row_idx, val in zip(df.index, df[col]):
            new_val, rules = fix_thai_chars(val)
            if isinstance(val, str) and new_val != val:
                new_val = re.sub(r"\s+", " ", new_val).strip() or None
                n_thai_fix[col] += 1
                for rule in rules:
                    n_thai_rule[rule] += 1
                    thai_rule_example.setdefault(rule, (col, val, as_text(new_val)))
                modified_indices.add(row_idx)
                thai_indices.add(row_idx)
                modified_row_reasons[row_idx].append(
                    f"{col}: '{val}' → '{as_text(new_val)}' "
                    f"[{describe_char_changes(val, new_val)}] ({', '.join(rules)})"
                )
                mod_logs_total += 1
            new_col_data.append(new_val)
        df[col] = new_col_data
    n_input_rule = defaultdict(int)  # {กฎ: จำนวนแถว addr_input}
    input_rule_example = {}          # {กฎ: (ก่อน, หลัง)}
    input_clean_vals = []
    for val in df['addr_input']:
        new_val, rules = fix_thai_chars(val)
        if isinstance(val, str) and new_val != val:
            for rule in rules:
                n_input_rule[rule] += 1
                input_rule_example.setdefault(rule, (val, new_val))
        input_clean_vals.append(new_val)
    addr_input_clean = pd.Series(input_clean_vals, index=df.index)
    n_input_fixed = sum(isinstance(a, str) and a != b for a, b in zip(df['addr_input'], addr_input_clean))
    print(f"5.2.2 [ใหม่] แก้อักษรไทยผิดตำแหน่ง (เเ→แ ก่อน, สระ/วรรณยุกต์ซ้ำเหลือ 1, "
          f"สระหน้าคนละตัวซ้อนไม่ลบ, สระ/วรรณยุกต์ลอย, สระหน้า/หลังไม่มีพยัญชนะ) เสร็จแล้ว: "
          f"จำนวน row ที่แก้ {fmt_counts(n_thai_fix)}")
    for rule, n in sorted(n_thai_rule.items(), key=lambda x: -x[1]):
        col, before, after = thai_rule_example[rule]
        print(f"      {rule}: {n:,} ค่า  เช่น {col}: '{before}' → '{after}'")
    print(f"5.2.3 [addr_input: ไม่เขียนทับ] อักษรไทยผิด {n_input_fixed:,} แถว — ไฟล์ส่งออกคงค่าเดิม "
          f"(ให้โมเดลเห็นคำผิดจริง, ใช้จับคู่ไฟล์แก้มือ)")
    print(f"      แก้ในสำเนา addr_input_clean แทน ใช้ใน Step 7-7.58: ค่า 11 key ที่เติมจาก addr_input จึงสะอาดด้วย "
          f"และเทียบกับ 11 key ได้ตรง (ถ้าไม่แก้ เช่น 'เเขวง' จะไม่ตรงกับ 'แขวง' ทำให้ตัดคำนำหน้า/เทียบค่าไม่ติด)")
    for rule, n in sorted(n_input_rule.items(), key=lambda x: -x[1]):
        before, after = input_rule_example[rule]
        print(f"      {rule}: {n:,} แถว  เช่น '{before}' → '{after}'")

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
            if kw == "ม.":
                field = moo_or_mooban(body)
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
            edit_indices.add(row_idx)
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

    n_clean = defaultdict(lambda: defaultdict(int))  # {ขั้นตอนใน CLEAN_STEPS: {key: จำนวน row ที่แก้}}

    def step(name, label, old, new):
        """ค่าเปลี่ยน → เก็บเหตุผลของค่านี้ และนับจำนวนตามขั้นตอน/key"""
        if new != old:
            reasons_list.append(label)
            n_clean[name][col] += 1
        return new

    cols_to_clean = [col for col in df.columns if df[col].dtype == 'string' or df[col].dtype == 'object']

    for col in cols_to_clean:
        if col in ['addr', 'addr_input', 'POSTCODE', 'postcode']:
            continue

        new_col_data = []

        for row_idx, val in zip(df.index, df[col]):
            current_val = val
            reasons_list = []

            if isinstance(current_val, str):
                # ไม่ลบอักขระนำหน้า: เก็บตามที่พิมพ์มา (PROTECTED_SPECIAL_CHARS)
                # วงเล็บลบเฉพาะที่ไม่ครบคู่ (ขั้น "bracket")
                # no: เลขโฉนด/น.ส.3/น.ส.3ก/น.ส.3ข/น.ส.4/ส.ป.ก. เก็บไว้ตามที่พิมพ์มา (ไม่ลบ)
                if col == 'no':
                    cleared = any(w in current_val for w in NO_CLEAR_WORDS)
                else:
                    cleared = current_val.strip() in CLEAR_VALUES
                if cleared:
                    current_val = step("clear", "เคลียร์ค่าที่แปลว่าไม่มี", current_val, "")
                elif col in PREFIX_RULES:
                    pattern, words = PREFIX_RULES[col]
                    current_val = step("prefix", f"ลบคำนำหน้า {words}", current_val,
                                       re.sub(pattern, '', current_val).strip())

                current_val = step("disallowed", "ลบอักขระพิเศษ", current_val,
                                   DISALLOWED_CHAR_RE.sub(' ', current_val))
                current_val = step("bracket", "ลบวงเล็บที่ไม่ครบคู่", current_val,
                                   strip_unmatched_brackets(current_val))
                current_val = step("space", "ตัดช่องว่างเกิน", current_val,
                                   re.sub(r'\s+', ' ', current_val).strip())
                if col in ADDRESS_FIELDS:
                    current_val = step("dedupe", "ตัดคำซ้ำ", current_val, dedupe_prefixed(current_val, col))

            # Check if value changed
            if is_changed(val, current_val):
                modified_indices.add(row_idx)
                clean_indices.add(row_idx)
                char_diff = describe_char_changes(val, current_val)
                modified_row_reasons[row_idx].append(
                    f"{col}: '{as_text(val)}' → '{as_text(current_val)}' "
                    f"[{char_diff}] ({', '.join(reasons_list)})"
                )
                mod_logs_total += 1

            new_col_data.append(current_val if not is_blank(current_val) else None)

        df[col] = new_col_data

    print(f"6. คลีนค่า: แก้ {len(modified_indices):,} row ({mod_logs_total:,} ค่า)")
    print(f"   เก็บอักขระพิเศษไว้: {' '.join(KEEP_SPECIAL_CHARS)}")
    print(f"   [ใหม่] () [] {{}} เก็บเฉพาะที่ครบคู่ ที่ไม่ครบคู่ลบทิ้ง (ใช้กับค่าที่เติมจาก addr_input ด้วย)")
    # พิมพ์เฉพาะขั้นตอนที่มี row ถูกแก้จริง
    for i, (name, desc) in enumerate(CLEAN_STEPS, 1):
        if not n_clean[name]:
            continue
        print(f"6.{i} {desc}: {fmt_counts(n_clean[name])}")
        if name == "prefix":
            for c, (_, words) in PREFIX_RULES.items():
                print(f"      {c}: {words}")

    # ==========================================
    # Step 7.5: Fill/overwrite keys from addr_input (rule base: คำนำหน้าของ 11 key)
    #           addr_input เป็นตัวตัดสิน: เจอค่า → เขียนทับ, เจอหลายค่า → รวมคั่นด้วย ", "
    # ==========================================
    try:
        df_names = pd.read_csv(DB_POSTCODE_PATH, dtype=str)
        known_names = {c: set(df_names[c].dropna().str.strip()) for c in DB_NAME_FIELDS}
    except FileNotFoundError:
        known_names = {c: set() for c in DB_NAME_FIELDS}

    def clean_filled(v, col):
        # ไม่ลบอักขระพิเศษใน PROTECTED_SPECIAL_CHARS แต่ลบวงเล็บที่ไม่ครบคู่ (เหมือน Step 6)
        v = strip_unmatched_brackets(DISALLOWED_CHAR_RE.sub(' ', v))
        v = re.sub(r'\s+', ' ', v).strip()
        leading_re = LEADING_NON_WORD_KEEP_AT_RE if col in KEEP_AT_COLUMNS else LEADING_NON_WORD_RE
        return leading_re.sub('', v).strip()

    def join_parts(parts, col):
        """รวมช่วงที่ติดกันของ key เดียวกันเป็นข้อความตามที่พิมพ์มา ตัดช่วงที่ซ้ำ
        คำระหว่างคำนำหน้าตัวแรกจนถึงคำนำหน้าของ key อื่น คือค่าของ key แรก
        (คำเชื่อม "และ" "," และคำนำหน้าซ้ำที่อยู่ระหว่างทาง เก็บไว้ตามที่พิมพ์มา)
        [("อาคาร","X "),("อาคาร","X ")] → "X"
        [("ชั้น","10และ"),("ชั้น"," 11")] → "10และชั้น 11"
        [("ห้อง"," ("),("ยูนิต"," 3-4)")] → "(ยูนิต 3-4)"
        [("มหาวิทยาลัย","จุฬา และ "),("อาคาร"," เอ ")] → "มหาวิทยาลัยจุฬา และ อาคาร เอ" """
        text, seen = "", set()
        for kw, body in parts:
            # KEEP_KEYWORDS เป็นส่วนของชื่อ ("ทรู ดิจิทัล พาร์ค" + อาคาร... body ว่างก็ยังเป็นค่า)
            keep = canon_keyword(kw) in KEEP_KEYWORDS
            key = clean_filled((kw + body if keep else body).rstrip(" ,"), col).replace(" ", "")
            if key and key in seen:
                text += " "
                continue
            if key:
                seen.add(key)
            # ยังไม่มีค่า (มีแต่ช่องว่าง/","): คำนำหน้านี้เป็นตัวแรก ตัดทิ้ง (ยกเว้น KEEP_KEYWORDS)
            # ตัวถัดไปเก็บคำนำหน้าตามที่พิมพ์มา
            if not WORD_CHAR_RE.search(text) and not BRACKET_RE.search(text):
                text = kw + body if keep else body
            else:
                text += kw + body
        return clean_filled(text, col).rstrip(" ,")

    def pick_known_name(col, v):
        """เอาคำยาวที่สุดที่ตรงกับชื่อใน DB ('เมือง ชลบุรี' → ไม่ตรง → 'เมือง')"""
        words = v.split(" ")
        for k in range(len(words), 1, -1):
            cand = " ".join(words[:k])
            if cand in known_names[col] or cand.replace(" ", "") in known_names[col]:
                return cand
        return words[0]

    def split_by_keywords(addr_input):
        """แยก addr_input เป็น {key: [ค่า]} ตามช่วงหลังคำนำหน้า จนถึงคำนำหน้าของ key อื่น
        ค่าของแต่ละ key:
          mooban/building/floor/room/soi/road → ทั้งช่วง ("ห้อง 4036,4172 ถนน..." → 4036,4172)
                     คำนำหน้าซ้ำของ key เดียวกันอยู่ในช่วงด้วย เก็บตามที่พิมพ์มา
                     ("ชั้น 10และชั้น 11 ห้อง..." → floor = 10และชั้น 11)
          no       → ก้อนเลขบ้านตาม NO_VALUE_RE ("เลขที่ 1,3 หมู่..." → 1,3, "440 และ 442 ซอย..." → 440 และ 442)
                     เลขโฉนด/น.ส.3/น.ส.3ก/น.ส.3ข/น.ส.4/ส.ป.ก. เก็บทั้งก้อนจนถึงตัวเลข (split_deed)
                     ("น.ส.3ก เลขที่ 123 บ้านเลขที่ 456" → no = ทั้งก้อน)
          moo      → เลขต้นช่วง ("หมู่ 3 บ้าน..." → 3)
          tambol/amphoe/province → ชื่อยาวสุดที่ตรงกับ DB_POSTCODE ไม่ตรงเอาคำแรก
        ทุก key: ช่วงท้ายสุดไม่มีคำนำหน้าปิด → จบที่ "," """
        found = defaultdict(list)
        if not isinstance(addr_input, str):
            return found
        # เลขโฉนด/น.ส.3/น.ส.3ก/น.ส.3ข/น.ส.4/ส.ป.ก. แยกออกก่อน เก็บทั้งก้อนเป็น no
        # ("โฉนดเลขที่ 123" → no = "โฉนดเลขที่ 123") ข้อความที่เหลือแยกตามคำนำหน้าตามปกติ
        text, deeds = split_deed(addr_input)
        deeds = [d for d in (clean_filled(d, "no") for d in deeds) if d]
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
            if kw == "ม.":
                col = moo_or_mooban(body)
            segs.append([col, raw, body])

        # 2) ช่วงที่ติดกันและเป็น key เดียวกันรวมเป็นค่าเดียว ("ชั้นที่ 7 ชั้น 7" → 7)
        groups = []
        for col, kw, body in segs:
            if groups and groups[-1][0] == col:
                groups[-1][1].append((kw, body))
            else:
                groups.append([col, [(kw, body)]])

        for gi, (col, parts) in enumerate(groups):
            if col == "_stop":
                continue
            # ข้อความระหว่างคำนำหน้าเป็นค่าทั้งช่วง ("ห้อง 4036,4172 ถนน..." → 4036,4172)
            # ช่วงท้ายสุดไม่มีคำนำหน้าปิด → จบที่ "," เหมือนเดิม ("จ.ขอนแก่น, 40000" → ขอนแก่น)
            # "," ระหว่างคำนำหน้าของ key เดียวกันเก็บไว้ ("ชั้น 10, ชั้น 11" → "10, ชั้น 11")
            cut = []
            for j, (kw, body) in enumerate(parts):
                if gi == len(groups) - 1 and j == len(parts) - 1:
                    body = body.split(",")[0]
                cut.append((kw, body))
            # วงเล็บครบคู่เก็บไว้ ไม่ครบคู่ลบใน clean_filled เหมือน Step 6
            # ("(จีพีเฮ้าส์) ถนน..." → (จีพีเฮ้าส์), "(ซอย 5) ..." → soi = 5)
            v = join_parts(cut, col)
            if not v:
                continue
            # "เลขที่ 1/ 1" → 1/ 1 (ไม่ตัดที่ช่องว่างหลัง /)
            no_value = NO_VALUE_RE.match(v) if col == "no" else None
            if no_value:
                v = no_value.group(0)
            elif col in SINGLE_TOKEN_FIELDS:
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
        #    อักษรไทย 1 ตัว (ไม่ใช่ ม) + เลข เก็บตามที่พิมพ์: "ข 49/3 ถ.วิไลเสนา" → no = "ข 49/3",
        #    "ค.2-ค.3 ถ.บางเจียน" → no = "ค.2-ค.3"
        if not found["no"]:
            head = text[:matches[0].start()] if matches else text
            leading_no = NO_VALUE_RE.match(head.lstrip())
            if leading_no:
                found["no"].append(leading_no.group(0))
                found["_leading_no"] = True
        # เลขโฉนดไว้หน้าค่า no อื่น (เลขบ้านต้น addr_input ยังเติมได้ ไม่นับเลขโฉนด)
        if deeds:
            found["no"] = deeds + [x for x in found["no"] if x not in deeds]
            found["_leading_no"] = False
        return found

    fill_cols = [c for c in OWN_KEYWORDS if c in df.columns]
    n_filled = defaultdict(int)
    n_overwritten = defaultdict(int)
    n_leading_no = 0
    new_data = {c: list(df[c]) for c in fill_cols}
    # ใช้ addr_input ที่แก้อักษรไทยแล้ว (Step 6.0) ค่าที่เติมจะสะอาดแบบเดียวกับ 11 key
    for pos, (row_idx, addr_input) in enumerate(zip(df.index, addr_input_clean)):
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
                fill_indices.add(row_idx)
            else:
                # addr_input เป็นตัวตัดสิน: เขียนทับค่าเดิม (แม้สะกดต่างจากเดิม เช่น เมือง → เมอง)
                reason = "เขียนทับตาม addr_input"
                n_overwritten[col] += 1
                edit_indices.add(row_idx)
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
    print(f"   [ใหม่] เลขโฉนด/น.ส.3/น.ส.3ก/น.ส.3ข/น.ส.4/ส.ป.ก. 4-01 เก็บทั้งก้อนจนถึงตัวเลขเป็น no "
          f"('โฉนดเลขที่ 123' → no = 'โฉนดเลขที่ 123', "
          f"'น.ส.3ก เลขที่ 123 บ้านเลขที่ 456' → no = 'น.ส.3ก เลขที่ 123 บ้านเลขที่ 456')")
    print(f"   [ใหม่] ทุก key: ข้อความตั้งแต่หลังคำนำหน้าตัวแรก จนถึงคำนำหน้าของ key อื่น คือค่าของ key แรก "
          f"เก็บตามที่พิมพ์มา รวม และ/, และคำนำหน้าซ้ำระหว่างทาง "
          f"('ชั้น 10และชั้น 11 ห้อง...' → floor = '10และชั้น 11', 'ห้อง (ยูนิต 3-4) ถนน...' → room = '(ยูนิต 3-4)')")
    print(f"   [ใหม่] no: 'และ' คั่นเลขบ้านเหมือน ',' ('440 และ 442 ซอย...' → no = '440 และ 442')")
    print(f"   [ใหม่] เลขที่ หลังคำว่า บ้าน → no, หลังคำว่า ห้อง → room (มีช่องว่างคั่นได้) "
          f"('บ้าน เลขที่ 12' → no = 12, 'ห้อง เลขที่ 5' → room = 5)")
    print(f"   [ใหม่] ห้อง + คำไทย 1 คำ + เลขที่ → room (บูธ/พื้นที่/ยูนิต/แผงค้า...) "
          f"('ห้อง บูธเลขที่ JB303' → room = JB303, 'ห้อง พื้นที่เลขที่ ทีเอ็นซี.เอฟ01.ซ' → room = ทีเอ็นซี.เอฟ01.ซ)")
    print(f"   [ใหม่] {'/'.join(ROOM_UNIT_WORDS)} + เลขที่ → room ไม่ใช่ no ({'/'.join(ROOM_UNIT_ALONE_WORDS)} ไม่มีเลขที่ก็เป็น room) "
          f"no ใช้เลขบ้านต้น addr_input แทน ('919/1 บูธเลขที่ บีบี017 อาคาร...' → no = 919/1, room = บีบี017)")
    print(f"   [ใหม่] วัด/ศาล ไม่ใช่คำนำหน้า building แล้ว ('ซอยวัดใหม่' → soi = วัดใหม่, "
          f"'วัดสระเกศ ถนน...', 'ศาลแขวงพระนครเหนือ' → ไม่เติม building)")
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
                if first_prefix_owns(col, word, addr_input_clean[row_idx]):
                    continue
                old_val = df.at[row_idx, col]
                df.at[row_idx, col] = None
                modified_indices.add(row_idx)
                edit_indices.add(row_idx)
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
        """ทำให้อยู่ในรูปเดียวกับค่าที่ผ่าน Step 7 แล้ว (แปลงเลขไทยเฉพาะตอนเทียบ ไม่เขียนกลับ)"""
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
    for row_idx, addr_input in zip(df.index, addr_input_clean):
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
            edit_indices.add(row_idx)
            modified_row_reasons[row_idx].append(
                f"{col}: '{old_val}' → '{as_text(new_val)}' "
                f"[{describe_char_changes(old_val, new_val)}] (ไม่ปรากฏใน addr_input)"
            )
            mod_logs_total += 1
            n_not_in_input[col] += 1
    print(f"7.5 [ใหม่] ลบค่าที่ไม่ปรากฏใน addr_input เสร็จแล้ว: {fmt_counts(n_not_in_input)}")
    print(f"      [ใหม่] no: เลขโฉนด/น.ส.3/น.ส.3ก/น.ส.3ข/น.ส.4/ส.ป.ก. ไม่ลบแล้ว เทียบกับ addr_input ทั้งข้อความ "
          f"('โฉนดเลขที่ 123' เก็บไว้)")

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
                edit_indices.add(row_idx)
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
                edit_indices.add(row_idx)
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

    # 4 ไฟล์: แก้ / เติม / คลีน / แก้อักษรไทย (แถวที่มีหลายประเภทอยู่ได้หลายไฟล์)
    for name, indices in [("edited", edit_indices), ("filled", fill_indices), ("cleaned", clean_indices),
                          ("thai_fixed", thai_indices)]:
        group_df = df[df.index.isin(indices)]
        group_df.to_parquet(GROUP_DIR / f"{name}.parquet", index=False)
        print(f"12. {name}.parquet: {len(group_df):,} rows")

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
    print(f"Removed by {PASS_CSV.name}: {n_pass_removed:,}")
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
