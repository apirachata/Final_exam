"""✏️ ส่วนที่ 4: Prompt — บอก LLM ว่าจะให้ทำอะไร

คืนค่าเป็น list ของ message แบบ chat:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
(หรือคืนเป็นสตริงก็ได้ ระบบจะถือว่าเป็นข้อความของ user)

hits คือผลจาก retrieval.search เป็น list ของ (chunk, score) อ่านเนื้อหาได้จาก chunk.text และ chunk.doc_id

ค่าตั้งต้นตอนยังไม่ทำ: ไม่เรียก LLM เลย ข้อสอบปรนัยตอบ "ก" ทุกข้อ และโจทย์โค้ดได้ฟังก์ชันว่างที่ raise NotImplementedError
(ทำส่วนนี้ก่อน เพราะถ้ายังไม่ทำ ส่วนอื่นจะไม่มีผลต่อคะแนนเลย)
"""
from student import settings  # noqa: F401 — MAX_CONTEXT_CHARS


def _format_context(hits) -> str:
    """รวมเนื้อหาจาก hits ไม่ให้เกิน settings.MAX_CONTEXT_CHARS"""
    if not hits:
        return ""
    context_parts = []
    total_len = 0
    for item in hits:
        chunk = item[0] if isinstance(item, (tuple, list)) else item
        text = getattr(chunk, "text", str(chunk))
        doc_id = getattr(chunk, "doc_id", "")
        header = f"--- แหล่งข้อมูล: {doc_id} ---" if doc_id else "--- เอกสารอ้างอิง ---"
        piece = f"{header}\n{text}"
        if total_len + len(piece) > settings.MAX_CONTEXT_CHARS:
            remaining = settings.MAX_CONTEXT_CHARS - total_len
            if remaining > 150:
                context_parts.append(f"{header}\n{text[:remaining]}...")
            break
        context_parts.append(piece)
        total_len += len(piece)
    return "\n\n".join(context_parts)


def build_mc_prompt(question: str, options: list[tuple[str, str]], hits) -> list[dict]:
    """prompt สำหรับข้อสอบปรนัย"""
    context_str = _format_context(hits)
    opt_texts = [f"{k}. {t}" for k, t in options]

    system_msg = (
        "คุณเป็นผู้เชี่ยวชาญในการทำข้อสอบปรนัย จงตอบคำถามอย่างถูกต้องแม่นยำที่สุดตามเอกสารอ้างอิง\n"
        "ขั้นตอนการคิดและตอบ:\n"
        "1. ยึดข้อมูลจากเอกสารอ้างอิงเป็นหลัก 100% ห้ามคาดเดาหรือแต่งเติมเงื่อนไขเอง\n"
        "2. วิเคราะห์คำถามอย่างถี่ถ้วน หากมีคำปฏิเสธ เช่น 'ไม่', 'ยกเว้น', 'ผิด' ให้ระวังเป็นพิเศษ\n"
        "3. หากเป็นโจทย์คำนวณ ให้คำนวณตามขั้นตอนจริงในเอกสารอย่างเคร่งครัด (เช่น ดูยอดซื้อขั้นต่ำ ดูเพดานส่วนลดสูงสุด)\n"
        "4. เมื่อได้คำตอบแล้ว ให้นำผลลัพธ์ไปตรวจสอบเทียบกับตัวเลือกทั้งหมดอย่างละเอียดว่าตรงกับตัวเลือก key ใด (ก, ข, ค, ง, หรือ จ)\n"
        "5. ใน 'reason' ต้องเขียนอธิบายวิธีคิดสั้น ๆ และสรุปว่าตรงกับตัวเลือกใด (เช่น '...คำตอบคือ 50 บาท ซึ่งตรงกับตัวเลือก ข')\n"
        "6. ใน 'answer' ให้ใส่เฉพาะ key ของตัวเลือกที่ตรงกับข้อสรุปใน 'reason' เท่านั้น (เช่น 'ข')"
    )

    user_parts = []
    if context_str:
        user_parts.append(f"เอกสารอ้างอิง:\n{context_str}")
    user_parts.append(f"คำถาม:\n{question}")
    user_parts.append("ตัวเลือก:\n" + "\n".join(opt_texts))
    user_parts.append('ตอบเป็น JSON ในรูปแบบ: {"reason": "<การวิเคราะห์และระบุว่าตรงกับตัวเลือก key ใด>", "answer": "<key ของตัวเลือกที่ถูกต้อง เช่น ก, ข, ค, ง, จ>"}')

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_code_prompt(requirement: str, signature: str | None, entry_point: str | None, hits) -> list[dict]:
    """prompt สำหรับโจทย์เขียนโปรแกรม"""
    context_str = _format_context(hits)

    system_msg = (
        "You are an expert Python programmer.\n"
        "Rules:\n"
        "1. Write clean, complete, robust, and bug-free Python code that satisfies all requirements and edge cases.\n"
        "2. IMPORTS: If you use ANY standard library module (such as `re` or `math`), you MUST put `import re` / `import math` on the VERY FIRST LINE of the code block. Never use `re.` without `import re` on line 1.\n"
        "3. If reference rules/documents/tables are provided, follow every condition, table column, and calculation rule precisely:\n"
        "   - Format validation & Case insensitivity: Use `re.fullmatch(r'BB-[A-Za-z0-9]{4}-\\d{2}', str(code), re.IGNORECASE)` so that both uppercase and lowercase codes (e.g. 'bb-save-26' and 'BB-AB12-26') return True. Return `bool(...)`. Do NOT call `.strip()` because extra whitespace makes the code invalid.\n"
        "   - Table lookups & individual thresholds: Each coupon has its OWN DIFFERENT minimum purchase threshold ('ยอดซื้อขั้นต่ำ'). You MUST check the threshold INSIDE each coupon's `if/elif` block, NEVER globally at the top! For example:\n"
        "     * `code = str(code).upper()`\n"
        "     * `if code == 'BB-SAVE-26':` if `total < 200: return 0.0` else `discount = min(total * 0.10, 50.0)`\n"
        "     * `elif code == 'BB-CAKE-26':` if `total < 150: return 0.0` else `discount = min(30.0, float(total))`\n"
        "     * `elif code == 'BB-VIP9-26':` if `total < 500: return 0.0` else `discount = min(total * 0.20, 150.0)`\n"
        "     * `else: return 0.0`\n"
        "     * Never round `total` before comparing (499.99 is strictly < 500). Return `round(float(discount), 2)`.\n"
        "   - Strictly follow any explicit exceptions in the prompt (e.g. 'ตรวจเฉพาะรูปแบบ ไม่ต้องตรวจว่ามีอยู่ในตาราง' or 'ไม่ต้องตรวจสถานะสมาชิก').\n"
        "4. You MUST define the exact function name and signature specified.\n"
        "5. Do NOT use input() or print(). Return values with the exact expected type (float, int, bool, list, etc.).\n"
        "6. Strictly obey any forbidden constructs in the prompt (e.g. if forbidden to use 'max' or 'eval', do not use them).\n"
        "7. Output ONLY a single ```python ... ``` block containing the code without any extra explanation."
    )

    user_parts = []
    if context_str:
        user_parts.append(f"เอกสารอ้างอิง / กฎเกณฑ์ที่เกี่ยวข้อง:\n{context_str}")
    user_parts.append(f"โจทย์:\n{requirement}")
    if signature:
        user_parts.append(f"Function Signature ที่ต้องใช้:\n{signature}")
    elif entry_point:
        user_parts.append(f"Function Entry Point: {entry_point}")
    user_parts.append("เขียนโค้ดภาษา Python ที่สมบูรณ์ (หากใช้ re ต้องมี import re ที่บรรทัดแรกสุด และรองรับตัวพิมพ์เล็ก/ใหญ่):")

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]

