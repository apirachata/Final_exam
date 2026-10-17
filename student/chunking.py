"""✏️ ส่วนที่ 1: Chunking — แบ่งเอกสารเป็นชิ้น (chunk)

เอกสารที่ได้รับเป็น markdown เช่น lecture note หรือ case study ที่มีหัวข้อ (#, ##), ตาราง และ code block
ชิ้นที่แบ่งแล้วจะถูกนำไปทำ embedding และใช้ค้นหาตอนตอบคำถาม
แบ่งดี = ค้นเจอเนื้อหาที่ตรงคำถามและครบถ้วน

ค่าตั้งต้นตอนยังไม่ทำ: ไม่แบ่งเลย ทั้งเอกสารเป็น chunk เดียว
ค่าที่ปรับได้ (CHUNK_SIZE, CHUNK_OVERLAP, ...) อยู่ใน student/settings.py
"""
import re
from student import settings


def _extract_sections(text: str) -> list[tuple[str, str]]:
    """แยกเนื้อหาตามลำดับหัวข้อ markdown (#, ##, ###)"""
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current_headings: dict[int, str] = {}
    current_content: list[str] = []

    def get_path() -> str:
        if not current_headings:
            return ""
        sorted_levels = sorted(current_headings.keys())
        return " > ".join(current_headings[lvl] for lvl in sorted_levels)

    heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$")
    in_code_block = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        match = heading_pattern.match(line)
        if match and not in_code_block:
            if current_content:
                body = "\n".join(current_content).strip()
                if body:
                    sections.append((get_path(), body))
                current_content = []

            level = len(match.group(1))
            title = match.group(2).strip()
            current_headings = {lvl: h for lvl, h in current_headings.items() if lvl < level}
            current_headings[level] = title
        else:
            current_content.append(line)

    if current_content:
        body = "\n".join(current_content).strip()
        if body:
            sections.append((get_path(), body))

    return sections


def _split_into_blocks(text: str) -> list[str]:
    """แยก text เป็น blocks ที่ไม่ควรตัดกลางคัน เช่น table, code block, paragraph"""
    lines = text.splitlines()
    blocks: list[str] = []
    current_block: list[str] = []
    mode = "normal"

    for line in lines:
        sline = line.strip()

        if sline.startswith("```"):
            if mode == "code":
                current_block.append(line)
                blocks.append("\n".join(current_block))
                current_block = []
                mode = "normal"
                continue
            else:
                if current_block:
                    blocks.append("\n".join(current_block))
                    current_block = []
                mode = "code"
                current_block.append(line)
                continue

        if mode == "code":
            current_block.append(line)
            continue

        if sline.startswith("|"):
            if mode != "table":
                if current_block:
                    blocks.append("\n".join(current_block))
                    current_block = []
                mode = "table"
            current_block.append(line)
            continue
        elif mode == "table":
            blocks.append("\n".join(current_block))
            current_block = []
            mode = "normal"

        if not sline:
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
            continue

        current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block))

    return [b.strip() for b in blocks if b.strip()]


def _chunk_text(header_prefix: str, text: str, max_size: int, overlap: int) -> list[str]:
    """รวม blocks เข้าด้วยกันให้ได้ขนาดใกล้เคียง max_size พร้อม overlap"""
    blocks = _split_into_blocks(text)
    if not blocks:
        return [header_prefix.strip()] if header_prefix else [text]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    prefix = f"[{header_prefix}]\n" if header_prefix and getattr(settings, "INCLUDE_HEADING", True) else ""
    prefix_len = len(prefix)

    for block in blocks:
        if len(block) + prefix_len > max_size:
            if current_chunk:
                chunks.append(prefix + "\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            step = max_size - prefix_len - overlap
            if step <= 0:
                step = max_size // 2
            start = 0
            while start < len(block):
                end = min(start + (max_size - prefix_len), len(block))
                chunks.append(prefix + block[start:end])
                if end == len(block):
                    break
                start += step
            continue

        if current_len + len(block) + prefix_len > max_size and current_chunk:
            chunks.append(prefix + "\n\n".join(current_chunk))
            last_block = current_chunk[-1]
            if len(last_block) <= overlap:
                current_chunk = [last_block, block]
                current_len = len(last_block) + len(block)
            else:
                current_chunk = [block]
                current_len = len(block)
        else:
            current_chunk.append(block)
            current_len += len(block)

    if current_chunk:
        chunks.append(prefix + "\n\n".join(current_chunk))

    return chunks


def split(doc_id: str, text: str) -> list[str]:
    """แบ่งเอกสาร 1 ชิ้นเป็น list ของข้อความ"""
    if not text.strip():
        return [f"doc: {doc_id}"]

    max_size = getattr(settings, "CHUNK_SIZE", 900)
    overlap = getattr(settings, "CHUNK_OVERLAP", 150)

    if getattr(settings, "SPLIT_BY_HEADING", True):
        sections = _extract_sections(text)
        if sections:
            chunks: list[str] = []
            for heading_path, section_text in sections:
                header = heading_path if heading_path else doc_id
                sub_chunks = _chunk_text(header, section_text, max_size, overlap)
                chunks.extend(sub_chunks)
            if chunks:
                return chunks

    return _chunk_text(doc_id, text, max_size, overlap)
