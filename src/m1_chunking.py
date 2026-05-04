"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

from pydoc import doc
import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)

_SEMANTIC_MODEL = None


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load markdown/text and PDF files from data/."""
    docs = []

    # Load .md files
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    # Load .pdf files
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        try:
            from pypdf import PdfReader
        except Exception:
            try:
                from PyPDF2 import PdfReader
            except Exception:
                continue

        try:
            reader = PdfReader(fp)
            pages = [(page.extract_text() or "").strip() for page in reader.pages]
            text = "\n\n".join(p for p in pages if p).strip()
            if text:
                docs.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": os.path.basename(fp),
                            "file_type": "pdf",
                            "num_pages": len(reader.pages),
                        },
                    }
                )
        except Exception:
            continue

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    # TODO: Implement semantic chunking
    # 1. Split text into sentences:
    #    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    #
    # 2. Encode sentences:
    #    from sentence_transformers import SentenceTransformer
    #    model = SentenceTransformer("all-MiniLM-L6-v2")  # fast
    #    embeddings = model.encode(sentences)
    #
    # 3. Compare consecutive sentences:
    #    from numpy import dot
    #    from numpy.linalg import norm
    #    def cosine_sim(a, b): return dot(a, b) / (norm(a) * norm(b))
    #
    # 4. Group sentences:
    #    current_group = [sentences[0]]
    #    for i in range(1, len(sentences)):
    #        sim = cosine_sim(embeddings[i-1], embeddings[i])
    #        if sim < threshold:
    #            chunks.append(Chunk(text=" ".join(current_group), metadata=...))
    #            current_group = []
    #        current_group.append(sentences[i])
    #    # Don't forget last group
    #
    # 5. Return chunks with metadata: {"chunk_index": i, "strategy": "semantic"}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n", text) if s.strip()]
    if not sentences:
        return []

    def lexical_cosine(a: str, b: str) -> float:
        a_tokens = set(re.findall(r"\w+", a.lower()))
        b_tokens = set(re.findall(r"\w+", b.lower()))
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / ((len(a_tokens) * len(b_tokens)) ** 0.5)

    chunks: list[Chunk] = []

    # Try embedding-based semantic similarity first; fallback to lexical similarity.
    use_embeddings = False
    embeddings = None
    try:
        from sentence_transformers import SentenceTransformer
        from numpy import dot
        from numpy.linalg import norm

        global _SEMANTIC_MODEL
        if _SEMANTIC_MODEL is None:
            _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = _SEMANTIC_MODEL.encode(sentences, show_progress_bar=False)
        use_embeddings = True

        def cosine_sim(vec_a, vec_b) -> float:
            if norm(vec_a) == 0 or norm(vec_b) == 0:
                return 0.0
            return float(dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b)))
    except Exception:
        use_embeddings = False

    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        if use_embeddings and embeddings is not None:
            sim = cosine_sim(embeddings[i - 1], embeddings[i])
        else:
            sim = lexical_cosine(sentences[i - 1], sentences[i])

        if sim < threshold:
            chunks.append(
                Chunk(
                    text=" ".join(current_group).strip(),
                    metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic", "similarity": sim},
                )
            )
            current_group = []
        current_group.append(sentences[i])

    if current_group:
        chunks.append(
            Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic","similarity": sim},
            )
        )
    print(chunks)
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


# def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
#                        child_size: int = HIERARCHICAL_CHILD_SIZE,
#                        metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
#     """
#     Parent-child hierarchy: retrieve child (precision) → return parent (context).
#     Đây là default recommendation cho production RAG.

#     Args:
#         text: Input text.
#         parent_size: Chars per parent chunk.
#         child_size: Chars per child chunk.
#         metadata: Metadata gắn vào mỗi chunk.

#     Returns:
#         (parents, children) — mỗi child có parent_id link đến parent.
#     """
#     metadata = metadata or {}
#     # TODO: Implement hierarchical chunking
#     # 1. Split text into parents:
#     #    paragraphs = text.split("\n\n")
#     #    Gom paragraphs cho đến khi đạt parent_size → 1 parent chunk
#     #    pid = f"parent_{p_index}"
#     #    parent = Chunk(text=parent_text, metadata={**metadata, "chunk_type": "parent", "parent_id": pid})
#     #
#     # 2. Split each parent into children:
#     #    Slide window child_size trên parent text
#     #    child = Chunk(text=child_text, metadata={**metadata, "chunk_type": "child"}, parent_id=pid)
#     #
#     # 3. Return (parents_list, children_list)
#     #
#     # Production pattern:
#     #   - Index CHILDREN vào vector DB (nhỏ → embedding chính xác)
#     #   - Khi retrieve child → lookup parent_id → trả parent cho LLM (đủ context)
#     if not text.strip():
#         return [], []

#     paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
#     parents: list[Chunk] = []
#     children: list[Chunk] = []

#     current_parent_parts: list[str] = []
#     current_len = 0
#     for para in paragraphs:
#         para_len = len(para)
#         if current_parent_parts and current_len + para_len > parent_size:
#             parent_text = "\n\n".join(current_parent_parts).strip()
#             pid = f"parent_{len(parents)}"
#             parents.append(
#                 Chunk(
#                     text=parent_text,
#                     metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
#                 )
#             )
#             current_parent_parts = []
#             current_len = 0

#         current_parent_parts.append(para)
#         current_len += para_len

#     if current_parent_parts:
#         parent_text = "\n\n".join(current_parent_parts).strip()
#         pid = f"parent_{len(parents)}"
#         parents.append(
#             Chunk(
#                 text=parent_text,
#                 metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
#             )
#         )

#     for parent in parents:
#         pid = parent.metadata["parent_id"]
#         parent_text = parent.text
#         for start in range(0, len(parent_text), child_size):
#             child_text = parent_text[start:start + child_size].strip()
#             if not child_text:
#                 continue
#             children.append(
#                 Chunk(
#                     text=child_text,
#                     metadata={**metadata, "chunk_type": "child", "chunk_index": len(children)},
#                     parent_id=pid,
#                 )
#             )

#     return parents, children

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_hierarchical(text: str, parent_size: int = 1000, 
                       child_size: int = 200, 
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    metadata = metadata or {}
    if not text.strip():
        return [], []

    # --- BƯỚC 1: TẠO PARENTS (Giữ nguyên logic của bạn) ---
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parents: list[Chunk] = []
    children: list[Chunk] = []

    current_parent_parts: list[str] = []
    current_len = 0
    
    for para in paragraphs:
        if current_parent_parts and current_len + len(para) > parent_size:
            pid = f"parent_{len(parents)}"
            parent_text = "\n\n".join(current_parent_parts).strip()
            parents.append(Chunk(text=parent_text, metadata={**metadata, "chunk_type": "parent", "parent_id": pid}))
            current_parent_parts, current_len = [], 0
        current_parent_parts.append(para)
        current_len += len(para)

    if current_parent_parts:
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(text="\n\n".join(current_parent_parts).strip(), metadata={**metadata, "chunk_type": "parent", "parent_id": pid}))

    # --- BƯỚC 2: TẠO CHILDREN THÔNG MINH (Sử dụng TextSplitter) ---
    # Cấu hình splitter để không cắt ngang từ
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=20, # Thêm một chút gối đầu để giữ ngữ cảnh giữa các child
        separators=["\n\n", "\n", ". ", " ", ""] # Thứ tự ưu tiên ngắt
    )

    for parent in parents:
        pid = parent.metadata["parent_id"]
        # Splitter sẽ tự động chia nhỏ parent thành các đoạn con hợp lý
        child_texts = child_splitter.split_text(parent.text)
        
        for i, c_text in enumerate(child_texts):
            children.append(
                Chunk(
                    text=c_text.strip(),
                    metadata={**metadata, "chunk_type": "child", "chunk_index": i},
                    parent_id=pid,
                )
            )

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    # TODO: Implement structure-aware chunking
    # 1. Split by markdown headers:
    #    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    #
    # 2. Pair headers with their content:
    #    chunks = []
    #    current_header = ""
    #    current_content = ""
    #    for part in sections:
    #        if re.match(r'^#{1,3}\s+', part):
    #            if current_content.strip():
    #                chunks.append(Chunk(
    #                    text=f"{current_header}\n{current_content}".strip(),
    #                    metadata={**metadata, "section": current_header, "strategy": "structure"}
    #                ))
    #            current_header = part.strip()
    #            current_content = ""
    #        else:
    #            current_content += part
    #    # Don't forget last section
    #
    # 3. Return chunks — mỗi chunk = 1 section hoàn chỉnh
    #
    # Ưu điểm: giữ nguyên tables, lists, code blocks
    # Dùng khi: corpus có structured documents (docs, API refs, manuals)
    if not text.strip():
        return []

    header_pattern = re.compile(r"^#{1,3}\s+.+$", flags=re.MULTILINE)
    sections = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    def flush_section() -> None:
        if not current_content.strip() and not current_header.strip():
            return
        section_name = current_header.strip() or "root"
        section_text = f"{current_header}\n{current_content}".strip()
        chunks.append(
            Chunk(
                text=section_text,
                metadata={
                    **metadata,
                    "section": section_name,
                    "strategy": "structure",
                    "chunk_index": len(chunks),
                },
            )
        )

    for part in sections:
        if not part:
            continue
        if header_pattern.match(part.strip()):
            flush_section()
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    flush_section()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    # TODO: Implement comparison
    # 1. For each doc, run: chunk_basic, chunk_semantic, chunk_hierarchical, chunk_structure_aware
    # 2. Collect stats: num_chunks, avg_length, min_length, max_length
    # 3. Print comparison table:
    #    Strategy      | Chunks | Avg Len | Min | Max
    #    basic         |   12   |   420   | 100 | 500
    #    semantic      |    8   |   580   | 200 | 900
    #    hierarchical  | 5p/15c |   256   | 100 | 2048
    #    structure     |   10   |   450   | 150 | 800
    # 4. Return results dict
    def stats(chunks: list[Chunk]) -> dict:
        lengths = [len(c.text) for c in chunks]
        if not lengths:
            return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
        return {
            "num_chunks": len(chunks),
            "avg_length": int(sum(lengths) / len(lengths)),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

    basic_chunks: list[Chunk] = []
    semantic_chunks: list[Chunk] = []
    structure_chunks: list[Chunk] = []
    hierarchical_parents: list[Chunk] = []
    hierarchical_children: list[Chunk] = []

    for doc in documents:
        text = doc.get("text", "")
        doc_meta = doc.get("metadata", {})
        basic_chunks.extend(chunk_basic(text, metadata=doc_meta))
        semantic_chunks.extend(chunk_semantic(text, metadata=doc_meta))
        parents, children = chunk_hierarchical(text, metadata=doc_meta)
        hierarchical_parents.extend(parents)
        hierarchical_children.extend(children)
        structure_chunks.extend(chunk_structure_aware(text, metadata=doc_meta))

    basic_stats = stats(basic_chunks)
    semantic_stats = stats(semantic_chunks)
    structure_stats = stats(structure_chunks)
    hierarchical_stats = stats(hierarchical_children)
    hierarchical_stats["parent_chunks"] = len(hierarchical_parents)
    hierarchical_stats["child_chunks"] = len(hierarchical_children)

    print("Strategy      | Chunks  | Avg Len | Min | Max")
    print("-" * 47)
    print(
        f"basic         | {basic_stats['num_chunks']:<7} | {basic_stats['avg_length']:<7} | "
        f"{basic_stats['min_length']:<3} | {basic_stats['max_length']}"
    )
    print(
        f"semantic      | {semantic_stats['num_chunks']:<7} | {semantic_stats['avg_length']:<7} | "
        f"{semantic_stats['min_length']:<3} | {semantic_stats['max_length']}"
    )
    print(
        f"hierarchical  | {hierarchical_stats['parent_chunks']}p/{hierarchical_stats['child_chunks']}c"
        f"{' ' * 2}| {hierarchical_stats['avg_length']:<7} | {hierarchical_stats['min_length']:<3} | "
        f"{hierarchical_stats['max_length']}"
    )
    print(
        f"structure     | {structure_stats['num_chunks']:<7} | {structure_stats['avg_length']:<7} | "
        f"{structure_stats['min_length']:<3} | {structure_stats['max_length']}"
    )

    return {
        "basic": basic_stats,
        "semantic": semantic_stats,
        "hierarchical": hierarchical_stats,
        "structure": structure_stats,
    }


if __name__ == "__main__":
    docs = load_documents(data_dir='data')
    for doc in docs:
        text = doc.get("text", "")
        doc_meta = doc.get("metadata", {})
        parents, children = chunk_hierarchical(text, metadata=doc_meta)
        print(f"Parents: {parents}")
        print(f"Children: {children}")
        # print(f"Loaded {len(docs)} documents")
    # print(f"Loaded {len(docs)} documents")
    # results = compare_strategies(docs)
    # for name, stats in results.items():
    #     print(f"  {name}: {stats}")
