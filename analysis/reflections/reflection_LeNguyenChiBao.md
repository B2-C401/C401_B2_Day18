# Individual Reflection - Lab 18

**Ten:** Le Nguyen Chi Bao  
**Module phu trach:** M1

---

## 1. Dong gop ky thuat

- Module da implement: M1 - Advanced Chunking Strategies.
- Cac ham/class chinh da viet: `load_documents()`, `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()`, `compare_strategies()`.
- So tests pass: 13/13 (`pytest tests/test_m1.py -q`).

## 2. Kien thuc hoc duoc

- Khai niem moi nhat: Hierarchical chunking (retrieve child de tang precision, tra parent de du context cho LLM).
- Dieu bat ngo nhat: Chat luong retrieval phu thuoc rat manh vao cach tach chunk; tach sai ranh gioi cau/section lam giam context precision.
- Ket noi voi bai giang (slide nao): slide 7 -> 12 

## 3. Kho khan & Cach giai quyet

- Kho khan lon nhat: Semantic chunking phu thuoc model embedding va thu vien ngoai.
- Cach giai quyet: Them fallback lexical similarity de van chay khi model khong san; cache model de tranh load lap.
- Thoi gian debug: Khoang 45-60 phut.

## 4. Neu lam lai

- Se lam khac dieu gi: Viet them benchmark dinh luong cho tung strategy (avg chunk length, retrieval hit-rate theo nhom cau hoi).
- Module nao muon thu tiep: M2 (Hybrid Search) de toi uu BM25 tieng Viet va RRF.

## 5. Tu danh gia

| Tieu chi | Tu cham (1-5) |
|----------|---------------|
| Hieu bai giang | 5 |
| Code quality | 5 |
| Teamwork | 5 |
| Problem solving | 5 |
