# Individual Reflection — Lab 18

**Tên:** Nguyễn Đức Trí
**Module phụ trách:** M2: Search (Hybrid Search — BM25 + Dense Search)

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** M2 Search (Hybrid Search — BM25 + Dense + RRF)
- **Các hàm/class chính đã viết:**
  - `segment_vietnamese`: Hỗ trợ tách từ tiếng Việt để BM25 hoạt động hiệu quả hơn.
  - `BM25Search`: Thực hiện lập chỉ mục (index) và tìm kiếm (search) dựa trên tần suất từ khóa BM25.
  - `DenseSearch`: Lập chỉ mục và tìm kiếm vector (Dense Search) sử dụng bộ mã hóa `all-MiniLM-L6-v2` và tính toán Cosine Similarity cục bộ (in-memory) bằng NumPy, giúp hệ thống chạy mượt mà mà không phụ thuộc vào kết nối tới Docker Qdrant.
  - `reciprocal_rank_fusion`: Gộp kết quả của BM25 và Dense Search theo thuật toán RRF.
- **Số tests pass:** 5/5 tests (`test_m2.py`).

## 2. Kiến thức học được

- **Khái niệm (Trang 17 - Hybrid Search & RRF Fusion):** Cách kết hợp giữa tìm kiếm truyền thống (BM25) và tìm kiếm ngữ nghĩa (Dense Vector). Thuật toán **Reciprocal Rank Fusion (RRF)** gộp 2 bảng xếp hạng riêng biệt thành một danh sách Top-K Results duy nhất bằng công thức:
  $$score(d) = \sum \frac{1}{k+rank_i(d)}$$
- **So sánh & Ứng dụng (Trang 18 - BM25 vs Dense):** Hiểu rõ ưu nhược điểm của từng phương pháp. BM25 thế mạnh ở việc tìm từ khóa chính xác, trong khi Dense Vector tốt ở ngữ nghĩa và từ đồng nghĩa. Với tiếng Việt, bước **tách từ (word segmentation)** bằng thư viện như `underthesea` là vô cùng quan trọng trước khi đánh chỉ mục BM25.
- **Tối ưu hóa nâng cao (Trang 19 - Beyond RRF):** Các kỹ thuật như Tensor Fusion, Late Interaction (ColBERT), và Learned Sparse (SPLADE) để tối ưu hóa tìm kiếm sau giai đoạn RRF. Trong đó, nếu không có dữ liệu có nhãn, sự kết hợp giữa **RRF** và **Cross-encoder** là hướng đi hiệu quả và tối ưu nhất.
- **Kết nối với code:**
  - `BM25Search` và `DenseSearch` là hiện thực hóa chính xác của 2 nhánh song song (Trang 17).
  - Sử dụng `underthesea.word_tokenize` trong `segment_vietnamese` giúp BM25 hoạt động hiệu quả cho tiếng Việt (Trang 18).
  - Thuật toán `reciprocal_rank_fusion` chính là hiện thực hóa của quy trình RRF Fusion (Trang 17).


## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** Lỗi kết nối tới server Qdrant do máy local không bật Docker Qdrant, cùng với lỗi lệch kích thước vector (`ValueError: could not broadcast input array`).
- **Cách giải quyết:** Đã chuyển đổi `DenseSearch` sang chế độ lưu trữ in-memory bằng NumPy (`np.save` file `.npy` và `.json`) để tính toán Cosine Similarity trực tiếp, và tính toán động kích thước vector (`vector_size`) dựa trên output của mô hình embedding.
- **Thời gian debug:** Khoảng 30 phút.

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:** Sẽ tìm hiểu sâu hơn về việc tối ưu hóa hiệu năng của NumPy khi tính toán cosine similarity trên tập dữ liệu hàng triệu chunks.
- **Module nào muốn thử tiếp:** Muốn thử tiếp module M3 (Rerank) để tối ưu lại Top-K sau khi lấy ra từ Hybrid Search.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5/5 |
| Code quality | 5/5 |
| Khả năng tự debug | 5/5 |
