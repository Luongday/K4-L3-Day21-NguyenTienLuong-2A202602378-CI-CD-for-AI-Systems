# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Nguyễn Tiến Lượng |
| MSSV | 2A202602378 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/Luongday/K4-L3-Day21-NguyenTienLuong-2A202602378-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |
| 4 | 200 | 0.1 | 3 | 0.7281 | 0.8820 |
| 5 | 100 | 0.2 | 3 | 0.7290 | 0.8840 |

**Bộ siêu tham số đã chọn:** `n_estimators=100`, `learning_rate=0.2`, `max_depth=3`.

**Lý do:** Bộ này có `f1_score` cao nhất (0,7290) và vượt ngưỡng 0,65 với biên an toàn. Lần có accuracy cao nhất ở đây trùng với lần có F1 cao nhất; holdout chỉ 500 mẫu nên chênh dưới 0,01 chủ yếu là nhiễu. F1 vẫn phân biệt rõ hơn: lần 2 kém lần 5 khoảng 0,04 accuracy nhưng kém tới 0,12 F1 (0,6051, sẽ bị quality gate chặn). Về đánh đổi, `learning_rate` thấp cần nhiều cây hơn: cùng 200 cây, `learning_rate=0.05` cho F1 0,7014 còn `0.1` cho 0,7281.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Chỉ khoảng 24,8% mẫu thuộc lớp thu nhập > 50K, nên một mô hình vô dụng luôn trả lời "thu nhập thấp" vẫn đạt accuracy 0,752 mà không bắt được người thu nhập cao nào. F1 của lớp dương là trung bình điều hòa của precision và recall trên đúng lớp thiểu số cần dự đoán, nên mô hình đó có F1 bằng 0. Không dùng `average="weighted"` hay `"macro"` vì weighted bị lớp đa số kéo lên cao, còn macro trộn cả điểm lớp âm vào; cả hai làm loãng điểm của lớp dương và khiến ngưỡng 0,65 mất ý nghĩa.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| `pip install` báo `CERTIFICATE_VERIFY_FAILED` | Phần mềm diệt virus chặn HTTPS bằng chứng chỉ riêng mà Python không tin | Ghép CA của phần mềm đó với certifi, truyền qua `PIP_CERT` và `SSL_CERT_FILE`, vẫn giữ xác thực SSL |
| Test lỗi `ImportError: FallbackAsyncAdaptedQueuePool` | `requirements.txt` không pin SQLAlchemy nên pip lấy 2.1.x, không tương thích mlflow 2.13.0 | Thêm `sqlalchemy<2.1` vào `requirements.txt` (nếu không, job Unit Test trên Actions cũng đỏ) |
| Secret `STORAGE_CREDENTIALS` làm Train lỗi `JSONDecodeError` | PowerShell 5.1 bỏ dấu `"` khi truyền chuỗi JSON cho `gh --body` | Đặt lại secret bằng cách pipe file qua stdin (`Get-Content -Raw ... \| gh secret set`) |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0,7290 | 0,8840 |
| Bước 3 (thêm `train_batch2`) | 0,7330 | 0,8820 |

**Nhận xét:** F1 tăng nhẹ 0,004 còn accuracy giảm 0,002 khi dữ liệu huấn luyện tăng gấp đôi (22.361 → 44.722 mẫu). Chênh lệch này nằm trong nhiễu của holdout chỉ 500 mẫu, vì hai nửa dữ liệu được chia ngẫu nhiên từ cùng một nguồn nên dữ liệu mới không mang thêm thông tin; mô hình đã học gần hết từ nửa đầu. Điều Bước 3 kiểm chứng là quy trình: commit file `.dvc` kích hoạt tự động cả bốn job và VM phục vụ model mới mà không cần thao tác thủ công.

---

## 5. Phần Bonus Đã Thực Hiện

- [x] Bonus 2 - Ngưỡng quyết định: ngưỡng tốt nhất 0,30 cho F1 0,7519, so với 0,7290 ở ngưỡng 0,5. Ngưỡng được chọn trên chính holdout 500 mẫu nên con số hơi lạc quan; API vẫn dùng ngưỡng 0,5 và quality gate vẫn đọc `f1_score` ở 0,5.
- [x] Bonus 3 - Precision / recall: `src/report_detail.py` ghi `outputs/detail.txt` trong CI và upload cùng `report.json`. Lớp thu nhập cao có precision 0,867 nhưng recall chỉ 0,629 (bỏ sót 46/124 người). Bỏ sót nhóm này thường tốn kém hơn gán nhầm vì đó là nhóm mô hình cần tìm ra.
- [x] Bonus 5 - Lệch dữ liệu: `train()` in cảnh báo nếu tỷ lệ lớp dương lệch quá 5 điểm phần trăm so với 24,8% và ghi `positive_rate` vào `report.json`.
