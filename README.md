# Demo KLTN 38: Phát hiện lỗi bề mặt sản phẩm

Ứng dụng Streamlit phát hiện và khoanh vùng lỗi bề mặt trên ảnh sản phẩm, dùng các mô hình YOLO đã huấn luyện trên bộ dữ liệu MVTec AD (bộ chia S1, 15 danh mục, 48 lớp lỗi). Repo chứa mã nguồn demo và toàn bộ trọng số đã huấn luyện.

## Chạy demo

**Trên Windows:** tải repo về (Code → Download ZIP hoặc `git clone`), rồi bấm đúp `chay_demo.bat`. Lần đầu chạy, script tự cài thư viện, sau đó trình duyệt mở `http://localhost:8501`.

**Chạy thủ công (mọi hệ điều hành, Python 3.10 trở lên):**

```bash
git clone https://github.com/mizzhau/demo-KLTN.git
cd demo-KLTN
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Demo chạy trên CPU, không cần GPU.

**Cách dùng:** ở thanh bên trái, chọn mô hình, tải ảnh lên hoặc chọn một trong 15 ảnh mẫu, rồi chỉnh ngưỡng τ. Khi đổi mô hình, τ tự đặt về giá trị τ\* chọn trên tập val của mô hình đó.

## Mô hình trong demo

Mỗi mô hình được huấn luyện với 5 seed (42–46). Demo dùng seed có **F1 cao nhất trên tập val tại ngưỡng τ\*** (`f1_val_at_tau_star` trong `thresholds.json`). Việc chọn seed chỉ dùng tập val, không dùng tập test, giống cách chọn τ\*.

| Mã | Mô hình | Seed | F1 val tại τ\* | τ\* |
|---|---|---|---|---|
| A0 | YOLO11n (baseline) | 43 | 0,775 | 0,12 |
| A1 | YOLO11n + CBAM-Backbone | 43 | 0,787 | 0,19 |
| A2 | YOLO11n + CBAM-Neck | 46 | 0,817 | 0,31 |
| A3 | YOLO11n + CBAM-PreDetect | 43 | 0,802 | 0,24 |
| A4 | YOLO11n + CBAM-Backbone+PreDetect | 45 | 0,804 | 0,26 |
| B1 | YOLO11n + SimAM | 45 | 0,778 | 0,13 |
| B2 | YOLO11n + EMA | 46 | 0,798 | 0,24 |
| B3 | YOLO11n + CBAM-PreDetect (khởi tạo mặc định) | 46 | 0,793 | 0,23 |
| D1 | YOLO11s | 43 | 0,792 | 0,12 |
| D2 | YOLOv8n | 45 | 0,772 | 0,11 |

## Cấu trúc repo

| Mục | Nội dung |
|---|---|
| `app.py` | Giao diện Streamlit |
| `core.py` | Suy luận: nạp mô hình, phát hiện, áp ngưỡng, vẽ khung lỗi. Bảng `SEEDS` quy định seed dùng cho từng mô hình |
| `cbam.py`, `attention.py` | Module CBAM, SimAM, EMA, cần để nạp A1–A4 và B1–B3 |
| `samples/` | 15 ảnh lỗi lấy nguyên từ tập test S1: mỗi danh mục MVTec AD 1 ảnh, 15 loại lỗi khác nhau |
| `trong_so/` | Toàn bộ trọng số đã huấn luyện (xem mục dưới) |
| `chay_demo.bat` | Chạy demo trên Windows |
| `requirements.txt`, `packages.txt`, `.streamlit/` | Thư viện Python, gói hệ thống (khi deploy lên Linux/Streamlit Cloud) và cấu hình Streamlit |

## Trọng số (`trong_so/`)

- `trong_so/<Mã>/s<seed>/best.pt` và `thresholds.json`: 10 mô hình A0–A4, B1–B3, D1, D2, mỗi mô hình 5 seed (42–46), tổng 50 lần huấn luyện. `tau_star` là ngưỡng chọn trên tập val.
- `trong_so/C3/s43`, `trong_so/C3/s44`: mô hình đối chứng DEIMv2-Nano (`best.pth`, `thresholds.json`, `config_c3_resolved.yml`). Mô hình này chạy bằng framework DEIMv2, không nạp bằng Ultralytics nên không có trong demo.
- C1 (PP-PicoDet-S), C2 (NanoDet-Plus-m), C4 (EfficientNet-B0), C5 (DeiT-Tiny): checkpoint sau huấn luyện không được lưu lại, chỉ còn kết quả dự đoán dùng cho số liệu đối chứng trong báo cáo.
- `trong_so/SHA256SUMS.txt`: checksum của 52 file trọng số.

Để nạp A1–A4, B1–B3 ngoài demo: đặt `cbam.py` và `attention.py` lên đầu `sys.path`, gọi `cbam.register()`, rồi mới `YOLO("trong_so/A1/s43/best.pt")`.
