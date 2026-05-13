# DuongThuyTram-24521802-Dự đoán điểm cuối kỳ
# Ứng dụng dự đoán điểm cuối kỳ bằng Linear Regression

Thực hành **[CS523] Cấu trúc dữ liệu và giải thuật nâng cao**.

## Nội dung

Chương trình mô phỏng một ứng dụng dự đoán điểm cuối kỳ của sinh viên dựa trên dataset điểm của một môn học. Chương trình hỗ trợ các thao tác chính:

- Nhập dataset từ máy người dùng.
- Hiển thị dataset sau khi tải lên.
- Tự động xác định cột điểm cuối kỳ cần dự đoán.
- Tự động lấy các cột điểm đầu vào để huấn luyện mô hình.
- Huấn luyện mô hình Linear Regression bằng Gradient Descent.
- Nhập điểm đầu vào và hiển thị kết quả điểm cuối kỳ dự đoán.

## Cấu trúc chương trình

Chương trình được xây dựng trong file chính là `app.py`. Nội dung chương trình được chia thành các phần chính:

- `read_uploaded_file`: đọc dataset từ máy người dùng với định dạng `CSV`, `XLSX` hoặc `XLS`.

- `numeric_columns`: lọc ra các cột có kiểu dữ liệu số trong dataset.

- `detect_target_column`: tự động xác định cột điểm cuối kỳ cần dự đoán dựa trên tên cột như `final`, `cuoi`, `ck`.

- `detect_feature_columns`: tự động lấy các cột điểm đầu vào, tức các cột số còn lại sau khi bỏ cột điểm cuối kỳ.

- `prepare_xy`: xử lý dữ liệu, tách dữ liệu đầu vào `X` và dữ liệu đầu ra `y`.

- `standardize_train_test`: chuẩn hóa dữ liệu trước khi huấn luyện mô hình.

- `train_linear_regression_gd`: huấn luyện mô hình Linear Regression bằng thuật toán Gradient Descent.

- `predict_one`: dự đoán điểm cuối kỳ dựa trên điểm đầu vào mà người dùng nhập.

## Giao diện

Giao diện chương trình được xây dựng bằng **Streamlit**. Đây là thư viện hỗ trợ tạo giao diện web đơn giản bằng Python.

Giao diện của chương trình gồm các phần chính:

- **Nhập dataset**: cho phép người dùng chọn file dataset từ máy.
- **Dataset**: hiển thị bảng dữ liệu sau khi tải lên.
- **Huấn luyện mô hình**: nút bấm để bắt đầu huấn luyện Linear Regression.
- **Dự đoán điểm**: cho phép người dùng nhập điểm đầu vào, ví dụ điểm giữa kỳ.
- **Kết quả**: hiển thị điểm cuối kỳ dự đoán.

## Chức năng chính

- Đọc và hiển thị dataset điểm của sinh viên.
- Tự động nhận diện cột điểm cuối kỳ và cột điểm đầu vào.
- Xử lý dữ liệu thiếu hoặc dữ liệu không hợp lệ.
- Chuẩn hóa dữ liệu trước khi huấn luyện.
- Huấn luyện mô hình hồi quy tuyến tính bằng Gradient Descent.
- Dự đoán điểm cuối kỳ và giới hạn kết quả trong khoảng từ `0` đến `10`.

## Phương pháp dự báo

Phương pháp dự báo được sử dụng trong chương trình là **hồi quy tuyến tính** (*Linear Regression*).

Mô hình dự đoán có dạng:

```text
y_hat = w^T x + w0
