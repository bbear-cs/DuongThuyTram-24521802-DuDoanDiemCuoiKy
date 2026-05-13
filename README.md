# DuongThuyTram-24521802-Dự đoán điểm cuối kỳ
\section*{Giới thiệu}
\addcontentsline{toc}{section}{Giới thiệu}

\textbf{Ứng dụng dự đoán điểm cuối kỳ bằng Linear Regression}

Thực hành \textbf{[CS523] Cấu trúc dữ liệu và giải thuật nâng cao}.

\subsection*{Nội dung}

Chương trình mô phỏng một ứng dụng dự đoán điểm cuối kỳ của sinh viên dựa trên dataset điểm của một môn học. Chương trình hỗ trợ các thao tác chính:

\begin{itemize}
    \item Nhập dataset từ máy người dùng.
    \item Hiển thị dataset sau khi tải lên.
    \item Tự động xác định cột điểm cuối kỳ cần dự đoán.
    \item Tự động lấy các cột điểm đầu vào để huấn luyện mô hình.
    \item Huấn luyện mô hình Linear Regression bằng Gradient Descent.
    \item Nhập điểm đầu vào và hiển thị kết quả điểm cuối kỳ dự đoán.
\end{itemize}

\subsection*{Cấu trúc chương trình}

Chương trình được xây dựng trong file chính là \texttt{app.py}. Nội dung chương trình được chia thành các phần chính:

\begin{itemize}
    \item \texttt{read\_uploaded\_file}: đọc dataset từ máy người dùng với định dạng \texttt{CSV}, \texttt{XLSX} hoặc \texttt{XLS}.
    
    \item \texttt{numeric\_columns}: lọc ra các cột có kiểu dữ liệu số trong dataset.
    
    \item \texttt{detect\_target\_column}: tự động xác định cột điểm cuối kỳ cần dự đoán dựa trên tên cột như \texttt{final}, \texttt{cuoi}, \texttt{ck}.
    
    \item \texttt{detect\_feature\_columns}: tự động lấy các cột điểm đầu vào, tức các cột số còn lại sau khi bỏ cột điểm cuối kỳ.
    
    \item \texttt{prepare\_xy}: xử lý dữ liệu, tách dữ liệu đầu vào $X$ và dữ liệu đầu ra $y$.
    
    \item \texttt{standardize\_train\_test}: chuẩn hóa dữ liệu trước khi huấn luyện mô hình.
    
    \item \texttt{train\_linear\_regression\_gd}: huấn luyện mô hình Linear Regression bằng thuật toán Gradient Descent.
    
    \item \texttt{predict\_one}: dự đoán điểm cuối kỳ dựa trên điểm đầu vào mà người dùng nhập.
\end{itemize}

\subsection*{Giao diện}

Giao diện chương trình được xây dựng bằng \textbf{Streamlit}. Đây là thư viện hỗ trợ tạo giao diện web đơn giản bằng Python. Giao diện của chương trình gồm các phần chính:

\begin{itemize}
    \item \textbf{Nhập dataset}: cho phép người dùng chọn file dataset từ máy.
    \item \textbf{Dataset}: hiển thị bảng dữ liệu sau khi tải lên.
    \item \textbf{Huấn luyện mô hình}: nút bấm để bắt đầu huấn luyện Linear Regression.
    \item \textbf{Dự đoán điểm}: cho phép người dùng nhập điểm đầu vào, ví dụ điểm giữa kỳ.
    \item \textbf{Kết quả}: hiển thị điểm cuối kỳ dự đoán.
\end{itemize}

\subsection*{Chức năng chính}

\begin{itemize}
    \item Đọc và hiển thị dataset điểm của sinh viên.
    \item Tự động nhận diện cột điểm cuối kỳ và cột điểm đầu vào.
    \item Xử lý dữ liệu thiếu hoặc dữ liệu không hợp lệ.
    \item Chuẩn hóa dữ liệu trước khi huấn luyện.
    \item Huấn luyện mô hình hồi quy tuyến tính bằng Gradient Descent.
    \item Dự đoán điểm cuối kỳ và giới hạn kết quả trong khoảng từ $0$ đến $10$.
\end{itemize}
