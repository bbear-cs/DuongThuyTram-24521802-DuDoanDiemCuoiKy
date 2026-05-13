import io
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
FIXED_TEST_SIZE = 0.2
FIXED_LEARNING_RATE = 0.05
FIXED_EPOCHS = 3000
FIXED_CLIP_PREDICTION = True

st.set_page_config(
    page_title="Dự đoán điểm cuối kỳ",
    page_icon="📘",
    layout="wide",
)


@dataclass
class TrainResult:
    weights_original: np.ndarray
    bias_original: float


def display_label(name: str) -> str:
    lower_name = str(name).strip().lower()

    if lower_name == "midterm":
        return "Điểm giữa kỳ"

    if lower_name in ["final", "finalterm", "final_term"]:
        return "Điểm cuối kỳ"

    return str(name)



def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(raw))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw), encoding="latin1")

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw))

    raise ValueError("Chỉ hỗ trợ file .csv, .xlsx hoặc .xls")


def numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def detect_target_column(num_cols: List[str]) -> str:
    keywords = [
        "cuối",
        "cuoi",
        "cuối cùng",
        "cuoi cung",
        "final",
        "ck",
        "diem_cuoi",
        "điểm cuối",
    ]

    for col in num_cols:
        lower_col = str(col).lower().strip()
        if any(keyword in lower_col for keyword in keywords):
            return col

    return num_cols[-1]


def detect_feature_columns(num_cols: List[str], target_col: str) -> List[str]:
    return [col for col in num_cols if col != target_col]


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    selected_cols = feature_cols + [target_col]

    clean_df = df[selected_cols].copy()
    clean_df = clean_df.apply(pd.to_numeric, errors="coerce")
    clean_df = clean_df.dropna()

    X = clean_df[feature_cols].to_numpy(dtype=float)
    y = clean_df[target_col].to_numpy(dtype=float)

    return X, y, clean_df


def standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)

    x_std[x_std == 0] = 1.0

    X_train_scaled = (X_train - x_mean) / x_std
    X_test_scaled = (X_test - x_mean) / x_std

    return X_train_scaled, X_test_scaled, x_mean, x_std


def train_linear_regression_gd(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = FIXED_TEST_SIZE,
    learning_rate: float = FIXED_LEARNING_RATE,
    epochs: int = FIXED_EPOCHS,
    random_state: int = 42,
) -> TrainResult:

    if len(X) < 5:
        raise ValueError("Dataset cần ít nhất 5 dòng hợp lệ để chia train/test.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    X_train_scaled, X_test_scaled, x_mean, x_std = standardize_train_test(
        X_train,
        X_test,
    )

    n_samples, n_features = X_train_scaled.shape

    weights = np.zeros(n_features)
    bias = 0.0

    for epoch in range(epochs):
        y_pred = X_train_scaled @ weights + bias
        error = y_pred - y_train

        grad_w = (2.0 / n_samples) * (X_train_scaled.T @ error)
        grad_b = (2.0 / n_samples) * np.sum(error)

        weights = weights - learning_rate * grad_w
        bias = bias - learning_rate * grad_b

    weights_original = weights / x_std
    bias_original = bias - np.sum(weights * x_mean / x_std)

    return TrainResult(
        weights_original=weights_original,
        bias_original=bias_original,
    )


def predict_one(
    values: List[float],
    weights: np.ndarray,
    bias: float,
    clip_score: bool = FIXED_CLIP_PREDICTION,
) -> float:

    values_array = np.array(values, dtype=float)
    pred = float(np.dot(values_array, weights) + bias)

    if clip_score:
        pred = float(np.clip(pred, 0, 10))

    return pred


st.title("Dự đoán điểm cuối kỳ")


with st.sidebar:
    st.header("Nhập dataset")

    uploaded_file = st.file_uploader(
        "Chọn dataset từ máy",
        type=["csv", "xlsx", "xls"],
    )


if uploaded_file is None:
    st.stop()

try:
    df = read_uploaded_file(uploaded_file)
except Exception as exc:
    st.error(f"Không đọc được dataset: {exc}")
    st.stop()


st.subheader("Dataset")
st.dataframe(df.head(20), use_container_width=True)

st.write(f"Số dòng: **{df.shape[0]}**")
st.write(f"Số cột: **{df.shape[1]}**")


num_cols = numeric_columns(df)

if len(num_cols) < 2:
    st.error(
        "Dataset cần ít nhất 2 cột số: "
        "một cột điểm đầu vào và một cột điểm cuối kỳ."
    )
    st.stop()

target_col = detect_target_column(num_cols)
feature_cols = detect_feature_columns(num_cols, target_col)

if len(feature_cols) == 0:
    st.error("Không tìm thấy cột đầu vào để dự đoán.")
    st.stop()

X, y, clean_df = prepare_xy(df, feature_cols, target_col)

if len(clean_df) < 5:
    st.error("Sau khi bỏ dòng thiếu dữ liệu, dataset còn quá ít dòng hợp lệ.")
    st.stop()


# Nếu người dùng đổi file, reset mô hình cũ
current_signature = (
    uploaded_file.name,
    tuple(df.columns),
    df.shape[0],
    tuple(feature_cols),
    target_col,
)

if st.session_state.get("data_signature") != current_signature:
    st.session_state.pop("result", None)
    st.session_state.pop("feature_cols", None)
    st.session_state.pop("target_col", None)
    st.session_state.pop("clean_df", None)
    st.session_state["data_signature"] = current_signature



if st.button("Huấn luyện mô hình", type="primary"):
    try:
        result = train_linear_regression_gd(
            X=X,
            y=y,
            test_size=FIXED_TEST_SIZE,
            learning_rate=FIXED_LEARNING_RATE,
            epochs=FIXED_EPOCHS,
        )

        st.session_state["result"] = result
        st.session_state["feature_cols"] = feature_cols
        st.session_state["target_col"] = target_col
        st.session_state["clean_df"] = clean_df

    except Exception as exc:
        st.error(f"Lỗi khi huấn luyện mô hình: {exc}")
        st.stop()



if "result" not in st.session_state:
    st.info("Bấm nút **Huấn luyện mô hình** trước để dự đoán điểm cuối kỳ.")
    st.stop()


result: TrainResult = st.session_state["result"]
feature_cols = st.session_state["feature_cols"]
clean_df = st.session_state["clean_df"]



st.divider()
st.subheader("Dự đoán điểm")

input_values = []

input_cols = st.columns(min(4, len(feature_cols)))

for idx, feature in enumerate(feature_cols):
    with input_cols[idx % len(input_cols)]:
        default_value = float(clean_df[feature].mean())

        value = st.number_input(
            label=display_label(feature),
            min_value=0.0,
            max_value=10.0,
            value=round(default_value, 2),
            step=0.25,
        )

        input_values.append(value)


if st.button("Dự đoán điểm cuối kỳ"):
    pred = predict_one(
        values=input_values,
        weights=result.weights_original,
        bias=result.bias_original,
        clip_score=FIXED_CLIP_PREDICTION,
    )

    st.success(f"Điểm cuối kỳ dự đoán: **{pred:.2f} / 10**")