from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from model_runtime import (
    basic_clean,
    load_bundle,
    missing_artifacts,
    predict_dataframe,
    read_deployment_config,
)


st.set_page_config(
    page_title="Peer Feedback Classifier",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.7rem; padding-bottom: 3rem;}
    .app-title {font-size: 2rem; font-weight: 750; margin-bottom: .2rem;}
    .app-subtitle {color: #59636e; margin-bottom: 1.25rem;}
    .label-card {border: 1px solid #e6e9ed; border-radius: 12px; padding: 1rem; background: #fafbfc;}
    div[data-testid="stMetric"] {border: 1px solid #eceff2; padding: .75rem; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts"))


@st.cache_resource(show_spinner="Memuat model dan tokenizer...")
def get_bundle(model_dir: str):
    return load_bundle(model_dir)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def score_table(row: pd.Series) -> pd.DataFrame:
    values = []
    for column in row.index:
        if column.startswith("score_") and pd.notna(row[column]):
            values.append({"Label": column.replace("score_", "", 1), "Probability": float(row[column])})
    return pd.DataFrame(values).sort_values("Probability", ascending=False)


st.markdown('<div class="app-title">Peer Feedback Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Implementasi model deep learning dari notebook klasifikasi komentar peer review.</div>',
    unsafe_allow_html=True,
)

config = read_deployment_config(MODEL_DIR)
missing = missing_artifacts(MODEL_DIR, config)

with st.sidebar:
    st.header("Model")
    st.caption(f"Folder artefak: `{MODEL_DIR}`")
    st.write(f"**Nama model:** {config.model_name}")
    st.write(f"**Maksimum token:** {config.max_len}")
    st.write("**Urutan prioritas target:**")
    st.caption(" > ".join(config.priority_order))
    st.divider()
    st.caption(
        "Model ini memprediksi satu label akhir. Data latih awalnya multi-label, lalu diubah "
        "menjadi single-label menggunakan urutan prioritas di atas."
    )

if missing:
    st.error("Artefak model belum tersedia: " + ", ".join(missing))
    st.info(
        "Jalankan notebook versi STREAMLIT_READY sampai selesai, lalu salin isi "
        "`streamlit_model_bundle/` ke folder `artifacts/` aplikasi ini."
    )
    st.stop()

try:
    bundle = get_bundle(str(MODEL_DIR))
except Exception as exc:
    st.exception(exc)
    st.stop()

single_tab, batch_tab, info_tab = st.tabs(["Prediksi Teks", "Prediksi CSV", "Informasi Model"])

with single_tab:
    example_text = "Penjelasannya sudah baik, tetapi bagian evaluasi masih kurang mendalam dan perlu ditambahkan contoh."
    text = st.text_area(
        "Masukkan komentar peer review",
        value="",
        placeholder=example_text,
        height=150,
    )
    col_predict, col_clear = st.columns([1, 5])
    predict_clicked = col_predict.button("Prediksi", type="primary", use_container_width=True)

    if predict_clicked:
        cleaned = basic_clean(text)
        if not cleaned:
            st.warning("Masukkan teks yang tidak kosong.")
        else:
            result = predict_dataframe(bundle, pd.DataFrame({"text": [text]}), "text")
            row = result.iloc[0]
            label = str(row["predicted_label"])
            confidence = float(row["confidence"])

            metric_left, metric_right = st.columns(2)
            metric_left.metric("Label prediksi", label)
            metric_right.metric("Confidence", f"{confidence:.2%}")

            st.subheader("Probabilitas per kelas")
            probabilities = score_table(row)
            st.dataframe(
                probabilities.style.format({"Probability": "{:.2%}"}),
                use_container_width=True,
                hide_index=True,
            )
            st.bar_chart(probabilities.set_index("Label"))

            with st.expander("Teks setelah basic cleaning"):
                st.code(row["cleaned_text"], language=None)

with batch_tab:
    uploaded = st.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded is not None:
        try:
            dataframe = pd.read_csv(uploaded)
        except UnicodeDecodeError:
            uploaded.seek(0)
            dataframe = pd.read_csv(uploaded, encoding="latin-1")
        except Exception as exc:
            st.error(f"CSV tidak dapat dibaca: {exc}")
            dataframe = None

        if dataframe is not None:
            st.write("Pratinjau data")
            st.dataframe(dataframe.head(20), use_container_width=True)
            text_column = st.selectbox(
                "Kolom teks",
                options=list(dataframe.columns),
                index=list(dataframe.columns).index(config.text_column)
                if config.text_column in dataframe.columns
                else 0,
            )

            if st.button("Proses seluruh data", type="primary"):
                with st.spinner("Melakukan inferensi..."):
                    predictions = predict_dataframe(bundle, dataframe, text_column)
                st.session_state["batch_predictions"] = predictions

    predictions = st.session_state.get("batch_predictions")
    if predictions is not None:
        valid_predictions = predictions.dropna(subset=["predicted_label"]).copy()
        total = len(predictions)
        valid = len(valid_predictions)
        avg_confidence = valid_predictions["confidence"].mean() if valid else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah baris", total)
        c2.metric("Berhasil diprediksi", valid)
        c3.metric("Rerata confidence", f"{avg_confidence:.2%}")

        if valid:
            distribution = (
                valid_predictions["predicted_label"]
                .value_counts()
                .rename_axis("Label")
                .reset_index(name="Count")
            )
            left, right = st.columns([1, 2])
            with left:
                st.subheader("Distribusi label")
                st.dataframe(distribution, use_container_width=True, hide_index=True)
            with right:
                st.subheader("Grafik distribusi")
                st.bar_chart(distribution.set_index("Label"))

        st.subheader("Hasil prediksi")
        st.dataframe(predictions, use_container_width=True)
        st.download_button(
            "Unduh hasil CSV",
            data=to_csv_bytes(predictions),
            file_name="peer_feedback_predictions.csv",
            mime="text/csv",
            type="primary",
        )

with info_tab:
    st.subheader("Konfigurasi deployment")
    info = {
        "Model": config.model_name,
        "Task": config.task_type,
        "Text column": config.text_column,
        "Label column": config.label_column,
        "Max length": config.max_len,
        "Priority order": " > ".join(config.priority_order),
    }
    st.dataframe(pd.DataFrame(info.items(), columns=["Item", "Value"]), hide_index=True, use_container_width=True)

    if bundle.run_summary:
        st.subheader("Ringkasan hasil training")
        summary = bundle.run_summary
        selected = {
            "Run ID": summary.get("run_id", "-"),
            "Best model": summary.get("best_model_name", config.model_name),
            "Final epochs": summary.get("final_epochs", "-"),
            "Train size": summary.get("n_train_full", "-"),
            "Test size": summary.get("n_test", "-"),
        }
        final_metrics = summary.get("final_test_metrics", {})
        selected.update(
            {
                "Test weighted F1": final_metrics.get("test_f1_weighted", "-"),
                "Test balanced accuracy": final_metrics.get("test_balanced_accuracy", "-"),
                "Test MCC": final_metrics.get("test_matthews_corrcoef", "-"),
            }
        )
        st.dataframe(pd.DataFrame(selected.items(), columns=["Item", "Value"]), hide_index=True, use_container_width=True)
