from __future__ import annotations

import io
import os
import platform
from pathlib import Path

# Reduce TensorFlow startup noise before TensorFlow is imported lazily.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import streamlit as st
from PIL import Image

from model_runtime import (
    basic_clean,
    load_bundle,
    missing_artifacts,
    predict_dataframe,
    read_deployment_config,
)


APP_NAME = "SLaT-PF-ID"
APP_FULL_NAME = "Semantic Label Transformation for Peer Feedback in Indonesian Language"
APP_DESCRIPTION = (
    "Aplikasi berbasis kecerdasan artifisial untuk menganalisis dan mengklasifikasikan "
    "umpan balik sejawat (peer feedback) berbahasa Indonesia secara otomatis."
)

APP_DIR = Path(__file__).resolve().parent
APP_LOGO_PATH = APP_DIR / "slat_pf_id_header_logo.png"
APP_PAGE_ICON_PATH = APP_DIR / "slat_pf_id_page_icon.png"
APP_ICON = Image.open(APP_PAGE_ICON_PATH) if APP_PAGE_ICON_PATH.exists() else "🏷️"

LABEL_DESCRIPTIONS = {
    "Appreciation": "Umpan balik yang menyampaikan apresiasi atau penilaian positif.",
    "Problem": "Umpan balik yang mengidentifikasi masalah, kekurangan, atau bagian yang perlu diperbaiki.",
    "Suggestion": "Umpan balik yang memberikan saran, rekomendasi, atau usulan perbaikan.",
    "Neutral": "Umpan balik yang bersifat netral atau tidak secara jelas termasuk kategori lainnya.",
}


st.set_page_config(
    page_title=f"{APP_NAME} | Peer Feedback Analytics",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    .app-title {
        font-size: 2rem;
        font-weight: 750;
        line-height: 1.35;
        padding-top: .12rem;
        margin: 0 0 .15rem 0;
        overflow: visible;
    }
    .app-full-name {font-size: 1.05rem; font-weight: 600; color: #39424e; margin-bottom: .35rem;}
    .app-subtitle {color: #59636e; margin-bottom: 1rem;}
    div[data-testid="stImage"] {
        margin-top: .15rem;
        margin-bottom: .6rem;
    }
    div[data-testid="stImage"] img {
        height: auto;
        object-fit: contain;
    }
    @media (max-width: 640px) {
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    }
    .app-note {
        border: 1px solid #e6e9ed;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        background: #fafbfc;
        margin-bottom: 1.25rem;
    }
    .label-card {border: 1px solid #e6e9ed; border-radius: 12px; padding: 1rem; background: #fafbfc;}
    div[data-testid="stMetric"] {border: 1px solid #eceff2; padding: .75rem; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts"))


@st.cache_resource(show_spinner="Memuat model dan tokenizer SLaT-PF-ID...")
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


def render_responsive_logo(image_path: Path):
    """Render header logo using Streamlit's native image component."""
    if not image_path.exists():
        return False

    # Center the logo on desktop while allowing it to shrink naturally on mobile.
    left_spacer, logo_col, right_spacer = st.columns([1, 8, 1])
    with logo_col:
        st.image(
            str(image_path),
            use_container_width=True,
        )
    return True


logo_rendered = render_responsive_logo(APP_LOGO_PATH)

if not logo_rendered:
    st.warning(
        f"Logo aplikasi tidak ditemukan di: {APP_LOGO_PATH.name}. "
        "Pastikan file logo berada pada folder yang sama dengan app.py."
    )
    st.markdown(f'<div class="app-full-name">{APP_FULL_NAME}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-subtitle">{APP_DESCRIPTION}</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-note">
    <b>SLaT-PF-ID</b> menerapkan pendekatan <i>Semantic Label Transformation</i> untuk membantu
    memetakan makna semantik komentar mahasiswa ke kategori <b>Appreciation, Problem,
    Suggestion,</b> dan <b>Neutral</b>. Hasil analisis ditujukan sebagai informasi pendukung
    bagi dosen dan peneliti dalam memahami pola serta karakteristik <i>peer feedback</i>.
    </div>
    """,
    unsafe_allow_html=True,
)

config = read_deployment_config(MODEL_DIR)
missing = missing_artifacts(MODEL_DIR, config)

with st.sidebar:
    st.header(APP_NAME)
    st.caption(APP_FULL_NAME)
    st.divider()

    st.subheader("Model & Konfigurasi")
    st.caption(f"Folder artefak: `{MODEL_DIR}`")
    st.caption(f"Python: `{platform.python_version()}`")
    st.write(f"**Nama model:** {config.model_name}")
    st.write(f"**Maksimum token:** {config.max_len}")
    st.write("**Urutan prioritas target:**")
    st.caption(" > ".join(config.priority_order))
    st.divider()
    st.caption(
        "Model memprediksi satu label akhir. Data latih awalnya multi-label, lalu diubah "
        "menjadi single-label menggunakan urutan prioritas target di atas."
    )
    st.info(
        "SLaT-PF-ID merupakan AI-assisted analytics tool. Hasil prediksi digunakan sebagai "
        "informasi pendukung dan tidak menggantikan penilaian akademik dosen."
    )

if missing:
    st.error("Artefak model SLaT-PF-ID belum tersedia: " + ", ".join(missing))
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

single_tab, batch_tab, info_tab = st.tabs(["Prediksi Teks", "Prediksi CSV", "Informasi SLaT-PF-ID"])

with single_tab:
    st.subheader("Klasifikasi satu komentar")
    st.caption(
        "Masukkan satu komentar peer feedback berbahasa Indonesia untuk memperoleh label prediksi, "
        "confidence, dan probabilitas setiap kelas."
    )

    example_text = "Penjelasannya sudah baik, tetapi bagian evaluasi masih kurang mendalam dan perlu ditambahkan contoh."
    text = st.text_area(
        "Masukkan komentar peer feedback",
        value="",
        placeholder=example_text,
        height=150,
    )
    col_predict, col_clear = st.columns([1, 5])
    predict_clicked = col_predict.button("Prediksi", type="primary", use_container_width=True)

    if predict_clicked:
        cleaned = basic_clean(text)
        if not cleaned:
            st.warning("Masukkan teks peer feedback yang tidak kosong.")
        else:
            result = predict_dataframe(bundle, pd.DataFrame({"text": [text]}), "text")
            row = result.iloc[0]
            label = str(row["predicted_label"])
            confidence = float(row["confidence"])

            metric_left, metric_right = st.columns(2)
            metric_left.metric("Label prediksi", label)
            metric_right.metric("Confidence", f"{confidence:.2%}")

            if label in LABEL_DESCRIPTIONS:
                st.caption(LABEL_DESCRIPTIONS[label])

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

            st.caption(
                "Catatan: hasil prediksi merupakan informasi pendukung untuk analisis peer feedback "
                "dan tetap perlu diinterpretasikan sesuai konteks akademik."
            )

with batch_tab:
    st.subheader("Klasifikasi kumpulan komentar")
    st.caption(
        "Unggah file CSV berisi komentar peer feedback, pilih kolom teks, lalu proses seluruh data "
        "untuk memperoleh hasil klasifikasi secara batch."
    )

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
                "Kolom teks peer feedback",
                options=list(dataframe.columns),
                index=list(dataframe.columns).index(config.text_column)
                if config.text_column in dataframe.columns
                else 0,
            )

            if st.button("Proses seluruh data", type="primary"):
                with st.spinner("SLaT-PF-ID sedang melakukan inferensi..."):
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

        st.subheader("Hasil prediksi SLaT-PF-ID")
        st.dataframe(predictions, use_container_width=True)
        st.download_button(
            "Unduh hasil CSV",
            data=to_csv_bytes(predictions),
            file_name="slat_pf_id_predictions.csv",
            mime="text/csv",
            type="primary",
        )
        st.caption(
            "Gunakan hasil klasifikasi sebagai informasi pendukung untuk memahami pola peer feedback. "
            "Interpretasi akhir tetap mempertimbangkan konteks aktivitas peer review dan penilaian dosen."
        )

with info_tab:
    st.subheader("Tentang SLaT-PF-ID")
    st.markdown(
        """
        **SLaT-PF-ID (Semantic Label Transformation for Peer Feedback in Indonesian Language)** adalah
        aplikasi berbasis kecerdasan artifisial yang dikembangkan untuk menganalisis dan mengklasifikasikan
        umpan balik sejawat (*peer feedback*) berbahasa Indonesia secara otomatis.

        Aplikasi menerapkan pendekatan **Semantic Label Transformation** untuk menangkap makna semantik
        komentar mahasiswa dan memetakannya ke kategori umpan balik yang relevan. Analisis ini membantu
        dosen dan peneliti memahami pola serta karakteristik umpan balik dalam kegiatan *peer review* secara
        lebih sistematis dan efisien.
        """
    )

    st.subheader("Kategori umpan balik")
    label_columns = st.columns(2)
    for idx, (label, description) in enumerate(LABEL_DESCRIPTIONS.items()):
        with label_columns[idx % 2]:
            st.markdown(
                f'<div class="label-card"><b>{label}</b><br>{description}</div>',
                unsafe_allow_html=True,
            )
            st.write("")

    st.subheader("Peran hasil analisis")
    st.write(
        "Hasil SLaT-PF-ID dapat digunakan sebagai informasi pendukung untuk memahami kualitas dan "
        "kecenderungan umpan balik mahasiswa, mengevaluasi pelaksanaan aktivitas peer review, dan "
        "memperoleh insight bagi peningkatan proses pembelajaran."
    )
    st.warning(
        "SLaT-PF-ID tidak dimaksudkan untuk menggantikan penilaian akademik dosen. Aplikasi ini berfungsi "
        "sebagai AI-assisted analytics tool untuk membantu mengolah dan menyajikan informasi dari data "
        "peer feedback secara lebih cepat, konsisten, dan mudah dipahami."
    )

    st.subheader("Konfigurasi deployment")
    info = {
        "Aplikasi": APP_NAME,
        "Nama lengkap": APP_FULL_NAME,
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
