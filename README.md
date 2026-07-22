# Streamlit Peer Feedback Classifier

Aplikasi ini mengimplementasikan artefak model dari notebook:
`Text_Classification_Compare_Models_PR_DL_FINAL_from_multilabel_priority_ablation.ipynb`.

## Perilaku model

Dataset sumber memiliki empat label multi-label: `Problem`, `Suggestion`, `Appreciation`, dan `Neutral`. Notebook mengubahnya menjadi target **single-label** menggunakan urutan prioritas:

`Problem > Suggestion > Appreciation > Neutral`

Karena itu, aplikasi mengeluarkan satu label utama, confidence, dan skor setiap kelas.

## 1. Hasilkan artefak model

Jalankan notebook versi `STREAMLIT_READY` sampai selesai. Notebook akan menyimpan:

- `best_model_final.keras`
- `best_tokenizer_final.pkl`
- `label_encoder.pkl`
- `run_summary.json`

Notebook versi siap Streamlit juga membuat folder `streamlit_model_bundle` di dalam `OUTPUT_DIR`.

Alternatifnya, dari folder aplikasi jalankan:

```bash
python export_deployment_bundle.py \
  --run-dir /path/ke/OUTPUT_DIR \
  --output-dir artifacts
```

## 2. Instal dependensi

Gunakan **Python 3.12**. Versi ini kompatibel dengan TensorFlow 2.21.0 dan harus dipilih saat membuat ulang aplikasi di Streamlit Community Cloud.

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 3. Jalankan aplikasi

```bash
streamlit run app.py
```

Secara default aplikasi membaca artefak dari folder `artifacts`. Lokasi lain dapat diberikan melalui variabel lingkungan:

```bash
MODEL_DIR=/path/ke/model_bundle streamlit run app.py
```

## Fitur

- Prediksi satu teks.
- Preprocessing yang sama dengan notebook: normalisasi kata kapital penuh, penghapusan URL, HTML, dan emoji.
- Probabilitas untuk semua kelas.
- Prediksi batch melalui CSV dengan pemilihan kolom teks.
- Ringkasan distribusi label dan rerata confidence.
- Unduh hasil prediksi dalam CSV UTF-8.
- Informasi konfigurasi dan metrik training dari `run_summary.json`.

## Struktur

```text
app.py
model_runtime.py
custom_layers.py
export_deployment_bundle.py
requirements.txt
sample_input.csv
artifacts/
```

## Deployment Streamlit Community Cloud

Log deployment lama menggunakan Python 3.14.6, yang belum didukung TensorFlow 2.21.0. Aplikasi harus dihapus lalu dibuat ulang melalui **Advanced settings** dengan **Python 3.12**. Instruksi lengkap tersedia di `DEPLOY_STREAMLIT_CLOUD.md`.

## Catatan kompatibilitas

`custom_layers.py` menyertakan implementasi `MultiHeadSelfAttention`, `TransformerBlock`, dan `TokenAndPositionEmbedding`, sehingga model Transformer dari notebook dapat dimuat. Notebook `STREAMLIT_READY` juga menambahkan `get_config()` pada `TransformerBlock` agar serialisasi Keras lebih stabil.
