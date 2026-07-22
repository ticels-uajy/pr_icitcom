# Deploy ke Streamlit Community Cloud

## Penyebab error sebelumnya

Log menunjukkan environment menggunakan **Python 3.14.6**, sedangkan TensorFlow yang dipakai aplikasi tidak menyediakan wheel untuk Python 3.14. Selain itu, notebook training menggunakan **TensorFlow 2.21.0**, tetapi `requirements.txt` lama membatasi TensorFlow pada versi `<2.19`.

## Perbaikan yang diterapkan

- Runtime target: **Python 3.12**.
- TensorFlow disamakan dengan notebook: **tensorflow-cpu==2.21.0**.
- Paket CPU dipakai agar ukuran instalasi lebih kecil daripada paket TensorFlow penuh.
- `runtime.txt`, `.python-version`, dan Dockerfile diselaraskan ke Python 3.12.

## Langkah redeploy wajib

Versi Python aplikasi yang sudah terlanjur dibuat tidak dapat diubah hanya dengan reboot.

1. Push semua file versi perbaikan ini ke branch GitHub yang digunakan aplikasi.
2. Di Streamlit Community Cloud, catat URL aplikasi dan Secrets bila ada.
3. Hapus aplikasi lama.
4. Buat/deploy ulang dari repository dan entrypoint `app.py` yang sama.
5. Buka **Advanced settings**.
6. Pilih **Python 3.12**.
7. Masukkan kembali Secrets bila sebelumnya digunakan.
8. Deploy.

## File minimal di root repository

```text
app.py
model_runtime.py
custom_layers.py
requirements.txt
runtime.txt
.python-version
artifacts/
```

Folder `artifacts/` harus memuat sedikitnya:

```text
best_model_final.keras
best_tokenizer_final.pkl
label_encoder.pkl
deployment_config.json
```

`run_summary.json` bersifat opsional.

## Pemeriksaan log yang diharapkan

Saat build, log seharusnya menunjukkan Python 3.12.x dan berhasil memasang:

```text
tensorflow-cpu==2.21.0
streamlit==1.60.0
```
