# Review: VQA_Robustness_BiomedCLIP.ipynb

Penjelasan notebook yang dibuat dari PRD "Evaluasi Ketahanan VQA Medis Berbasis BiomedCLIP".

## 1. Apa yang dibuat

Satu notebook (`VQA_Robustness_BiomedCLIP.ipynb`, 35 sel) yang menjalankan seluruh eksperimen PRD secara berurutan di GPU T4 (Colab atau Kaggle). Tidak ada berkas lain yang dibutuhkan. Semua parameter ada di satu sel konfigurasi (`CFG`), sesuai R-01.

Status: sintaks semua sel sudah diperiksa, dan fungsi bootstrap, metrik, typo, serta parafrase sudah diuji lokal. Bagian yang butuh GPU, internet, atau paket berat (BiomedCLIP, OSF, torchattacks, TRADES, Gradio) **belum pernah dijalankan**.

## 2. Alur notebook

| Bagian | Isi | Kebutuhan PRD |
| --- | --- | --- |
| 0 | Instal paket, konfigurasi, seed, log lingkungan | R-01, R-02, R-12, NFR-03 |
| 1 | Unduh VQA-RAD (OSF, cadangan Hugging Face), split resmi, validasi per citra, distribusi kelas | FR-01 sampai FR-04, CP0 |
| 2 | Muat BiomedCLIP, latih Model 1 (3 seed), cek lebih baik dari menebak mayoritas | FR-05, CP1 |
| 3 | Evaluator serangan, penamaan dan penyimpanan hasil, smoke test 20 sampel | FR-08, FR-09, NFR-02, R-07 |
| 4 | Perturbasi teks: typo, parafrase, terjemahan Indonesia | FR-10, FR-11 |
| 5 | Skenario 1 sampai 4 pada Model 1, sanity otomatis | CP2, CP3, R-08, R-09 |
| 6 | Model 2 (TRADES), pilih β di validasi, skenario 5, gradient masking | FR-06, FR-12, CP4 |
| 7 | Tabel (CI bootstrap), kurva robust accuracy, pembekuan hasil | FR-14 sampai FR-16, CP5 |
| 8 | Demo Gradio dengan disclaimer | FR-17, FR-18, R-13, CP6 |

Sel *checkpoint* (CP0 sampai CP6) mencetak LULUS atau GAGAL per syarat PRD 9.1. Hanya CP0 sampai CP6 yang berupa pemeriksaan, bukan penghentian paksa (`hard=False`), jadi Anda tetap perlu membaca hasilnya.

## 3. Cara kerja inti

**Model.** `VQANet` menerima citra di [0,1] dan embedding pertanyaan. Normalisasi ada di dalam model, jadi ε terukur di ruang piksel (FR-04, R-04).
- Model 1: encoder BiomedCLIP beku, embedding citra (512) + teks (512) digabung, lalu MLP. Hanya MLP yang dilatih.
- Model 2: encoder citra dibuka dan dilatih dengan loss TRADES. Kepala diinisialisasi dari Model 1 dengan seed yang sama.

**Serangan.** `Wrap2` membungkus model menjadi fungsi citra → 2 logit [no, yes] dengan embedding pertanyaan ditahan tetap. Pembungkus ini yang masuk ke `torchattacks` (FGSM, PGD, AutoAttack, Square). Ini mengikuti saran integrasi di PRD bagian 7.

**Hasil dan resume.** Setiap kombinasi (model, serangan, ε, seed, jenis teks) disimpan ke `results/*.json` beserta prediksi per sampel. Bila berkas sudah ada, kombinasi dilewati. Karena itu sesi yang terputus bisa dilanjutkan dengan menjalankan ulang dari atas.

**Statistik.** Bootstrap persentil 1.000 resampel pada tingkat citra (bukan pertanyaan), dengan metrik dirata-ratakan dari 3 seed (FR-15, NFR-04). Fungsi ini sudah diuji lokal dan menghasilkan interval yang masuk akal.

**Gradient masking.** Empat pemeriksaan FR-12 dijalankan otomatis:
- (a) Square Attack tidak boleh lebih kuat dari serangan white-box.
- (b) RA tidak naik saat ε naik dan tidak melebihi akurasi bersih.
- (c) PGD pada ε=64/255 harus membuat akurasi mendekati nol.
- (d) Menggandakan iterasi PGD tidak boleh menaikkan ASR.

Hasilnya masuk ke `gradient_masking_report.csv`.

## 4. Keputusan yang saya ambil sendiri

PRD punya pertanyaan terbuka. Saya pilih default berikut agar notebook bisa dibangun. Semuanya bisa diubah.

| Keputusan | Pilihan | Alasan |
| --- | --- | --- |
| Pustaka serangan | `torchattacks` | Pertanyaan terbuka no. 7. Syaratnya citra [0,1] dan normalisasi di dalam model cocok dengan rancangan PRD. **Harus masuk daftar pustaka.** |
| Evaluasi ya/tidak | Klasifikasi 2 arah (logit no dan yes) | Membuat serangan margin (APGD-T, FAB-T) bekerja pada 2 kelas, dan metrik biner (AUC, MCC, FNR) jelas |
| Pertanyaan terbuka | Exact match, kosakata penuh, hanya data bersih | FR-16 meminta dilaporkan terpisah. Menyerang terbuka tidak diminta |
| Model kontrol | Ada, default mati (`run_control`) | Pertanyaan terbuka no. 2. Biayanya satu putaran pelatihan tambahan |
| TRADES | ε=4/255, 3 langkah, 5 epoch, β dari {1, 3, 6} | PRD tidak menetapkan. Ini tebakan masuk akal, bukan hasil tuning |
| Pilih checkpoint Model 2 | Rata-rata akurasi bersih dan robust PGD di validasi | Agar tidak memilih model yang robust tetapi akurasi bersihnya runtuh |
| Hasil gagal sanity | Otomatis dipindah ke `results_rejected/` | R-09. Dipindah, tidak dihapus, alasan dicatat |
| Terjemahan Indonesia | MarianMT `Helsinki-NLP/opus-mt-en-id` | Model terbuka, tidak perlu API berbayar |
| Parafrase | Aturan templat, bukan adversarial | Sesuai R-11, diberi label "uji robustness" |

## 5. Risiko dan hal yang belum pasti

1. **Belum dijalankan end-to-end.** Bug kecil di bagian GPU dan pustaka eksternal sangat mungkin ada. Jalankan bertahap dan lihat tiap sel checkpoint.
2. **URL unduhan OSF belum terverifikasi.** Bila gagal, notebook beralih ke mirror Hugging Face dan mencatatnya di `data_source.json`. Mirror itu tidak punya kolom `phrase_type`, tetapi split latih dan uji-nya sudah ada.
3. **Waktu AutoAttack** di T4 bisa besar. Lihat estimasi dari smoke test. Ingat bahwa smoke test sebelum Model 1 terlatih memakai kepala acak, jadi estimasi AutoAttack bisa meleset. Ulangi setelah pelatihan (R-07).
4. **Perbandingan Model 1 dan Model 2 belum bersih.** Model 2 mengubah dua hal sekaligus (encoder dibuka + TRADES). Tanpa `run_control=True`, kenaikan robustness tidak bisa dipastikan berasal dari TRADES.
5. **Teks Indonesia** akan hampir pasti menurunkan akurasi tajam, karena encoder teks BiomedCLIP berbahasa Inggris. Itu temuan, bukan bug.
6. **Penyimpanan.** Satu checkpoint Model 2 per epoch sekitar 350 MB. Dengan 3 seed, 5 epoch, dan sesi pemilihan β, Drive bisa terisi beberapa GB.
7. **Tidak dikerjakan:** BPDA/EOT (FR-13, opsional), Model 3 generatif (FR-07, opsional), dan koreksi Tabel 3.4 proposal (urusan naskah).

## 6. Hal di notebook yang bukan dari PRD

- Fungsi `trades_loss` ditulis sendiri mengikuti repositori resmi TRADES (MIT) dengan tambahan penskalaan loss ×128 agar gradien tidak underflow di fp16.
- Aturan sanity (a), yaitu akurasi di ε=0 sama dengan akurasi bersih, adalah saran PRD sendiri (R-08).
- Toleransi sanity 0,02 (`gm_tol`) dan ambang "mendekati nol" 0,05 (`gm_zero_thr`) adalah angka saya. Sesuaikan bila terlalu ketat atau longgar mengingat set uji kecil.
- Ambang "runtuh" untuk menambah titik ε=0,5/255 adalah RA ≤ 0,25 (`collapse_thr`), juga angka saya.

## 7. Langkah berikutnya yang disarankan

1. Buka di Colab, pilih GPU T4, jalankan sampai CP0 dan perbaiki bila ada galat unduhan.
2. Latih Model 1 dan cek CP1, lalu jalankan ulang smoke test.
3. Putuskan `aa_n_samples` dari estimasi waktu, baru jalankan skenario 1 sampai 4.
4. Sepakati target K1 sampai K3 dengan dosen (masih usulan di PRD).
5. Putuskan apakah model kontrol dipakai sebelum melatih Model 2.
