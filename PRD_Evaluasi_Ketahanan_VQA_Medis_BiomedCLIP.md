# PRD: Evaluasi Ketahanan Model VQA Medis Berbasis BiomedCLIP terhadap Serangan Adversarial pada VQA-RAD

15 November 2026 · DIKAMEO

## 1. Ringkasan

Proyek ini membangun protokol evaluasi yang mengukur seberapa tahan model VQA medis berbasis BiomedCLIP terhadap serangan citra dan perubahan teks pada VQA-RAD, lalu menguji pertahanan adversarial fine-tuning (TRADES) dan menampilkannya dalam demo web Gradio. Penelitian dikerjakan sebagai Final Project mata kuliah Temu Kembali Citra, Informatika UMM, oleh tiga mahasiswa.

**Mengapa perlu.** Model VQA medis bisa sangat akurat pada data bersih tetapi runtuh saat citra diberi gangguan yang tak terlihat mata. Dua referensi utama proposal menunjukkan hal itu:

- **SafeMed-R1** (Pramana dkk., 2025): model fine-tune biasa turun dari 95,43% pada data bersih ke 26,07% di bawah PGD. Model adversarial mencapai 82,27% (Instruct) dan 84,45% (Think). Penulisnya hanya menguji PGD, dan certified robustness lewat randomized smoothing tidak dievaluasi.
- **Fime dkk.** (*Scientific Reports*, 2026): CLIP ViT-B/32 beku pada MedMNIST dalam federated learning. Serangan iteratif (PGD, BIM) membuat tingkat keberhasilan serangan sering di atas 90%, dan dua pertahanan test-time melemah pada serangan kuat. Tugasnya klasifikasi, bukan VQA.

**Celah yang diisi.** Belum ada studi yang (1) mengukur VQA radiologi dengan encoder BiomedCLIP terhadap FGSM, PGD, dan AutoAttack, (2) menggabungkan serangan citra dengan perubahan teks, termasuk pertanyaan berbahasa Indonesia, dan (3) memeriksa gradient masking pada pertahanan yang diuji.

**Keluaran produk.**

| # | Keluaran | Isi |
| --- | --- | --- |
| 1 | Protokol evaluasi | Pipeline serangan citra, perubahan teks, dan metrik, dengan 3 seed dan interval kepercayaan bootstrap |
| 2 | Model | Model 1 (BiomedCLIP beku + kepala fusi), Model 2 (TRADES), Model 3 (pembanding generatif, opsional) |
| 3 | Demo web | Aplikasi Gradio yang menampilkan jawaban, confidence, dan efek serangan, dengan disclaimer bukan alat diagnosis |

## 2. Tujuan, non-tujuan, dan kriteria keberhasilan

Proyek dianggap berhasil bila kelima tujuan proposal terpenuhi dengan angka yang bisa direproduksi pada GPU T4. Target numerik di bawah ini adalah **usulan** untuk dikonfirmasi tim dan dosen, karena proposal belum menetapkannya.

| # | Tujuan (dari proposal, subbab 1.3) | Kriteria keberhasilan | Status target |
| --- | --- | --- | --- |
| G1 | Ukur performa bersih Model 1 pada VQA-RAD: accuracy, balanced accuracy, sensitivitas per kelas | Hasil dari minimal 3 seed, dengan interval kepercayaan bootstrap | Dari proposal |
| G2 | Ukur kurva akurasi terhadap epsilon untuk FGSM, PGD, AutoAttack, plus satu konfigurasi PGD ala SafeMed-R1 | Kurva untuk ε = 1, 2, 4, 8 per 255, dan satu titik PGD (ε = 0,01; α = 0,002; 5 langkah) | Dari proposal |
| G3 | Ukur efek perturbasi teks dan gabungan citra dengan teks | Tabel untuk typo, parafrase, terjemahan ke bahasa Indonesia, dan kombinasi dengan serangan citra | Dari proposal |
| G4 | Uji TRADES, periksa gradient masking, dan serangan adaptif bila ada preprocessing | Robust accuracy Model 2 dibandingkan Model 1 pada skenario yang sama, disertai hasil pemeriksaan gradient masking | Dari proposal |
| G5 | Bangun demo web Gradio dengan disclaimer jelas | Demo berjalan di Colab atau Kaggle dan menampilkan jawaban, confidence, serta efek serangan | Dari proposal |
| K1 | Pemulihan robustness | Robust accuracy Model 2 lebih tinggi dari Model 1 pada PGD dan AutoAttack di ε menengah dan besar | Usulan |
| K2 | Biaya akurasi bersih | Penurunan akurasi bersih Model 2 terhadap Model 1 dilaporkan, dengan batas yang disepakati (misalnya di bawah 5 poin) | Usulan |
| K3 | Reproduksibilitas | Seluruh angka bisa diulang dari satu notebook dengan seed tetap | Usulan |

**Hipotesis yang diuji** (dari proposal): robust accuracy turun makin tajam saat ε naik, dan adversarial fine-tuning ringan memulihkan sebagian akurasi robust dengan penurunan akurasi bersih yang kecil.

**Non-tujuan.**

- Tidak mengalahkan angka SafeMed-R1. Dataset, model, dan ruang perturbasinya berbeda, sehingga perbandingan hanya kualitatif.
- Tidak membuat alat diagnosis dan tidak untuk keputusan klinis.
- Tidak mengimplementasikan GRPO maupun randomized smoothing.
- Tidak melatih VLM generatif besar. Model 3 hanya pembanding opsional bila muat di T4.

## 3. Pengguna, use case, dan ruang lingkup

Pengguna utama adalah tim peneliti sendiri. Dosen pengampu dan pembaca laporan memakai hasilnya, dan demo web melayani pembelajar yang ingin melihat efek serangan secara langsung.

| Pengguna | Kebutuhan | Fitur yang melayani |
| --- | --- | --- |
| Tim peneliti (3 mahasiswa) | Menjalankan eksperimen yang bisa diulang dan menghasilkan tabel untuk laporan serta artikel | Pipeline eksperimen, skenario 1–5, tabel hasil |
| Dosen pengampu dan penguji | Menilai kebaruan, kebenaran angka, dan kejujuran perbandingan | Tabel perbandingan dengan penelitian terdahulu, interval kepercayaan, catatan keterbatasan |
| Pembelajar atau pembaca | Memahami bahwa model akurat belum tentu tahan serangan | Demo Gradio dengan disclaimer |

**Use case utama.**

1. Peneliti melatih Model 1, lalu mengukur akurasi bersih dan akurasi di bawah FGSM, PGD, dan AutoAttack.
2. Peneliti mengubah pertanyaan (typo, parafrase, terjemahan Indonesia) dan mengukur penurunan akurasi, sendiri maupun bersama serangan citra.
3. Peneliti melatih Model 2 dengan TRADES dan mengulang semua skenario untuk membandingkan.
4. Pengguna demo mengunggah atau memilih citra, mengajukan pertanyaan, lalu melihat jawaban, confidence, dan perubahan jawaban setelah serangan.

**Dalam lingkup** (batasan masalah proposal, subbab 1.4):

- Dataset hanya VQA-RAD dengan split resmi. Metrik utama pada pertanyaan tertutup (ya/tidak), sedangkan pertanyaan terbuka dilaporkan terpisah.
- Encoder BiomedCLIP, beku pada baseline dan dibuka hanya pada adversarial fine-tuning citra.
- Serangan citra dengan norma L-tak-hingga, ε maksimum 8/255, diukur di ruang piksel [0,1] dengan normalisasi di dalam model.
- Perturbasi teks diberi label "uji robustness", bukan "adversarial", kecuali parafrase dicari secara adversarial lewat query ke model.
- Eksperimen di GPU kelas T4 (Colab atau Kaggle).

**Di luar lingkup:** dataset medis lain, pelatihan VLM generatif besar, penggunaan klinis, GRPO, dan randomized smoothing.

## 4. Requirements fungsional

Sistem terdiri dari 18 kebutuhan fungsional dalam lima kelompok: data, model, serangan, pemeriksaan pertahanan, dan evaluasi serta demo. Kolom sumber menunjuk subbab proposal. Prioritas (Must, Should, Could) adalah usulan untuk disepakati tim.

| ID | Kebutuhan | Sumber | Prioritas |
| --- | --- | --- | --- |
| **Data** | | | |
| FR-01 | Unduh VQA-RAD dari OSF (sumber resmi). Mirror Hugging Face hanya cadangan. Hitung jumlah citra dan pasangan tanya-jawab | 3.2, 3.3 | Must |
| FR-02 | Pisahkan pertanyaan tertutup dan terbuka dengan split resmi tanpa perubahan | 3.2, 1.4 | Must |
| FR-03 | Hitung ulang distribusi kelas ya/tidak pada partisi latih dan uji sebelum membaca akurasi | 2.4 | Must |
| FR-04 | Ubah ukuran citra sesuai BiomedCLIP. Normalisasi dilakukan di dalam model supaya ε terukur di ruang piksel [0,1] | 3.2 | Must |
| **Model** | | | |
| FR-05 | Model 1: encoder citra dan teks BiomedCLIP beku, embedding digabung (konkatenasi), kepala fusi berlapis linear, keluaran kelas jawaban dari kosakata tetap, loss cross-entropy. Hanya kepala fusi yang dilatih, minimal 3 seed | 3.4.1, 2.2.2 | Must |
| FR-06 | Model 2: arsitektur Model 1 dengan encoder citra dibuka, dilatih dengan loss TRADES (β dipilih pada data validasi). Perturbasi hanya pada citra. Hemat memori: encoder teks beku, batch kecil, mixed precision | 3.4.2, 2.2.4 | Must |
| FR-07 | Model 3 (opsional): VLM generatif 2–4 miliar parameter (kelas Qwen3-VL-4B) atau BiomedGPT sebagai pembanding, hanya bila muat di T4 | 3.4.3 | Could |
| **Serangan** | | | |
| FR-08 | Jalankan FGSM, PGD (awal acak, proyeksi ke bola ε), dan AutoAttack (APGD-CE, APGD-T, FAB-T, Square Attack) pada ε = 1, 2, 4, 8 per 255. Tambah titik ε di bawah 1/255 (misalnya 0,5/255) bila Model 1 runtuh pada 1/255 | 3.6, 2.2.3, 2.4 | Must |
| FR-09 | Jalankan satu konfigurasi PGD dengan parameter pelatihan SafeMed-R1: ε = 0,01; α = 0,002; 5 langkah | 1.3, 3.6 | Must |
| FR-10 | Perturbasi teks: typo, parafrase, terjemahan ke bahasa Indonesia. Diberi label "uji robustness" | 1.4, 3.6 | Must |
| FR-11 | Serangan gabungan: perturbasi citra bersama perubahan teks | 3.6 | Must |
| **Pemeriksaan pertahanan** | | | |
| FR-12 | Pemeriksaan gradient masking: (a) serangan black-box tidak boleh mengungguli white-box, dengan Square Attack sebagai serangan tanpa gradien, (b) robust accuracy tidak naik saat ε naik dan tidak melebihi akurasi bersih, (c) satu ε jauh lebih besar, misalnya 64/255, harus membuat akurasi mendekati nol, (d) gandakan iterasi PGD dan lihat apakah keberhasilan serangan naik | 2.2.5 | Must |
| FR-13 | BPDA dan EOT dijalankan hanya bila peneliti menambahkan preprocessing yang tidak diferensiabel atau acak | 2.2.5 | Could |
| **Evaluasi dan demo** | | | |
| FR-14 | Laporkan confusion matrix, accuracy, balanced accuracy, precision, recall, F1, AUC, MCC, robust accuracy RA(ε), ASR(ε), dan false negative rate | 2.4, 3.5 | Must |
| FR-15 | Interval kepercayaan 95% dengan bootstrap persentil pada tingkat citra (1.000 resampel), dari minimal 3 seed | 2.4 | Must |
| FR-16 | Pertanyaan terbuka dinilai dengan exact match dan dilaporkan terpisah | 2.4, 1.4 | Should |
| FR-17 | Demo Gradio: pilih atau unggah citra, ajukan pertanyaan, tampilkan jawaban dan confidence, lalu tampilkan citra serta jawaban sebelum dan sesudah serangan dengan pengatur ε | 1.2, 1.3 | Must |
| FR-18 | Disclaimer "bukan alat diagnosis" selalu terlihat di demo | 1.4, 1.1 | Must |

## 5. Requirements non-fungsional

Batasan utamanya adalah satu GPU kelas T4 dan set uji yang kecil, sehingga komputasi dan statistik menentukan desain.

| ID | Kategori | Kebutuhan | Cara mengukur |
| --- | --- | --- | --- |
| NFR-01 | Komputasi | Semua eksperimen inti berjalan di GPU kelas T4 (Colab atau Kaggle) | Satu notebook selesai tanpa kehabisan memori GPU |
| NFR-02 | Estimasi waktu | Smoke test 20 sampel memperkirakan waktu GPU sebelum eksperimen penuh. AutoAttack bisa dipersempit (subset ε atau subset data) bila waktu tidak cukup | Estimasi waktu dicatat sebelum run penuh |
| NFR-03 | Reproduksibilitas | Seed tetap, versi pustaka dicatat, dan seluruh angka dapat diulang dari kode | Menjalankan ulang menghasilkan angka yang sama dalam toleransi interval kepercayaan |
| NFR-04 | Statistik | Minimal 3 seed. Set uji resmi hanya 451 pasangan, sehingga satu pertanyaan mengubah akurasi sekitar 0,22 poin. Bootstrap mengambil ulang citra, bukan pertanyaan, karena pertanyaan pada citra yang sama saling bergantung | Setiap angka dilaporkan dengan interval 95% |
| NFR-05 | Integritas evaluasi | Hiperparameter seperti β dipilih pada data validasi, bukan set uji | Tidak ada tuning pada split uji |
| NFR-06 | Kejujuran perbandingan | Perbandingan dengan angka SafeMed-R1 diberi label kualitatif, karena dataset, model, dan parameter serangan evaluasinya berbeda | Setiap tabel perbandingan memuat catatan itu |
| NFR-07 | Keamanan penggunaan | Demo ditandai bukan alat diagnosis dan tidak dipakai untuk keputusan klinis | Disclaimer terlihat di setiap layar demo |
| NFR-08 | Sumber data | VQA-RAD diunduh dari sumber resmi di OSF. Mirror pihak ketiga hanya cadangan | Tautan sumber dan tanggal unduh dicatat |
| NFR-09 | Keterbukaan kode | Kode eksperimen dan demo dirapikan agar bisa dibagikan. Paper SafeMed-R1 tidak mencantumkan tautan kode, jadi kelengkapan kode sendiri menjadi nilai tambah | Repositori dengan README dan petunjuk menjalankan |

## 6. Desain eksperimen

Eksperimen berjalan dalam lima skenario yang membandingkan Model 1 (belum dilatih adversarial) dengan Model 2 (TRADES) pada serangan yang sama.

| Skenario | Model | Perlakuan | Keluaran |
| --- | --- | --- | --- |
| 1. Data bersih | Model 1 | Tanpa gangguan | Accuracy, balanced accuracy, precision, recall, F1, AUC, MCC, sensitivitas per kelas |
| 2. Serangan citra | Model 1 | FGSM, PGD, AutoAttack pada ε = 1, 2, 4, 8/255, ditambah satu konfigurasi PGD ala SafeMed-R1 (ε = 0,01; α = 0,002; 5 langkah) | Kurva robust accuracy dan ASR terhadap ε |
| 3. Perturbasi teks | Model 1 | Typo, parafrase, terjemahan ke bahasa Indonesia | Penurunan akurasi per jenis perubahan |
| 4. Gabungan | Model 1 | Serangan citra bersama perubahan teks | Akurasi dan ASR gabungan |
| 5. Pertahanan | Model 2 | Ulangi skenario 2–4, lalu pemeriksaan gradient masking | Perbandingan dengan Model 1, hasil pemeriksaan gradient masking |

**Template hasil utama.** Tabel ini diisi setelah eksperimen, sebagai robust accuracy (%) dengan rata-rata dan interval kepercayaan 95% dari 3 seed.

| Model | Serangan | ε = 1/255 | ε = 2/255 | ε = 4/255 | ε = 8/255 |
| --- | --- | --- | --- | --- | --- |
| Model 1 | FGSM | | | | |
| Model 1 | PGD | | | | |
| Model 1 | AutoAttack | | | | |
| Model 2 | FGSM | | | | |
| Model 2 | PGD | | | | |
| Model 2 | AutoAttack | | | | |

**Aturan membaca hasil.**

- Bila Model 1 sudah runtuh pada ε = 1/255, kurva tidak informatif. Tambahkan ε di bawah 1/255.
- Robust accuracy tidak boleh naik saat ε naik atau melebihi akurasi bersih. Bila itu terjadi, curigai gradient masking.
- Pada pertanyaan ya/tidak, laporkan false negative rate secara terpisah karena kesalahan "tidak" pada temuan yang ada bermakna klinis.
- Pertanyaan terbuka dilaporkan terpisah dengan exact match.

**Catatan desain.** Model 1 memakai encoder beku, sedangkan Model 2 membuka encoder citra dan memakai TRADES. Dua perubahan itu terjadi bersamaan, sehingga kenaikan robustness belum tentu berasal dari TRADES saja. Lihat pertanyaan terbuka di bagian 8 tentang model kontrol.

## 7. Referensi kode

Dari 40 referensi proposal, 11 punya kode atau aset yang bisa dipakai langsung, dan dua referensi utama (SafeMed-R1 dan Fime dkk.) tidak menyediakan kode eksperimen. Tautan diperiksa pada 15 November 2026. Sebagian hanya lewat sumber sekunder atau kutipan dari paper, dan itu ditandai di kolom catatan. Kolom "catatan" memuat hal yang perlu diwaspadai sebelum dipakai.

**A. Kode dan aset dari referensi proposal**

| Ref. | Nama (tautan) | Dipakai untuk | Catatan |
| --- | --- | --- | --- |
| [11] | [BiomedCLIP, kartu model Hugging Face](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | Encoder citra dan teks Model 1 dan Model 2 (FR-05, FR-06) | Lisensi MIT. Dimuat lewat `open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')` dengan konteks teks 256 token. Repositori [BiomedCLIP_data_pipeline](https://github.com/microsoft/BiomedCLIP_data_pipeline) hanya untuk menyiapkan data pralatih, bukan kode pelatihan |
| [26] | [TRADES, yaodongyu/TRADES](https://www.github.com/yaodongyu/TRADES) | Loss TRADES untuk Model 2 (FR-06) | Lisensi MIT, kode resmi ICML 2019. Fungsi `trades_loss()` ada di `trades.py` dengan argumen `beta` (disarankan 1–10), `epsilon`, `step_size`, `num_steps`, dan `distance` (`l_inf` atau `l_2`). Skrip contoh hanya untuk CIFAR-10 dan MNIST, jadi harus diadaptasi ke masukan citra + pertanyaan |
| [25] | [AutoAttack, fra31/auto-attack](https://github.com/fra31/auto-attack) | Serangan AutoAttack (FR-08) | Repositori resmi, 756 bintang, terakhir diperbarui Mei 2024 (menurut [ringkasan repositori](https://relatedrepos.com/gh/fra31/auto-attack)). Lisensi dan perintah instal tidak terverifikasi dari halaman yang bisa dibuka, cek langsung di GitHub |
| [34] | [Obfuscated gradients, anishathalye/obfuscated-gradients](https://github.com/anishathalye/obfuscated-gradients) | Acuan pemeriksaan gradient masking, BPDA dan EOT bila perlu (FR-12, FR-13) | Berisi implementasi serangan pada 7 pertahanan ICLR 2018 (folder `baseline`, `defensegan`, `inputtransformations`, `lid`, `pixeldefend`, `randomization`, `sap`, `thermometer`). README tidak menyebut BPDA atau EOT secara eksplisit dan tidak menampilkan lisensi. Gunakan sebagai referensi bacaan, bukan pustaka |
| [4] | [VQA-RAD di OSF](https://osf.io/89kps/) (lewat kartu [mirror Hugging Face](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)) | Dataset utama (FR-01, FR-02) | Lisensi CC0 1.0. Mirror melaporkan 2.248 pasangan tanya-jawab dan 315 citra (2.244 pasangan pada 314 citra setelah 4 duplikat dibuang). Uji: 451 pasangan, cocok dengan angka di proposal. Mirror pihak ketiga hanya cadangan, unduh dari OSF |
| [8] | [BiomedGPT, taokz/BiomedGPT](https://github.com/taokz/BiomedGPT) | Pembanding generatif alternatif, Model 3 (FR-07) | Skrip VQA-RAD tersedia: `scripts/vqa/train_vqa_rad_beam.sh`, `evaluate_vqa_rad_beam.sh`. Lisensi Apache-2.0 tetapi README membatasi untuk riset akademik. Lingkungannya lama (conda Python 3.7.4, pip 21.2.4), kemungkinan sulit dijalankan di Colab modern. Ada bobot kompatibel Hugging Face |
| [6] | [PubMedCLIP, sarahESL/PubMedCLIP](https://github.com/sarahESL/PubMedCLIP) | Referensi arsitektur CLIP + VQA-RAD, pembanding opsional | Tautan kode dikutip dari Subbab 2.1 paper ([ar5iv](https://ar5iv.labs.arxiv.org/html/2112.13906)). Bobot ada di [Hugging Face](https://huggingface.co/sarahESL/PubMedCLIP) (lisensi "cc", varian tidak disebut). Halaman repositori GitHub itu sendiri tidak berhasil dibuka saat pengecekan |
| [20] | [SURE-VQA, IML-DKFZ/sure-vqa](https://github.com/IML-DKFZ/sure-vqa) | Acuan desain evaluasi ketahanan VQA medis | Tautan dikutip dari abstrak paper ([arXiv](https://arxiv.org/html/2411.19688v2)). Datasetnya SLAKE, OVQA, MIMIC-CXR-VQA, **bukan VQA-RAD**, dan fokusnya pergeseran distribusi, bukan serangan gradien |
| [5] | [vqa_medical, vuhoangminh/vqa_medical](https://www.github.com/vuhoangminh/vqa_medical) | Referensi model QCR | Dibangun di atas MUTAN. README tidak menyebut VQA-RAD dan tidak menampilkan lisensi. Nilai praktisnya rendah untuk proposal ini |
| [21] | [prompt_injection_attacks, katherlab](https://github.com/katherlab/prompt_injection_attacks) | Hanya bahan bacaan | Lisensi CC BY-NC-SA 4.0. Skripnya memanggil API model berbayar (Claude, GPT-4, Reka-Core) dan memakai 90 citra. Tidak relevan untuk serangan gradien |
| [19] | [MedMNIST, MedMNIST/MedMNIST](https://github.com/MedMNIST/MedMNIST) | Hanya bila ingin mereplikasi Fime dkk. | Dataset yang dipakai paper Fime. `pip install medmnist`, kode Apache-2.0, dataset CC BY 4.0 (kecuali DermaMNIST, CC BY-NC 4.0). Tidak dibutuhkan untuk VQA-RAD |

**B. Referensi yang tidak menyediakan kode**

| Ref. | Paper | Temuan |
| --- | --- | --- |
| [22] | SafeMed-R1 | Tidak ada tautan kode di naskah PDF. Parameter serangan saat evaluasi tidak dinyatakan, sehingga hasilnya tidak bisa direproduksi dari paper |
| [19] | Fime dkk. | Artikel hanya memuat pernyataan ketersediaan data (MedMNIST). Tidak ada pernyataan ketersediaan kode yang ditemukan di teks yang dibaca |
| [7] | Bazi dkk. | Pernyataan data: "available on request". Tidak ada tautan kode |

**C. Implementasi pihak ketiga dari metode yang dikutip (tidak ada di daftar pustaka)**

| Nama (tautan) | Cakupan | Catatan |
| --- | --- | --- |
| [torchattacks, Harry24k](https://pypi.org/project/torchattacks/3.0.0) | FGSM, PGD, AutoAttack, APGD, APGDT, FAB, Square, TPGD | Lisensi MIT. `pip install torchattacks`. Mensyaratkan citra di [0,1] dan normalisasi di dalam model, persis seperti rancangan proposal (FR-04). Tidak dikutip di proposal, jadi bila dipakai harus ditambahkan ke daftar pustaka |

**Catatan integrasi (saran, belum diuji).** Pustaka serangan dan `trades_loss()` mengharapkan model yang memetakan satu tensor citra ke logit. Model VQA menerima citra dan pertanyaan, sehingga perlu pembungkus yang menahan embedding pertanyaan tetap saat citra diserang. Untuk serangan gabungan (FR-11), embedding pertanyaan dihitung ulang dari teks yang sudah diubah sebelum pembungkus dibuat.

**Belum diperiksa:** repositori resmi CLIP [10], PubMedBERT [30], PLIP [31], serta kode untuk [14], [15], [16], [33], dan FGSM serta PGD asli [23], [24]. Repositori `MadryLab/cifar10_challenge` tidak muncul di hasil pencarian, jadi tidak dicantumkan.

## 8. Risiko, lini masa, dan pertanyaan terbuka

Risiko terbesar adalah waktu: seluruh eksperimen harus selesai dalam empat minggu di satu GPU T4.

### Risiko dan mitigasi

| Risiko | Dampak | Mitigasi |
| --- | --- | --- |
| AutoAttack terlalu lambat di T4 | Eksperimen tidak selesai sebelum penulisan | Smoke test 20 sampel (NFR-02). Persempit ε atau subset data bila perlu |
| Model 1 runtuh sudah pada ε = 1/255 | Kurva tidak informatif | Tambah titik ε di bawah 1/255, misalnya 0,5/255 (FR-08) |
| Set uji kecil (451 pasangan) | Angka bergoyang, satu pertanyaan = 0,22 poin | 3 seed dan bootstrap pada tingkat citra (FR-15) |
| Robustness tampak naik karena gradient masking | Klaim pertahanan keliru | Empat pemeriksaan pada FR-12 |
| Model 2 kehabisan memori GPU karena encoder dibuka | Pelatihan TRADES gagal | Encoder teks beku, batch kecil, mixed precision (FR-06) |
| Kode referensi tidak langsung cocok (TRADES hanya contoh CIFAR/MNIST, BiomedGPT berlingkungan lama) | Waktu integrasi membengkak | Adaptasi `trades_loss()` dengan pembungkus model. Model 3 tetap opsional |
| Angka SafeMed-R1 dibaca sebagai target yang harus dikalahkan | Kesimpulan menyesatkan | Label "kualitatif" di setiap tabel perbandingan (NFR-06) |

### Lini masa

Jadwal mengikuti Tabel 3.5 proposal, dari minggu 7 sampai 14. Eksperimen harus selesai di akhir minggu 11, lalu penulisan dimulai.

| Kegiatan | Minggu |
| --- | --- |
| Studi literatur, referensi utama, dataset | 7 |
| Penulisan proposal | 7 |
| Implementasi skenario pengujian | 8–11 |
| Penulisan artikel publikasi | 12–13 |
| Penulisan laporan | 12–13 |
| Presentasi dan demo | 14 |

### Pertanyaan terbuka dan perbaikan proposal

1. **Tabel 3.4 salah melabeli SafeMed-R1.** Angka 95,43% (bersih) dan 26,07% (PGD) milik model Clean FT di Paper 1, bukan SafeMed-R1. Angka SafeMed-R1 Instruct adalah 92,77% (bersih) dan 82,27% (PGD), dan varian Think 84,45% (PGD). Bab 1.1 proposal sudah menulisnya dengan benar, sehingga tabel dan teks saling bertentangan.
2. **Perlukah model kontrol?** Model 2 membuka encoder sekaligus memakai TRADES. Menambah satu model fine-tune encoder terbuka tanpa TRADES memisahkan efek keduanya, dengan biaya satu putaran pelatihan tambahan.
3. **Jumlah pasangan VQA-RAD.** Bab 2.1 proposal menulis 3.515 pasangan tanya-jawab (angka makalah Lau dkk.), sedangkan mirror Hugging Face melaporkan 2.248 pada rilis publik. Konfirmasi dari berkas OSF saat mengunduh dan perbarui teks proposal.
4. **Salah ketik di Tabel 3.4.** "iomedGPT-B" seharusnya "BiomedGPT-B".
5. **Angka pembanding lain di Tabel 3.4** (Bazi dkk. 77,27%; BiomedGPT-B 73,2%; BiomedCLIP 72,7%) belum diverifikasi terhadap paper aslinya.
6. **Target numerik K1–K3** pada bagian 2 masih usulan. Perlu disepakati tim dan dosen.
7. **Pustaka serangan.** Pakai AutoAttack resmi dan implementasi sendiri untuk FGSM dan PGD, atau satu paket seperti torchattacks? Paket itu lebih cepat dipasang tetapi harus ditambahkan ke daftar pustaka.

## 9. Checkpoint dan aturan menjalankan program

Program dijalankan berurutan mengikuti 11 tahap rancangan penelitian (proposal, subbab 3.2), dan tahap berikutnya baru dimulai setelah checkpoint sebelumnya lulus. Bagian ini mengatur tiga hal: checkpoint proyek, checkpoint model untuk menyimpan dan melanjutkan pelatihan, serta aturan eksekusi. Pembagian minggu per checkpoint adalah usulan, karena proposal hanya menetapkan implementasi pada minggu 8–11.

### 9.1 Checkpoint proyek

| CP | Akhir minggu | Tahap proposal | Syarat lulus | Bila tidak lulus |
| --- | --- | --- | --- | --- |
| CP0 | 7 | 1–3: unduh VQA-RAD, pisah pertanyaan tertutup dan terbuka, ubah ukuran dan normalisasi | Jumlah citra dan pasangan terhitung. Split resmi terverifikasi (uji 451 pasangan). Distribusi kelas ya/tidak dihitung. Smoke test 20 sampel memberi estimasi waktu GPU | Jangan mulai melatih. Selesaikan data dulu, karena semua hasil bergantung padanya |
| CP1 | 8 | 4–5: latih Model 1 (3 seed) dan ukur performa bersih | Skenario 1 selesai dengan interval kepercayaan. Akurasi bersih lebih baik dari menebak kelas mayoritas | Periksa kepala fusi, kosakata jawaban, dan learning rate. Jangan menyerang model yang belum belajar |
| CP2 | 9 | 6: FGSM, PGD, AutoAttack, dan satu konfigurasi PGD ala SafeMed-R1 | Skenario 2 selesai. Semua sanity check pada 9.3 lulus. Titik ε di bawah 1/255 ditambahkan bila Model 1 runtuh pada 1/255 | Perbaiki implementasi serangan. Bila AutoAttack terlalu lambat, pakai fallback pada NFR-02 |
| CP3 | 10 | 7–8: perturbasi teks dan serangan gabungan | Skenario 3 dan 4 selesai. Perubahan teks berlabel "uji robustness" | Pangkas jenis perubahan teks, bukan jumlah seed |
| CP4 | 11 | 9–10: TRADES dan pemeriksaan gradient masking | Model 2 terlatih (3 seed). Skenario 5 selesai. Empat pemeriksaan FR-12 lulus. Ini batas akhir eksperimen | Tidak ada eksperimen baru setelah minggu 11. Laporkan apa yang selesai dan catat sisanya sebagai keterbatasan |
| CP5 | 13 | Penulisan artikel dan laporan | Tabel dan gambar final dibekukan sebelum menulis. Setiap angka di naskah bisa ditelusuri ke berkas hasil | Jangan mengganti angka di naskah secara manual. Jalankan ulang dari berkas hasil |
| CP6 | 14 | 11: demo Gradio, presentasi | Demo berjalan di Colab atau Kaggle dengan disclaimer terlihat dan model final dimuat dari checkpoint | Demo cukup menampilkan satu model dan satu serangan. Fitur tambahan dipotong |

### 9.2 Checkpoint model dan hasil

Sesi Colab dan Kaggle bisa terputus kapan saja, jadi pelatihan dan serangan harus bisa dilanjutkan tanpa mengulang dari awal. Semua isi tabel ini adalah saran, bukan bagian proposal.

| Yang disimpan | Kapan | Berkas | Catatan |
| --- | --- | --- | --- |
| Bobot Model 1 (hanya kepala fusi) | Tiap akhir epoch, plus yang terbaik di validasi | `model1_seed{s}_ep{e}.pt`, `model1_seed{s}_best.pt` | Encoder BiomedCLIP beku tidak disimpan ulang. Cukup catat id `hf-hub` dan versi `open_clip` |
| Bobot Model 2 (encoder citra + kepala fusi) | Tiap epoch, karena TRADES lambat | `model2_seed{s}_ep{e}.pt`, `model2_seed{s}_best.pt` | Sertakan state optimizer, epoch, seed, β, ε, dan jumlah langkah serangan saat pelatihan |
| Hasil serangan | Setelah tiap kombinasi (model, serangan, ε, seed) selesai | `results/{model}_{serangan}_eps{k}_seed{s}.json` | Bila berkas sudah ada, kombinasi itu dilewati. Inilah yang membuat run terputus bisa dilanjutkan |
| Prediksi per sampel | Bersama hasil serangan | ID pertanyaan, jawaban benar, prediksi, confidence (bersih dan terserang) | Diperlukan agar bootstrap pada tingkat citra dan AUC bisa dihitung ulang tanpa menjalankan serangan lagi |
| Konfigurasi dan lingkungan | Awal tiap run | `config.yaml`, versi pustaka, hash kode | Memenuhi NFR-03 (reproduksibilitas) |

Aturan penyimpanan:

- Simpan ke penyimpanan persisten (Google Drive atau output Kaggle), bukan ke disk sesi.
- Pilih checkpoint terbaik dari data validasi, bukan dari set uji. Menurut kartu mirror Hugging Face, split resmi VQA-RAD hanya memuat latih dan uji, sehingga validasi harus dipisah dari data latih. Pisahkan per citra, karena pertanyaan pada citra yang sama saling bergantung.
- Jangan menimpa checkpoint lama. Beri nomor epoch atau cap waktu pada nama berkas.

### 9.3 Aturan menjalankan program

Kolom "Asal" menunjukkan apakah aturan berasal dari proposal atau saran tambahan.

| ID | Aturan | Asal |
| --- | --- | --- |
| R-01 | Semua parameter (ε, α, langkah serangan, β, seed, batch, learning rate) berada di satu berkas konfigurasi, tidak ditulis langsung di sel notebook. Berkas itu disimpan bersama hasil | Saran |
| R-02 | Seed diset untuk Python, NumPy, dan PyTorch di setiap run, lalu dicatat. Daftar seed (minimal 3) ditetapkan sebelum eksperimen dimulai | Proposal 2.4, 3.5 (minimal 3 seed). Detail: saran |
| R-03 | Model dalam mode eval saat menyerang dan mengevaluasi. Mode train hanya saat melatih | Saran |
| R-04 | Citra berada di rentang [0,1], normalisasi dilakukan di dalam model, dan hasil perturbasi dipotong kembali ke [0,1] | Proposal 1.4, 2.2.3 |
| R-05 | ε ditulis sebagai k/255 di semua tabel dan berkas hasil. Konfigurasi SafeMed-R1 (ε = 0,01) juga ditulis sebagai ≈2,55/255 | Proposal 2.2.3 |
| R-06 | Tidak ada tuning pada set uji. β, learning rate, dan pilihan checkpoint ditentukan di validasi. Set uji dibuka hanya untuk konfigurasi final | Proposal 3.4.2 (β di validasi). Detail: saran |
| R-07 | Smoke test 20 sampel dijalankan sebelum run penuh dan setiap kali kode serangan atau model berubah | Proposal 3.2. Perluasan: saran |
| R-08 | Sanity check otomatis sebelum hasil diterima: (a) robust accuracy pada ε = 0 sama dengan akurasi bersih, (b) tidak melebihi akurasi bersih, (c) tidak naik saat ε naik, (d) ε jauh lebih besar seperti 64/255 menurunkan akurasi mendekati nol, (e) menggandakan iterasi PGD tidak menaikkan keberhasilan serangan secara berarti. Bila (e) gagal, serangan awal belum cukup kuat | Proposal 2.2.5 untuk (b)–(e). (a): saran |
| R-09 | Hasil yang gagal sanity check tidak masuk tabel final. Simpan di folder terpisah dan catat alasannya, jangan dihapus | Saran |
| R-10 | Bila sesi terputus, lanjutkan dari checkpoint atau berkas hasil terakhir. Jangan mulai dari awal dan jangan menimpa hasil yang sudah ada | Saran |
| R-11 | Perubahan teks diberi label "uji robustness", kecuali parafrase dicari secara adversarial lewat query ke model | Proposal 1.4 |
| R-12 | Setiap run mencatat waktu GPU, versi pustaka (torch, open_clip, pustaka serangan), hash kode, seed, dan konfigurasi | Saran (mendukung NFR-03) |
| R-13 | Demo hanya memuat checkpoint final, tidak melatih atau mengubah model, dan menampilkan disclaimer "bukan alat diagnosis" | Proposal 1.4 |
