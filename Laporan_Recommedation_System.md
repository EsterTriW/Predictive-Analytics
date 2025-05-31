# Laporan Proyek Recommendation System
## Judul proyek : Implementasi Sistem Rekomendasi Buku Berbasis User-Based Collaborative Filtering dan Content-Based Filtering
## Ditulis oleh : Ester Tri Wahyuningsih -  MC002D5X0841

## Project Overview
### Latar Belakang
Di era digital saat ini, kemajuan teknologi informasi telah mengubah cara masyarakat dalam mengakses dan mengonsumsi konten literasi. Platform seperti Goodreads, Amazon, dan Google Books menyediakan jutaan judul buku dari berbagai kategori yang dapat diakses kapan saja. Namun, tingginya jumlah pilihan justru menimbulkan masalah baru, yaitu information overload, di mana pengguna kesulitan menemukan buku yang sesuai dengan minat dan kebutuhannya secara efisien (Resnick & Varian, 1997).

Salah satu solusi untuk permasalahan ini adalah penerapan sistem rekomendasi (recommender system). Sistem ini berperan penting dalam menyaring informasi dan memberikan rekomendasi personal berdasarkan preferensi atau perilaku pengguna. Tidak hanya di bidang e-commerce dan layanan hiburan seperti Netflix dan Spotify, sistem rekomendasi juga memiliki potensi besar di sektor literasi digital untuk meningkatkan keterlibatan dan kepuasan pengguna (Ricci et al., 2015).

Meski demikian, tantangan utama dalam pengembangan sistem rekomendasi adalah sparsity problem, yaitu kondisi di mana data interaksi pengguna dan buku cenderung jarang (Jarada et al., 2022). Selain itu, menentukan metode rekomendasi yang paling efektif untuk konteks literasi digital juga menjadi persoalan yang menarik untuk diteliti.

Oleh karena itu, diperlukan pengembangan sistem rekomendasi buku yang efektif dan adaptif dengan pendekatan yang sesuai. Proyek ini bertujuan untuk membangun dan membandingkan dua metode rekomendasi populer, yaitu User-Based Collaborative Filtering dan Content-Based Filtering, guna mengatasi masalah sparsity serta meningkatkan relevansi rekomendasi bagi pengguna platform literasi digital.

### Tujuan Proyek
Proyek ini penting dilakukan untuk 
- Mengembangkan sistem rekomendasi buku berbasis machine learning dengan menerapkan User-Based Collaborative Filtering dan Content-Based Filtering untuk memberikan rekomendasi personal yang relevan bagi pengguna.
- Mengatasi permasalahan sparsity pada data rekomendasi dengan melakukan filtering terhadap pengguna dan buku aktif.
- Membandingkan performa kedua metode rekomendasi menggunakan metrik evaluasi berbasis Precision@K dan Recall@K guna menentukan pendekatan yang paling optimal untuk diterapkan di platform literasi digital.

---

## Business Understanding

### Problem Statements
Beberapa pertanyaan penting yang akan dijawab melalui proyek ini:
- Bagaimana cara membangun sistem rekomendasi buku yang dapat memberikan rekomendasi personalisasi sesuai minat pengguna?
- Metode rekomendasi mana yang memiliki performa lebih baik dalam memberikan rekomendasi top-N buku?
- Bagaimana cara menangani permasalahan data yang sparse dan cold-start user di sistem rekomendasi?

### Goals
Dengan menjawab problem statements di atas, proyek menetapkan target berikut:
- Mengembangkan sistem rekomendasi buku berbasis User-Based Collaborative Filtering dan Content-Based Filtering.
- Melakukan evaluasi performa model menggunakan metrik Precision@K dan Recall@K.
- Menangani permasalahan data sparse dengan melakukan filter pengguna dan buku aktif.

### Solution Statement
Solusi yang dipilih meliputi:
1. Content-Based Filtering (CBF)
- Metode: Menggunakan metadata buku seperti judul, genre, dan deskripsi yang diolah dengan TF-IDF Vectorizer. Kemudian dihitung cosine similarity antar buku untuk merekomendasikan buku dengan konten paling mirip.
- Output: Top‑N daftar buku paling mirip dengan buku yang pernah disukai pengguna.

2. User-Based Collaborative Filtering (UBCF)
- Metode: Menghitung cosine similarity antar pengguna berdasarkan riwayat rating. Sistem merekomendasikan buku yang disukai pengguna-pengguna dengan pola preferensi serupa.
- Output: Top‑N buku berdasarkan prediksi preferensi dari pengguna yang mirip.

Strategi tambahan lainnya :
- Menangani data sparse dengan memfilter pengguna aktif dan buku populer berdasarkan jumlah rating minimum.
- Mengevaluasi hasil rekomendasi menggunakan metrik Precision@5 dan Recall@5.
- Membandingkan hasil performa kedua metode untuk menentukan pendekatan yang paling sesuai.

---

## Data Understanding

### 1. Sumber & Link Dataset  
Dataset yang digunakan adalah **Book-Crossing Dataset**, tersedia secara publik di:  
- [Kaggle](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)   

Dataset ini berisi file 'Ratings', 'Users' dan 'Books'

### 2. Ukuran & Struktur Data  
Setelah proses pra-pemrosesan, data terdiri dari tiga komponen utama:

| Dataset  | Jumlah Baris | Kolom                                                        | Keterangan                                  |
|----------|--------------|--------------------------------------------------------------|---------------------------------------------|
| Ratings  | 1149780    | `UserID`, `ISBN`, `BookRating`                               | Rating yang diberikan pengguna terhadap buku |
| Users    | 278858     | `UserID`, `Location`, `Age`                                  | Data demografi pengguna                      |
| Books    | 271360      | `ISBN`, `Book-Title`, `Book-Author`, `Year-Of-Publication`, `Publisher`, `Image-URL-S`, `Image-URL-M`, `Image-URL-L` | Metadata buku                               |

### 3. Penjelasan Fitur  

**Ratings:**  
- `UserID` (integer): ID unik pengguna  
- `ISBN` (string): Kode unik buku  
- `BookRating` (integer): Skor rating dari 1 hingga 10  

**Users:**  
- `UserID` (integer)  
- `Location` (string): Kota atau negara pengguna  
- `Age` (float): Usia pengguna  

**Books:**  
- `ISBN` (string) : Kode unik buku (International Standard Book Number)
- `Book-Title` (string): Judul buku
- `Book-Author` (string): Nama penulis buku
- `Year-Of-Publication` (string): Tahun terbit buku
- `Publisher` (string): Nama penerbit
- `Image-URL-S` (string): Link ke thumbnail (ukuran kecil) cover buku
- `Image-URL-M` (string): Link ke gambar cover buku ukuran sedang
- `Image-URL-L` (string): Link ke gambar cover buku ukuran besar


### 4. Missing Values dan Data Duplikat

**Missing Values:**

- **Ratings**
  - `User-ID`        : 0
  - `ISBN`           : 0
  - `Book-Rating`    : 0

- **Users**
  - `User-ID`        : 0
  - `Location`       : 0
  - `Age`            : 110.762

- **Books**
  - `ISBN`                   : 0
  - `Book-Title`             : 0
  - `Book-Author`            : 2
  - `Year-Of-Publication`    : 0
  - `Publisher`              : 2
  - `Image-URL-S`            : 0
  - `Image-URL-M`            : 0
  - `Image-URL-L`            : 3


### 5. Statistik Deskriptif

#### Statistik Deskriptif **Users**

| Statistik     | Value         |
|---------------|---------------|
| Count         | 278.858 users |
| Mean Age      | 34.75         |
| Std Age       | 14.43         |
| Range Age     | 0 – 244       |

#### Statistik Deskriptif **Books**

| Statistik               | Value                    |
|-------------------------|--------------------------|
| Count                   | 271.360 books            |
| Unique Titles           | 242.135                  |
| Unique Authors          | 102.022                  |
| Most Frequent Author    | *Agatha Christie* (632 books) |
| Range Year Published    | ‘-1’ – ‘2020’            |

#### Statistik Deskriptif **Ratings**

| Statistik       | Value             |
|-----------------|-------------------|
| Count           | 1.149.780 ratings |
| Mean Rating     | 2.87              |
| Std Rating      | 3.85              |
| Range Rating    | 0 – 10            |


---

## Exploratory Data Analysis
Untuk memahami karakteristik awal dataset, mengenali pola, outlier, dan distribusi nilai yang ada sebelum masuk ke tahap preprocessing dan modeling Langkah-langkah EDA yang digunakan:
- Menghitung jumlah pengguna dan buku (untuk mengetahui total unique user dan buku dalam dataset.)
  Hasil :
  * Total users : 105283
  * Total items books : 340556
    
- Distribusi rating buku (untuk melihat persebaran nilai rating agar mengetahui apakah data seimbang atau skewed)
  * Rating 0 mendominasi jumlah (lebih dari 700.000 data!):
    Ini menunjukkan banyak pengguna memberi rating 0, yang kemungkinan besar berarti “belum menilai” atau “tidak tertarik”, bukan penilaian kualitas. Ini adalah noise dan      tidak informatif untuk sistem rekomendasi akan dihapus dari data sebelum modeling.
  * Distribusi rating 1–10 cukup normal:
    Nilai rating paling sering diberikan adalah 8, diikuti 7, 10, dan 5.
    
- Menampilkan 10 pengguna paling aktif (untuk mengidentifikasi user yang paling sering memberi rating, untuk filtering dan analisis behavior.)
  * Pengguna dengan rating terbanyak ada pada user-id 11676 dengan 13602 rating.
  
- Menampilkan 10 buku paling sering dirating (untuk mengetahui buku apa yang paling populer di dataset.)
  * Buku paling sering dirating (paling populer) ada pada ISBN 0971880107 dengan 2502 rating.
  
- Distribusi buku berdasarkan tahun terbit (untuk mengecek apakah ada data yang tidak valid dan melihat tren jumlah buku per tahun.)
  * Terjadi peningkatan signifikan jumlah buku yang diterbitkan dari tahun 1986 hingga puncaknya sekitar tahun 2002. Namun, ada pola anomali di tahun 1999. Jumlah publikasi tahun 1999 jauh lebih rendah dibanding 1998 dan 2000. Hal tersebut disebabkan oleh kesalahan input data atau missing data di tahun tesebeut
  * Tahun 2000–2002 merupakan periode dengan jumlah publikasi tertinggi.
  * Setelah tahun 2002, jumlah publikasi menurun drastis (mungkin karena keterbatasan data atau belum ter-update).
    
- Top 20 Publisher dan Author (untuk mengidentifikasi penerbit dan penulis yang paling produktif.)
  * Harlequin, Silhouette, dan Pocket adalah tiga penerbit dengan jumlah buku terbanyak.
  * Agatha Christie, William Shakespeare, dan Stephen King merupakan penulis dengan jumlah buku terbanyak.

---
## Data Preparation

Tujuan Data Preparation:
Membersihkan dan merapikan data agar siap dipakai untuk membangun model rekomendasi, mengurangi noise, menangani missing value, dan memfilter data yang relevan untuk mengurangi sparsity matrix.

Langkah-langkah Data Preparation:
- Menangani missing value dan data duplikat
  * Mengisi nilai kosong di kolom Book-Author dan Publisher dengan label default.
  * Menghapus kolom Image-URL-S, Image-URL-M, dan Image-URL-L yang tidak relevan dan tidak dipakai.
  * Menghapus kolom Age dari dataset Users karena memiliki banyak nilai anomali dan tidak digunakan dalam sistem rekomendasi ini.
  * Menghapus rating 0 (Rating 0 tidak memberikan informasi preferensi.)
  * Buku dengan judul sama dihapus duplikatnya untuk menjaga konsistensi.

- Filter User dan Item
  * Filter pengguna aktif: Hanya mengambil user yang memberi lebih dari 50 rating, agar data lebih stabil dan tidak terlalu sparse.
  * Filter buku populer: Mengambil 2000 buku dengan jumlah rating terbanyak, supaya sistem fokus pada buku yang paling banyak dikenal dan dinilai.
  * Data buku dan user kemudian disesuaikan (merge) berdasarkan hasil filter dari dataset rating yang digunakan, agar hanya data yang relevan yang masuk ke proses modeling.
    
- Ekstraksi Fitur untuk Content-Based Filtering (CBF)
  Untuk pendekatan Content-Based Filtering, dilakukan ekstraksi fitur sebagai berikut:
  * Fitur gabungan dibuat dari kolom Book-Title, Book-Author, dan Publisher.
  * Digunakan teknik TF-IDF Vectorization (Term Frequency-Inverse Document Frequency), yaitu metode representasi teks menjadi vektor numerik:
    ** TF (Term Frequency): Mengukur seberapa sering kata muncul dalam satu buku.
    ** IDF (Inverse Document Frequency): Mengukur keunikan kata di seluruh korpus buku.
* Dengan TF-IDF, kata-kata yang umum di semua buku (seperti "the", "book", "author") akan memiliki bobot rendah, sementara kata-kata yang unik akan diberi bobot lebih tinggi.
* Hasil dari proses ini adalah matriks fitur vektor yang digunakan untuk menghitung kemiripan antar buku berdasarkan konten.

- Pembagian Data (Train-Test Split)untuk evaluasi
* Dataset hasil preprocessing dibagi menjadi dua: 80% untuk data latih dan 20% untuk data uji
* Pembagian ini penting untuk mengevaluasi performa sistem rekomendasi secara adil dan menghindari overfitting.
  
---

## Model & Result
Proyek ini membangun sistem rekomendasi buku dengan dua pendekatan:
1. User-Based Collaborative Filtering (User-Based CF): merekomendasikan buku berdasarkan kesamaan preferensi antar pengguna.
2. Content-Based Filtering (CBF): merekomendasikan buku berdasarkan kemiripan konten buku (judul, penulis, dan penerbit).

Dataset yang digunakan adalah data rating buku yang telah difilter, dengan rating skala 0–10.

### 1. User-Based Collaborative Filtering (User-Based CF)
Definisi:
Merekomendasikan buku berdasarkan kesamaan antar pengguna. Jika dua pengguna memiliki pola rating yang mirip, buku yang disukai pengguna lain bisa direkomendasikan ke pengguna target.

Cara kerja User-Based Collaborative Filtering :
- Split data menjadi train dan test.
- Bentuk rating matrix (users × books).
- Hitung cosine similarity antar pengguna.
- Untuk setiap user:
  * Cari pengguna serupa.
  * Hitung skor prediksi untuk buku yang belum pernah dirating.
  * Rekomendasikan buku dengan skor prediksi tertinggi.

Parameter Utama
- rating_matrix_train → matrix rating user terhadap buku (sparse)
- user_similarity_df_train → similarity matrix antar user
- top_n → jumlah buku yang direkomendasikan (default 5)
- k (di evaluasi) → jumlah rekomendasi saat evaluasi precision@k, recall@k
- max_users → jumlah user untuk proses evaluasi

Kelebihan:
- Sederhana, mudah diimplementasikan.
- Bisa menangkap pola rating antar pengguna.
- Dapat mengeksplorasi buku di luar preferensi konten pengguna.
- Tidak memerlukan metadata buku.

Kekurangan:
- Cold-start problem untuk user baru.
- Sparse data problem jika banyak rating kosong.
- Komputasi similarity bisa mahal di dataset besar.
- Memerlukan data rating yang cukup.
- Cold-start problem untuk pengguna/item baru.
    
### 2. Content-Based Filtering (CBF)
Definisi:
Merekomendasikan buku berdasarkan kemiripan konten (judul, penulis, publisher) dengan buku-buku yang pernah disukai user.

Cara kerja Content-Based Filtering (CBF) :
- Hitung kemiripan antar buku menggunakan cosine similarity.
- Untuk user, ambil buku dengan rating ≥ 7 → cari buku mirip berdasarkan similarity.
- Rekomendasikan buku dengan skor kemiripan tertinggi.

Parameter Utama
- combined_features → gabungan fitur buku (title + author + publisher)
- cosine_sim_books → similarity matrix antar buku
- top_n → jumlah buku yang direkomendasikan (default 5)
- k, max_users → parameter saat evaluasi

Kelebihan:
- Tidak tergantung data user lain.
- Bisa merekomendasikan item baru yang belum dirating banyak orang.
- Rekomendasi spesifik sesuai preferensi konten.

Kekurangan:
- Terbatas pada item yang mirip secara konten.
- Cold-start problem untuk item baru tanpa deskripsi lengkap.

---

## Hasil Rekomendasi

### User-Based Collaborative Filtering

**Rekomendasi untuk User ID `4017`:**

| Judul Buku                                                      | ISBN        | Skor Prediksi |
|:----------------------------------------------------------------|:------------|:---------------|
| Charlotte's Web (Trophy Newbery)                                | 0064400557  | 10.00          |
| The Lost World                                                   | 034540288X  | 10.00          |
| The Second Summer of the Sisterhood                              | 0385729340  | 10.00          |
| Charlotte's Web                                                  | 059030271X  | 10.00          |
| A Confederacy of Dunces (Evergreen Book)                         | 0802130208  | 10.00          |

---

### Content-Based Filtering

#### Rekomendasi Berdasarkan ISBN `0440234743`:

| Judul Buku                           | ISBN        | Skor Kemiripan |
|:-------------------------------------|:------------|:----------------|
| The Rainmaker                        | 044022165X  | 0.53             |
| A Time to Kill                       | 0440211727  | 0.49             |
| The Summons                          | 0440241073  | 0.47             |
| The Street Lawyer                    | 0440225701  | 0.45             |
| The Runaway Jury                     | 0440221471  | 0.44             |

---

#### Rekomendasi Berdasarkan User ID `4017`:

| Judul Buku                                                                 | ISBN        | Skor Kemiripan |
|:---------------------------------------------------------------------------|:------------|:----------------|
| Dr. Death: A Novel                                                         | 0679459618  | 0.18             |
| Portrait in Death                                                          | 0425189031  | 0.15             |
| Naked in Death                                                             | 0425148297  | 0.15             |
| Prodigal Summer: A Novel                                                   | 0060959037  | 0.15             |
| The Professor and the Madman: A Tale of Murder, Insanity, and the Making of The Oxford English Dictionary | 006099486X  | 0.51             |

---

## Evaluasi Sistem Rekomendasi
Pada proyek ini, evaluasi performa sistem rekomendasi dilakukan menggunakan dua metrik utama, yaitu Precision@5 dan Recall@5.

Dalam sistem rekomendasi, sangat penting untuk mengetahui seberapa relevan item yang direkomendasikan kepada pengguna. Oleh karena itu, precision dan recall digunakan untuk mengukur kualitas hasil rekomendasi:

- Precision@k digunakan untuk mengetahui proporsi item yang relevan dari total item yang direkomendasikan di posisi teratas.
  * Rumus :  (Jumlah item relevan di top-k rekomendasi) / k
- Recall@k digunakan untuk mengetahui seberapa banyak item relevan yang berhasil ditemukan dari seluruh item relevan yang tersedia.
  * Rumus : (Jumlah item relevan di top-k rekomendasi) / (Jumlah total item relevan untuk user)

Kedua metrik ini dipilih karena dapat memberikan gambaran menyeluruh mengenai kualitas rekomendasi yang diberikan, khususnya ketika jumlah data relevan dalam dataset tergolong sedikit (data sparsity).

---

### Hasil Evaluasi  

Berikut hasil evaluasi yang diperoleh dari sistem rekomendasi yang dikembangkan:

| Metode               | Precision@5 | Recall@5 |
|:--------------------|:------------|:----------|
| User-Based CF        | 0.009        | 0.002     |
| Content-Based CF     | 0.032        | 0.108     |

---

### Analisis Hasil  

Berdasarkan hasil evaluasi di atas, dapat disimpulkan bahwa nilai precision dan recall dari kedua metode masih tergolong rendah. Beberapa faktor yang menyebabkan hal ini antara lain:

1. **Data Sparsity**  
   Dataset Book-Crossing memiliki tingkat sparsity yang tinggi, di mana sebagian besar pengguna hanya memberikan sedikit rating. Hal ini menyebabkan overlap antara preferensi pengguna sangat kecil, sehingga metode User-Based Collaborative Filtering sulit menemukan user yang benar-benar mirip.

2. **Cold Start Problem**  
   Terdapat banyak buku dan pengguna dengan jumlah rating yang minim. Kondisi ini menyulitkan sistem dalam menghasilkan rekomendasi yang akurat karena kurangnya data historis.

3. **Top-K Recommendation**  
   Karena sistem hanya merekomendasikan 5 item teratas dari sekian banyak buku, peluang item relevan untuk muncul di posisi tersebut menjadi sangat kecil.

4. **Keterbatasan Metadata**  
   Pada metode Content-Based, informasi buku yang digunakan hanya terbatas pada Title dan Author. Akibatnya, nilai similarity antar item juga terbatas dan kurang bervariasi.

---

## Kesimpulan

Berdasarkan hasil evaluasi sistem rekomendasi menggunakan metode User-Based Collaborative Filtering dan Content-Based Filtering, diperoleh bahwa nilai precision dan recall pada top-5 rekomendasi masih relatif rendah. Hal ini menunjukkan bahwa sistem belum mampu memberikan rekomendasi yang sangat relevan dan memadai bagi pengguna. Meski demikian, metode Content-Based Filtering menunjukkan performa yang sedikit lebih baik dibandingkan User-Based CF, terutama pada metrik recall. Ini mengindikasikan bahwa informasi konten buku walaupun terbatas masih dapat membantu dalam memberikan rekomendasi yang relevan.

Faktor utama penyebab rendahnya performa adalah data sparsity, cold start problem, serta keterbatasan fitur metadata yang digunakan dalam metode content-based. Untuk menangani data sparsity dan cold start problem, sudah diterapkan filtering sparsity yang mengambil hanya user dengan ≥50 rating dan top-2000 ISBN menjadikan rating matrix lebih padat, sehingga perhitungan similarity lebih stabil dan fallback Cold-Start yang menyediakan sistem rekomendasi berbasis buku paling populer, menjaga minimalisasi daftar rekomendasi kosong untuk user tanpa neighbor (CF) atau tanpa seed book (CBF).

---
## Rekomendasi Tindak Lanjut

Untuk meningkatkan performa sistem rekomendasi, berikut beberapa rekomendasi yang dapat dilakukan:

1. **Perluasan Metadata Buku**  
   Menambahkan fitur metadata yang lebih kaya seperti genre, sinopsis, tahun terbit, dan penerbit agar similarity antar item dapat dihitung dengan lebih akurat.

2. **Implementasi Hybrid Recommendation System**  
   Menggabungkan metode User-Based Collaborative Filtering dan Content-Based Filtering untuk mengatasi kelemahan masing-masing metode.

3. **Penggunaan Teknik Matrix Factorization**  
   Menerapkan teknik seperti Singular Value Decomposition (SVD) atau Alternating Least Squares (ALS) untuk menangani masalah sparsity dan meningkatkan akurasi prediksi.

4. **Pengelolaan Data Cold Start**  
   Mengadopsi strategi khusus untuk menangani pengguna atau item baru dengan sedikit data, seperti rekomendasi berbasis popularitas atau menggunakan metadata.

5. **Eksperimen dengan Parameter dan Metode Evaluasi**  
   Melakukan tuning parameter model dan memperluas metrik evaluasi untuk memperoleh gambaran performa yang lebih komprehensif.

Dengan langkah-langkah tersebut, diharapkan sistem rekomendasi dapat memberikan hasil yang lebih relevan dan memuaskan pengguna.
---


Referensi:
- Resnick, P., & Varian, H. R. (1997). Recommender systems. Communications of the ACM, 40(3), 56-58.
- Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer.
- Jarada, T. N., Senan, M. A., & Jaafar, J. (2022). A comprehensive survey of collaborative filtering-based recommender systems: trends, challenges, and future directions. Artificial Intelligence Review, 55(2), 1201-1252.
