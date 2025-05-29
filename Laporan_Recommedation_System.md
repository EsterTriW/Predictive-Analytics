# Laporan Proyek Recommendation System
## Judul proyek : Implementasi Sistem Rekomendasi Buku Berbasis User-Based Collaborative Filtering dan Content-Based Filtering
## Ditulis oleh : Ester Tri Wahyuningsih -  MC002D5X0841

## Domain Proyek
### Latar Belakang
Di era digital saat ini, kemajuan teknologi informasi telah mengubah cara masyarakat dalam mengakses dan mengonsumsi konten literasi. Platform seperti Goodreads, Amazon, dan Google Books menyediakan jutaan judul buku dari berbagai kategori yang dapat diakses kapan saja. Namun, tingginya jumlah pilihan justru menimbulkan masalah baru, yaitu information overload, di mana pengguna kesulitan menemukan buku yang sesuai dengan minat dan kebutuhannya secara efisien (Resnick & Varian, 1997).

Salah satu solusi untuk permasalahan ini adalah penerapan sistem rekomendasi (recommender system). Sistem ini berperan penting dalam menyaring informasi dan memberikan rekomendasi personal berdasarkan preferensi atau perilaku pengguna. Tidak hanya di bidang e-commerce dan layanan hiburan seperti Netflix dan Spotify, sistem rekomendasi juga memiliki potensi besar di sektor literasi digital untuk meningkatkan keterlibatan dan kepuasan pengguna (Ricci et al., 2015).

Dalam sistem rekomendasi buku, terdapat dua metode utama yang banyak digunakan, yaitu:

- Collaborative Filtering, yang memanfaatkan pola perilaku pengguna lain dengan preferensi serupa.
- Content-Based Filtering, yang memberikan rekomendasi berdasarkan kemiripan atribut konten buku, seperti genre, deskripsi, dan penulis.

Meski demikian, tantangan utama dalam pengembangan sistem rekomendasi adalah sparsity problem, yaitu kondisi di mana data interaksi pengguna dan buku cenderung jarang (Jarada et al., 2022). Selain itu, menentukan metode rekomendasi yang paling efektif untuk konteks literasi digital juga menjadi persoalan yang menarik untuk diteliti.

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
- Kelebihan:
  * Tidak bergantung pada data pengguna lain.
  * Rekomendasi spesifik sesuai preferensi konten.
- Kekurangan:
  * Rentan menghasilkan filter bubble (konten terlalu mirip).

2. User-Based Collaborative Filtering (UBCF)
- Metode: Menghitung cosine similarity antar pengguna berdasarkan riwayat rating. Sistem merekomendasikan buku yang disukai pengguna-pengguna dengan pola preferensi serupa.
- Output: Top‑N buku berdasarkan prediksi preferensi dari pengguna yang mirip.
- Kelebihan:
  * Dapat mengeksplorasi buku di luar preferensi konten pengguna.
  * Tidak memerlukan metadata buku.
- Kekurangan:
  * Memerlukan data rating yang cukup.
  * Cold-start problem untuk pengguna/item baru.

Strategi tambahan lainnya :
- Menangani data sparse dengan memfilter pengguna aktif dan buku populer berdasarkan jumlah rating minimum.
- Mengevaluasi hasil rekomendasi menggunakan metrik Precision@5 dan Recall@5.
- Membandingkan hasil performa kedua metode untuk menentukan pendekatan yang paling sesuai.





Referensi:
- Resnick, P., & Varian, H. R. (1997). Recommender systems. Communications of the ACM, 40(3), 56-58.
- Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer.
- Jarada, T. N., Senan, M. A., & Jaafar, J. (2022). A comprehensive survey of collaborative filtering-based recommender systems: trends, challenges, and future directions. Artificial Intelligence Review, 55(2), 1201-1252.
