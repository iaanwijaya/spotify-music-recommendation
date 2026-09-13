🎵 Spotify Music Recommendation & Analytics Studio

Spotify Studio adalah aplikasi web interaktif berbasis Machine Learning yang memadukan K-Means Clustering dan Vector Cosine Similarity untuk memberikan rekomendasi lagu berdasarkan atribut audio numerik, serta menyajikan dashboard eksplorasi data musik interaktif. Proyek ini dikembangkan sebagai Final Project Program GDGoC Universitas Sriwijaya 2026.
![alt text](<Screenshot 2026-09-13 202338.png>)
![alt text](<Screenshot 2026-09-13 202420.png>)

🌟 Fitur Aplikasi (app.py)

🎧 1. Sistem Rekomendasi Lagu
- Pencarian Spesifik: Pencarian lagu acuan berdasarkan format unik track_name - track_artist.
- Filter Genre Acuan: Pengguna dapat menyaring lagu acuan berdasarkan genre tertentu atau seluruh katalog.
- Kluster-Based Cosine Similarity: Rekomendasi dihitung menggunakan kemiripan vektor Cosine Similarity khusus pada lagu dalam segmen cluster mood yang sama untuk efisiensi komputasi.
- Jumlah Rekomendasi Dinamis: Pilihan jumlah rekomendasi dari 3 hingga 15 lagu.
- Direct Spotify Link: Setiap lagu hasil rekomendasi dilengkapi tombol langsung menuju pencarian aplikasi Spotify (open.spotify.com).
- Ekspor Data (.CSV): Fitur mengunduh hasil rekomendasi lagu ke dalam format berkas CSV.

📊 2. Dashboard Analytics & Visualisasi Data
- Metric KPI Cards: Menampilkan Total Lagu Filtered, Rata-rata Popularitas, Rata-rata Mood (Valence), dan Rata-rata Durasi.
- Pie Chart Cluster Mood: Visualisasi sebaran 4 segmen cluster (Chill & Relaxed, High Energy Party, Acoustic Vibes, Dark & Melancholic).
- Horizontal Bar Chart Top Artis: Visualisasi 10 Artis terpopuler berdasarkan skor track_popularity.
- Scatter Plot Mood vs Popularitas: Analisis hubungan valence (mood) terhadap track_popularity dengan ukuran gelembung berdasarkan duration_min.
- Filter Sidebar Interaktif: Filter multiselect genre dan slider rentang popularitas lagu.

🎨 3. UI Spotify Dark Theme
Desain kustom CSS dengan skema warna khas Spotify (#121212 background & #1DB954 Spotify Green accent) dengan kontras tinggi pada dropdown, tab, dan slider.

🧠 Metodologi & Arsitektur Sistem
[Raw Dataset] -> [StandardScaler Normalization (X_scaled.npy)]
                      |
                      v
             [K-Means Clustering (kmeans_model.pkl)]
                      |
                      v
             [Filtered Cluster Subset]
                      |
                      v
             [Cosine Similarity Calculation] -> [Top-N Recommendations & CSV Download]


Preprocessing (scaler.pkl & X_scaled.npy):
1. Atribut numerik audio distandarisasi menggunakan StandardScaler agar seluruh variabel memiliki bobot yang seimbang.
2. K-Means Clustering (kmeans_model.pkl):
Katalog lagu dikelompokkan ke dalam 4 mood cluster utama.
3. Similarity Engine:
Sistem menghitung kemiripan sudut antar-vektor lagu A dan B

🚀 Langkah Menjalankan Proyek di Lokal
1. Clone Repositori:
git clone https://github.com/iaanwijaya/spotify-music-recommendation.git
cd spotify-music-recommendation

2. Install Dependensi:
pip install -r requirements.txt

3. Jalankan Aplikasi Streamlit:
streamlit run app.py
