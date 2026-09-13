import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi halaman wide mode
st.set_page_config(
    page_title="Spotify Studio - Music Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling Theme
spotify_css = """
<style>
    /* Base background */
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
        font-family: 'CircularStd', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Header Bar */
    header[data-testid="stHeader"], [data-testid="stHeader"] {
        background-color: #121212 !important;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 100% !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid #282828;
    }
    
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #E0E0E0 !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #FFFFFF !important;
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* FIX KONTRAS: Selectbox, Multiselect & Popover Menu Dropdown */
    div[data-baseweb="select"] > div {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] * {
        color: #FFFFFF !important;
    }
    div[data-baseweb="popover"] div {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
    }
    li[role="option"] {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
    }
    li[role="option"]:hover, li[aria-selected="true"] {
        background-color: #1DB954 !important;
        color: #000000 !important;
        font-weight: bold;
    }

    /* FIX KONTRAS: Tabs */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #B3B3B3 !important;
        font-weight: 600 !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 8px 16px !important;
    }
    button[aria-selected="true"] {
        color: #1DB954 !important;
        border-bottom: 2px solid #1DB954 !important;
    }

    /* FIX KONTRAS: Slider */
    div[data-baseweb="slider"] * {
        color: #1DB954 !important;
    }
    
    /* Headers & Text */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        line-height: 1.3 !important;
    }
    .spotify-green {
        color: #1DB954 !important;
    }
    
    /* Spotify Cards */
    .spotify-card {
        background-color: #181818;
        border: 1px solid #282828;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    .spotify-card:hover {
        border-color: #1DB954;
        box-shadow: 0 8px 24px rgba(29, 185, 84, 0.2);
    }

    /* KPI Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1f1f1f 0%, #141414 100%);
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #282828;
        border-left: 5px solid #1DB954;
        text-align: left;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1DB954;
        margin: 2px 0;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #B3B3B3;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    /* Recommendation Track Cards */
    .rec-card {
        background: #181818;
        border: 1px solid #282828;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .rec-card:hover {
        transform: translateY(-2px);
        border-color: #1DB954;
        background: #202020;
    }
    
    /* Custom Green Button */
    .stButton > button {
        background-color: #1DB954 !important;
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border-radius: 50px !important;
        padding: 12px 32px !important;
        border: none !important;
        width: 100%;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(29, 185, 84, 0.3);
    }
    .stButton > button:hover {
        background-color: #1ed760 !important;
        transform: scale(1.02);
        cursor: pointer;
    }

    /* Custom Download Button */
    .stDownloadButton > button {
        background-color: transparent !important;
        color: #1DB954 !important;
        border: 2px solid #1DB954 !important;
        border-radius: 50px !important;
        font-weight: 700 !important;
    }
    .stDownloadButton > button:hover {
        background-color: #1DB954 !important;
        color: #000000 !important;
    }

    /* Badge Tags */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        background-color: #282828;
        color: #1DB954;
        margin-right: 6px;
    }

    /* Fix Streamlit Table readability */
    div[data-testid="stDataFrame"] {
        background-color: #181818 !important;
        border-radius: 8px;
        padding: 10px;
    }
</style>
"""
st.markdown(spotify_css, unsafe_allow_html=True)

@st.cache_data
def load_data_and_artifacts():
    try:
        df = pd.read_csv('spotify_processed_with_clusters.csv')
        X_scaled = np.load('X_scaled.npy')
        kmeans_model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        for feat in ['danceability', 'energy', 'acousticness']:
            if feat not in df.columns:
                df[feat] = np.round(np.random.uniform(0.2, 0.95, len(df)), 2)
                
        return df, X_scaled, kmeans_model, scaler
    except Exception:
        np.random.seed(42)
        n_samples = 200
        genres = ['Pop', 'Dance/EDM', 'Hip-Hop', 'Rock', 'R&B', 'Indie']
        clusters = [0, 1, 2, 3]
        cluster_names = ['Chill & Relaxed', 'High Energy Party', 'Acoustic Vibes', 'Dark & Melancholic']
        
        songs = [f"Song Track {i+1}" for i in range(n_samples)]
        artists = [f"Artist {chr(65 + (i % 20))}" for i in range(n_samples)]
        
        data = {
            'track_name': songs,
            'track_artist': artists,
            'playlist_genre': np.random.choice(genres, n_samples),
            'track_popularity': np.random.randint(30, 100, n_samples),
            'valence': np.round(np.random.uniform(0.1, 0.99, n_samples), 2),
            'danceability': np.round(np.random.uniform(0.2, 0.95, n_samples), 2),
            'energy': np.round(np.random.uniform(0.3, 0.98, n_samples), 2),
            'acousticness': np.round(np.random.uniform(0.05, 0.85, n_samples), 2),
            'duration_min': np.round(np.random.uniform(2.5, 4.5, n_samples), 2),
            'cluster': np.random.choice(clusters, n_samples)
        }
        df_dummy = pd.DataFrame(data)
        df_dummy['cluster_name'] = df_dummy['cluster'].map(lambda x: cluster_names[x])
        
        X_scaled_dummy = np.random.randn(n_samples, 8)
        return df_dummy, X_scaled_dummy, None, None

df, X_scaled, kmeans_model, scaler = load_data_and_artifacts()

# Sidebar Header
st.sidebar.markdown("""
<div style="text-align: center; padding: 15px 0;">
    <h2 style="color: #1DB954; margin: 0; font-size: 1.6rem;">🎵 Spotify Studio</h2>
    <p style="color: #B3B3B3; font-size: 0.85rem; margin-top: 4px;">Music Intelligence & Analytics</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
page = st.sidebar.radio("📌 Pilih Halaman:", ["🎧 Sistem Rekomendasi", "📊 Dashboard Analytics"])

def apply_spotify_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF', family='CircularStd, Inter, sans-serif'),
        coloraxis_colorbar=dict(title_font_color='#FFFFFF', tickfont_color='#B3B3B3'),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_xaxes(gridcolor='#282828', tickfont=dict(color='#B3B3B3'))
    fig.update_yaxes(gridcolor='#282828', tickfont=dict(color='#B3B3B3'))
    return fig

if page == "🎧 Sistem Rekomendasi":
    st.markdown("""
    <div class="spotify-card" style="background: linear-gradient(90deg, #181818 0%, #0d2818 100%); border-left: 6px solid #1DB954;">
        <h1 style="margin:0; font-size: 2.2rem;">🎧 Spotify Song Recommendation</h1>
        <p style="color: #B3B3B3; margin-top: 6px; font-size: 1rem;">
            Temukan lagu-lagu favorit baru yang memiliki kemiripan audio, ritme, dan karakter musik secara akurat!
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Buat label unik 'Nama Lagu - Artis' agar lagu versi remix/duplikat terpilih secara akurat
    df['track_display'] = df['track_name'] + " - " + df['track_artist']

    # Filter genre
    col_filter, col_input1, col_input2 = st.columns([1, 2, 1])
    
    with col_filter:
        st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
        st.subheader("Genre")
        genre_options = ["Semua Genre"] + list(df['playlist_genre'].unique())
        selected_genre_filter = st.selectbox("Genre Acuan:", genre_options)
        st.markdown('</div>', unsafe_allow_html=True)

    # Filter dataframe berdasarkan genre terpilih jika ada
    if selected_genre_filter != "Semua Genre":
        filtered_song_df = df[df['playlist_genre'] == selected_genre_filter]
    else:
        filtered_song_df = df

    with col_input1:
        st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
        st.subheader("Lagu Acuan")
        song_list = filtered_song_df['track_display'].sort_values().unique()
        selected_display = st.selectbox("Pilih atau ketik nama lagu (Judul - Artis):", song_list)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_input2:
        st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
        st.subheader("Jumlah")
        top_n = st.slider("Jumlah Rekomendasi:", min_value=3, max_value=15, value=5)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Dapatkan Rekomendasi Musik"):
        song_pos = df[df['track_display'] == selected_display].index[0]
        selected_song = df.loc[song_pos, 'track_name']
        target_cluster = df.loc[song_pos, 'cluster']
        cluster_name = df.loc[song_pos, 'cluster_name']

        # Header Lagu Acuan Terpilih
        st.markdown(f"""
        <div class="spotify-card" style="border: 1px solid #1DB954; background: linear-gradient(135deg, #181818 0%, #11261a 100%);">
            <span class="badge">Lagu Acuan Terpilih</span>
            <h2 style="margin-top: 8px; color: #1DB954; font-size: 1.8rem;">{df.loc[song_pos, 'track_name']}</h2>
            <div style="display: flex; gap: 24px; flex-wrap: wrap; margin-top: 12px; font-size: 0.95rem;">
                <div>🎤 Artis: <b>{df.loc[song_pos, 'track_artist']}</b></div>
                <div>🎷 Genre: <b>{df.loc[song_pos, 'playlist_genre']}</b></div>
                <div>🔥 Popularitas: <b>{df.loc[song_pos, 'track_popularity']}/100</b></div>
                <div>🏷️ Mood Cluster: <span class="badge">{cluster_name}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Similarity calculation
        same_cluster_mask = (df['cluster'] == target_cluster)
        same_cluster_positions = df[same_cluster_mask].index.values

        target_vector = X_scaled[song_pos].reshape(1, -1)
        cluster_vectors = X_scaled[same_cluster_positions]

        similarity_scores = cosine_similarity(target_vector, cluster_vectors).flatten()

        rec_df = pd.DataFrame({
            'Lagu': df.loc[same_cluster_positions, 'track_name'].values,
            'Artis': df.loc[same_cluster_positions, 'track_artist'].values,
            'Genre': df.loc[same_cluster_positions, 'playlist_genre'].values,
            'Popularitas': df.loc[same_cluster_positions, 'track_popularity'].values,
            'Valence (Mood)': df.loc[same_cluster_positions, 'valence'].values,
            'Danceability': df.loc[same_cluster_positions, 'danceability'].values,
            'Energy': df.loc[same_cluster_positions, 'energy'].values,
            'Acousticness': df.loc[same_cluster_positions, 'acousticness'].values,
            'Durasi (Menit)': df.loc[same_cluster_positions, 'duration_min'].values,
            'Similarity Score': similarity_scores
        })

        results = (
            rec_df[rec_df['Lagu'] != selected_song]
            .sort_values(by='Similarity Score', ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        st.subheader(f"✨ Top {top_n} Rekomendasi Lagu Untukmu")
        
        # Radar Spider Chart Profil Audio Lagu Pilihan vs Rekomendasi
        col_rec_left, col_rec_right = st.columns([3, 2])

        with col_rec_right:
            st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
            st.subheader("Perbandingan Audio")
            
            categories = ['Valence (Mood)', 'Danceability', 'Energy', 'Acousticness']
            song_vals = [
                df.loc[song_pos, 'valence'],
                df.loc[song_pos, 'danceability'],
                df.loc[song_pos, 'energy'],
                df.loc[song_pos, 'acousticness']
            ]
            rec_avg_vals = [
                results['Valence (Mood)'].mean(),
                results['Danceability'].mean(),
                results['Energy'].mean(),
                results['Acousticness'].mean()
            ]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=song_vals,
                theta=categories,
                fill='toself',
                name='Lagu Acuan',
                line_color='#1DB954'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=rec_avg_vals,
                theta=categories,
                fill='toself',
                name='Rata-rata Rekomendasi',
                line_color='#3d5af1'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor='#282828', tickfont=dict(color='#B3B3B3')),
                    angularaxis=dict(gridcolor='#282828', tickfont=dict(color='#FFFFFF'))
                ),
                showlegend=True,
                legend=dict(font=dict(color='#FFFFFF'))
            )
            fig_radar = apply_spotify_dark_theme(fig_radar)
            st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_rec_left:
            tab_cards, tab_table = st.tabs(["List Rekomendasi Lagu", "Tabel Detail & Ekspor"])
            
            with tab_cards:
                for idx, row in results.iterrows():
                    sim_pct = int(row['Similarity Score'] * 100)
                    search_query = f"{row['Lagu']} {row['Artis']}".replace(" ", "%20")
                    spotify_url = f"https://open.spotify.com/search/{search_query}"
                    
                    st.markdown(f"""
                    <div class="rec-card">
                        <div style="flex: 2;">
                            <span class="badge">#{idx+1} Match</span>
                            <h3 style="margin: 4px 0; font-size: 1.15rem; color: #FFFFFF;">{row['Lagu']}</h3>
                            <p style="color: #B3B3B3; margin: 0; font-size: 0.88rem;">
                                🎤 <b>{row['Artis']}</b> • Genre: <span class="spotify-green">{row['Genre']}</span>
                            </p>
                            <div style="margin-top: 8px;">
                                <a href="{spotify_url}" target="_blank" style="text-decoration: none; font-size: 0.8rem; color: #1DB954; font-weight: 700;">
                                    ▶️ Lihat di Spotify ↗
                                </a>
                            </div>
                        </div>
                        <div style="flex: 1; text-align: center;">
                            <div style="font-size: 0.75rem; color: #B3B3B3;">Durasi & Mood</div>
                            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.9rem;">⏱️ {row['Durasi (Menit)']:.2f} min</div>
                            <div style="font-weight: 700; color: #FF9100; font-size: 0.85rem;">😊 Mood: {row['Valence (Mood)']*100:.0f}%</div>
                        </div>
                        <div style="flex: 1; text-align: right;">
                            <div style="font-size: 0.75rem; color: #B3B3B3;">Kemiripan Audio</div>
                            <div style="font-size: 1.6rem; font-weight: 800; color: #1DB954;">{sim_pct}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with tab_table:
                results_display = results.copy()
                results_display.index = range(1, len(results_display) + 1)
                st.dataframe(
                    results_display.style.format({
                        'Similarity Score': '{:.2%}',
                        'Durasi (Menit)': '{:.2f}',
                        'Valence (Mood)': '{:.2f}',
                        'Danceability': '{:.2f}',
                        'Energy': '{:.2f}',
                        'Acousticness': '{:.2f}'
                    }),
                    use_container_width=True
                )
                
                # Download hasil rekomendasi sebagai CSV
                csv_data = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh Daftar Rekomendasi (CSV)",
                    data=csv_data,
                    file_name=f"Spotify_Recommendations_{selected_song}.csv",
                    mime="text/csv"
                )

            

elif page == "📊 Dashboard Analytics":
    st.markdown("""
    <div class="spotify-card" style="background: linear-gradient(90deg, #181818 0%, #112536 100%); border-left: 6px solid #3d5af1;">
        <h1 style="margin:0; font-size: 2.2rem;">📊 Spotify Data Analytics Dashboard</h1>
        <p style="color: #B3B3B3; margin-top: 6px; font-size: 1rem;">
            Eksplorasi wawasan mendalam, distribusi genre, popularitas artis, dan tren audio dari katalog Spotify.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Interactive Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Dashboard")
    selected_genres = st.sidebar.multiselect(
        "Filter Genre Musik:",
        options=df['playlist_genre'].unique(),
        default=df['playlist_genre'].unique()
    )

    min_pop, max_pop = st.sidebar.slider(
        "Rentang Popularitas Lagu:",
        min_value=0, max_value=100, value=(0, 100)
    )

    filtered_df = df[
        (df['playlist_genre'].isin(selected_genres)) &
        (df['track_popularity'] >= min_pop) &
        (df['track_popularity'] <= max_pop)
    ]

    # Metric KPI Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Lagu Filtered</div>
            <div class="metric-value">{len(filtered_df):,}</div>
            <div style="font-size: 0.75rem; color: #B3B3B3;">Dari {len(df):,} total katalog</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        mean_pop = filtered_df['track_popularity'].mean() if not filtered_df.empty else 0
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3d5af1;">
            <div class="metric-label">Rata-rata Popularitas</div>
            <div class="metric-value" style="color: #3d5af1;">{mean_pop:.1f}</div>
            <div style="font-size: 0.75rem; color: #B3B3B3;">Skala 0 - 100</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        mean_val = filtered_df['valence'].mean() if not filtered_df.empty else 0
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #ff9100;">
            <div class="metric-label">Rata-rata Mood (Valence)</div>
            <div class="metric-value" style="color: #ff9100;">{mean_val:.2f}</div>
            <div style="font-size: 0.75rem; color: #B3B3B3;">0 (Sedih) - 1 (Ceria)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi4:
        mean_dur = filtered_df['duration_min'].mean() if not filtered_df.empty else 0
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #e040fb;">
            <div class="metric-label">Rata-rata Durasi</div>
            <div class="metric-value" style="color: #e040fb;">{mean_dur:.2f}m</div>
            <div style="font-size: 0.75rem; color: #B3B3B3;">Menit per lagu</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("⚠️ Tidak ada data yang sesuai dengan filter. Silakan sesuaikan kembali filter di sidebar.")
    else:
        # Visualizations Row 1
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
            st.subheader("Sebaran Kelompok Musik")
            fig_cluster = px.pie(
                filtered_df, 
                names='cluster_name', 
                hole=0.45,
                color_discrete_sequence=['#1DB954', '#3d5af1', '#ff9100', '#e040fb', '#00e5ff']
            )
            fig_cluster.update_traces(textposition='inside', textinfo='percent+label', textfont_size=12)
            fig_cluster = apply_spotify_dark_theme(fig_cluster)
            st.plotly_chart(fig_cluster, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
            st.subheader("🎤 Top 10 Artis Terpopuler")
            top_artists = (
                filtered_df.groupby('track_artist')['track_popularity']
                .mean()
                .reset_index()
                .sort_values(by='track_popularity', ascending=False)
                .head(10)
            )
            fig_artist = px.bar(
                top_artists, 
                x='track_popularity', 
                y='track_artist', 
                orientation='h',
                color='track_popularity',
                color_continuous_scale=['#143820', '#1DB954', '#1ed760']
            )
            fig_artist.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
            fig_artist = apply_spotify_dark_theme(fig_artist)
            st.plotly_chart(fig_artist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Visualizations Row 2
        st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
        st.subheader("Hubungan Mood vs Popularitas Lagu")
        fig_scatter = px.scatter(
            filtered_df, 
            x='valence', 
            y='track_popularity', 
            color='playlist_genre',
            size='duration_min',
            hover_data=['track_name', 'track_artist'],
            labels={'valence': 'Mood Positive (Valence)', 'track_popularity': 'Popularitas (0-100)'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_scatter = apply_spotify_dark_theme(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #B3B3B3; font-size: 0.85rem; margin-top: 40px; padding: 20px; border-top: 1px solid #282828;">
    Spotify Recommendation System • Designed by M. Ian Wijaya
</div>
""", unsafe_allow_html=True)