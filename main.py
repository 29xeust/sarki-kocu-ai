import os
import librosa
import numpy as np
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
import pypdf

app = FastAPI(title="Mix ve Prodüksiyon Koçu AI - Link & Ton Pro")

client = genai.Client()
REFERANSLAR_KLASORU = "referanslar"
VERSIYON_GECMISI = []

def urlden_metin_cek(url):
    """Verilen URL'deki web sayfasının ana metin içeriklerini temiz bir şekilde çeker"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Gereksiz etiketleri temizle
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
                
            metin = soup.get_text(separator=' ', strip=True)
            return metin[:15000] # Çok uzun sayfalarda token sınırını korumak için sınır koyuyoruz
    except Exception as e:
        return f"[Link Okuma Hatası ({url}): {str(e)}]"
    return ""

def referanslari_oku(ek_linkler=None):
    toplam_icerik = ""
    
    # 1. Klasördeki Dosyaları Oku (PDF ve TXT)
    if not os.path.exists(REFERANSLAR_KLASORU):
        os.makedirs(REFERANSLAR_KLASORU)
    else:
        for dosya_adi in os.listdir(REFERANSLAR_KLASORU):
            dosya_yolu = os.path.join(REFERANSLAR_KLASORU, dosya_adi)
            if dosya_adi.endswith(".txt"):
                try:
                    with open(dosya_yolu, "r", encoding="utf-8") as f:
                        toplam_icerik += f"\n--- KILAVUZ (TXT): {dosya_adi} ---\n" + f.read() + "\n"
                except Exception:
                    pass
            elif dosya_adi.endswith(".pdf"):
                try:
                    pdf_metin = ""
                    reader = pypdf.PdfReader(dosya_yolu)
                    for sayfa in reader.pages:
                        metin = sayfa.extract_text()
                        if metin:
                            pdf_metin += metin + "\n"
                    toplam_icerik += f"\n--- KILAVUZ (PDF): {dosya_adi} ---\n" + pdf_metin + "\n"
                except Exception:
                    pass

    # 2. Kullanıcının Eklediği Linkleri Oku
    if ek_linkler:
        link_listesi = [l.strip() for l in ek_linkler.split(",") if l.strip()]
        for link in link_listesi:
            toplam_icerik += f"\n--- WEB KILAVUZ (LINK): {link} ---\n" + urlden_metin_cek(link) + "\n"

    return toplam_icerik if toplam_icerik else "Referans kılavuz veya link bulunamadı."

def ton_ve_akor_tahmini(y, sr):
    """Librosa kullanarak şarkının tonal merkezini (Key & Mode) tahmin eder"""
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        notalar = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        en_guclu_nota_idx = int(np.argmax(chroma_mean))
        kok_nota = notalar[en_guclu_nota_idx]
        
        major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        
        best_score = -1
        mode = "Majör"
        for i in range(12):
            shifted_chroma = np.roll(chroma_mean, -i)
            maj_corr = np.corrcoef(shifted_chroma, major_template)[0, 1]
            min_corr = np.corrcoef(shifted_chroma, minor_template)[0, 1]
            
            if maj_corr > best_score:
                best_score = maj_corr
                mode = "Majör"
            if min_corr > best_score:
                best_score = min_corr
                mode = "Minör"
                
        return f"{kok_nota} {mode}"
    except Exception:
        return "Tespit Edilemedi"

def librosa_ile_sesi_olcutle(dosya_yolu):
    try:
        y, sr = librosa.load(dosya_yolu, sr=None)
        
        peak = float(np.max(np.abs(y)))
        peak_db = 20 * np.log10(peak) if peak > 0 else -100
        
        rms = float(np.sqrt(np.mean(y**2)))
        rms_db = 20 * np.log10(rms) if rms > 0 else -100
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, "__len__"):
            tempo = float(tempo[0])
        else:
            tempo = float(tempo)
            
        ton = ton_ve_akor_tahmini(y, sr)
            
        return {
            "peak_db": round(peak_db, 2),
            "rms_db": round(rms_db, 2),
            "tahmini_bpm": round(tempo, 1),
            "tahmini_ton": ton
        }
    except Exception as e:
        return {"hata": str(e)}

@app.get("/", response_class=HTMLResponse)
async def anasayfa():
    return """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <title>Mix ve Prodüksiyon Koçu AI - Link & Ton Pro</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #e0e0e0; max-width: 950px; margin: 40px auto; padding: 20px; }
            h1 { color: #bb86fc; text-align: center; margin-bottom: 30px; }
            .card { background: #1e1e1e; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); margin-bottom: 20px; }
            label { font-weight: bold; color: #03dac6; display: block; margin-top: 15px; }
            textarea, input[type="file"], input[type="text"] { width: 100%; padding: 12px; margin-top: 8px; border-radius: 6px; border: 1px solid #333; background: #2d2d2d; color: #fff; box-sizing: border-box; }
            
            .mode-selector { display: flex; gap: 15px; margin-bottom: 20px; background: #252525; padding: 15px; border-radius: 8px; }
            .mode-option { flex: 1; text-align: center; cursor: pointer; padding: 10px; border-radius: 6px; border: 1px solid #444; transition: 0.3s; font-weight: bold; color: #aaa; }
            .mode-option.active { background: #bb86fc; color: #121212; border-color: #bb86fc; }
            
            .upload-section { display: none; }
            .upload-section.active { display: block; }
            
            .file-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px; }
            .file-group { background: #252525; padding: 12px; border-radius: 8px; border: 1px dashed #444; }
            .file-group label { margin-top: 0; font-size: 0.9em; color: #bb86fc; }
            
            button { background: #bb86fc; color: #121212; border: none; padding: 14px 20px; font-size: 16px; font-weight: bold; border-radius: 6px; cursor: pointer; width: 100%; margin-top: 25px; transition: 0.3s; }
            button:hover { background: #9965f4; }
            button:disabled { background: #555; cursor: not-allowed; }
            
            .results-container { display: none; margin-top: 25px; }
            .sonuc-box { background: #1e1e1e; padding: 25px; border-radius: 12px; white-space: pre-wrap; line-height: 1.6; border-left: 5px solid #03dac6; margin-bottom: 20px; }
            
            .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }
            .metric-card { background: #252525; padding: 15px; border-radius: 8px; border-top: 3px solid #03dac6; text-align: center; }
            .metric-card h4 { margin: 0; color: #aaa; font-size: 0.8em; }
            .metric-card p { margin: 8px 0 0 0; font-size: 1.1em; font-weight: bold; color: #fff; }

            .history-card { background: #1e1e1e; padding: 20px; border-radius: 12px; margin-bottom: 20px; border-top: 4px solid #ffb74d; }
            .history-card h3 { color: #ffb74d; margin-top: 0; }
            .history-item { background: #252525; padding: 10px 15px; border-radius: 6px; margin-top: 8px; display: flex; justify-content: space-between; align-items: center; font-size: 0.9em; }

            .download-btn { background: #03dac6; color: #121212; width: auto; padding: 10px 20px; font-size: 14px; margin-top: 0; display: inline-block; text-align: center; text-decoration: none; border-radius: 6px; font-weight: bold; }
            .download-btn:hover { background: #02b8a6; }
            .loading { text-align: center; color: #ffb74d; display: none; font-weight: bold; margin-top: 15px; font-size: 1.05em; }
        </style>
    </head>
    <body>
        <h1>🎛️ Mix ve Prodüksiyon Koçu AI - Link & Ton Pro</h1>
        
        <div class="card">
            <form id="mixForm">
                <label for="versiyon_adi">Şarkı / Versiyon Adı:</label>
                <input type="text" id="versiyon_adi" required placeholder="Örn: Demo_Beat_v2">

                <label for="parca_detaylari">Parça Detayları (Tarz, Hedef, Melodi İsteği):</label>
                <textarea id="parca_detaylari" rows="2" required placeholder="Örn: Synthwave tarzında. Linkteki makalede geçen kompresyon tekniklerini bu parçaya uyarla."></textarea>
                
                <label for="referans_linkler">Ek Web Kılavuz Linkleri (Birden fazla ise virgülle ayır):</label>
                <input type="text" id="referans_linkler" placeholder="Örn: https://ornek-muzik-blogu.com/mix-rehberi, https://site.com/akorlar">

                <label style="margin-top: 20px;">Analiz Modu Seçin:</label>
                <div class="mode-selector">
                    <div id="btnSingle" class="mode-option active" onclick="setMode('single')">🎵 1. Tek Dosya (Mix + Ton & Melodi)</div>
                    <div id="btnMulti" class="mode-option" onclick="setMode('multi')">🎚️ 2. Çoklu Kanal (Stem Analizi)</div>
                </div>
                <input type="hidden" id="analysis_mode" value="single">

                <div id="sectionSingle" class="upload-section active">
                    <label for="ses_dosyasi">Master Şarkı Dosyası (MP3 / WAV):</label>
                    <input type="file" id="ses_dosyasi" accept="audio/*">
                </div>

                <div id="sectionMulti" class="upload-section">
                    <label>Ayrı Stem Dosyaları:</label>
                    <div class="file-grid">
                        <div class="file-group"><label>Kick Stem</label><input type="file" id="kick_dosyasi" accept="audio/*"></div>
                        <div class="file-group"><label>Bass Stem</label><input type="file" id="bass_dosyasi" accept="audio/*"></div>
                        <div class="file-group"><label>Vokal Stem</label><input type="file" id="vokal_dosyasi" accept="audio/*"></div>
                        <div class="file-group"><label>Enstrüman Stem</label><input type="file" id="enstruman_dosyasi" accept="audio/*"></div>
                    </div>
                </div>
                
                <button type="submit" id="submitBtn">Linkleri Oku, Tonu Çöz ve Koçluğu Başlat</button>
            </form>
            <div id="loading" class="loading">⏳ [1/4] Ses dosyası sunucuya yükleniyor...</div>
        </div>

        <div id="history-section" class="history-card" style="display:none;">
            <h3>📂 Proje Versiyon Geçmişi</h3>
            <div id="history-list"></div>
        </div>

        <div id="results-container" class="results-container">
            <div class="metrics-grid" id="metrics-grid" style="display:none;">
                <div class="metric-card"><h4>Tespit Edilen Ton</h4><p id="m-ton">-</p></div>
                <div class="metric-card"><h4>Tespit Edilen BPM</h4><p id="m-bpm">-</p></div>
                <div class="metric-card"><h4>Peak (dB)</h4><p id="m-peak">-</p></div>
                <div class="metric-card"><h4>RMS Enerji</h4><p id="m-rms">-</p></div>
            </div>

            <div id="sonuc" class="sonuc-box"></div>
            
            <div style="text-align: right;">
                <a id="downloadLink" class="download-btn" style="display:none;" download="mix_link_raporu.txt">📥 Raporu .TXT Olarak İndir</a>
            </div>
        </div>

        <script>
            function setMode(mode) {
                document.getElementById('analysis_mode').value = mode;
                if(mode === 'single') {
                    document.getElementById('btnSingle').classList.add('active');
                    document.getElementById('btnMulti').classList.remove('active');
                    document.getElementById('sectionSingle').classList.add('active');
                    document.getElementById('sectionMulti').classList.remove('active');
                } else {
                    document.getElementById('btnMulti').classList.add('active');
                    document.getElementById('btnSingle').classList.remove('active');
                    document.getElementById('sectionMulti').classList.add('active');
                    document.getElementById('sectionSingle').classList.remove('active');
                }
            }

            document.getElementById('mixForm').onsubmit = async function(e) {
                e.preventDefault();
                const btn = document.getElementById('submitBtn');
                const loading = document.getElementById('loading');
                const resultsContainer = document.getElementById('results-container');
                const sonucDiv = document.getElementById('sonuc');
                const downloadLink = document.getElementById('downloadLink');
                const metricsGrid = document.getElementById('metrics-grid');
                const historySection = document.getElementById('history-section');
                const historyList = document.getElementById('history-list');
                
                btn.disabled = true;
                btn.innerText = "İşleniyor...";
                loading.style.display = 'block';
                resultsContainer.style.display = 'none';
                sonucDiv.innerText = '';

                // --- ADIM ADIM İLERLEME MESAJLARI ---
                const adimlar = [
                    "⏳ [1/4] Ses dosyası sunucuya yükleniyor...",
                    "🎵 [2/4] Librosa ile frekans, BPM ve Ton analizi yapılıyor...",
                    "🧠 [3/4] Yapay zeka kılavuzları okuyor ve önerileri hazırlıyor...",
                    "✨ [4/4] Rapor tamamlanmak üzere, son dokunuşlar yapılıyor..."
                ];
                
                let adimIndex = 0;
                loading.innerText = adimlar[0];

                // Her 8 saniyede bir ekrandaki yazıyı sıradaki adıma geçir
                const loadingInterval = setInterval(() => {
                    adimIndex++;
                    if (adimIndex < adimlar.length) {
                        loading.innerText = adimlar[adimIndex];
                    }
                }, 8000);

                const formData = new FormData();
                formData.append('versiyon_adi', document.getElementById('versiyon_adi').value);
                formData.append('parca_detaylari', document.getElementById('parca_detaylari').value);
                formData.append('referans_linkler', document.getElementById('referans_linkler').value);
                const mode = document.getElementById('analysis_mode').value;
                formData.append('analysis_mode', mode);

                if(mode === 'single') {
                    const sesFile = document.getElementById('ses_dosyasi').files[0];
                    if (sesFile) formData.append('ses_dosyasi', sesFile);
                } else {
                    const kick = document.getElementById('kick_dosyasi').files[0];
                    const bass = document.getElementById('bass_dosyasi').files[0];
                    const vokal = document.getElementById('vokal_dosyasi').files[0];
                    const enst = document.getElementById('enstruman_dosyasi').files[0];

                    if (kick) formData.append('kick_dosyasi', kick);
                    if (bass) formData.append('bass_dosyasi', bass);
                    if (vokal) formData.append('vokal_dosyasi', vokal);
                    if (enst) formData.append('enstruman_dosyasi', enst);
                }

                try {
                    const response = await fetch('/mix-analiz/', {
                        method: 'POST',
                        body: formData
                    });

                    // İşlem bitince sayacı durdur
                    clearInterval(loadingInterval);

                    const data = await response.json();
                    
                    loading.style.display = 'none';
                    btn.disabled = false;
                    btn.innerText = "Linkleri Oku, Tonu Çöz ve Koçluğu Başlat";
                    resultsContainer.style.display = 'block';

                    if (data.mix_ve_produksiyon_tavsiyeleri) {
                        sonucDiv.innerText = data.mix_ve_produksiyon_tavsiyeleri;

                        if(data.ses_metrikleri) {
                            metricsGrid.style.display = 'grid';
                            document.getElementById('m-ton').innerText = data.ses_metrikleri.tahmini_ton;
                            document.getElementById('m-bpm').innerText = data.ses_metrikleri.tahmini_bpm + " BPM";
                            document.getElementById('m-peak').innerText = data.ses_metrikleri.peak_db + " dB";
                            document.getElementById('m-rms').innerText = data.ses_metrikleri.rms_db + " dB";
                        } else {
                            metricsGrid.style.display = 'none';
                        }

                        if(data.gecmis_ozeti) {
                            historySection.style.display = 'block';
                            historyList.innerHTML = '';
                            data.gecmis_ozeti.forEach(item => {
                                historyList.innerHTML += `<div class="history-item"><span><b>${item.ad}</b></span><span>Ton: ${item.ton} | BPM: ${item.bpm}</span></div>`;
                            });
                        }

                        const blob = new Blob([data.mix_ve_produksiyon_tavsiyeleri], { type: 'text/plain;charset=utf-8' });
                        downloadLink.href = URL.createObjectURL(blob);
                        downloadLink.style.display = 'inline-block';
                    } else {
                        sonucDiv.innerText = "Hata: " + JSON.stringify(data, null, 2);
                    }
                } catch (err) {
                    clearInterval(loadingInterval);
                    loading.style.display = 'none';
                    btn.disabled = false;
                    btn.innerText = "Linkleri Oku, Tonu Çöz ve Koçluğu Başlat";
                    resultsContainer.style.display = 'block';
                    sonucDiv.innerText = "Bağlantı hatası oluştu: " + err;
                }
            };
        </script>
    </body>
    </html>
    """

@app.post("/mix-analiz/")
async def analyze_mix(
    versiyon_adi: str = Form("Versiyon 1"),
    parca_detaylari: str = Form(...),
    referans_linkler: str = Form(""),
    analysis_mode: str = Form("single"),
    ses_dosyasi: UploadFile = File(None),
    kick_dosyasi: UploadFile = File(None),
    bass_dosyasi: UploadFile = File(None),
    vokal_dosyasi: UploadFile = File(None),
    enstruman_dosyasi: UploadFile = File(None)
):
    try:
        otomatik_referanslar = referanslari_oku(referans_linkler)
        icerik_listesi = []
        librosa_sonuclar = None

        system_instruction = """
        Sen profesyonel bir kıdemli ses mühendisi, aranjör ve müzik prodüktörüsün. Sana yüklenen ses dosyasının 
        Librosa ile otomatik tespit edilen tonu (Key/Mode), temposu (BPM) ve ses seviyesi (Peak/RMS) verilecek.
        Ayrıca kullanıcı tarafından sağlanan harici web kılavuzları (linkler) ve yerel belgeler referans alınacaktır.
        Tavsiyelerini bu somut verilere, ton merkezine ve web'den çekilen kılavuzlara dayandırarak ver. ÖNEMLİ KURAL: Kullanıcının sorusu ne olursa olsun, cevabının en sonunda MUTLAKA o konuya veya tarza uygun 2-3 adet gerçek referans şarkı/parça önerisinde bulun. Referans parça önermeyi asla unutma. 
        Asla düz paragraf yazma; net başlıklar ve maddeler kullan.
        """

        gecmis_metni = "Daha önceki yüklenen versiyon yok."
        if VERSIYON_GECMISI:
            gecmis_metni = "Önceki Versiyonlar:\n" + "\n".join([f"- {v['ad']} (Ton: {v['ton']}, BPM: {v['bpm']})" for v in VERSIYON_GECMISI])

        librosa_bilgi_metni = ""
        if analysis_mode == "single" and ses_dosyasi:
            gecici_yol = "gecici_master_link.mp3"
            try:
                icerik_bytes = await ses_dosyasi.read()
                with open(gecici_yol, "wb") as f:
                    f.write(icerik_bytes)
                
                librosa_sonuclar = librosa_ile_sesi_olcutle(gecici_yol)
                if "hata" not in librosa_sonuclar:
                    librosa_bilgi_metni = f"""
                    [PYTHON / LIBROSA TONA VE SES VERİLERİ]:
                    - Tespit Edilen Ton / Mod: {librosa_sonuclar['tahmini_ton']}
                    - Tespit Edilen Tempo: {librosa_sonuclar['tahmini_bpm']} BPM
                    - Tepe Noktası (Peak): {librosa_sonuclar['peak_db']} dB
                    - Ortalama Enerji (RMS): {librosa_sonuclar['rms_db']} dB
                    """
                
                uploaded_file = client.files.upload(file=gecici_yol)
                icerik_listesi.append("[Master Şarkı Dosyası]:")
                icerik_listesi.append(uploaded_file)
            finally:
                if os.path.exists(gecici_yol):
                    os.remove(gecici_yol)

        prompt = f"""
        [OTOMATİK KILAVUZLAR VE WEB LİNKLERİ]:
        {otomatik_referanslar}

        {librosa_bilgi_metni}

        [PROJE GEÇMİŞİ]:
        {gecmis_metni}

        [KULLANICI TALEBİ / HEDEFİ]:
        Versiyon: {versiyon_adi}
        Not: {parca_detaylari}

        Lütfen yukarıdaki verileri, web linklerinden gelen bilgileri ve ton bilgisini kullanarak şu başlıklarla profesyonel rapor oluştur:

        ### 🎵 Tonal ve Teknik Analiz Özeti
        - (Tespit edilen ton, tempo ve genel karakter üzerine değerlendirme)

        ### 🎹 1. Akor Yürüyüşü ve Melodi Fikirleri
        - (Tespit edilen tona, tarza ve kılavuzlara uygun olarak yeni melodi ve akor önerileri)

        ### 🎛️ 2. Mix ve Frekans Dengeleme Tavsiyeleri
        - ...

        ### 📉 3. Dinamik ve Alan Kontrolü
        - ...

        ### 🚀 4. Hemen Uygulanacak 3 Kritik Adım
        1. ...
        2. ...
        3. ...
        """
        icerik_listesi.append(prompt)

        if analysis_mode == "multi":
            dosyalar = {
                "Kick Stem": kick_dosyasi,
                "Bass Stem": bass_dosyasi,
                "Vokal Stem": vokal_dosyasi,
                "Enstrüman Stem": enstruman_dosyasi
            }
            for isim, dosya in dosyalar.items():
                if dosya:
                    gecici_yol = f"gecici_{isim.lower().replace(' ', '_')}.mp3"
                    try:
                        icerik_bytes = await dosya.read()
                        with open(gecici_yol, "wb") as f:
                            f.write(icerik_bytes)
                        uploaded_stem = client.files.upload(file=gecici_yol)
                        icerik_listesi.append(f"[{isim} Analizi]:")
                        icerik_listesi.append(uploaded_stem)
                    finally:
                        if os.path.exists(gecici_yol):
                            os.remove(gecici_yol)

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=icerik_listesi,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.5,
            ),
        )

        yeni_gecmis_kaydi = {
            "ad": versiyon_adi,
            "ton": librosa_sonuclar['tahmini_ton'] if librosa_sonuclar and "hata" not in librosa_sonuclar else "N/A",
            "bpm": librosa_sonuclar['tahmini_bpm'] if librosa_sonuclar and "hata" not in librosa_sonuclar else "N/A"
        }
        VERSIYON_GECMISI.append(yeni_gecmis_kaydi)

        return {
            "mix_ve_produksiyon_tavsiyeleri": response.text,
            "ses_metrikleri": librosa_sonuclar if (librosa_sonuclar and "hata" not in librosa_sonuclar) else None,
            "gecmis_ozeti": VERSIYON_GECMISI
        }

    except Exception as genel_hata:
        return {"hata_detayi": f"API veya Sunucu Hatası: {str(genel_hata)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)