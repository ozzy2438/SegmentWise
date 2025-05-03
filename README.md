# SegmentWise: Akıllı Müşteri Segmentasyon Platformu

<p align="center">
  <img src="img/segmentwise_logo.png" alt="SegmentWise Logo" width="200"/>
</p>

<p align="center">
  <b>Müşteri verilerinizden otomatik segmentler oluşturun ve yapay zeka destekli pazarlama stratejileri geliştirin</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/Flask-2.0.1-green" alt="Flask 2.0.1"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.0.2-orange" alt="scikit-learn 1.0.2"/>
  <img src="https://img.shields.io/badge/OpenAI-API-blueviolet" alt="OpenAI API"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"/>
</p>

## 📊 Proje Tanıtımı

SegmentWise, işletmelerin müşteri verilerini otomatik olarak analiz ederek anlamlı müşteri segmentleri oluşturan, her segment için yapay zeka destekli pazarlama stratejileri sunan ve görsel veri analizleri sağlayan kapsamlı bir platformdur.

### ✨ Demo Görüntüleri

<p align="center">
  <img src="img/dashboard.png" alt="SegmentWise Dashboard" width="800"/>
  <br/>
  <em>Ana Dashboard: Segmentler ve Pazarlama Stratejileri</em>
</p>

<p align="center">
  <img src="img/segments.png" alt="Segment Analizi" width="800"/>
  <br/>
  <em>Segment Detayları ve Dağılım Analizleri</em>
</p>

## 🚀 Temel Özellikler

- **🔍 Otomatik Veri Algılama**: CSV dosyalarından otomatik olarak ilgili müşteri verilerini tespit eder
- **🧹 Akıllı Veri Temizleme**: Eksik verileri otomatik doldurma ve hatalı verileri filtreleme
- **📊 Dinamik Segmentasyon**: K-means algoritması kullanarak en optimal segment sayısını belirler
- **📈 İnteraktif Grafikler**: Segmentlerin dağılımı ve karakteristik özellikleri için görselleştirmeler
- **🤖 AI Pazarlama Önerileri**: Her segment için OpenAI API kullanarak özel pazarlama stratejileri
- **📱 Duyarlı Tasarım**: Farklı cihazlarda sorunsuz çalışan modern arayüz

## 🛠️ Teknolojiler

- **Backend**: Python, Flask, Pandas, NumPy, scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Veri Görselleştirme**: Plotly, Seaborn
- **Yapay Zeka**: OpenAI GPT API entegrasyonu

## ⚙️ Kurulum

### Ön Gereksinimler

- Python 3.8 veya üzeri
- pip (Python paket yöneticisi)
- OpenAI API anahtarı (isteğe bağlı)

### Adım Adım Kurulum

1. Projeyi klonlayın:
   ```
   git clone https://github.com/ozzy2438/SegmentWise.git
   cd SegmentWise
   ```

2. Sanal ortam oluşturun ve aktifleştirin:
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac için
   venv\Scripts\activate     # Windows için
   ```

3. Gerekli paketleri yükleyin:
   ```
   pip install -r requirements.txt
   ```

4. (İsteğe bağlı) OpenAI API entegrasyonu için `.env` dosyası oluşturun:
   ```
   touch .env
   echo "OPENAI_API_KEY=sizin_api_anahtarınız" >> .env
   ```

5. Uygulamayı başlatın:
   ```
   python src/app.py
   ```

6. Tarayıcınızda şu adresi açın: `http://localhost:8080`

## 📊 Kullanım Kılavuzu

### 1. Veri Yükleme ve Segmentasyon

1. Ana sayfada "CSV Yükle" butonuna tıklayın
2. Müşteri verilerinizi içeren CSV dosyasını seçin
3. İstediğiniz segment sayısını seçin veya otomatik belirleme için boş bırakın
4. "Analiz Başlat" butonuna tıklayarak segmentasyon işlemini başlatın

### 2. Segmentleri İnceleme

- Oluşturulan her segment için detaylı profiller görüntülenir
- Demografik ve davranışsal özellikler grafik ve tablolarla sunulur
- Segment dağılımları ve büyüklükleri görsel olarak gösterilir

### 3. Pazarlama Stratejileri

- Her segment için AI tarafından önerilen:
  - Kampanya fikirleri
  - İletişim kanalları
  - Özel teklifler ve indirimler
  - Strateji gerekçeleri

## 📋 Proje Yapısı

```
SegmentWise/
├── src/                  # Kaynak kodları
│   ├── app.py            # Flask uygulaması ve route'lar
│   ├── data_processor.py # Veri işleme ve temizleme
│   ├── segmentation.py   # Segment oluşturma algoritmaları
│   └── recommendation.py # OpenAI API entegrasyonu
├── templates/            # HTML şablonları  
├── static/               # CSS, JS ve görseller
├── uploads/              # Yüklenen CSV dosyaları
├── requirements.txt      # Bağımlılıklar
└── README.md             # Proje dokümantasyonu
```

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Özellik istekleri, hata raporları veya pull request'ler için GitHub üzerinden iletişime geçebilirsiniz.

## 📜 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasını inceleyebilirsiniz.

## 🙏 Teşekkürler

Bu proje aşağıdaki açık kaynak kütüphanelerinin yardımıyla geliştirilmiştir:
- [Flask](https://flask.palletsprojects.com/)
- [scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Plotly](https://plotly.com/)
- [OpenAI](https://openai.com/)

---

<p align="center">
  <a href="https://linkedin.com/in/your-linkedin-profile">LinkedIn'de bağlantı kurun</a> • 
  <a href="mailto:your-email@example.com">İletişim</a>
</p> 