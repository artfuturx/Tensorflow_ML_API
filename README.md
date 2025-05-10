# Müşteri Sipariş Risk Analizi Modeli

Bu proje, müşteri siparişlerinin iade riskini tahmin eden bir makine öğrenmesi modelidir. PostgreSQL veritabanından alınan veriler üzerinde çalışarak, müşterilerin iade yapma olasılığını tahmin eder.

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. PostgreSQL veritabanı ayarlarını yapılandırın:
   - Veritabanı adı: northwind
   - Kullanıcı adı: *******
   - Şifre: (veritabanı şifreniz)
   - Host: localhost
   - Port: 5432

3. Veritabanı kullanıcısına gerekli yetkileri verin:
```sql
GRANT ALL PRIVILEGES ON DATABASE northwind TO sevgi;
```

## Proje Yapısı

```
├── main.py                 # Ana uygulama dosyası
├── requirements.txt        # Proje bağımlılıkları
├── models/                 # Eğitilmiş modellerin kaydedileceği dizin
└── src/
    ├── config.py          # Yapılandırma ayarları
    ├── database.py        # Veritabanı işlemleri
    ├── model.py           # Model tanımı ve eğitim
    └── utils.py           # Yardımcı fonksiyonlar
```

## Model Mimarisi

Model, aşağıdaki katmanlardan oluşan bir derin öğrenme ağıdır:

```python
Sequential([
    Dense(64, activation="relu", input_shape=(input_shape,)),
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dropout(0.5),
    Dense(16, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])
```

- Giriş katmanı: 64 nöron, ReLU aktivasyon
- Dropout katmanı: 0.5 oranında dropout
- Gizli katman 1: 32 nöron, ReLU aktivasyon
- Dropout katmanı: 0.5 oranında dropout
- Gizli katman 2: 16 nöron, ReLU aktivasyon
- Dropout katmanı: 0.5 oranında dropout
- Çıkış katmanı: 1 nöron, Sigmoid aktivasyon

## Callback Kullanımı

Model eğitimi sırasında callback'ler opsiyonel olarak kullanılabilir. Callback'leri aktif etmek için `src/model.py` dosyasındaki ilgili kısmın yorum satırlarını kaldırın:

```python
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint(filepath="models/best_model.keras", monitor="val_loss", save_best_only=True)
]
```

### Callback'lerin Özellikleri

1. **EarlyStopping:**
   - `monitor="val_loss"`: Doğrulama kaybını izler
   - `patience=10`: 10 epoch boyunca iyileşme olmazsa durur
   - `restore_best_weights=True`: En iyi ağırlıkları geri yükler

2. **ModelCheckpoint:**
   - `filepath="models/best_model.keras"`: En iyi modeli kaydeder
   - `monitor="val_loss"`: Doğrulama kaybını izler
   - `save_best_only=True`: Sadece en iyi modeli kaydeder

## Model Performans Karşılaştırması

### Callback'li Durum
```python
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint(filepath="models/best_model.keras", monitor="val_loss", save_best_only=True)
]
```

**Sonuçlar:**
- Loss: 0.0026881550438702106
- Accuracy: 1.0
- Eğitim süresi: Daha kısa (EarlyStopping sayesinde)
- Kaydedilen model: En iyi performans gösteren model

### Callback'siz Durum
```python
# Callback'ler yorum satırına alındı
# callbacks = [
#     EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
#     ModelCheckpoint(filepath="models/best_model.keras", monitor="val_loss", save_best_only=True)
# ]
```

**Sonuçlar:**
- Loss: 0.0023631304502487183
- Accuracy: 1.0
- Eğitim süresi: Tüm epoch'lar (50) tamamlandı
- Kaydedilen model: Son epoch'taki model

## Karşılaştırma Analizi

1. **Performans:**
   - Her iki durumda da %100 doğruluk elde edildi
   - Callback'siz durumda loss değeri biraz daha düşük (0.0023 vs 0.0026)
   - Bu, veri setinin nispeten basit olduğunu ve modelin kolayca öğrenebildiğini gösteriyor

2. **Eğitim Süreci:**
   - Callback'li durum: EarlyStopping sayesinde gereksiz epoch'lar atlanır
   - Callback'siz durum: Tüm epoch'lar çalıştırılır, daha fazla hesaplama gücü kullanılır

3. **Model Kaydı:**
   - Callback'li durum: En iyi performans gösteren model kaydedilir
   - Callback'siz durum: Son epoch'taki model kaydedilir

## Kullanım

Uygulamayı çalıştırmak için:
```bash
python main.py
```

## Sonuç

Bu projede, callback'lerin kullanılması veya kullanılmaması modelin performansını önemli ölçüde etkilemedi. Bu durum:
- Veri setinin iyi yapılandırıldığını
- Modelin uygun şekilde tasarlandığını
- Öğrenme sürecinin stabil olduğunu gösteriyor

Ancak, daha karmaşık veri setleri veya modeller için callback'lerin kullanılması önerilir, çünkü:
- Gereksiz hesaplama gücü kullanımını önler
- Aşırı öğrenmeyi (overfitting) engeller
- En iyi modeli otomatik olarak kaydeder

## Gereksinimler

Projenin çalışması için gerekli kütüphaneler:
- pg8000==1.30.3
- pandas==2.1.4
- numpy==1.26.3
- tensorflow==2.15.0
- scikit-learn==1.4.0
- python-dotenv==1.0.0
- SQLAlchemy==2.0.25 
