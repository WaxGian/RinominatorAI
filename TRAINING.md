# 🎓 Sistema di Training Automatico

Documentazione completa per il sistema di training automatico da PDF pre-rinominati.

## 📋 Panoramica

Il sistema di training permette di:
1. **Apprendere** da PDF già rinominati correttamente
2. **Estrarre** pattern dai nomi file (denominazione, numero documento, data)
3. **Addestrare** modelli ML per riconoscimento automatico
4. **Applicare** il modello per rinominare nuovi PDF automaticamente

## 🚀 Quick Start

### 1. Preparazione Dati di Training

Crea una cartella con PDF già rinominati seguendo uno di questi pattern:

```
pdf_rinominati/
├── Fornitore ABC 12345 del 15-01-2024.pdf
├── Ditta XYZ 67890 del 20-03-2024.pdf
├── Azienda Test 999 del 01-12-2023.pdf
└── ...
```

**Pattern supportati:**
- `Denominazione NumDoc del Data.pdf` ✅ (consigliato)
- `Denominazione_NumDoc_Data.pdf` ⚠️  (limitato)
- `Denominazione - Numero - Data.pdf` ⚠️  (limitato)

**Requisiti minimi:**
- Almeno 5 PDF (consigliato: 10-20)
- Nomi file consistenti
- PDF leggibili e non corrotti

### 2. Training

```bash
# Training base
python train.py --training-folder ./pdf_rinominati

# Training con opzioni custom
python train.py \
  --training-folder ./data/training \
  --model ./models/custom_model.pkl \
  --languages it en \
  --no-gpu
```

**Output del training:**
```
🎓 TRAINING MODE
📂 Trovati 20 PDF già rinominati

📄 Analisi: Fornitore ABC 12345 del 15-01-2024.pdf
  ✓ Denominazione: Fornitore ABC
  ✓ Numero: 12345
  ✓ Data: 15-01-2024
  ✓ OCR completato: 42 elementi trovati

...

✅ Training completato!
📊 Campi identificati: denominazione, numero_documento, data_documento
💾 Modello salvato: models/trained_model.pkl
```

### 3. Predizione

```bash
# Applica il modello addestrato
python predict.py --input ./pdf_nuovi --output ./output

# Con modello custom
python predict.py \
  --input ./docs \
  --output ./renamed \
  --model ./models/custom_model.pkl
```

**Output della predizione:**
```
🤖 PREDICTION MODE
📂 Trovati 10 PDF da rinominare

[1/10] 📄 Elaborazione: documento_001.pdf
  🔍 OCR...
  🤖 Predizione con AI...
  ✅ Rinominato: Fornitore XYZ 67890 del 20-03-2024.pdf

...

✅ Processo completato!
   Successi: 9/10
   Errori: 1/10
```

## 🔧 Architettura Tecnica

### Moduli Creati

#### `src/trainer.py`
- **PDFTrainer**: Classe per training del modello
- **Funzionalità:**
  - `parse_filename()`: Estrae ground truth dai nomi file
  - `train_from_folder()`: Addestra su cartella di PDF
  - `_train_model()`: ML training (KMeans + Random Forest + TF-IDF)
  - `save_model()`: Salva modello addestrato

#### `src/predictor.py`
- **PDFPredictor**: Classe per applicare modello
- **Funzionalità:**
  - `predict_fields()`: Predice campi da OCR
  - `_find_candidates()`: Trova candidati per ogni campo
  - `predict_from_folder()`: Applica a cartella di PDF

#### `src/pdf_to_image.py`
- Conversione PDF → Immagine (prima pagina)
- Usa PyMuPDF con DPI configurabile

#### `src/ocr_extractor.py`
- Estrazione testo con EasyOCR
- Tracking posizioni normalizzate (0-1)
- Cache del reader OCR per performance

#### `src/field_detector.py`
- Rilevamento campi basato su regole
- Zone detection (top-left, center, center-right)
- Pattern matching per numeri e date

#### `src/file_renamer.py`
- Sanitizzazione nomi file
- Gestione caratteri speciali
- Gestione duplicati

### Algoritmi ML Utilizzati

1. **Zone Patterns (Statistics)**
   - Calcola statistiche posizionali per ogni campo
   - Range X/Y, media, deviazione standard
   - Usato per filtrare candidati in fase di predizione

2. **Text Classification (TF-IDF + Random Forest)**
   - TF-IDF Vectorizer: estrae features testuali
   - Random Forest Classifier: 50 estimatori
   - Threshold di confidenza: 0.3

3. **Combined Scoring**
   ```python
   score = (position_score * 0.6 + text_score * 0.4) * ocr_confidence
   ```

### Struttura Modello Salvato

```python
{
    'zone_patterns': {
        'denominazione': {
            'x_range': (0.0, 0.3),
            'y_range': (0.0, 0.2),
            'x_mean': 0.15,
            'y_mean': 0.1,
            'x_std': 0.05,
            'y_std': 0.03
        },
        # ... altri campi
    },
    'text_classifiers': {
        'main': RandomForestClassifier(...)
    },
    'vectorizers': {
        'main': TfidfVectorizer(...)
    },
    'training_stats': {
        'n_samples': 20,
        'fields_found': ['denominazione', 'numero_documento', 'data_documento'],
        'samples_per_field': {...}
    }
}
```

## 📊 Performance e Requisiti

### Performance Attese
- **Training:** ~5-10 secondi per PDF (con GPU)
- **Predizione:** ~2-5 secondi per PDF (con GPU)
- **Accuratezza:** 85-95% (dipende da qualità e consistenza)

### Requisiti Sistema
- **Python:** 3.8+
- **RAM:** 4GB+ (8GB+ consigliato)
- **GPU:** Opzionale ma consigliata (CUDA compatible)
- **Storage:** ~2GB per modelli base

### Dipendenze
```txt
PyMuPDF>=1.23.0          # PDF processing
Pillow>=10.0.0           # Image handling
easyocr>=1.7.0           # OCR engine
torch>=2.0.0             # Deep learning
numpy>=1.24.0            # Numerical computing
scikit-learn>=1.3.0      # Machine learning
PyYAML>=6.0              # Configuration
```

## 🔍 Troubleshooting

### Errore: "Numero insufficiente di PDF"
```
❌ Errore: Numero insufficiente di PDF.
   Trovati: 3 PDF
   Richiesti: almeno 5 PDF per training affidabile
```

**Soluzione:** Rinomina manualmente almeno 10-20 PDF seguendo il pattern corretto.

### Errore: "Nome file non parsificabile"
```
⚠️  Nome file non parsificabile, saltato
```

**Cause:**
- Pattern non riconosciuto
- Nome file non segue convenzioni

**Soluzione:** Usa il pattern "Denominazione NumDoc del Data.pdf"

### Errore: "Modello non trovato"
```
❌ Errore: Modello non trovato: models/trained_model.pkl
```

**Soluzione:** Esegui prima il training con `train.py`

### Bassa Accuratezza

**Possibili cause:**
1. **Training insufficiente:** < 10 campioni
2. **Documenti inconsistenti:** Layout molto variabile
3. **OCR di bassa qualità:** PDF scansionati male

**Soluzioni:**
- Aumenta numero di esempi di training (20-50 PDF)
- Usa PDF di qualità migliore
- Raggruppa documenti simili e addestra modelli separati

### GPU Out of Memory

**Soluzioni:**
```bash
# Usa CPU invece di GPU
python train.py --training-folder ./data --no-gpu
python predict.py --input ./data --output ./result --no-gpu
```

## 🎯 Best Practices

### 1. Preparazione Dati
- ✅ Rinomina almeno 10-20 PDF rappresentativi
- ✅ Usa pattern consistente
- ✅ Verifica che i nomi siano corretti
- ❌ Non mescolare pattern diversi

### 2. Training
- ✅ Usa GPU se disponibile
- ✅ Verifica output del training
- ✅ Salva modelli con nomi descrittivi
- ❌ Non addestrare su PDF corrotti

### 3. Predizione
- ✅ Testa prima su piccolo batch
- ✅ Verifica risultati manualmente
- ✅ Usa threshold di confidenza appropriato
- ❌ Non applicare a documenti troppo diversi dal training

### 4. Manutenzione
- 🔄 Ri-addestra periodicamente con nuovi esempi
- 🔄 Mantieni backup dei modelli funzionanti
- 🔄 Monitora accuratezza nel tempo

## 📚 Esempi Avanzati

### Training con Validazione
```bash
# Prepara dati
mkdir -p data/training data/validation

# Training
python train.py --training-folder data/training --model models/v1.pkl

# Test su validation set
python predict.py --input data/validation --output data/results --model models/v1.pkl
```

### Modelli Specializzati
```bash
# Modello per fatture
python train.py --training-folder fatture/ --model models/fatture.pkl

# Modello per DDT
python train.py --training-folder ddt/ --model models/ddt.pkl

# Usa il modello appropriato
python predict.py --input nuove_fatture/ --output out/ --model models/fatture.pkl
```

## 🔗 Integrazione con Main.py

Il sistema di training è complementare a `main.py`:

- **main.py:** Usa regole predefinite (config.yaml)
- **train.py + predict.py:** Impara dai tuoi dati

**Quando usare training:**
- Hai molti documenti simili
- Le regole standard non funzionano bene
- Vuoi personalizzare per il tuo caso d'uso

**Quando usare main.py:**
- Documenti singoli o pochi
- Vuoi controllo completo sulle regole
- Non hai esempi per training

## 📝 Note

- Il modello è salvato in formato pickle
- I file temporanei sono creati in temp_images/
- Supporto multi-lingua per OCR (default: italiano e inglese)
- Gestione automatica di GPU/CPU

## 🆘 Supporto

Per problemi o domande:
1. Verifica di avere tutte le dipendenze installate
2. Controlla che i PDF siano leggibili
3. Verifica i log di errore
4. Apri una issue su GitHub con i dettagli

---

**Creato per RinominatorAI** - Sistema intelligente di rinominazione PDF
