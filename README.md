# 🤖 RinominatorAI

Sistema di intelligenza artificiale per rinominare automaticamente documenti PDF estraendo informazioni chiave dalla prima pagina.

## 🎯 Obiettivo

Processa PDF di fatture/documenti estraendo:
- **Denominazione** del fornitore (alto a sinistra)
- **Numero Documento** (centro)
- **Data Documento** (centro a destra)

Rinomina i file come: `<Denominazione> <Numero Documento> del <Data Documento>.pdf`

## 🚀 Come Usare

### Opzione 1: Google Colab (Consigliato - GPU Gratuita)

1. Apri il notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WaxGianil/RinominatorAI/blob/main/RinominatorAI_Colab.ipynb)

2. Carica i tuoi PDF nella cartella `input/` su Google Drive

3. Esegui tutte le celle del notebook

4. Trova i file rinominati nella cartella `output/`

### Opzione 2: Locale (Python)

```bash
# Installa dipendenze
pip install -r requirements.txt

# Esegui il processore
python src/main.py --input ./input --output ./output
```

## 🛠️ Tecnologie Utilizzate

- **EasyOCR** - OCR multilingua con deep learning
- **PyMuPDF (fitz)** - Conversione PDF a immagini
- **PaddleOCR** - Layout analysis per identificare posizioni
- **Transformers** - Modelli AI per document understanding
- **Google Colab** - GPU gratuita T4

## 📁 Struttura Progetto

```
RinominatorAI/
├── RinominatorAI_Colab.ipynb    # Notebook principale Colab
├── src/
│   ├── main.py                   # Script principale
│   ├── pdf_to_image.py          # Conversione PDF
│   ├── ocr_extractor.py         # Estrazione testo con OCR
│   ├── field_detector.py        # Rilevamento campi AI
│   └── file_renamer.py          # Rinominazione file
├── models/
│   └── fine_tuned/              # Modelli personalizzati
├── config/
│   └── config.yaml              # Configurazione
├── examples/
│   └── sample_documents/        # PDF di esempio
├── requirements.txt
└── README.md
```

## ⚙️ Configurazione

Modifica `config/config.yaml` per personalizzare:

```yaml
ocr:
  languages: ['it', 'en']
  gpu: true

extraction:
  fields:
    denominazione:
      position: "top-left"
      keywords: ["denominazione", "ragione sociale", "fornitore"]
    
    numero_documento:
      position: "center"
      keywords: ["n.", "numero", "fattura n", "doc"]
    
    data_documento:
      position: "center-right"
      keywords: ["data", "del", "emissione"]

output:
  format: "{denominazione} {numero} del {data}.pdf"
  clean_names: true
```

## 🎓 Addestramento Personalizzato (Opzionale)

Se hai molti documenti simili, puoi addestrare un modello custom:

1. Prepara 20-50 documenti annotati
2. Esegui `notebooks/Train_Custom_Model.ipynb`
3. Il modello migliorerà l'accuratezza sui tuoi documenti specifici

## 📊 Performance

- **Velocità**: ~2-5 secondi per documento (con GPU)
- **Accuratezza**: 85-95% (dipende dalla qualità del PDF)
- **Lingue supportate**: Italiano, Inglese, Multilingua

## 🐛 Troubleshooting

### Errore: "CUDA out of memory"
Riduci batch_size in config.yaml o usa CPU

### Errore: "Field not found"
Controlla che i campi siano effettivamente presenti nel documento
Verifica la qualità dell'immagine generata

### Bassa accuratezza
Prova ad addestrare un modello personalizzato sui tuoi documenti

## 📝 Licenza

MIT License - Usa liberamente per progetti personali e commerciali

## 🤝 Contribuire

Pull requests benvenute! Per modifiche importanti, apri prima un'issue.

---

**Creato con ❤️ per automatizzare la gestione documentale**