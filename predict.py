#!/usr/bin/env python3
"""
Script per applicare il modello addestrato a nuovi PDF
"""

import argparse
from pathlib import Path
import shutil
import sys
import re

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from predictor import PDFPredictor


def clean_filename(text):
    """Pulisce il testo per nome file valido."""
    # Rimuovi caratteri non validi
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    # Limita lunghezza
    text = text[:150]
    return text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="🤖 RinominatorAI - Predizione con modello addestrato"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Cartella con PDF da rinominare"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Cartella dove salvare PDF rinominati"
    )
    parser.add_argument(
        "--model", "-m",
        default="models/trained_model.pkl",
        help="Path al modello addestrato (default: models/trained_model.pkl)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("🤖 MODALITÀ PREDIZIONE")
    print("="*60)
    
    # Carica modello
    try:
        print(f"📂 Caricamento modello: {args.model}")
        predictor = PDFPredictor(args.model)
        print(f"✓ Modello caricato (training: {predictor.stats['n_samples']} documenti, acc: {predictor.stats['accuracy']:.1%})")
    except Exception as e:
        print(f"\n❌ ERRORE caricamento modello: {e}")
        print("\n💡 Devi prima addestrare il modello:")
        print("   python train.py --training-folder ./pdf_training\n")
        sys.exit(1)
    
    # Trova PDF
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"\n⚠️  Nessun PDF trovato in: {input_path}")
        sys.exit(1)
    
    print(f"\n📂 Trovati {len(pdf_files)} PDF da rinominare\n")
    
    # Processa ogni PDF
    successes = 0
    failures = 0
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"📄 [{i}/{len(pdf_files)}] Elaborazione: {pdf_path.name}")
        
        try:
            # Converti e predici
            print("  🔍 Conversione PDF → Immagine...")
            print("  🔍 OCR in corso...")
            print("  🤖 Estrazione con AI...")
            
            fields, confidences = predictor.process_pdf(pdf_path)
            
            # Mostra risultati
            print(f"    ✓ Denominazione: {fields['denominazione']} (confidenza: {confidences['denominazione']:.0%})")
            print(f"    ✓ Numero: {fields['numero_documento']} (confidenza: {confidences['numero_documento']:.0%})")
            print(f"    ✓ Data: {fields['data_documento']} (confidenza: {confidences['data_documento']:.0%})")
            
            # Crea nuovo nome
            new_name = f"{fields['denominazione']} {fields['numero_documento']} del {fields['data_documento']}.pdf"
            new_name = clean_filename(new_name)
            
            # Copia file rinominato
            new_path = output_path / new_name
            shutil.copy2(pdf_path, new_path)
            
            print(f"  ✅ Rinominato: {new_name}\n")
            successes += 1
            
        except Exception as e:
            print(f"  ❌ Errore: {e}\n")
            failures += 1
    
    # Risultati finali
    print("="*60)
    print("✅ Processo completato!")
    print("="*60)
    print(f"📊 RISULTATI:")
    print(f"  - PDF processati: {len(pdf_files)}")
    print(f"  - Successi: {successes} ({successes/len(pdf_files)*100:.0%})")
    print(f"  - Falliti: {failures} ({failures/len(pdf_files)*100:.0%})")
    print(f"  - File salvati in: {output_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
