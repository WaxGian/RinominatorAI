#!/usr/bin/env python3
"""
Train Script
Script principale per addestrare il modello da PDF già rinominati
"""

import argparse
import sys
from pathlib import Path

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from trainer import PDFTrainer


def main():
    parser = argparse.ArgumentParser(
        description="🎓 RinominatorAI - Training da PDF pre-rinominati",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python train.py --training-folder ./pdf_rinominati
  python train.py --training-folder ./examples --model ./models/custom_model.pkl
  python train.py --training-folder ./data --no-gpu

I PDF nella cartella devono essere già rinominati correttamente usando pattern come:
  - "Fornitore ABC 12345 del 15-01-2024.pdf"
  - "Ditta XYZ - 67890 - 20/03/2024.pdf"
  - "Azienda_987_31-12-2023.pdf"
        """
    )
    
    parser.add_argument(
        '--training-folder',
        required=True,
        help='Cartella contenente i PDF già rinominati per il training'
    )
    
    parser.add_argument(
        '--model',
        default='models/trained_model.pkl',
        help='Path dove salvare il modello addestrato (default: models/trained_model.pkl)'
    )
    
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['it', 'en'],
        help='Lingue per OCR (default: it en)'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disabilita uso GPU per OCR'
    )
    
    args = parser.parse_args()
    
    # Valida input
    training_folder = Path(args.training_folder)
    if not training_folder.exists():
        print(f"❌ Errore: Cartella non trovata: {training_folder}")
        sys.exit(1)
    
    pdf_files = list(training_folder.glob("*.pdf"))
    if len(pdf_files) < 5:
        print(f"❌ Errore: Numero insufficiente di PDF.")
        print(f"   Trovati: {len(pdf_files)} PDF")
        print(f"   Richiesti: almeno 5 PDF per training affidabile")
        print(f"\n💡 Suggerimento: Rinomina manualmente almeno 10-20 PDF seguendo il pattern:")
        print(f"   'Denominazione NumDoc del Data.pdf'")
        sys.exit(1)
    
    # Crea trainer
    trainer = PDFTrainer()
    
    try:
        # Esegui training
        stats = trainer.train_from_folder(
            training_folder=args.training_folder,
            languages=args.languages,
            gpu=not args.no_gpu
        )
        
        # Salva modello
        trainer.save_model(args.model)
        
        print(f"\n{'='*60}")
        print(f"🎉 Training completato con successo!")
        print(f"   Campioni utilizzati: {stats['n_samples']}/{stats['n_total_files']}")
        print(f"   Modello salvato: {args.model}")
        print(f"\n💡 Per usare il modello:")
        print(f"   python predict.py --input ./pdf_nuovi --output ./output")
        
    except Exception as e:
        print(f"\n❌ Errore durante training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
