#!/usr/bin/env python3
"""
Predict Script
Script principale per applicare il modello addestrato a nuovi PDF
"""

import argparse
import sys
from pathlib import Path

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def main():
    parser = argparse.ArgumentParser(
        description="🤖 RinominatorAI - Predizione con modello addestrato",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python predict.py --input ./pdf_nuovi --output ./output
  python predict.py --input ./docs --output ./renamed --model ./models/custom_model.pkl
  python predict.py --input ./data --output ./result --no-gpu

Il modello deve essere stato addestrato prima usando train.py
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Cartella contenente i PDF da rinominare'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Cartella dove salvare i PDF rinominati'
    )
    
    parser.add_argument(
        '--model',
        default='models/trained_model.pkl',
        help='Path al modello addestrato (default: models/trained_model.pkl)'
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
    input_folder = Path(args.input)
    if not input_folder.exists():
        print(f"❌ Errore: Cartella input non trovata: {input_folder}")
        sys.exit(1)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Errore: Modello non trovato: {model_path}")
        print(f"\n💡 Devi prima addestrare il modello usando:")
        print(f"   python train.py --training-folder ./pdf_rinominati")
        sys.exit(1)
    
    pdf_files = list(input_folder.glob("*.pdf"))
    if len(pdf_files) == 0:
        print(f"⚠️  Attenzione: Nessun PDF trovato in {input_folder}")
        sys.exit(0)
    
    # Import predictor (lazy to allow --help without dependencies)
    try:
        from predictor import PDFPredictor
    except ImportError as e:
        print(f"❌ Errore: Dipendenze mancanti: {e}")
        print(f"\n💡 Installa le dipendenze con:")
        print(f"   pip install -r requirements.txt")
        sys.exit(1)
    
    try:
        # Crea predictor
        predictor = PDFPredictor(args.model)
        
        # Esegui predizione
        stats = predictor.predict_from_folder(
            input_folder=args.input,
            output_folder=args.output,
            languages=args.languages,
            gpu=not args.no_gpu
        )
        
        # Mostra statistiche finali
        success_rate = (stats['n_success'] / stats['n_processed'] * 100) if stats['n_processed'] > 0 else 0
        
        print(f"\n📊 Statistiche finali:")
        print(f"   Tasso di successo: {success_rate:.1f}%")
        
        if stats['n_errors'] > 0:
            print(f"\n💡 Alcuni file hanno avuto errori. Possibili cause:")
            print(f"   - PDF corrotti o non leggibili")
            print(f"   - Qualità immagine troppo bassa")
            print(f"   - Layout molto diverso dai dati di training")
        
    except Exception as e:
        print(f"\n❌ Errore durante predizione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
