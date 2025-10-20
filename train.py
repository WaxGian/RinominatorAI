#!/usr/bin/env python3
"""
Script per addestrare RinominatorAI da PDF già rinominati
"""

import argparse
from pathlib import Path
import sys

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from trainer import PDFTrainer


def main():
    parser = argparse.ArgumentParser(
        description="🎓 RinominatorAI - Training da PDF già rinominati"
    )
    parser.add_argument(
        "--training-folder", "-t",
        required=True,
        help="Cartella contenente PDF già rinominati correttamente"
    )
    parser.add_argument(
        "--output-model", "-o",
        default="models/trained_model.pkl",
        help="Dove salvare il modello addestrato (default: models/trained_model.pkl)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Mostra solo statistiche del modello esistente"
    )
    
    args = parser.parse_args()
    
    # Inizializza trainer
    trainer = PDFTrainer()
    
    # Training
    try:
        trainer.train_from_folder(args.training_folder)
        
        # Salva modello
        trainer.save_model(args.output_model)
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
