#!/usr/bin/env python3
"""
RinominatorAI - Main Script
Versione per esecuzione locale (senza Colab)
"""

import argparse
from pathlib import Path
from pdf_to_image import convert_pdf_first_page
from ocr_extractor import extract_text_with_ocr
from field_detector import extract_fields
from file_renamer import rename_and_copy

def main():
    parser = argparse.ArgumentParser(
        description="RinominatorAI - Rinomina automatica PDF"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Cartella contenente i PDF da processare"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Cartella dove salvare i PDF rinominati"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/config.yaml",
        help="File di configurazione"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_path.glob("*.pdf"))
    
    print(f"🤖 RinominatorAI")
    print(f"📂 Trovati {len(pdf_files)} PDF da processare\n")
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_file.name}")
        
        try:
            # Converti PDF
            image_path = convert_pdf_first_page(pdf_file)
            
            # OCR
            extracted_data = extract_text_with_ocr(image_path)
            
            # Estrai campi
            fields = extract_fields(extracted_data)
            
            # Rinomina
            new_path = rename_and_copy(
                pdf_file, 
                output_path, 
                fields
            )
            
            print(f"  ✅ → {new_path.name}\n")
            
        except Exception as e:
            print(f"  ❌ Errore: {e}\n")
    
    print(f"✅ Processo completato!")
    print(f"📁 File salvati in: {output_path}")

if __name__ == "__main__":
    main()