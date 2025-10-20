#!/usr/bin/env python3
"""
PDF Predictor
Applica il modello addestrato per rinominare nuovi PDF
"""

import pickle
import re
from pathlib import Path
import numpy as np

from pdf_to_image import convert_pdf_first_page
from ocr_extractor import extract_text_with_ocr
from file_renamer import rename_and_copy


class PDFPredictor:
    """Classe per applicare il modello di rinominazione addestrato"""
    
    def __init__(self, model_path):
        """
        Inizializza il predictor caricando il modello.
        
        Args:
            model_path: Path al file del modello (.pkl)
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modello non trovato: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"✅ Modello caricato: {model_path.name}")
        print(f"   Training samples: {self.model['training_stats']['n_samples']}")
        print(f"   Campi: {', '.join(self.model['training_stats']['fields_found'])}\n")
    
    def predict_fields(self, ocr_data):
        """
        Predice i campi da dati OCR usando il modello addestrato.
        
        Args:
            ocr_data: Lista di elementi estratti da OCR
            
        Returns:
            Dict con: denominazione, numero_documento, data_documento
        """
        results = {
            'denominazione': None,
            'numero_documento': None,
            'data_documento': None
        }
        
        # Per ogni campo, trova il miglior candidato
        for field in ['denominazione', 'numero_documento', 'data_documento']:
            candidates = self._find_candidates(field, ocr_data)
            
            if candidates:
                # Prendi il candidato con score più alto
                best_candidate = max(candidates, key=lambda x: x['score'])
                results[field] = best_candidate['text']
        
        return results
    
    def _find_candidates(self, field, ocr_data):
        """
        Trova candidati per un campo specifico.
        
        Args:
            field: Nome del campo
            ocr_data: Dati OCR
            
        Returns:
            Lista di candidati con score
        """
        candidates = []
        
        if field not in self.model['zone_patterns']:
            return candidates
        
        zone = self.model['zone_patterns'][field]
        
        # Filtra elementi nella zona appropriata
        for element in ocr_data:
            center = element['center']
            
            # Calcola quanto l'elemento è vicino alla zona del campo
            x_dist = self._distance_to_range(center['x'], zone['x_range'])
            y_dist = self._distance_to_range(center['y'], zone['y_range'])
            
            # Score basato su distanza (più vicino = score più alto)
            position_score = 1.0 / (1.0 + x_dist + y_dist)
            
            # Score basato su classificatore testuale (se disponibile)
            text_score = 0.5
            if 'main' in self.model['text_classifiers']:
                try:
                    vectorizer = self.model['vectorizers']['main']
                    classifier = self.model['text_classifiers']['main']
                    
                    X = vectorizer.transform([element['text']])
                    proba = classifier.predict_proba(X)[0]
                    
                    # Trova indice della classe
                    classes = classifier.classes_
                    if field in classes:
                        field_idx = list(classes).index(field)
                        text_score = proba[field_idx]
                except:
                    pass
            
            # Score combinato
            combined_score = (position_score * 0.6 + text_score * 0.4) * element['confidence']
            
            # Applica filtri specifici per campo
            if field == 'numero_documento':
                # Deve contenere numeri
                if not re.search(r'\d', element['text']):
                    combined_score *= 0.1
            
            elif field == 'data_documento':
                # Deve matchare pattern data
                if not re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', element['text']):
                    combined_score *= 0.1
            
            elif field == 'denominazione':
                # Deve essere testo significativo (non solo numeri)
                if element['text'].isdigit():
                    combined_score *= 0.1
                # Deve essere abbastanza lungo
                if len(element['text']) < 3:
                    combined_score *= 0.5
            
            # Threshold minimo
            if combined_score >= 0.3:
                candidates.append({
                    'text': element['text'].strip(),
                    'score': combined_score,
                    'element': element
                })
        
        return candidates
    
    def _distance_to_range(self, value, range_tuple):
        """
        Calcola la distanza di un valore da un range.
        
        Args:
            value: Valore da testare
            range_tuple: (min, max)
            
        Returns:
            Distanza (0 se dentro il range)
        """
        min_val, max_val = range_tuple
        
        if min_val <= value <= max_val:
            return 0.0
        elif value < min_val:
            return min_val - value
        else:
            return value - max_val
    
    def predict_from_folder(self, input_folder, output_folder, languages=['it', 'en'], gpu=True):
        """
        Applica il modello a tutti i PDF in una cartella.
        
        Args:
            input_folder: Cartella con PDF da rinominare
            output_folder: Cartella dove salvare PDF rinominati
            languages: Lingue per OCR
            gpu: Usa GPU per OCR
            
        Returns:
            Dict con statistiche
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        if not input_folder.exists():
            raise ValueError(f"Cartella non trovata: {input_folder}")
        
        pdf_files = list(input_folder.glob("*.pdf"))
        
        if len(pdf_files) == 0:
            print(f"⚠️  Nessun PDF trovato in {input_folder}")
            return {'n_processed': 0, 'n_success': 0, 'n_errors': 0}
        
        print(f"🤖 PREDICTION MODE")
        print(f"📂 Trovati {len(pdf_files)} PDF da rinominare\n")
        
        n_success = 0
        n_errors = 0
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"[{i}/{len(pdf_files)}] 📄 Elaborazione: {pdf_file.name}")
            
            try:
                # Converti PDF a immagine
                print(f"  🔍 OCR...")
                image_path = convert_pdf_first_page(pdf_file)
                
                # Esegui OCR
                ocr_data = extract_text_with_ocr(image_path, languages=languages, gpu=gpu)
                
                # Predici campi
                print(f"  🤖 Predizione con AI...")
                fields = self.predict_fields(ocr_data)
                
                # Rinomina file
                new_path = rename_and_copy(pdf_file, output_folder, fields)
                
                print(f"  ✅ Rinominato: {new_path.name}\n")
                n_success += 1
                
            except Exception as e:
                print(f"  ❌ Errore: {e}\n")
                n_errors += 1
        
        print(f"\n{'='*60}")
        print(f"✅ Processo completato!")
        print(f"   Successi: {n_success}/{len(pdf_files)}")
        print(f"   Errori: {n_errors}/{len(pdf_files)}")
        print(f"📁 File salvati in: {output_folder}")
        
        return {
            'n_processed': len(pdf_files),
            'n_success': n_success,
            'n_errors': n_errors
        }
