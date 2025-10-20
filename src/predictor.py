#!/usr/bin/env python3
"""
RinominatorAI - Sistema di Predizione
Usa il modello addestrato per rinominare nuovi PDF
"""

import pickle
from pathlib import Path
import numpy as np
import logging

# Importa moduli esistenti
try:
    from pdf_to_image import convert_pdf_first_page
    from ocr_extractor import extract_text_with_ocr
except ImportError:
    import sys
    sys.path.append('src')
    from pdf_to_image import convert_pdf_first_page
    from ocr_extractor import extract_text_with_ocr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFPredictor:
    """Usa il modello addestrato per predire campi in nuovi PDF."""
    
    def __init__(self, model_path):
        """
        Carica un modello addestrato.
        
        Args:
            model_path: Path al file .pkl del modello
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modello non trovato: {model_path}\n"
                f"Devi prima addestrare il modello con: python train.py --training-folder ./pdf_training"
            )
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.zone_patterns = model_data['zone_patterns']
        self.text_classifiers = model_data['text_classifiers']
        self.vectorizers = model_data['vectorizers']
        self.stats = model_data['stats']
        
        logger.info(f"✓ Modello caricato (training: {self.stats['n_samples']} doc, acc: {self.stats['accuracy']:.1%})")
    
    def predict_fields(self, ocr_data):
        """
        Predice i campi da dati OCR.
        
        Args:
            ocr_data: Lista di dict con 'text', 'x', 'y', 'confidence'
        
        Returns:
            dict: {'denominazione': str, 'numero_documento': str, 'data_documento': str}
        """
        fields = {}
        confidences = {}
        
        for field_name in ['denominazione', 'numero_documento', 'data_documento']:
            value, confidence = self._predict_field(field_name, ocr_data)
            fields[field_name] = value
            confidences[field_name] = confidence
        
        return fields, confidences
    
    def _predict_field(self, field_name, ocr_data):
        """Predice un singolo campo."""
        
        if field_name not in self.text_classifiers:
            return self._fallback_prediction(field_name, ocr_data), 0.0
        
        classifier = self.text_classifiers[field_name]
        vectorizer = self.vectorizers[field_name]
        zone = self.zone_patterns.get(field_name)
        
        # Filtra candidati per zona
        candidates = []
        
        for item in ocr_data:
            # Controlla se è nella zona giusta
            in_zone = True
            if zone:
                x, y = item['x'], item['y']
                x_min, x_max = zone['x_range']
                y_min, y_max = zone['y_range']
                
                # Tolleranza del 20%
                margin = 0.2
                in_zone = (
                    x_min - margin <= x <= x_max + margin and
                    y_min - margin <= y <= y_max + margin
                )
            
            if in_zone:
                candidates.append(item)
        
        if not candidates:
            candidates = ocr_data  # Fallback: usa tutto
        
        # Predici con ML
        texts = [c['text'] for c in candidates]
        
        try:
            X = vectorizer.transform(texts)
            probabilities = classifier.predict_proba(X)[:, 1]
            
            # Prendi il migliore
            best_idx = np.argmax(probabilities)
            best_prob = probabilities[best_idx]
            
            if best_prob >= 0.6:  # Soglia di confidenza
                return candidates[best_idx]['text'], best_prob
        
        except Exception as e:
            logger.error(f"Errore predizione {field_name}: {e}")
        
        # Fallback
        return self._fallback_prediction(field_name, candidates), 0.5
    
    def _fallback_prediction(self, field_name, ocr_data):
        """Fallback se ML non funziona."""
        
        if field_name == 'denominazione':
            # Prendi il primo testo grande in alto a sinistra
            candidates = [item for item in ocr_data if item['y'] < 0.3 and len(item['text']) > 3]
            if candidates:
                return candidates[0]['text']
            return "Fornitore_Sconosciuto"
        
        elif field_name == 'numero_documento':
            # Cerca numeri
            import re
            for item in ocr_data:
                if re.search(r'\d{3,}', item['text']):
                    return item['text']
            return "NUM_NF"
        
        elif field_name == 'data_documento':
            # Cerca date
            import re
            date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
            for item in ocr_data:
                if re.search(date_pattern, item['text']):
                    return item['text']
            return "DATA_NF"
        
        return "N/A"
    
    def process_pdf(self, pdf_path):
        """
        Processa un PDF e predice i campi.
        
        Args:
            pdf_path: Path al PDF
        
        Returns:
            dict: Campi predetti
        """
        # Converti PDF
        image_path = convert_pdf_first_page(pdf_path)
        
        # OCR
        ocr_data = extract_text_with_ocr(image_path)
        
        # Predici
        fields, confidences = self.predict_fields(ocr_data)
        
        return fields, confidences
