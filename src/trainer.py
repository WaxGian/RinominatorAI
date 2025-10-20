#!/usr/bin/env python3
"""
RinominatorAI - Sistema di Training
Impara dai PDF già rinominati correttamente
"""

import re
import pickle
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import logging

# Importa moduli esistenti
try:
    from pdf_to_image import convert_pdf_first_page
    from ocr_extractor import extract_text_with_ocr
except ImportError:
    # Fallback se eseguito da root
    import sys
    sys.path.append('src')
    from pdf_to_image import convert_pdf_first_page
    from ocr_extractor import extract_text_with_ocr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFTrainer:
    """Sistema che impara dai PDF già rinominati."""
    
    def __init__(self):
        self.training_data = []
        self.zone_patterns = {}
        self.text_classifiers = {}
        self.vectorizers = {}
        self.stats = {
            'n_samples': 0,
            'accuracy': 0.0,
            'date_patterns': Counter(),
            'number_patterns': Counter(),
            'denominazione_lengths': []
        }
    
    def parse_filename(self, filename):
        """
        Estrae denominazione, numero, data dal nome file.
        
        Supporta pattern:
        - "Denominazione 12345 del 15-01-2024.pdf"
        - "Denominazione_12345_15-01-2024.pdf"
        - "Denominazione - 12345 - 15/01/2024.pdf"
        
        Returns:
            dict: {'denominazione': str, 'numero_documento': str, 'data_documento': str}
        """
        # Rimuovi estensione
        name = filename.replace('.pdf', '').replace('.PDF', '')
        
        # Pattern 1: "Nome NumDoc del Data"
        pattern1 = r'^(.+?)\s+([A-Z0-9\-/]+)\s+del\s+(.+)$'
        match = re.match(pattern1, name, re.IGNORECASE)
        if match:
            return {
                'denominazione': match.group(1).strip(),
                'numero_documento': match.group(2).strip(),
                'data_documento': match.group(3).strip()
            }
        
        # Pattern 2: "Nome_NumDoc_Data" o "Nome - NumDoc - Data"
        pattern2 = r'^(.+?)[\s_\-]+([A-Z0-9\-/]+)[\s_\-]+(.+)$'
        match = re.match(pattern2, name)
        if match:
            parts = [p.strip() for p in [match.group(1), match.group(2), match.group(3)]]
            
            # Cerca data (ultimo elemento che assomiglia a data)
            date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
            
            if re.search(date_pattern, parts[2]):
                return {
                    'denominazione': parts[0],
                    'numero_documento': parts[1],
                    'data_documento': parts[2]
                }
        
        # Pattern 3: Prova a estrarre con logica
        # Cerca data
        date_match = re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', name)
        # Cerca numero (sequenza di cifre o alfanumerico)
        number_match = re.search(r'\b[A-Z]{0,3}\d{3,}\b|\b\d{3,}\b', name)
        
        if date_match and number_match:
            data = date_match.group(0)
            numero = number_match.group(0)
            
            # Denominazione = tutto prima del numero
            denom_end = name.find(numero)
            denominazione = name[:denom_end].strip(' -_')
            
            return {
                'denominazione': denominazione,
                'numero_documento': numero,
                'data_documento': data
            }
        
        logger.warning(f"⚠️  Impossibile parsificare: {filename}")
        return None
    
    def fuzzy_find_text_in_ocr(self, search_text, ocr_data, threshold=0.7):
        """
        Cerca un testo nell'OCR con matching fuzzy.
        
        Returns:
            dict: Elemento OCR più simile o None
        """
        search_text = search_text.lower().strip()
        search_words = set(search_text.split())
        
        best_match = None
        best_score = 0
        
        for item in ocr_data:
            item_text = item['text'].lower().strip()
            item_words = set(item_text.split())
            
            # Matching esatto
            if search_text in item_text or item_text in search_text:
                return item
            
            # Jaccard similarity
            if not item_words or not search_words:
                continue
            
            intersection = len(search_words & item_words)
            union = len(search_words | item_words)
            score = intersection / union
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = item
        
        return best_match
    
    def train_from_folder(self, training_folder):
        """
        Addestra il modello da una cartella di PDF già rinominati.
        
        Args:
            training_folder: Path alla cartella con PDF rinominati
        """
        training_path = Path(training_folder)
        
        if not training_path.exists():
            raise FileNotFoundError(f"Cartella non trovata: {training_folder}")
        
        pdf_files = list(training_path.glob("*.pdf"))
        
        if len(pdf_files) < 5:
            logger.warning(f"⚠️  Trovati solo {len(pdf_files)} PDF. Consigliati almeno 10-15 per un buon training!")
        
        print("\n" + "="*60)
        print("🎓 MODALITÀ TRAINING")
        print("="*60)
        print(f"📂 Trovati {len(pdf_files)} PDF già rinominati\n")
        
        # Processa ogni PDF
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"📄 [{i}/{len(pdf_files)}] Analisi: {pdf_path.name}")
            
            try:
                # 1. Parsifica nome file (ground truth)
                fields = self.parse_filename(pdf_path.name)
                
                if not fields:
                    print("  ❌ Nome file non parsificabile, saltato\n")
                    continue
                
                print(f"  ✓ Denominazione: {fields['denominazione']}")
                print(f"  ✓ Numero: {fields['numero_documento']}")
                print(f"  ✓ Data: {fields['data_documento']}")
                
                # 2. Converti PDF e fai OCR
                image_path = convert_pdf_first_page(pdf_path)
                ocr_data = extract_text_with_ocr(image_path)
                
                print(f"  ✓ OCR completato: {len(ocr_data)} elementi trovati")
                
                # 3. Associa ground truth con OCR
                associations = self._associate_fields_with_ocr(fields, ocr_data)
                
                if associations:
                    self.training_data.append({
                        'filename': pdf_path.name,
                        'fields': fields,
                        'ocr_data': ocr_data,
                        'associations': associations
                    })
                    print("  ✓ Associazioni create")
                else:
                    print("  ⚠️  Associazioni parziali")
                
                print()
                
            except Exception as e:
                print(f"  ❌ Errore: {e}\n")
                logger.error(f"Errore processing {pdf_path.name}: {e}")
        
        # 4. Training modelli ML
        if len(self.training_data) < 3:
            raise ValueError(f"Troppi pochi esempi validi ({len(self.training_data)}). Servono almeno 5 PDF!")
        
        print("\n🤖 Training modelli Machine Learning...")
        self._train_ml_models()
        
        print("\n" + "="*60)
        print("✅ Training completato con successo!")
        print("="*60)
        self._print_statistics()
    
    def _associate_fields_with_ocr(self, fields, ocr_data):
        """Associa i campi del nome file con elementi OCR."""
        associations = {}
        
        # Cerca denominazione
        denom_match = self.fuzzy_find_text_in_ocr(fields['denominazione'], ocr_data, threshold=0.5)
        if denom_match:
            associations['denominazione'] = denom_match
        
        # Cerca numero
        num_match = self.fuzzy_find_text_in_ocr(fields['numero_documento'], ocr_data, threshold=0.8)
        if num_match:
            associations['numero_documento'] = num_match
        
        # Cerca data
        date_match = self.fuzzy_find_text_in_ocr(fields['data_documento'], ocr_data, threshold=0.7)
        if date_match:
            associations['data_documento'] = date_match
        
        return associations
    
    def _train_ml_models(self):
        """Addestra i modelli di Machine Learning."""
        
        # 1. Clustering posizionale per zone
        print("  ├─ Clustering posizionale...")
        self._train_zone_clustering()
        
        # 2. Classificatori per ogni campo
        print("  ├─ Training classificatore denominazione...")
        acc_denom = self._train_field_classifier('denominazione')
        
        print("  ├─ Training classificatore numero...")
        acc_num = self._train_field_classifier('numero_documento')
        
        print("  ├─ Training classificatore data...")
        acc_date = self._train_field_classifier('data_documento')
        
        # Statistiche
        self.stats['n_samples'] = len(self.training_data)
        self.stats['accuracy'] = (acc_denom + acc_num + acc_date) / 3
        
        print(f"  ✓ Accuratezza media: {self.stats['accuracy']:.1%}")
    
    def _train_zone_clustering(self):
        """Usa KMeans per trovare le zone tipiche dei campi."""
        
        zones = {'denominazione': [], 'numero_documento': [], 'data_documento': []}
        
        for example in self.training_data:
            for field_name, ocr_item in example['associations'].items():
                if ocr_item:
                    zones[field_name].append([ocr_item['x'], ocr_item['y']])
        
        # Calcola zone medie
        for field_name, positions in zones.items():
            if len(positions) > 0:
                positions = np.array(positions)
                
                x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
                y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
                
                # Normalizza (assumendo documento 0-1)
                self.zone_patterns[field_name] = {
                    'x_range': (float(x_min), float(x_max)),
                    'y_range': (float(y_min), float(y_max)),
                    'x_mean': float(positions[:, 0].mean()),
                    'y_mean': float(positions[:, 1].mean())
                }
    
    def _train_field_classifier(self, field_name):
        """Addestra un classificatore per un campo specifico."""
        
        texts = []
        labels = []
        
        for example in self.training_data:
            ground_truth = example['fields'][field_name]
            
            for ocr_item in example['ocr_data']:
                text = ocr_item['text']
                texts.append(text)
                
                # Label: 1 se match con ground truth, 0 altrimenti
                is_match = self._is_fuzzy_match(text, ground_truth)
                labels.append(1 if is_match else 0)
        
        if len(set(labels)) < 2:
            logger.warning(f"Dati insufficienti per {field_name}")
            return 0.0
        
        # TF-IDF + Random Forest
        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        
        # Cross-validation per stimare accuratezza
        scores = cross_val_score(clf, X, labels, cv=min(3, len(self.training_data)))
        accuracy = scores.mean()
        
        # Training finale
        clf.fit(X, labels)
        
        # Salva
        self.vectorizers[field_name] = vectorizer
        self.text_classifiers[field_name] = clf
        
        return accuracy
    
    def _is_fuzzy_match(self, text1, text2, threshold=0.6):
        """Matching fuzzy tra due stringhe."""
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if text1 == text2 or text1 in text2 or text2 in text1:
            return True
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return (intersection / union) >= threshold
    
    def save_model(self, model_path):
        """Salva il modello addestrato."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'zone_patterns': self.zone_patterns,
            'text_classifiers': self.text_classifiers,
            'vectorizers': self.vectorizers,
            'stats': self.stats,
            'training_date': datetime.now().isoformat()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Modello salvato in: {model_path}")
    
    def _print_statistics(self):
        """Stampa statistiche del training."""
        print(f"📊 STATISTICHE:")
        print(f"  - Documenti analizzati: {self.stats['n_samples']}")
        print(f"  - Accuratezza media: {self.stats['accuracy']:.1%}")
        print(f"  - Zone identificate: {len(self.zone_patterns)}")
        print("="*60)
