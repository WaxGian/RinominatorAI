#!/usr/bin/env python3
"""
PDF Trainer
Modulo per il training da PDF già rinominati correttamente
"""

import re
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

from pdf_to_image import convert_pdf_first_page
from ocr_extractor import extract_text_with_ocr


class PDFTrainer:
    """Classe per il training del modello di rinominazione PDF"""
    
    def __init__(self):
        self.training_data = []
        self.model = None
    
    def parse_filename(self, filename):
        """
        Estrae denominazione, numero, data dal nome file.
        
        Supporta pattern:
        - "Denominazione NumDoc del Data.pdf"
        - "Denominazione_NumDoc_Data.pdf"
        - "Denominazione - Numero - Data.pdf"
        
        Args:
            filename: Nome del file (senza path)
            
        Returns:
            Dict con: denominazione, numero_documento, data_documento
            None se non riesce a parsificare
        """
        # Rimuovi estensione
        name = Path(filename).stem
        
        # Pattern 1: "Denominazione NumDoc del Data"
        pattern1 = r'^(.+?)\s+(\S+)\s+del\s+(.+)$'
        match = re.match(pattern1, name)
        if match:
            return {
                'denominazione': match.group(1).strip(),
                'numero_documento': match.group(2).strip(),
                'data_documento': match.group(3).strip()
            }
        
        # Pattern 2: "Denominazione - Numero - Data"
        pattern2 = r'^(.+?)\s*-\s*(\S+)\s*-\s*(.+)$'
        match = re.match(pattern2, name)
        if match:
            return {
                'denominazione': match.group(1).strip(),
                'numero_documento': match.group(2).strip(),
                'data_documento': match.group(3).strip()
            }
        
        # Pattern 3: "Denominazione_Numero_Data" (separato da underscore)
        # Cerca pattern data per identificarla
        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
        date_match = re.search(date_pattern, name)
        
        if date_match:
            data_str = date_match.group()
            # Dividi sul pattern della data
            parts = name.split(data_str)
            if len(parts) >= 2:
                before_date = parts[0].strip('_- ')
                # Cerca l'ultimo numero prima della data
                num_pattern = r'(\S+\d+\S*)$'
                num_match = re.search(num_pattern, before_date)
                if num_match:
                    numero = num_match.group(1)
                    denominazione = before_date[:num_match.start()].strip('_- ')
                    return {
                        'denominazione': denominazione,
                        'numero_documento': numero,
                        'data_documento': data_str
                    }
        
        return None
    
    def train_from_folder(self, training_folder, languages=['it', 'en'], gpu=True):
        """
        Addestra il modello da una cartella di PDF già rinominati.
        
        Args:
            training_folder: Path alla cartella con PDF rinominati
            languages: Lingue per OCR
            gpu: Usa GPU per OCR
            
        Returns:
            Dict con statistiche del training
        """
        training_folder = Path(training_folder)
        
        if not training_folder.exists():
            raise ValueError(f"Cartella non trovata: {training_folder}")
        
        pdf_files = list(training_folder.glob("*.pdf"))
        
        if len(pdf_files) < 5:
            raise ValueError(f"Numero insufficiente di PDF per training. Trovati {len(pdf_files)}, minimo 5 richiesti.")
        
        print(f"🎓 TRAINING MODE")
        print(f"📂 Trovati {len(pdf_files)} PDF già rinominati\n")
        
        # Raccogli dati di training
        valid_samples = 0
        
        for pdf_file in pdf_files:
            print(f"📄 Analisi: {pdf_file.name}")
            
            # Parsifica nome file per ottenere ground truth
            ground_truth = self.parse_filename(pdf_file.name)
            
            if ground_truth is None:
                print(f"  ⚠️  Nome file non parsificabile, saltato\n")
                continue
            
            print(f"  ✓ Denominazione: {ground_truth['denominazione']}")
            print(f"  ✓ Numero: {ground_truth['numero_documento']}")
            print(f"  ✓ Data: {ground_truth['data_documento']}")
            
            try:
                # Converti PDF a immagine
                image_path = convert_pdf_first_page(pdf_file)
                
                # Esegui OCR
                ocr_data = extract_text_with_ocr(image_path, languages=languages, gpu=gpu)
                
                print(f"  ✓ OCR completato: {len(ocr_data)} elementi trovati\n")
                
                # Salva dati di training
                self.training_data.append({
                    'filename': pdf_file.name,
                    'ground_truth': ground_truth,
                    'ocr_data': ocr_data
                })
                
                valid_samples += 1
                
            except Exception as e:
                print(f"  ❌ Errore durante OCR: {e}\n")
                continue
        
        if valid_samples < 5:
            raise ValueError(f"Training fallito: solo {valid_samples} campioni validi raccolti (minimo 5)")
        
        print(f"✅ Dati raccolti: {valid_samples} campioni validi")
        print(f"🤖 Addestramento modello in corso...\n")
        
        # Addestra modello
        self._train_model()
        
        return {
            'n_samples': valid_samples,
            'n_total_files': len(pdf_files)
        }
    
    def _train_model(self):
        """
        Addestra il modello ML sui dati raccolti.
        Usa KMeans per zone detection e Random Forest per text classification.
        """
        # Prepara dati per training
        positions_data = defaultdict(list)
        text_data = defaultdict(list)
        labels_data = defaultdict(list)
        
        for sample in self.training_data:
            ground_truth = sample['ground_truth']
            ocr_data = sample['ocr_data']
            
            # Per ogni elemento OCR, cerca corrispondenze con ground truth
            for element in ocr_data:
                text = element['text'].strip()
                center = element['center']
                
                # Determina a quale campo appartiene questo testo
                field = None
                
                # Check denominazione (fuzzy match)
                if ground_truth['denominazione'].lower() in text.lower() or \
                   text.lower() in ground_truth['denominazione'].lower():
                    field = 'denominazione'
                
                # Check numero
                elif ground_truth['numero_documento'] in text:
                    field = 'numero_documento'
                
                # Check data
                elif ground_truth['data_documento'].replace('-', '/') in text.replace('-', '/') or \
                     ground_truth['data_documento'].replace('-', '') in text.replace('-', '').replace('/', ''):
                    field = 'data_documento'
                
                if field:
                    positions_data[field].append([center['x'], center['y']])
                    text_data[field].append(text)
                    labels_data[field].append(field)
        
        # Addestra modelli per ogni campo
        self.model = {
            'zone_patterns': {},
            'text_classifiers': {},
            'vectorizers': {},
            'training_stats': {}
        }
        
        # Zone patterns (statistiche posizionali)
        for field in ['denominazione', 'numero_documento', 'data_documento']:
            if field in positions_data and len(positions_data[field]) > 0:
                positions = np.array(positions_data[field])
                
                # Calcola statistiche posizionali
                x_coords = positions[:, 0]
                y_coords = positions[:, 1]
                
                self.model['zone_patterns'][field] = {
                    'x_range': (float(np.min(x_coords)), float(np.max(x_coords))),
                    'y_range': (float(np.min(y_coords)), float(np.max(y_coords))),
                    'x_mean': float(np.mean(x_coords)),
                    'y_mean': float(np.mean(y_coords)),
                    'x_std': float(np.std(x_coords)),
                    'y_std': float(np.std(y_coords))
                }
        
        # Text classifiers (TF-IDF + Random Forest)
        all_texts = []
        all_labels = []
        
        for field in text_data:
            all_texts.extend(text_data[field])
            all_labels.extend([field] * len(text_data[field]))
        
        if len(all_texts) > 0:
            # TF-IDF Vectorizer
            vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
            X = vectorizer.fit_transform(all_texts)
            
            # Random Forest Classifier
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X, all_labels)
            
            self.model['vectorizers']['main'] = vectorizer
            self.model['text_classifiers']['main'] = clf
        
        # Training stats
        self.model['training_stats'] = {
            'n_samples': len(self.training_data),
            'fields_found': list(positions_data.keys()),
            'samples_per_field': {field: len(positions_data[field]) for field in positions_data}
        }
        
        print(f"✅ Training completato!")
        print(f"📊 Campi identificati: {', '.join(self.model['training_stats']['fields_found'])}")
    
    def save_model(self, output_path):
        """
        Salva il modello addestrato.
        
        Args:
            output_path: Path dove salvare il modello
        """
        if self.model is None:
            raise ValueError("Nessun modello da salvare. Esegui prima train_from_folder()")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"💾 Modello salvato: {output_path}")
        print(f"   Dimensione: {output_path.stat().st_size / 1024:.1f} KB")
