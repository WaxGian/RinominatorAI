#!/usr/bin/env python3
"""
OCR Text Extractor
Estrae testo e posizioni da immagini usando EasyOCR
"""

import easyocr
import numpy as np
from pathlib import Path


class OCRExtractor:
    """Classe per gestire l'estrazione OCR"""
    
    def __init__(self, languages=['it', 'en'], gpu=True):
        """
        Inizializza il reader OCR.
        
        Args:
            languages: Lista delle lingue da riconoscere
            gpu: Usa GPU se disponibile
        """
        self.reader = easyocr.Reader(languages, gpu=gpu)
    
    def extract(self, image_path):
        """
        Estrae testo e posizioni dall'immagine.
        
        Args:
            image_path: Path all'immagine
            
        Returns:
            Lista di dict con: text, bbox, confidence, normalized_bbox
        """
        image_path = Path(image_path)
        
        # Esegui OCR
        results = self.reader.readtext(str(image_path))
        
        # Ottieni dimensioni immagine
        from PIL import Image
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        # Formatta risultati
        extracted_data = []
        for bbox, text, confidence in results:
            # bbox è una lista di 4 punti: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # Calcola bounding box normalizzato (0-1)
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)
            
            # Normalizza coordinate
            norm_x_min = x_min / img_width
            norm_x_max = x_max / img_width
            norm_y_min = y_min / img_height
            norm_y_max = y_max / img_height
            
            # Calcola centro
            center_x = (norm_x_min + norm_x_max) / 2
            center_y = (norm_y_min + norm_y_max) / 2
            
            extracted_data.append({
                'text': text,
                'confidence': confidence,
                'bbox': {
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max
                },
                'normalized_bbox': {
                    'x_min': norm_x_min,
                    'y_min': norm_y_min,
                    'x_max': norm_x_max,
                    'y_max': norm_y_max
                },
                'center': {
                    'x': center_x,
                    'y': center_y
                }
            })
        
        return extracted_data


# Istanza globale per riutilizzo
_ocr_instance = None


def extract_text_with_ocr(image_path, languages=['it', 'en'], gpu=True):
    """
    Funzione helper per estrarre testo da immagine.
    Riutilizza l'istanza OCR per evitare di ricaricare il modello.
    
    Args:
        image_path: Path all'immagine
        languages: Lingue da riconoscere
        gpu: Usa GPU
        
    Returns:
        Lista di elementi estratti
    """
    global _ocr_instance
    
    if _ocr_instance is None:
        _ocr_instance = OCRExtractor(languages=languages, gpu=gpu)
    
    return _ocr_instance.extract(image_path)
