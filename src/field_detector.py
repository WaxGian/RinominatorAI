#!/usr/bin/env python3
"""
Field Detector
Estrae campi specifici (denominazione, numero, data) dai dati OCR
"""

import re
from datetime import datetime
import yaml
from pathlib import Path


def load_config(config_path="config/config.yaml"):
    """Carica configurazione"""
    config_path = Path(config_path)
    if not config_path.exists():
        # Usa configurazione di default
        return {
            'extraction': {
                'fields': {
                    'denominazione': {
                        'zone': {'x_max': 0.4, 'y_max': 0.3},
                        'keywords': ['denominazione', 'ragione sociale', 'ditta', 'fornitore']
                    },
                    'numero_documento': {
                        'zone': {'x_min': 0.3, 'x_max': 0.7, 'y_max': 0.5},
                        'keywords': ['n.', 'numero', 'n°', 'nr', 'fattura', 'doc'],
                        'patterns': [r'\b\d{3,}\b']
                    },
                    'data_documento': {
                        'zone': {'x_min': 0.5, 'y_max': 0.5},
                        'keywords': ['data', 'del', 'emissione'],
                        'patterns': [r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}']
                    }
                }
            }
        }
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def is_in_zone(element, zone_config):
    """
    Controlla se un elemento è nella zona specificata.
    
    Args:
        element: Dizionario con 'center' e 'normalized_bbox'
        zone_config: Configurazione della zona (x_min, x_max, y_min, y_max)
        
    Returns:
        True se l'elemento è nella zona
    """
    center = element['center']
    x, y = center['x'], center['y']
    
    x_min = zone_config.get('x_min', 0)
    x_max = zone_config.get('x_max', 1)
    y_min = zone_config.get('y_min', 0)
    y_max = zone_config.get('y_max', 1)
    
    return x_min <= x <= x_max and y_min <= y <= y_max


def contains_keyword(text, keywords):
    """Controlla se il testo contiene una delle keywords"""
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def matches_pattern(text, patterns):
    """Controlla se il testo matcha uno dei pattern regex"""
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def extract_date(text):
    """
    Estrae una data dal testo.
    
    Returns:
        Data formattata come stringa (gg-mm-aaaa) o None
    """
    # Pattern comuni per date
    date_patterns = [
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # gg/mm/aaaa o gg-mm-aaaa
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{2})',   # gg/mm/aa o gg-mm-aa
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            day, month, year = match.groups()
            
            # Converti anno a 2 cifre in 4 cifre
            if len(year) == 2:
                year = '20' + year if int(year) < 50 else '19' + year
            
            # Valida data
            try:
                date_obj = datetime(int(year), int(month), int(day))
                return date_obj.strftime('%d-%m-%Y')
            except ValueError:
                continue
    
    return None


def extract_number(text):
    """
    Estrae un numero documento dal testo.
    
    Returns:
        Numero come stringa o None
    """
    # Cerca numeri di almeno 3 cifre
    match = re.search(r'\b\d{3,}\b', text)
    if match:
        return match.group()
    
    # Cerca pattern tipo "FT2024-001" o "N.12345"
    match = re.search(r'[A-Z]*\.?\s*\d{3,}[-/]?\d*', text, re.IGNORECASE)
    if match:
        return match.group().strip()
    
    return None


def extract_fields(ocr_data, config_path="config/config.yaml"):
    """
    Estrae i campi denominazione, numero, data dai dati OCR.
    
    Args:
        ocr_data: Lista di elementi estratti da OCR
        config_path: Path al file di configurazione
        
    Returns:
        Dizionario con: denominazione, numero_documento, data_documento
    """
    config = load_config(config_path)
    field_configs = config['extraction']['fields']
    
    results = {
        'denominazione': None,
        'numero_documento': None,
        'data_documento': None
    }
    
    # Estrai denominazione (primo testo significativo in alto a sinistra)
    denom_config = field_configs['denominazione']
    for element in ocr_data:
        if is_in_zone(element, denom_config['zone']):
            text = element['text'].strip()
            # Prendi testi con almeno 3 caratteri e alta confidenza
            if len(text) >= 3 and element['confidence'] > 0.5:
                # Escludi se è solo un numero o una data
                if not text.isdigit() and not extract_date(text):
                    results['denominazione'] = text
                    break
    
    # Estrai numero documento
    num_config = field_configs['numero_documento']
    for element in ocr_data:
        if is_in_zone(element, num_config['zone']):
            text = element['text'].strip()
            # Cerca nel testo corrente o vicino a keywords
            numero = extract_number(text)
            if numero:
                results['numero_documento'] = numero
                break
            # Se contiene keyword, cerca nel prossimo elemento
            if contains_keyword(text, num_config['keywords']):
                # Trova elementi vicini
                for other in ocr_data:
                    if other == element:
                        continue
                    # Controlla se è vicino (stessa riga circa)
                    y_diff = abs(other['center']['y'] - element['center']['y'])
                    if y_diff < 0.05:  # 5% dell'altezza
                        numero = extract_number(other['text'])
                        if numero:
                            results['numero_documento'] = numero
                            break
                if results['numero_documento']:
                    break
    
    # Estrai data documento
    data_config = field_configs['data_documento']
    for element in ocr_data:
        if is_in_zone(element, data_config['zone']):
            text = element['text'].strip()
            data = extract_date(text)
            if data:
                results['data_documento'] = data
                break
    
    # Se non trovato nella zona, cerca in tutto il documento
    if not results['data_documento']:
        for element in ocr_data:
            data = extract_date(element['text'])
            if data:
                results['data_documento'] = data
                break
    
    return results
