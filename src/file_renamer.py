#!/usr/bin/env python3
"""
File Renamer
Rinomina e copia file PDF in base ai campi estratti
"""

import shutil
import re
from pathlib import Path


def clean_filename(text):
    """
    Pulisce un testo per renderlo valido come nome file.
    
    Args:
        text: Testo da pulire
        
    Returns:
        Testo pulito
    """
    if not text:
        return ""
    
    # Rimuovi caratteri non validi per nomi file
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        text = text.replace(char, '')
    
    # Rimuovi spazi multipli
    text = re.sub(r'\s+', ' ', text)
    
    # Rimuovi spazi all'inizio e alla fine
    text = text.strip()
    
    # Limita lunghezza
    max_length = 50
    if len(text) > max_length:
        text = text[:max_length].strip()
    
    return text


def format_filename(fields, format_template="{denominazione} {numero} del {data}.pdf"):
    """
    Formatta il nome file in base ai campi estratti.
    
    Args:
        fields: Dizionario con denominazione, numero_documento, data_documento
        format_template: Template per il nome file
        
    Returns:
        Nome file formattato
    """
    # Estrai campi e puliscili
    denominazione = clean_filename(fields.get('denominazione', 'Sconosciuto'))
    numero = clean_filename(fields.get('numero_documento', '000'))
    data = fields.get('data_documento', '01-01-2000')
    
    # Formatta usando il template
    filename = format_template.format(
        denominazione=denominazione,
        numero=numero,
        data=data
    )
    
    return filename


def rename_and_copy(source_path, output_dir, fields, format_template=None):
    """
    Rinomina e copia un file PDF nell'output directory.
    
    Args:
        source_path: Path al file sorgente
        output_dir: Directory di destinazione
        fields: Campi estratti (denominazione, numero_documento, data_documento)
        format_template: Template opzionale per il nome
        
    Returns:
        Path al file rinominato
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Formatta nome file
    if format_template is None:
        format_template = "{denominazione} {numero} del {data}.pdf"
    
    new_filename = format_filename(fields, format_template)
    
    # Path di destinazione
    dest_path = output_dir / new_filename
    
    # Se il file esiste già, aggiungi un suffisso
    counter = 1
    base_path = dest_path.parent / dest_path.stem
    while dest_path.exists():
        dest_path = base_path.parent / f"{base_path.stem}_{counter}{dest_path.suffix}"
        counter += 1
    
    # Copia file
    shutil.copy2(source_path, dest_path)
    
    return dest_path
