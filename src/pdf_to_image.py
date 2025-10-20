#!/usr/bin/env python3
"""
PDF to Image Converter
Converte la prima pagina di un PDF in immagine
"""

import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
import tempfile


def convert_pdf_first_page(pdf_path, dpi=144, output_dir=None):
    """
    Converte la prima pagina di un PDF in un'immagine.
    
    Args:
        pdf_path: Path al file PDF
        dpi: Risoluzione dell'immagine (default: 144)
        output_dir: Directory dove salvare l'immagine (default: temp)
        
    Returns:
        Path all'immagine creata
    """
    pdf_path = Path(pdf_path)
    
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "rinominator_temp"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apri PDF
    doc = fitz.open(pdf_path)
    
    # Prendi prima pagina
    page = doc[0]
    
    # Calcola zoom per ottenere il DPI desiderato
    zoom = dpi / 72  # 72 è il DPI standard
    mat = fitz.Matrix(zoom, zoom)
    
    # Renderizza pagina
    pix = page.get_pixmap(matrix=mat)
    
    # Salva come immagine
    image_path = output_dir / f"{pdf_path.stem}_page1.png"
    pix.save(str(image_path))
    
    doc.close()
    
    return image_path
