import os
import sys
import logging

# Set logging to avoid console spam from Docling
logging.getLogger("docling").setLevel(logging.ERROR)

try:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
except ImportError:
    print("Error: Docling not found.")
    print("Run: pip install docling")
    sys.exit(1)

print("Initializing Docling Converter (CPU Optimized)...")

# --- CPU OPTIMIZATION CONFIG ---
# We configure Docling to use less memory by restricting table structure recognition
# if it gets too heavy. For 8GB RAM, standard settings are usually okay, 
# but we are being safe.
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True  # Enable OCR for scanned PDFs
pipeline_options.do_table_structure = True # Great for RAG
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE 

# Initialize the converter ONCE to avoid reloading models
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
print("✓ Docling Ready.")

def parse_hybrid_pdf(pdf_path):
    """
    Parses PDF using IBM Docling to Markdown.
    Includes robust error handling for low-RAM environments.
    """
    filename = os.path.basename(pdf_path)
    print(f"   Analysing layout & tables: {filename}...")

    try:
        # 1. Run Conversion
        conv_result = doc_converter.convert(pdf_path)
        
        # 2. Export to Markdown (Best format for RAG)
        # Docling creates perfect Markdown tables which helps the AI understand data.
        md_text = conv_result.document.export_to_markdown()
        
        if not md_text.strip():
            return "[WARNING] Docling finished but found no text."

        return md_text

    except RuntimeError as e:
        if "memory" in str(e).lower():
            return "[ERROR] System ran out of RAM processing this file."
        return f"[ERROR] Docling failed: {e}"
    except Exception as e:
        return f"[ERROR] Processing error: {e}"

# ... (Keep all your existing code above) ...

# --- Test Block ---
if __name__ == "__main__":
    # Test with a dummy file if you run this script directly
    test_pdf = "Tata Code Of Conduct.pdf" 
    
    if os.path.exists(test_pdf):
        print(f"Processing '{test_pdf}'...")
        full_text = parse_hybrid_pdf(test_pdf)
        
        # --- NEW CODE TO SAVE FILE ---
        output_filename = "docling_output.md"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(full_text)
            
        print(f"\nSuccess! Full output saved to: {output_filename}")
        print("--- Preview (First 500 chars) ---")
        print(full_text[:500])
        # -----------------------------
        
    else:
        print(f"File '{test_pdf}' not found in this folder.")