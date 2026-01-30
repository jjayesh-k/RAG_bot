import os
import sys
import io
import logging
import warnings
import fitz  # PyMuPDF
import pymupdf4llm 

# Suppress warnings and logs
warnings.filterwarnings('ignore')
logging.getLogger("docling").setLevel(logging.ERROR)
logging.getLogger("fitz").setLevel(logging.ERROR)

try:
    from docling.datamodel.base_models import InputFormat, DocumentStream
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
except ImportError:
    print("❌ Error: Docling not found.")
    print("   Install with: pip install docling")
    sys.exit(1)

print("⏳ Initializing Hybrid Docling (Smart Table Detection)...")

# --- 1. CONFIG: MAX SPEED FOR CPU ---
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False  
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.mode = TableFormerMode.FAST 

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

print("✓ Hybrid Parser Ready.")


def detect_tables_improved(page):
    """
    Enhanced table detection using multiple strategies
    Returns: (has_tables: bool, detection_method: str)
    """
    
    # Strategy 1: PyMuPDF's find_tables with multiple strategies
    tables_lines = page.find_tables(strategy='lines')
    tables_text = page.find_tables(strategy='text')
    
    # Check lines strategy
    for tab in tables_lines.tables:
        if tab.row_count >= 2 and tab.col_count >= 2:
            return True, f"lines ({tab.row_count}x{tab.col_count})"
    
    # Check text strategy
    for tab in tables_text.tables:
        if tab.row_count >= 2 and tab.col_count >= 2:
            return True, f"text ({tab.row_count}x{tab.col_count})"
    
    # Strategy 2: Look for table-like patterns in text
    text = page.get_text()
    lines = text.split('\n')
    
    # Count lines with multiple tabs or pipe characters (common in tables)
    table_indicators = 0
    for line in lines:
        if line.count('\t') >= 2 or line.count('|') >= 2:
            table_indicators += 1
    
    if table_indicators >= 3:  # At least 3 lines with table-like structure
        return True, f"text_pattern ({table_indicators} lines)"
    
    # Strategy 3: Check for images that might be scanned tables
    images = page.get_images()
    if len(images) > 0:
        # Check if the image covers a significant portion of the page
        page_area = abs(page.rect)
        for img_info in images:
            xref = img_info[0]
            try:
                img_bbox = page.get_image_bbox(img_info[7])  # Get image bounding box
                img_area = abs(img_bbox)
                
                # If image covers more than 30% of page, might be a scanned table
                if img_area > page_area * 0.3:
                    return True, f"large_image ({len(images)} images)"
            except:
                pass
    
    # Strategy 4: Detect dense text blocks (potential tables)
    blocks = page.get_text("dict")["blocks"]
    dense_blocks = 0
    
    for block in blocks:
        if block.get("type") == 0:  # Text block
            bbox = block.get("bbox", [0, 0, 0, 0])
            text_content = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_content += span.get("text", "")
            
            # Calculate text density
            block_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if block_area > 0:
                char_density = len(text_content) / block_area
                # High character density might indicate tabular data
                if char_density > 0.5 and len(text_content) > 50:
                    dense_blocks += 1
    
    if dense_blocks >= 2:
        return True, f"dense_blocks ({dense_blocks} blocks)"
    
    return False, "none"


def has_scanned_content(page):
    """
    Detect if page is likely scanned (image-based)
    Returns: (is_scanned: bool, reason: str)
    """
    
    # Check 1: Very little extractable text
    text = page.get_text().strip()
    if len(text) < 50:  # Less than 50 characters
        images = page.get_images()
        if len(images) > 0:
            return True, f"low_text_with_images ({len(text)} chars, {len(images)} imgs)"
    
    # Check 2: Large images covering most of the page
    images = page.get_images()
    if len(images) > 0:
        page_area = abs(page.rect)
        total_image_area = 0
        
        for img_info in images:
            try:
                img_bbox = page.get_image_bbox(img_info[7])
                total_image_area += abs(img_bbox)
            except:
                pass
        
        # If images cover more than 70% of page, likely scanned
        if total_image_area > page_area * 0.7:
            return True, f"high_image_coverage ({total_image_area/page_area*100:.1f}%)"
    
    return False, "none"


def parse_hybrid_pdf(pdf_path, debug=False):
    """
    Hybrid PDF parser: Uses Docling for complex pages (tables/scans), 
    PyMuPDF4LLM for simple text pages
    
    Args:
        pdf_path: Path to PDF file
        debug: If True, prints detailed detection info for each page
    """
    filename = os.path.basename(pdf_path)
    print(f"   🚀 Smart-Parsing: {filename}...")
    final_markdown = ""
    
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            docling_pages = []
            stats = {
                'tables': 0,
                'scans': 0,
                'simple': 0,
                'detection_methods': {}
            }
            
            # First pass: Analyze all pages
            print(f"   📊 Analyzing {total_pages} pages...")
            
            for i, page in enumerate(doc):
                has_tables, table_method = detect_tables_improved(page)
                is_scanned, scan_reason = has_scanned_content(page)
                
                use_docling = has_tables or is_scanned
                
                if use_docling:
                    docling_pages.append(i)
                    if has_tables:
                        stats['tables'] += 1
                        stats['detection_methods'][table_method] = stats['detection_methods'].get(table_method, 0) + 1
                    if is_scanned:
                        stats['scans'] += 1
                else:
                    stats['simple'] += 1
                
                if debug:
                    status = "🔬 DOCLING" if use_docling else "⚡ FAST"
                    reasons = []
                    if has_tables:
                        reasons.append(f"table:{table_method}")
                    if is_scanned:
                        reasons.append(f"scan:{scan_reason}")
                    reason_str = ", ".join(reasons) if reasons else "text_only"
                    print(f"      [Page {i+1:3d}] {status} - {reason_str}")
            
            # Print summary
            print(f"\n   📈 Analysis Complete:")
            print(f"      • Pages with tables: {stats['tables']}")
            print(f"      • Scanned pages: {stats['scans']}")
            print(f"      • Simple text pages: {stats['simple']}")
            print(f"      • Using Docling for: {len(docling_pages)}/{total_pages} pages")
            
            if stats['detection_methods']:
                print(f"      • Detection methods: {dict(stats['detection_methods'])}")
            
            print(f"\n   🔄 Processing pages...")
            
            # Second pass: Parse pages
            for i, page in enumerate(doc):
                if i in docling_pages:
                    # --- HEAVY PARSE (Docling) ---
                    try:
                        new_doc = fitz.open()
                        new_doc.insert_pdf(doc, from_page=i, to_page=i)
                        pdf_bytes = new_doc.tobytes()
                        new_doc.close()
                        
                        source = DocumentStream(
                            name=f"page_{i+1}.pdf", 
                            stream=io.BytesIO(pdf_bytes)
                        )
                        conv_result = doc_converter.convert(source)
                        page_text = conv_result.document.export_to_markdown()
                        
                        final_markdown += f"\n\n<!-- Page {i+1} (Docling) -->\n{page_text}\n\n"
                        
                        if debug:
                            print(f"      ✓ Page {i+1} parsed with Docling")
                    except Exception as e:
                        print(f"Page {i+1} Docling failed, falling back: {e}")
                        # Fallback to fast parser
                        page_text = pymupdf4llm.to_markdown(pdf_path, pages=[i])
                        final_markdown += f"\n\n<!-- Page {i+1} (Fallback) -->\n{page_text}\n\n"
                else:
                    # --- INSTANT PARSE (PyMuPDF4LLM) ---
                    page_text = pymupdf4llm.to_markdown(pdf_path, pages=[i])
                    final_markdown += f"\n\n<!-- Page {i+1} (Fast) -->\n{page_text}\n\n"
                    
                    if debug and (i + 1) % 10 == 0:
                        print(f"      ✓ Processed {i+1}/{total_pages} pages...")
            
            print(f"   ✅ Finished. Used Docling on {len(docling_pages)}/{total_pages} pages.")
            
        return final_markdown
        
    except Exception as e:
        print(f"   ❌ Error in hybrid parsing: {e}")
        print(f"   ⚠️  Falling back to full PyMuPDF4LLM parsing...")
        return pymupdf4llm.to_markdown(pdf_path)


# --- TESTING ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n" + "="*60)
        print("USAGE:")
        print("="*60)
        print(f"  python {os.path.basename(__file__)} <pdf_file> [--debug]")
        print("\nEXAMPLES:")
        print(f"  python {os.path.basename(__file__)} document.pdf")
        print(f"  python {os.path.basename(__file__)} document.pdf --debug")
        print("\nOPTIONS:")
        print("  --debug    Show detailed page-by-page analysis")
        print("="*60 + "\n")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    debug_mode = "--debug" in sys.argv
    
    if not os.path.exists(pdf_file):
        print(f"❌ File not found: {pdf_file}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"HYBRID PDF PARSER - SMART TABLE DETECTION")
    print(f"{'='*60}\n")
    
    markdown_output = parse_hybrid_pdf(pdf_file, debug=debug_mode)
    
    # Save output
    output_file = pdf_file.replace('.pdf', '_parsed.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_output)
    
    print(f"\n   💾 Output saved to: {output_file}")
    print(f"   📄 Total length: {len(markdown_output):,} characters")
    print(f"\n{'='*60}\n")