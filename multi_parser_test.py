# """
# Smart PDF Parser for Multi-Column Documents + Small LLMs
# =========================================================

# Uses pymupdf4llm (handles multi-column layouts correctly)
# + Post-processing to clean tables for phi3.5

# Perfect for:
# - Multi-column PDFs (academic papers, reports)
# - 8GB RAM systems
# - Small LLMs like phi3.5:3.8b
# """

# import pymupdf4llm
# import re
# import json
# import sys
# import time
# from typing import List, Dict
# from dataclasses import dataclass


# @dataclass
# class ParsedChunk:
#     id: int
#     page_num: int
#     chunk_type: str
#     content: str
#     metadata: Dict


# class SmartMultiColumnParser:
#     """
#     Uses pymupdf4llm for proper layout detection
#     but cleans up tables for small LLM comprehension
#     """
    
#     def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.chunk_counter = 0
    
#     def parse_pdf(self, pdf_path: str, verbose: bool = True) -> List[Dict]:
#         """
#         Parse PDF using pymupdf4llm (respects multi-column layouts)
        
#         Args:
#             pdf_path: Path to PDF file
#             verbose: Print progress
            
#         Returns:
#             List of page dictionaries with content
#         """
#         if verbose:
#             print(f"📄 Parsing: {pdf_path}")
#             print("   Using pymupdf4llm (multi-column aware)...")
        
#         # Parse with page separation
#         pages_data = pymupdf4llm.to_markdown(
#             pdf_path,
#             page_chunks=True,      # Get pages separately
#             write_images=False,    # Skip images for speed
#             show_progress=verbose
#         )
        
#         result = []
#         for page_data in pages_data:
#             page_num = page_data['metadata']['page']
#             content = page_data['text']
            
#             # Clean the markdown tables
#             clean_content = self._clean_markdown_tables(content)
            
#             result.append({
#                 'page_num': page_num,
#                 'content': clean_content,
#                 'metadata': page_data['metadata']
#             })
        
#         if verbose:
#             print(f"✅ Parsed {len(result)} pages")
        
#         return result
    
#     def _clean_markdown_tables(self, text: str) -> str:
#         """
#         Convert messy markdown tables to clean text format
#         """
#         lines = text.split('\n')
#         result = []
#         i = 0
#         table_num = 1
        
#         while i < len(lines):
#             line = lines[i]
            
#             # Check if this line starts a markdown table
#             if self._is_table_header(line):
#                 # Extract the entire table
#                 table_lines = []
#                 while i < len(lines) and self._is_table_line(lines[i]):
#                     table_lines.append(lines[i])
#                     i += 1
                
#                 # Convert to clean format
#                 clean_table = self._convert_markdown_table(table_lines, table_num)
#                 if clean_table:
#                     result.append(clean_table)
#                     table_num += 1
#             else:
#                 result.append(line)
#                 i += 1
        
#         return '\n'.join(result)
    
#     def _is_table_header(self, line: str) -> bool:
#         """Check if line looks like a markdown table header"""
#         stripped = line.strip()
#         return (
#             stripped.startswith('|') and 
#             stripped.endswith('|') and 
#             stripped.count('|') >= 3 and
#             not all(c in '|-: ' for c in stripped)  # Not a separator line
#         )
    
#     def _is_table_line(self, line: str) -> bool:
#         """Check if line is part of a markdown table"""
#         stripped = line.strip()
#         return stripped.startswith('|') and stripped.endswith('|')
    
#     def _convert_markdown_table(self, table_lines: List[str], table_num: int) -> str:
#         """Convert markdown table to clean text format"""
#         if len(table_lines) < 2:
#             return ""
        
#         # Parse rows
#         rows = []
#         for line in table_lines:
#             # Split by | and clean
#             cells = line.split('|')
#             cells = [c.strip() for c in cells if c.strip()]
            
#             # Skip separator rows (---)
#             if cells and all(set(c) <= set('-: ') for c in cells):
#                 continue
            
#             if cells:
#                 rows.append(cells)
        
#         if len(rows) < 2:
#             return ""
        
#         # First row is headers
#         headers = rows[0]
#         data_rows = rows[1:]
        
#         # Build clean output
#         result = [f"\n--- TABLE {table_num} ---\n"]
        
#         for row_idx, row in enumerate(data_rows, 1):
#             result.append(f"Row {row_idx}:")
            
#             for header, value in zip(headers, row):
#                 # Clean the cell value
#                 clean_value = self._clean_cell_value(value)
#                 clean_header = self._clean_cell_value(header)
                
#                 if clean_value:
#                     result.append(f"  • {clean_header}: {clean_value}")
            
#             result.append("")  # Empty line between rows
        
#         result.append("--- END TABLE ---\n")
        
#         return '\n'.join(result)
    
#     def _clean_cell_value(self, value: str) -> str:
#         """Clean cell value: remove HTML and markdown formatting"""
#         if not value:
#             return ""
        
#         # Split by <br> to handle multi-value cells
#         parts = re.split(r'<br\s*/?>', value)
        
#         cleaned_parts = []
#         for part in parts:
#             # Remove markdown formatting
#             part = re.sub(r'_([^_]+)_', r'\1', part)  # _italic_ -> italic
#             part = re.sub(r'\*\*([^*]+)\*\*', r'\1', part)  # **bold** -> bold
#             part = re.sub(r'\*([^*]+)\*', r'\1', part)  # *italic* -> italic
            
#             # Remove extra whitespace
#             part = ' '.join(part.split())
            
#             if part.strip():
#                 cleaned_parts.append(part.strip())
        
#         # Join multiple values with " | "
#         return ' | '.join(cleaned_parts)
    
#     def create_smart_chunks(self, pages_data: List[Dict]) -> List[ParsedChunk]:
#         """Create chunks optimized for small LLMs"""
#         chunks = []
#         self.chunk_counter = 0
        
#         for page_data in pages_data:
#             page_num = page_data['page_num']
#             content = page_data['content']
            
#             # Split into sections (text and tables)
#             sections = self._split_content(content)
            
#             current_chunk = ""
            
#             for section in sections:
#                 is_table = section['is_table']
#                 text = section['text']
                
#                 if is_table:
#                     # Save current text chunk
#                     if current_chunk.strip():
#                         chunks.append(self._create_chunk(
#                             page_num, 'text', current_chunk
#                         ))
#                         current_chunk = ""
                    
#                     # Add table as separate chunk
#                     chunks.append(self._create_chunk(
#                         page_num, 'table', text
#                     ))
#                 else:
#                     # Accumulate text
#                     if len(current_chunk) + len(text) < self.chunk_size:
#                         current_chunk += text + "\n\n"
#                     else:
#                         # Save chunk
#                         if current_chunk.strip():
#                             chunks.append(self._create_chunk(
#                                 page_num, 'text', current_chunk
#                             ))
                        
#                         # Start new chunk with overlap
#                         overlap = self._get_overlap(current_chunk)
#                         current_chunk = overlap + text + "\n\n"
            
#             # Save remaining
#             if current_chunk.strip():
#                 chunks.append(self._create_chunk(
#                     page_num, 'text', current_chunk
#                 ))
        
#         return chunks
    
#     def _split_content(self, content: str) -> List[Dict]:
#         """Split content into text and table sections"""
#         sections = []
#         lines = content.split('\n')
        
#         current_text = []
#         in_table = False
#         table_lines = []
        
#         for line in lines:
#             if line.strip().startswith('--- TABLE'):
#                 # Start of table
#                 if current_text:
#                     sections.append({
#                         'is_table': False,
#                         'text': '\n'.join(current_text).strip()
#                     })
#                     current_text = []
#                 in_table = True
#                 table_lines = [line]
#             elif line.strip().startswith('--- END TABLE'):
#                 # End of table
#                 table_lines.append(line)
#                 sections.append({
#                     'is_table': True,
#                     'text': '\n'.join(table_lines).strip()
#                 })
#                 table_lines = []
#                 in_table = False
#             elif in_table:
#                 table_lines.append(line)
#             else:
#                 current_text.append(line)
        
#         # Add remaining text
#         if current_text:
#             sections.append({
#                 'is_table': False,
#                 'text': '\n'.join(current_text).strip()
#             })
        
#         return [s for s in sections if s['text']]
    
#     def _create_chunk(self, page_num: int, chunk_type: str, content: str) -> ParsedChunk:
#         """Create a ParsedChunk object"""
#         chunk = ParsedChunk(
#             id=self.chunk_counter,
#             page_num=page_num,
#             chunk_type=chunk_type,
#             content=content.strip(),
#             metadata={'page': page_num, 'type': chunk_type}
#         )
#         self.chunk_counter += 1
#         return chunk
    
#     def _get_overlap(self, text: str) -> str:
#         """Get overlap text for context preservation"""
#         if len(text) <= self.chunk_overlap:
#             return text
#         return text[-self.chunk_overlap:]
    
#     def parse_and_chunk(self, pdf_path: str, verbose: bool = True) -> List[ParsedChunk]:
#         """Complete pipeline: parse and chunk"""
#         # Parse PDF
#         pages_data = self.parse_pdf(pdf_path, verbose=verbose)
        
#         # Create chunks
#         if verbose:
#             print("📦 Creating chunks...")
        
#         chunks = self.create_smart_chunks(pages_data)
        
#         if verbose:
#             print(f"✅ Created {len(chunks)} chunks")
#             text_chunks = sum(1 for c in chunks if c.chunk_type == 'text')
#             table_chunks = sum(1 for c in chunks if c.chunk_type == 'table')
#             print(f"   • Text chunks: {text_chunks}")
#             print(f"   • Table chunks: {table_chunks}")
        
#         return chunks
    
#     def save_chunks(self, chunks: List[ParsedChunk], output_path: str):
#         """Save chunks to JSONL"""
#         with open(output_path, 'w', encoding='utf-8') as f:
#             for chunk in chunks:
#                 chunk_dict = {
#                     'id': chunk.id,
#                     'page_num': chunk.page_num,
#                     'type': chunk.chunk_type,
#                     'content': chunk.content,
#                     'metadata': chunk.metadata
#                 }
#                 f.write(json.dumps(chunk_dict, ensure_ascii=False) + '\n')
        
#         print(f"💾 Saved to: {output_path}")


# def demonstrate_cleaning():
#     """Show the table cleaning transformation"""
#     print("\n" + "="*70)
#     print("TABLE CLEANING DEMONSTRATION")
#     print("="*70)
    
#     print("\n❌ BEFORE (Messy markdown from pymupdf4llm):")
#     print("-"*70)
#     messy = """
# |Col1|Q1 FY22|Q2 FY22|Q3 FY22|
# |---|---|---|---|
# |Global wholesales (K units)|<br>213|<br>252|<br>286|
# |Revenue<br>_EBITDA (%)_<br>_EBIT (%)_|66.4<br>_8.3%_<br>_-1.3%_|61.4<br>_8.4%_<br>_-1.5%_|72.2<br>_10.2%_<br>_1.7%_|
#     """
#     print(messy)
    
#     print("\n✅ AFTER (Clean text for phi3.5):")
#     print("-"*70)
#     clean = """
# --- TABLE 1 ---

# Row 1:
#   • Col1: Global wholesales (K units)
#   • Q1 FY22: 213
#   • Q2 FY22: 252
#   • Q3 FY22: 286

# Row 2:
#   • Col1: Revenue | EBITDA (%) | EBIT (%)
#   • Q1 FY22: 66.4 | 8.3% | -1.3%
#   • Q2 FY22: 61.4 | 8.4% | -1.5%
#   • Q3 FY22: 72.2 | 10.2% | 1.7%

# --- END TABLE ---
#     """
#     print(clean)
    
#     print("\n💡 Key Improvements:")
#     print("-"*70)
#     print("✓ No markdown syntax (|---|---|)")
#     print("✓ No HTML tags (<br>)")
#     print("✓ No formatting markup (_text_)")
#     print("✓ Multi-values joined with ' | '")
#     print("✓ Clear key-value pairs")
#     print("✓ Easy for phi3.5 to understand!")
#     print("="*70 + "\n")


# # Main execution
# if __name__ == "__main__":
#     import sys
#     import time
#     import os  # <--- Added os to check if file exists
    
#     # Show cleaning demo
#     demonstrate_cleaning()
    
#     # --- HARDCODED FILE (Use this instead of sys.argv) ---
#     pdf_file = "q4fy22-presentation.pdf"
#     # -----------------------------------------------------

#     # Check if the file actually exists to avoid errors
#     if not os.path.exists(pdf_file):
#         print(f"❌ Error: File '{pdf_file}' not found in this folder.")
#         print("   Please make sure the PDF is in the same directory as this script.")
#         sys.exit(1)
    
#     print("="*70)
#     print("SMART MULTI-COLUMN PDF PARSER")
#     print("="*70)
#     print(f"\n📄 File: {pdf_file}\n")
    
#     # Parse and chunk
#     parser = SmartMultiColumnParser(chunk_size=1000, chunk_overlap=200)
    
#     start = time.time()
#     chunks = parser.parse_and_chunk(pdf_file)
#     end = time.time()
    
#     print(f"\n⏱️  Total time: {end - start:.2f} seconds")
    
#     # Show sample chunks
#     print("\n" + "="*70)
#     print("SAMPLE CHUNKS")
#     print("="*70)
    
#     for i, chunk in enumerate(chunks[:3], 1):
#         print(f"\n--- Chunk {i} (Page {chunk.page_num}, Type: {chunk.chunk_type}) ---")
#         content = chunk.content
#         if len(content) > 400:
#             print(content[:400] + "\n... (truncated)")
#         else:
#             print(content)
    
#     # Save JSONL
#     output_file = pdf_file.replace('.pdf', '_phi35_clean.jsonl')
#     parser.save_chunks(chunks, output_file)
    
#     # Save Viewable Text File
#     viewable_file = pdf_file.replace('.pdf', '_viewable.txt')
#     try:
#         with open(viewable_file, 'w', encoding='utf-8') as f:
#             for chunk in chunks:
#                 f.write(f"\n--- [Page {chunk.page_num} | {chunk.chunk_type} | Chunk ID: {chunk.id}] ---\n")
#                 f.write(chunk.content)
#                 f.write("\n" + "-"*80 + "\n")
#         print(f"👀 Readable output saved to: {viewable_file}")
#     except Exception as e:
#         print(f"⚠️ Could not save viewable file: {e}")
    
#     print("\n" + "="*70)
#     print("✅ SUCCESS!")
#     print("="*70)

"""
Hybrid PDF Parser (Smart Tables + Lost Text Recovery)
=====================================================
1. Uses pymupdf4llm to get clean Markdown tables.
2. Uses raw text extraction to find text skipped by the Smart Parser.
3. Appends the missing text to ensure RAG doesn't miss rules.
"""

import pymupdf4llm
import fitz  # PyMuPDF
import re
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class ParsedChunk:
    id: int
    page_num: int
    chunk_type: str
    content: str
    metadata: Dict

class SmartMultiColumnParser:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_counter = 0

    def _normalize(self, text):
        """Removes whitespace/formatting for comparison"""
        return re.sub(r'\s+', '', text).lower()

    def parse_and_chunk(self, pdf_path: str, verbose: bool = True) -> List[ParsedChunk]:
        if verbose: print(f"📄 Parsing (Hybrid Mode): {pdf_path}")
        
        # 1. Open Doc for Raw Extraction
        doc = fitz.open(pdf_path)
        
        # 2. Get Smart Markdown (Page by Page)
        # We process page-by-page to keep alignment with raw text
        md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        
        all_chunks = []
        self.chunk_counter = 0

        for i, md_data in enumerate(md_pages):
            page_num = i + 1
            smart_text = md_data['text']
            
            # --- THE MAGIC: RECOVER LOST TEXT ---
            raw_page = doc[i]
            # Get text blocks (x, y, w, h, text, block_no, block_type)
            raw_blocks = raw_page.get_text("blocks")
            
            missing_text = []
            smart_text_norm = self._normalize(smart_text)
            
            for b in raw_blocks:
                block_text = b[4].strip()
                # Skip tiny fragments or page numbers
                if len(block_text) < 10: continue
                
                # Check if this block exists in the smart text
                if self._normalize(block_text) not in smart_text_norm:
                    missing_text.append(block_text)

            # Combine: Smart Text + Separator + Recovered Text
            final_page_content = smart_text
            if missing_text:
                recovered_str = "\n".join(missing_text)
                final_page_content += f"\n\n--- [ADDITIONAL NOTES / SIDEBARS] ---\n{recovered_str}"
                if verbose:
                    print(f"   + Page {page_num}: Recovered {len(missing_text)} missing text blocks.")

            # --- CHUNKING ---
            # Now we chunk the *Combined* text
            page_chunks = self._create_sliding_window_chunks(final_page_content, page_num)
            all_chunks.extend(page_chunks)

        if verbose: print(f"✅ Extracted {len(all_chunks)} chunks (Tables preserved + Text recovered).")
        return all_chunks

    def _create_sliding_window_chunks(self, text: str, page_num: int) -> List[ParsedChunk]:
        """Standard sliding window chunker"""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            if end < text_len:
                last_space = text.rfind(' ', start, end)
                if last_space != -1:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(ParsedChunk(
                    id=self.chunk_counter,
                    page_num=page_num,
                    chunk_type="text",
                    content=chunk_text,
                    metadata={'page': page_num}
                ))
                self.chunk_counter += 1
            
            start = end - self.chunk_overlap
            if start >= end: start = end # Safety

        return chunks