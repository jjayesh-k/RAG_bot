"""
Hybrid PDF Parser (Smart Tables + Lost Text Recovery)
=====================================================
FIXED VERSION:
1. Lowers text filter threshold (catches 'Unity', 'Integrity').
2. Forces visual sorting (sort=True) to keep headers above text.
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
        if verbose: print(f"Parsing (Hybrid Mode): {pdf_path}")
        
        # 1. Open Doc for Raw Extraction
        doc = fitz.open(pdf_path)
        
        # 2. Get Smart Markdown (Page by Page)
        md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        
        all_chunks = []
        self.chunk_counter = 0

        for i, md_data in enumerate(md_pages):
            page_num = i + 1
            smart_text = md_data['text']
            
            # --- THE MAGIC: RECOVER LOST TEXT ---
            raw_page = doc[i]
            
            # FIX #1: Added sort=True to force Reading Order (Header -> Body)
            raw_blocks = raw_page.get_text("blocks", sort=True)
            
            missing_text = []
            smart_text_norm = self._normalize(smart_text)
            
            for b in raw_blocks:
                block_text = b[4].strip()
                
                # FIX #2: Lowered threshold from 10 to 3
                # This ensures we catch headers like "UNITY" (5 chars) or "Note" (4 chars)
                if len(block_text) < 3: continue
                
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
            page_chunks = self._create_sliding_window_chunks(final_page_content, page_num)
            all_chunks.extend(page_chunks)

        if verbose: print(f"Extracted {len(all_chunks)} chunks (Tables preserved + Text recovered).")
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