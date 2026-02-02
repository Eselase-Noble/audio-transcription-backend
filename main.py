"""
Professional Audio Transcription API
Built with FastAPI and Anthropic Claude AI

A production-ready system for converting audio recordings into
professionally formatted meeting documents.
"""

import os
import base64
import subprocess
import sys
import tempfile
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Document generation
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.section import WD_SECTION
import anthropic
from openai import OpenAI
import whisper
from docx2pdf import convert as docx2pdf_convert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.ogg', '.webm', '.flac', '.mp4', '.mpeg'}
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
STATIC_DIR = Path("static")

# Create directories
for directory in [UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR]:
    directory.mkdir(exist_ok=True, parents=True)


class TranscriptionRequest(BaseModel):
    """Request model for transcription API"""
    title: str = Field(default="Meeting Transcription", max_length=200)
    date: Optional[str] = None
    participants: Optional[str] = None
    include_summary: bool = True
    include_action_items: bool = True
    confidentiality_level: str = Field(default="Internal", pattern="^(Internal|Confidential|Public)$")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    api_key_configured: bool
    timestamp: str
    version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for startup/shutdown events"""
    # Startup
    logger.info("🚀 Starting Audio Transcription API")
    logger.info(f"📁 Upload directory: {UPLOAD_DIR.absolute()}")
    logger.info(f"📄 Output directory: {OUTPUT_DIR.absolute()}")

    yield

    # Shutdown
    logger.info("🛑 Shutting down Audio Transcription API")
    # Cleanup temporary files
    for temp_file in UPLOAD_DIR.glob("temp_*"):
        try:
            temp_file.unlink()
        except Exception:
            pass


app = FastAPI(
    title="Professional Audio Transcription API",
    description="AI-powered audio transcription with professional document generation for business meetings",
    version="2.0.0",
    lifespan=lifespan,
    contact={
        "name": "AI Transcription Services",
        "email": "support@transcription.ai",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize API clients
anthropic_client = None
openai_client = None

if ANTHROPIC_API_KEY:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)


def get_media_type(filename: str) -> str:
    """Determine media type from filename extension"""
    ext = Path(filename).suffix.lower()
    media_types = {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.m4a': 'audio/mp4',
        '.ogg': 'audio/ogg',
        '.webm': 'audio/webm',
        '.flac': 'audio/flac',
        '.mp4': 'audio/mp4',
        '.mpeg': 'audio/mpeg'
    }
    return media_types.get(ext, 'audio/mpeg')


def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Check content type
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an audio file."
        )


async def transcribe_audio(audio_data: bytes, media_type: str) -> str:
    """
    Transcribe audio file using Whisper.
    Returns clean, punctuated text transcription.
    """
    temp_filename = None

    try:
        # Determine file extension
        ext_mapping = {
            'audio/wav': '.wav',
            'audio/mpeg': '.mp3',
            'audio/mp4': '.m4a',
            'audio/ogg': '.ogg',
            'audio/webm': '.webm',
            'audio/flac': '.flac'
        }
        suffix = ext_mapping.get(media_type, '.mp3')

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(audio_data)
            temp_filename = tmp_file.name

        # Use OpenAI Whisper API
        if openai_client:
            try:
                with open(temp_filename, "rb") as audio_file:
                    response = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text",
                        language="en"  # Optional: specify language
                    )
                    transcription = response
            except Exception as e:
                logger.warning(f"OpenAI Whisper failed: {e}. Falling back to local model.")
                return await transcribe_local(temp_filename)
        else:
            # Use local whisper model
            return await transcribe_local(temp_filename)

        return transcription.strip()

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    finally:
        # Cleanup temporary file
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception:
                pass


async def transcribe_local(temp_filename: str) -> str:
    """Transcribe using local Whisper model"""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(
            temp_filename,
            language="en",
            fp16=False  # Disable if no GPU
        )
        return result["text"].strip()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Local transcription failed: {str(e)}"
        )

logger = logging.getLogger(__name__)

def generate_meeting_analysis(transcription: str) -> Dict[str, str]:
    if not openai_client:
        return {
            "summary": "AI summary unavailable. Please configure OpenAI API key.",
            "action_items": "",
            "key_decisions": "",
            "next_steps": ""
        }

    try:
        prompt = f"""
Analyze this meeting transcription and provide:

1. EXECUTIVE SUMMARY: A concise 4-6 (depending on the content) paragraph summary of the entire meeting
2. KEY DECISIONS: Bulleted list of all decisions made
3. ACTION ITEMS: Tabular format with:
   - Task Description
   - Responsible Person
   - Deadline
   - Priority (High/Medium/Low)
4. NEXT STEPS: Recommended follow-up actions

5. CONCLUSION.  A concise 3-5 paragraph conclusion of the discussed topic and key components

Transcription:
{transcription}

Format the response with clear section headings in ALL CAPS.
"""

        # messages as plain dicts
        messages = [
            {"role": "system", "content": "You are an assistant that summarizes meetings professionally."},
            {"role": "user", "content": prompt}
        ]

        response =  openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )

        response_text = response.choices[0].message.content

        sections = parse_ai_response(response_text)

        return sections

    except Exception as e:
        logger.error(f"AI analysis failed: {str(e)}")
        return {
            "summary": "Summary generation failed. Please check API configuration.",
            "action_items": "",
            "key_decisions": "",
            "next_steps": "",
            "conclusion": ""
        }


def parse_ai_response(response_text: str) -> Dict[str, str]:
    """Parse AI response into structured sections"""
    sections = {
        "summary": "",
        "action_items": "",
        "key_decisions": "",
        "next_steps": "",
        "conclusion": ""
    }

    current_section = None
    lines = response_text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for section headers
        if "EXECUTIVE SUMMARY" in line.upper():
            current_section = "summary"
            continue
        elif "KEY DECISIONS" in line.upper():
            current_section = "key_decisions"
            continue
        elif "ACTION ITEMS" in line.upper():
            current_section = "action_items"
            continue
        elif "NEXT STEPS" in line.upper():
            current_section = "next_steps"
            continue
        elif "CONCLUSION" in line.upper():
            current_section = "conclusion"
            continue

        # Add content to current section
        if current_section and current_section in sections:
            if sections[current_section]:
                sections[current_section] += "\n" + line
            else:
                sections[current_section] = line

    return sections


def setup_document_styles(doc: Document) -> None:
    """Configure professional document styles"""
    # Set default font to Times New Roman
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(12)
    style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    style.paragraph_format.space_after = Pt(6)

    # Create heading styles
    for level in range(1, 4):
        try:
            style = doc.styles[f'Heading {level}']
            style.font.name = 'Times New Roman'
            style.font.bold = True
            style.font.color.rgb = RGBColor(31, 73, 125)  # Dark blue
            if level == 1:
                style.font.size = Pt(16)
            elif level == 2:
                style.font.size = Pt(14)
            else:
                style.font.size = Pt(12)
        except KeyError:
            pass


def add_footer_with_page_numbers(doc: Document, title: str):
    """Add professional footer with page numbers"""
    try:
        # Access the first section
        section = doc.sections[0]
        footer = section.footer

        # Clear any existing paragraphs
        for paragraph in footer.paragraphs:
            paragraph.clear()

        # Add centered footer with page number
        footer_para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Add document title and page info
        footer_text = f"{title} - Page "
        footer_run = footer_para.add_run(footer_text)
        footer_run.font.size = Pt(9)
        footer_run.font.name = 'Times New Roman'
        footer_run.font.italic = True

        # Add page number field using simpler method
        from docx.oxml.shared import OxmlElement
        from docx.oxml.ns import qn

        # Create page number field
        fld_char1 = OxmlElement('w:fldChar')
        fld_char1.set(qn('w:fldCharType'), 'begin')

        instr_text = OxmlElement('w:instrText')
        instr_text.text = "PAGE"
        instr_text.set(qn('xml:space'), 'preserve')

        fld_char2 = OxmlElement('w:fldChar')
        fld_char2.set(qn('w:fldCharType'), 'end')

        # Add elements to run
        footer_run._r.append(fld_char1)
        footer_run._r.append(instr_text)
        footer_run._r.append(fld_char2)

        # Add " of " separator
        footer_para.add_run(" of ")

        # Add total pages field
        fld_char3 = OxmlElement('w:fldChar')
        fld_char3.set(qn('w:fldCharType'), 'begin')

        instr_text2 = OxmlElement('w:instrText')
        instr_text2.text = "NUMPAGES"
        instr_text2.set(qn('xml:space'), 'preserve')

        fld_char4 = OxmlElement('w:fldChar')
        fld_char4.set(qn('w:fldCharType'), 'end')

        # Add elements to run
        footer_run = footer_para.add_run()
        footer_run._r.append(fld_char3)
        footer_run._r.append(instr_text2)
        footer_run._r.append(fld_char4)

    except Exception as e:
        logger.warning(f"Could not add page numbers to footer: {e}")
        # Add simple footer without page numbers
        try:
            section = doc.sections[0]
            footer = section.footer
            if footer.paragraphs:
                footer.paragraphs[0].clear()
            else:
                footer.add_paragraph()
            footer_para = footer.paragraphs[0]
            footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            footer_run = footer_para.add_run(f"{title}")
            footer_run.font.size = Pt(9)
            footer_run.font.name = 'Times New Roman'
            footer_run.font.italic = True
        except Exception:
            pass  # Skip footer if it fails


def create_professional_word_document(
    title: str,
    date: str,
    participants: Optional[str],
    transcription: str,
    analysis: Dict[str, str],
    confidentiality_level: str,
    output_path: Path
) -> None:
    """
    Create a professionally formatted Word document with Times New Roman font,
    justified text, and proper business formatting.
    """
    doc = Document()

    # Setup document styles
    setup_document_styles(doc)

    # Set document margins (1 inch = 914400 EMU)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Add confidentiality header
    if confidentiality_level != "Public":
        conf_header = doc.add_paragraph()
        conf_header.alignment = WD_ALIGN_PARAGRAPH.CENTER
        conf_run = conf_header.add_run(f"{confidentiality_level.upper()}")
        conf_run.font.size = Pt(10)
        conf_run.font.bold = True
        conf_run.font.color.rgb = RGBColor(192, 0, 0)  # Red color
        conf_header.paragraph_format.space_after = Pt(12)

    # Title
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run(title.upper())
    title_run.font.size = Pt(18)
    title_run.font.bold = True
    title_run.font.name = 'Times New Roman'
    title_run.font.color.rgb = RGBColor(0, 51, 102)  # Dark blue
    title_para.paragraph_format.space_after = Pt(18)

    # Meeting Details Section
    details_heading = doc.add_heading('MEETING DETAILS', level=2)
    details_heading.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # Create details table
    table = doc.add_table(rows=0, cols=2)
    table.style = 'Light Grid Accent 1'
    table.autofit = False

    # Set column widths
    for row in table.rows:
        row.cells[0].width = Inches(1.5)
        row.cells[1].width = Inches(5)

    # Date row
    date_row = table.add_row()
    date_cell = date_row.cells[0]
    date_cell.text = "Date:"
    date_cell.paragraphs[0].runs[0].font.bold = True
    date_cell.paragraphs[0].runs[0].font.name = 'Times New Roman'
    date_row.cells[1].text = date

    # Participants row
    if participants:
        part_row = table.add_row()
        part_cell = part_row.cells[0]
        part_cell.text = "Participants:"
        part_cell.paragraphs[0].runs[0].font.bold = True
        part_cell.paragraphs[0].runs[0].font.name = 'Times New Roman'
        part_row.cells[1].text = participants

    # Confidentiality row
    conf_row = table.add_row()
    conf_cell = conf_row.cells[0]
    conf_cell.text = "Classification:"
    conf_cell.paragraphs[0].runs[0].font.bold = True
    conf_cell.paragraphs[0].runs[0].font.name = 'Times New Roman'
    conf_row.cells[1].text = confidentiality_level

    # Document ID row
    doc_id_row = table.add_row()
    doc_id_cell = doc_id_row.cells[0]
    doc_id_cell.text = "Document ID:"
    doc_id_cell.paragraphs[0].runs[0].font.bold = True
    doc_id_cell.paragraphs[0].runs[0].font.name = 'Times New Roman'
    doc_id_row.cells[1].text = f"TRANS-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Add spacing
    doc.add_paragraph().paragraph_format.space_before = Pt(24)

    # Executive Summary Section
    if analysis.get("summary"):
        summary_heading = doc.add_heading('EXECUTIVE SUMMARY', level=2)
        summary_heading.alignment = WD_ALIGN_PARAGRAPH.LEFT

        # Add summary paragraphs with justification
        summary_paragraphs = analysis["summary"].split('\n\n')
        for para_text in summary_paragraphs:
            if para_text.strip():
                para = doc.add_paragraph(para_text.strip())
                para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                para.paragraph_format.space_after = Pt(8)
                # Ensure Times New Roman font
                for run in para.runs:
                    run.font.name = 'Times New Roman'

        doc.add_paragraph().paragraph_format.space_before = Pt(18)

    # Key Decisions Section
    if analysis.get("key_decisions"):
        decisions_heading = doc.add_heading('KEY DECISIONS', level=2)
        decisions_heading.alignment = WD_ALIGN_PARAGRAPH.LEFT

        # Parse bullet points
        lines = analysis["key_decisions"].split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                # Remove bullet and add to document
                content = line[1:].strip()
                if content:
                    para = doc.add_paragraph(style='List Bullet')
                    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    run = para.add_run(content)
                    run.font.name = 'Times New Roman'
                    para.paragraph_format.left_indent = Inches(0.25)
                    para.paragraph_format.space_after = Pt(4)
            elif line:  # Handle non-bulleted lines
                para = doc.add_paragraph(line)
                para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                para.paragraph_format.space_after = Pt(4)
                for run in para.runs:
                    run.font.name = 'Times New Roman'

        doc.add_paragraph().paragraph_format.space_before = Pt(18)

    # Action Items Section
    if analysis.get("action_items"):
        actions_heading = doc.add_heading('ACTION ITEMS', level=2)
        actions_heading.alignment = WD_ALIGN_PARAGRAPH.LEFT

        # Check if action items are in table format
        if '|' in analysis["action_items"]:
            # Parse table format
            lines = analysis["action_items"].split('\n')
            table_lines = [line for line in lines if '|' in line]

            if len(table_lines) > 0:
                # Count columns from first line
                col_count = table_lines[0].count('|') + 1

                # Create action items table
                action_table = doc.add_table(rows=1, cols=col_count)
                action_table.style = 'Light Grid Accent 1'

                # Parse header row
                header_cells = table_lines[0].split('|')
                for i, header in enumerate(header_cells[:col_count]):
                    cell = action_table.rows[0].cells[i]
                    cell.text = header.strip()
                    cell.paragraphs[0].runs[0].font.bold = True
                    cell.paragraphs[0].runs[0].font.name = 'Times New Roman'

                # Parse data rows
                for line in table_lines[1:]:
                    cells = line.split('|')
                    if len(cells) >= col_count:
                        row = action_table.add_row()
                        for i, cell_text in enumerate(cells[:col_count]):
                            cell = row.cells[i]
                            cell.text = cell_text.strip()
                            cell.paragraphs[0].runs[0].font.name = 'Times New Roman'
        else:
            # Add as bullet points
            lines = analysis["action_items"].split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    para = doc.add_paragraph(style='List Bullet')
                    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    run = para.add_run(line)
                    run.font.name = 'Times New Roman'
                    para.paragraph_format.left_indent = Inches(0.25)
                    para.paragraph_format.space_after = Pt(4)

        doc.add_paragraph().paragraph_format.space_before = Pt(18)

    # Next Steps Section
    if analysis.get("next_steps"):
        steps_heading = doc.add_heading('NEXT STEPS', level=2)
        steps_heading.alignment = WD_ALIGN_PARAGRAPH.LEFT

        lines = analysis["next_steps"].split('\n')
        for line in lines:
            line = line.strip()
            if line:
                para = doc.add_paragraph(style='List Bullet')
                para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                run = para.add_run(line)
                run.font.name = 'Times New Roman'
                para.paragraph_format.left_indent = Inches(0.25)
                para.paragraph_format.space_after = Pt(4)

        doc.add_paragraph().paragraph_format.space_before = Pt(24)


    # Executive Summary Section
    if analysis.get("conclusion"):
        conclusion_heading = doc.add_heading('CONCLUSION', level=2)
        conclusion_heading.alignment = WD_ALIGN_PARAGRAPH.LEFT

        # Add summary paragraphs with justification
        conclusion_paragraphs = analysis["conclusion"].split('\n\n')
        for para_text in conclusion_paragraphs:
            if para_text.strip():
                para = doc.add_paragraph(para_text.strip())
                para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                para.paragraph_format.space_after = Pt(8)
                # Ensure Times New Roman font
                for run in para.runs:
                    run.font.name = 'Times New Roman'

        doc.add_paragraph().paragraph_format.space_before = Pt(18)


    # Full Transcription Section
    # transcription_heading = doc.add_heading('CONCLUSION', level=2)
    # transcription_heading.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # Add transcription with proper formatting
    # transcription_paragraphs = transcription.split('\n\n')
    # for i, para_text in enumerate(transcription_paragraphs):
    #     if para_text.strip():
    #         # Add paragraph number for longer transcriptions
    #         if len(transcription_paragraphs) > 5:
    #             para = doc.add_paragraph(f"[{i+1}] {para_text.strip()}")
    #         else:
    #             para = doc.add_paragraph(para_text.strip())
    #
    #         para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    #         para.paragraph_format.space_after = Pt(8)
    #
    #         # Ensure Times New Roman font
    #         for run in para.runs:
    #             run.font.name = 'Times New Roman'

    # Add footer with page numbers
    add_footer_with_page_numbers(doc, title)

    # Save document
    doc.save(output_path)
    logger.info(f"Document created: {output_path}")


def create_professional_documents(
    title: str,
    date: str,
    document_time: str,
    participants: Optional[str],
    transcription: str,
    analysis: Dict[str, str],
    confidentiality_level: str,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Create both Word and PDF documents professionally.
    Returns a dictionary with paths.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    docx_filename = f"{safe_title}_{document_time}.docx"
    pdf_filename = f"{safe_title}_{document_time}.pdf"
    docx_path = output_dir / docx_filename
    pdf_path = output_dir / pdf_filename

    # Create Word document
    create_professional_word_document(
        title=title,
        date=date,
        participants=participants,
        transcription=transcription,
        analysis=analysis,
        confidentiality_level=confidentiality_level,
        output_path=docx_path
    )

    # Convert Word to PDF
    try:
        if sys.platform in ("win32", "darwin"):
            docx2pdf_convert(str(docx_path), str(pdf_path))
            logger.info(f"PDF created: {pdf_path}")
        else:
            pdf_path = convert_docx_to_pdf_linux(docx_path, pdf_path)
            logger.info(f"PDF created: {pdf_path}")
    except Exception as e:
        logger.warning(f"PDF conversion failed: {e}")
        pdf_path = None

    return {"docx": docx_path, "pdf": pdf_path}


def convert_docx_to_pdf_linux(docx_path: Path, pdf_path: Path):
    """Convert DOCX to PDF using LibreOffice (Linux-compatible)."""
    try:
        subprocess.run([
            "/usr/bin/libreoffice",
            "--headless",
            "--convert-to", "pdf",
            "--outdir", str(pdf_path.parent),
            str(docx_path)
        ], check=True)
        logger.info(f"PDF created: {pdf_path}")
        return pdf_path
    except subprocess.CalledProcessError as e:
        logger.warning(f"PDF conversion failed: {e}")
        return None

def cleanup_old_files(directory: Path, max_age_hours: int = 24):
    """Cleanup old files in the given directory"""
    cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)

    for file_path in directory.glob("*"):
        if file_path.is_file():
            try:
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    logger.debug(f"Cleaned up old file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "api_key_configured": bool(ANTHROPIC_API_KEY) or bool(OPENAI_API_KEY),
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "api_key_configured": bool(ANTHROPIC_API_KEY) or bool(OPENAI_API_KEY),
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }


@app.get("/api/cleanup")
async def cleanup_files(background_tasks: BackgroundTasks):
    """Trigger cleanup of old files"""
    background_tasks.add_task(cleanup_old_files, UPLOAD_DIR, 24)
    background_tasks.add_task(cleanup_old_files, OUTPUT_DIR, 24)
    return {"message": "Cleanup scheduled"}


@app.post("/api/transcribe")
async def transcribe_meeting(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    title: str = Form("Meeting Transcription"),
    date: Optional[str] = Form(None),
    participants: Optional[str] = Form(None),
    include_summary: bool = Form(True),
    include_action_items: bool = Form(True),
    confidentiality_level: str = Form("Internal")
):
    """
    Transcribe audio file and generate professional Word document

    - **audio**: Audio file (MP3, WAV, M4A, OGG, WEBM, FLAC)
    - **title**: Document title
    - **date**: Meeting date (format: YYYY-MM-DD or text)
    - **participants**: Comma-separated list of participants
    - **include_summary**: Generate AI executive summary
    - **include_action_items**: Extract action items from meeting
    - **confidentiality_level**: Document classification (Internal/Confidential/Public)
    """

    temp_audio_path = None
    output_file_path = None

    try:
        logger.info(f"Processing transcription request: {audio.filename}")

        # Validate file
        validate_file(audio)

        # Read audio file
        audio_data = await audio.read()

        # Validate file size
        if len(audio_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds limit. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"
            )

        # Save temporary file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = "".join(c for c in audio.filename if c.isalnum() or c in (' ', '.', '-', '_')).rstrip()
        temp_audio_path = UPLOAD_DIR / f"temp_{timestamp}_{safe_filename}"

        with open(temp_audio_path, "wb") as f:
            f.write(audio_data)

        # Get media type
        media_type = get_media_type(audio.filename)

        # Transcribe audio
        logger.info(f"Transcribing {audio.filename}...")
        transcription = await transcribe_audio(audio_data, media_type)

        # Generate AI analysis if requested
        analysis = {}
        if include_summary or include_action_items:
            logger.info("Generating meeting analysis...")
            analysis =  generate_meeting_analysis(transcription)

            # If action items not requested, remove them
            if not include_action_items:
                analysis["action_items"] = ""

        # Set default date if not provided
        if not date:
            date = datetime.now().strftime("%B %d, %Y")

        # Generate Word document
        logger.info("Creating professional document...")
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        output_filename = f"{safe_title}_{timestamp}.docx"
        output_file_path = OUTPUT_DIR / output_filename

        # create_professional_word_document(
        #     title=title,
        #     date=date,
        #     participants=participants,
        #     transcription=transcription,
        #     analysis=analysis,
        #     confidentiality_level=confidentiality_level,
        #     output_path=output_file_path
        # )

        document_id = timestamp

        # Create both Word and PDF
        documents = create_professional_documents(
            title=title,
            date=date,
            document_time=document_id,
            participants=participants,
            transcription=transcription,
            analysis=analysis,
            confidentiality_level=confidentiality_level,
            output_dir=OUTPUT_DIR
        )


        logger.info(f"Document created successfully: {output_filename}")

        # Schedule cleanup
        background_tasks.add_task(cleanup_old_files, UPLOAD_DIR, 1)  # Cleanup after 1 hour

        # Return the file
        # return FileResponse(
        #     path=output_file_path,
        #     media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        #     filename=output_filename,
        #     headers={
        #         "Content-Disposition": f"attachment; filename={output_filename}",
        #         "X-Document-ID": f"TRANS-{timestamp}"
        #     }
        # )

        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()

        print("Document_ID" , document_id)

        return {
            "message": "Transcription complete",
            "document_id": document_id,
            "files": {
                "word": documents["docx"].name if documents["docx"] else None,
                "pdf": documents["pdf"].name if documents["pdf"] else None
            }
        }


    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

    finally:
        # Cleanup temporary audio file
        if temp_audio_path and temp_audio_path.exists():
            try:
                temp_audio_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


@app.get("/api/document/{document_id}")
async def get_document(document_id: str):
    """Retrieve a previously generated document"""
    # Find document by ID pattern
    for doc_file in OUTPUT_DIR.glob(f"*_{document_id}.docx"):
        if doc_file.exists():
            return FileResponse(
                path=doc_file,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                filename=doc_file.name
            )

    raise HTTPException(status_code=404, detail="Document not found")

@app.get("/api/document/{document_id}/{file_type}")
async def download_document(document_id: str, file_type: str):
    """
    Download a previously generated document
    - file_type: 'word' or 'pdf'
    """
    if file_type not in ("word", "pdf"):
        raise HTTPException(status_code=400, detail="file_type must be 'word' or 'pdf'")

    ext = ".docx" if file_type == "word" else ".pdf"
    for doc_file in OUTPUT_DIR.glob(f"*_{document_id}{ext}"):
        if doc_file.exists():
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if ext == ".docx" else "application/pdf"
            return FileResponse(
                path=doc_file,
                media_type=media_type,
                filename=doc_file.name
            )

    raise HTTPException(status_code=404, detail="Document not found")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎙️  PROFESSIONAL AUDIO TRANSCRIPTION API")
    print("="*60)
    print(f"📁 Upload directory: {UPLOAD_DIR.absolute()}")
    print(f"📄 Output directory: {OUTPUT_DIR.absolute()}")
    print(f"🔗 API Documentation: http://localhost:8000/docs")
    print(f"🌐 Health check: http://localhost:8000/api/health")
    print("="*60)

    if not ANTHROPIC_API_KEY and not OPENAI_API_KEY:
        print("⚠️  WARNING: No API keys configured. Limited functionality.")
        print("   Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env file")
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )