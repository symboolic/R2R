# type: ignore
import asyncio
import base64
import json
import logging
import string
import time
import unicodedata
from io import BytesIO
from typing import AsyncGenerator
import hashlib
import uuid

import pdf2image
from mistralai.models import OCRResponse
from pypdf import PdfReader
import pdfplumber
from PIL import Image, ImageDraw, ImageFont

from core.base.abstractions import GenerationConfig
from core.base.parsers.base_parser import AsyncParser
from core.base.providers import (
    CompletionProvider,
    DatabaseProvider,
    IngestionConfig,
    OCRProvider,
)

logger = logging.getLogger()


class OCRPDFParser(AsyncParser[str | bytes]):
    """
    A parser for PDF documents using Mistral's OCR for page processing.

    Mistral supports directly processing PDF files, so this parser is a simple wrapper around the Mistral OCR API.
    """

    def __init__(
        self,
        config: IngestionConfig,
        database_provider: DatabaseProvider,
        llm_provider: CompletionProvider,
        ocr_provider: OCRProvider,
    ):
        self.config = config
        self.database_provider = database_provider
        self.ocr_provider = ocr_provider

    async def ingest(
        self, data: str | bytes, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Ingest PDF data and yield text from each page."""
        try:
            logger.info("Starting PDF ingestion using MistralOCRParser")

            if isinstance(data, str):
                response: OCRResponse = await self.ocr_provider.process_pdf(
                    file_path=data
                )
            else:
                response: OCRResponse = await self.ocr_provider.process_pdf(
                    file_content=data
                )

            for page in response.pages:
                yield {
                    "content": page.markdown,
                    "page_number": page.index + 1,  # Mistral is 0-indexed
                }

        except Exception as e:
            logger.error(f"Error processing PDF with Mistral OCR: {str(e)}")
            raise


class VLMPDFParser(AsyncParser[str | bytes]):
    """A parser for PDF documents using vision models for page processing."""

    def __init__(
        self,
        config: IngestionConfig,
        database_provider: DatabaseProvider,
        llm_provider: CompletionProvider,
        ocr_provider: OCRProvider,
    ):
        self.database_provider = database_provider
        self.llm_provider = llm_provider
        self.config = config
        self.vision_prompt_text = None
        self.vlm_batch_size = self.config.vlm_batch_size or 5
        self.vlm_max_tokens_to_sample = (
            self.config.vlm_max_tokens_to_sample or 1024
        )
        self.max_concurrent_vlm_tasks = (
            self.config.max_concurrent_vlm_tasks or 5
        )
        self.semaphore = None

    async def process_page(self, image, page_num: int) -> dict[str, str]:
        """Process a single PDF page using the vision model."""
        page_start = time.perf_counter()
        try:
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format="JPEG")
            image_data = img_byte_arr.getvalue()
            # Convert image bytes to base64
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            model = self.config.app.vlm

            # Configure generation parameters
            generation_config = GenerationConfig(
                model=self.config.vlm or self.config.app.vlm,
                stream=False,
                max_tokens_to_sample=self.vlm_max_tokens_to_sample,
            )

            is_anthropic = model and "anthropic/" in model

            # Prepare message with image content
            if is_anthropic:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.vision_prompt_text},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64,
                                },
                            },
                        ],
                    }
                ]
            else:
                # Use OpenAI format
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.vision_prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ]

            logger.debug(f"Sending page {page_num} to vision model.")

            if is_anthropic:
                response = await self.llm_provider.aget_completion(
                    messages=messages,
                    generation_config=generation_config,
                    apply_timeout=False,
                    tools=[
                        {
                            "name": "parse_pdf_page",
                            "description": "Parse text content from a PDF page",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "page_content": {
                                        "type": "string",
                                        "description": "Extracted text from the PDF page, transcribed into markdown",
                                    },
                                    "thoughts": {
                                        "type": "string",
                                        "description": "Any thoughts or comments on the text",
                                    },
                                },
                                "required": ["page_content"],
                            },
                        }
                    ],
                    tool_choice={"type": "tool", "name": "parse_pdf_page"},
                )

                if (
                    response.choices
                    and response.choices[0].message
                    and response.choices[0].message.tool_calls
                ):
                    tool_call = response.choices[0].message.tool_calls[0]
                    args = json.loads(tool_call.function.arguments)
                    content = args.get("page_content", "")
                    page_elapsed = time.perf_counter() - page_start
                    logger.debug(
                        f"Processed page {page_num} in {page_elapsed:.2f} seconds."
                    )
                    return {"page": str(page_num), "content": content}
                else:
                    logger.warning(
                        f"No valid tool call in response for page {page_num}, document might be missing text."
                    )
                    return {"page": str(page_num), "content": ""}
            else:
                response = await self.llm_provider.aget_completion(
                    messages=messages,
                    generation_config=generation_config,
                    apply_timeout=False,
                )

                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                    page_elapsed = time.perf_counter() - page_start
                    logger.debug(
                        f"Processed page {page_num} in {page_elapsed:.2f} seconds."
                    )
                    return {"page": str(page_num), "content": content}
                else:
                    msg = f"No response content for page {page_num}"
                    logger.error(msg)
                    return {"page": str(page_num), "content": ""}
        except Exception as e:
            logger.error(
                f"Error processing page {page_num} with vision model: {str(e)}"
            )
            # Return empty content rather than raising to avoid failing the entire batch
            return {
                "page": str(page_num),
                "content": f"Error processing page: {str(e)}",
            }

    def replace_figure_numbers_with_uuids(self, content: str, page_num: int) -> str:
        """Replace [FIG-N] tags with [FIG-uuid] tags."""
        if not hasattr(self, 'image_uuid_mapping') or page_num not in self.image_uuid_mapping:
            return content
        
        import re
        
        # Create a mapping from figure numbers to UUIDs for this page
        fig_to_uuid = {}
        for img_info in self.image_uuid_mapping[page_num]:
            fig_to_uuid[img_info['fig_num']] = img_info['uuid']
        
        # Replace [FIG-N] and [/FIG-N] tags with UUID versions
        def replace_fig_tag(match):
            fig_num = int(match.group(1))
            if fig_num in fig_to_uuid:
                uuid = fig_to_uuid[fig_num]
                if match.group(0).startswith('[/'):
                    return f'[/FIG: {uuid}]'
                else:
                    return f'[FIG: {uuid}]'
            return match.group(0)  # Return original if no UUID found
        
        # Replace opening and closing tags
        content = re.sub(r'\[FIG-(\d+)\]', replace_fig_tag, content)
        content = re.sub(r'\[/FIG-(\d+)\]', replace_fig_tag, content)
        
        return content

    def add_all_figure_numbers_to_page(self, image, page_num: int) -> Image.Image:
        """Add figure number overlays for all images on a page."""
        if not hasattr(self, 'image_uuid_mapping') or page_num not in self.image_uuid_mapping:
            return image
        
        processed_image = image.copy()
        
        # Get page dimensions from the rendered image
        page_width, page_height = processed_image.size
        
        # Add overlays for each figure on this page
        for idx, img_info in enumerate(self.image_uuid_mapping[page_num]):
            # Convert PDF coordinates to image pixel coordinates
            # PDF coordinates are typically in points (72 DPI), but we rendered at 150 DPI
            scale_factor = 150 / 72  # DPI conversion factor
            
            # Calculate pixel position (PDF y-coordinates are bottom-up, image coordinates are top-down)
            pixel_x = int(img_info['x0'] * scale_factor)
            pixel_y = int((page_height / scale_factor - img_info['y1']) * scale_factor)  # Flip Y coordinate
            
            # Ensure coordinates are within image bounds
            pixel_x = max(0, min(pixel_x, page_width - 80))  # Leave space for text
            pixel_y = max(0, min(pixel_y, page_height - 30))  # Leave space for text
            
            # Add overlay at the calculated position
            processed_image = add_figure_numbers_to_image(
                processed_image, 
                img_info['fig_num'], 
                position_x=pixel_x,
                position_y=pixel_y
            )
        
        return processed_image

    async def process_and_yield(self, image, page_num: int):
        """Process a page and yield the result."""
        async with self.semaphore:
            # Add figure numbers to the image if there are embedded images on this page
            processed_image = image
            if hasattr(self, 'image_uuid_mapping') and page_num in self.image_uuid_mapping:
                # Add figure number overlays to the image
                processed_image = self.add_all_figure_numbers_to_page(image, page_num)
            
            result = await self.process_page(processed_image, page_num)
            content = result.get("content", "") or ""
            
            # Replace figure numbers with UUIDs in the content
            if hasattr(self, 'image_uuid_mapping') and page_num in self.image_uuid_mapping:
                content = self.replace_figure_numbers_with_uuids(content, page_num)
            
            return {
                "content": content,
                "page_number": page_num,
                
            }

    async def ingest(
        self, data: str | bytes, **kwargs
    ) -> AsyncGenerator[dict[str, str | int], None]:
        """Process PDF as images using pdf2image."""
        ingest_start = time.perf_counter()
        logger.info("Starting PDF ingestion using VLMPDFParser.")

        if not self.vision_prompt_text:
            self.vision_prompt_text = (
                await self.database_provider.prompts_handler.get_cached_prompt(
                    prompt_name="vision_pdf"
                )
            )
            logger.info("Retrieved vision prompt text from database.")

        self.semaphore = asyncio.Semaphore(self.max_concurrent_vlm_tasks)
        
        # Get document_id from kwargs if available
        document_id = kwargs.get('document_id')
        
        # Extract embedded images from PDF and store them
        pdf_data = data if isinstance(data, bytes) else open(data, 'rb').read()
        extracted_images = extract_embedded_images_from_pdf_data(pdf_data)
        
        # Store images in database and create mapping
        image_uuid_mapping = {}  # Maps page_num -> list of {fig_num, uuid}
        
        for image_info in extracted_images:
            try:
                # Store image in database
                if hasattr(self.database_provider, 'images_handler'):
                    stored_uuid = await self.database_provider.images_handler.store_image(
                        image_data=image_info["image_data"],
                        mime_type=image_info["mime_type"],
                        width=image_info["width"],
                        height=image_info["height"],
                        metadata={
                            "page_number": str(image_info["page_number"]),
                            "x0": str(image_info["x0"]),
                            "y0": str(image_info["y0"]),
                            "x1": str(image_info["x1"]),
                            "y1": str(image_info["y1"]),
                        },
                        document_id=document_id,  # Pass document_id to establish relationship
                    )
                    
                    page_num = image_info["page_number"]
                    if page_num not in image_uuid_mapping:
                        image_uuid_mapping[page_num] = []
                    
                    fig_num = len(image_uuid_mapping[page_num]) + 1
                    image_uuid_mapping[page_num].append({
                        'fig_num': fig_num,
                        'uuid': str(stored_uuid),
                        'x0': image_info["x0"],
                        'y0': image_info["y0"],
                        'x1': image_info["x1"],
                        'y1': image_info["y1"],
                    })
                    
                    logger.info(f"Stored image {stored_uuid} as FIG-{fig_num} from page {page_num} for document {document_id}")
                    
            except Exception as e:
                logger.warning(f"Failed to store image from page {image_info['page_number']}: {str(e)}")
        
        # Store the mapping for later use in processing
        self.image_uuid_mapping = image_uuid_mapping

        try:
            if isinstance(data, str):
                pdf_info = pdf2image.pdfinfo_from_path(data)
            else:
                pdf_bytes = BytesIO(data)
                pdf_info = pdf2image.pdfinfo_from_bytes(pdf_bytes.getvalue())

            max_pages = pdf_info["Pages"]
            logger.info(f"PDF has {max_pages} pages to process")

            # Create a task queue to process pages in order
            pending_tasks = []
            completed_tasks = []
            next_page_to_yield = 1

            # Process pages with a sliding window, in batches
            for batch_start in range(1, max_pages + 1, self.vlm_batch_size):
                batch_end = min(
                    batch_start + self.vlm_batch_size - 1, max_pages
                )
                logger.debug(
                    f"Preparing batch of pages {batch_start}-{batch_end}/{max_pages}"
                )

                # Convert the batch of pages to images
                if isinstance(data, str):
                    images = pdf2image.convert_from_path(
                        data,
                        dpi=150,
                        first_page=batch_start,
                        last_page=batch_end,
                    )
                else:
                    pdf_bytes = BytesIO(data)
                    images = pdf2image.convert_from_bytes(
                        pdf_bytes.getvalue(),
                        dpi=150,
                        first_page=batch_start,
                        last_page=batch_end,
                    )

                # Create tasks for each page in the batch
                for i, image in enumerate(images):
                    page_num = batch_start + i
                    task = asyncio.create_task(
                        self.process_and_yield(image, page_num)
                    )
                    task.page_num = page_num  # Store page number for sorting
                    pending_tasks.append(task)

                # Check if any tasks have completed and yield them in order
                while pending_tasks:
                    # Get the first done task without waiting
                    done_tasks, pending_tasks_set = await asyncio.wait(
                        pending_tasks,
                        timeout=0.01,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if not done_tasks:
                        break

                    # Add completed tasks to our completed list
                    pending_tasks = list(pending_tasks_set)
                    completed_tasks.extend(iter(done_tasks))

                    # Sort completed tasks by page number
                    completed_tasks.sort(key=lambda t: t.page_num)

                    # Yield results in order
                    while (
                        completed_tasks
                        and completed_tasks[0].page_num == next_page_to_yield
                    ):
                        task = completed_tasks.pop(0)
                        yield await task
                        next_page_to_yield += 1

            # Wait for and yield any remaining tasks in order
            while pending_tasks:
                done_tasks, _ = await asyncio.wait(pending_tasks)
                completed_tasks.extend(done_tasks)
                pending_tasks = []

                # Sort and yield remaining completed tasks
                completed_tasks.sort(key=lambda t: t.page_num)

                # Yield results in order
                while (
                    completed_tasks
                    and completed_tasks[0].page_num == next_page_to_yield
                ):
                    task = completed_tasks.pop(0)
                    yield await task
                    next_page_to_yield += 1

            total_elapsed = time.perf_counter() - ingest_start
            logger.info(
                f"Completed PDF conversion in {total_elapsed:.2f} seconds"
            )

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise


class BasicPDFParser(AsyncParser[str | bytes]):
    """A parser for PDF data."""

    def __init__(
        self,
        config: IngestionConfig,
        database_provider: DatabaseProvider,
        llm_provider: CompletionProvider,
    ):
        self.database_provider = database_provider
        self.llm_provider = llm_provider
        self.config = config
        self.PdfReader = PdfReader

    async def ingest(
        self, data: str | bytes, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Ingest PDF data and yield text from each page."""
        if isinstance(data, str):
            raise ValueError("PDF data must be in bytes format.")
        pdf = self.PdfReader(BytesIO(data))
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text is not None:
                page_text = "".join(
                    filter(
                        lambda x: (
                            unicodedata.category(x)
                            in [
                                "Ll",
                                "Lu",
                                "Lt",
                                "Lm",
                                "Lo",
                                "Nl",
                                "No",
                            ]  # Keep letters and numbers
                            or "\u4e00" <= x <= "\u9fff"  # Chinese characters
                            or "\u0600" <= x <= "\u06ff"  # Arabic characters
                            or "\u0400" <= x <= "\u04ff"  # Cyrillic letters
                            or "\u0370" <= x <= "\u03ff"  # Greek letters
                            or "\u0e00" <= x <= "\u0e7f"  # Thai
                            or "\u3040" <= x <= "\u309f"  # Japanese Hiragana
                            or "\u30a0" <= x <= "\u30ff"  # Katakana
                            or "\uff00"
                            <= x
                            <= "\uffef"  # Halfwidth and Fullwidth Forms
                            or x in string.printable
                        ),
                        page_text,
                    )
                )  # Keep characters in common languages ; # Filter out non-printable characters
                yield page_text


class PDFParserUnstructured(AsyncParser[str | bytes]):
    def __init__(
        self,
        config: IngestionConfig,
        database_provider: DatabaseProvider,
        llm_provider: CompletionProvider,
        ocr_provider: OCRProvider,
    ):
        self.database_provider = database_provider
        self.llm_provider = llm_provider
        self.config = config
        try:
            from unstructured.partition.pdf import partition_pdf

            self.partition_pdf = partition_pdf

        except ImportError as e:
            logger.error("PDFParserUnstructured ImportError :  ", e)

    async def ingest(
        self,
        data: str | bytes,
        partition_strategy: str = "hi_res",
        chunking_strategy="by_title",
    ) -> AsyncGenerator[str, None]:
        # partition the pdf
        elements = self.partition_pdf(
            file=BytesIO(data),
            partition_strategy=partition_strategy,
            chunking_strategy=chunking_strategy,
        )
        for element in elements:
            yield element.text


def extract_embedded_images_from_pdf_data(pdf_data: bytes) -> list[dict]:
    """Extract embedded images from PDF data using pdfplumber."""
    extracted_images = []
    
    try:
        # Open PDF with pdfplumber
        with pdfplumber.open(BytesIO(pdf_data)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Get images from the page
                for img_obj in page.images:
                    try:
                        
                        # Get image dimensions
                        width = img_obj.get('width', 0)
                        height = img_obj.get('height', 0)

                        bbox = [img_obj.get('x0', 0), page.cropbox[3]-img_obj.get('y1', 0),  img_obj.get('x1', 0), page.cropbox[3]-img_obj.get('y0', 0)]
                        img_page = page.crop(bbox=bbox)
                        img = img_page.to_image(resolution=2048)
                        with BytesIO() as output:
                            img.original.save(output, format='PNG')
                            image_data = output.getvalue()
                            del img

                        content_hash = hashlib.sha256(image_data).hexdigest()
                        image_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, content_hash))
                        
                        
                        # Create image metadata
                        image_info = {
                            'uuid': image_uuid,
                            'content_hash': content_hash,
                            'image_data': image_data,
                            'mime_type': 'image/png',
                            'width': width,
                            'height': height,
                            'page_number': page_num,
                            'x0': img_obj.get('x0', 0),
                            'y0': img_obj.get('y0', 0),
                            'x1': img_obj.get('x1', 0),
                            'y1': img_obj.get('y1', 0),
                        }
                        
                        extracted_images.append(image_info)
                        logger.info(f"Extracted image {image_uuid} from page {page_num}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image from page {page_num}: {str(e)}")
                        continue
                        
    except Exception as e:
        logger.error(f"Error extracting images from PDF: {str(e)}")
        
    return extracted_images


def add_figure_numbers_to_image(image_pil, figure_number: int, position_x: int = 10, position_y: int = 10) -> Image.Image:
    """Add a figure number overlay to an image at the specified position."""
    try:
        # Create a copy of the image to avoid modifying the original
        img_copy = image_pil.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Figure label
        fig_text = f"FIG-{figure_number}"
        
        # Get image dimensions
        width, height = img_copy.size
        
        # Use provided position, but ensure it's within bounds
        x = max(5, min(position_x, width - 80))
        y = max(5, min(position_y, height - 25))
        
        # Draw background rectangle for better visibility
        if font:
            bbox = draw.textbbox((x, y), fig_text, font=font)
            # Add padding around text
            padding = 3
            draw.rectangle([
                bbox[0] - padding, 
                bbox[1] - padding, 
                bbox[2] + padding, 
                bbox[3] + padding
            ], fill="white", outline="red", width=2)
            draw.text((x, y), fig_text, fill="red", font=font)
        else:
            # Fallback without font
            draw.rectangle([x-3, y-3, x+70, y+22], fill="white", outline="red", width=2)
            draw.text((x, y), fig_text, fill="red")
        
        return img_copy
        
    except Exception as e:
        logger.warning(f"Failed to add figure number to image: {str(e)}")
        return image_pil  # Return original if overlay fails
