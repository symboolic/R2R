# type: ignore
import logging
import time
from typing import Any, AsyncGenerator, Optional

from core import parsers
from core.base import (
    AsyncParser,
    ChunkingStrategy,
    Document,
    DocumentChunk,
    DocumentType,
    IngestionConfig,
    IngestionProvider,
    R2RDocumentProcessingError,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)
from core.providers.database import PostgresDatabaseProvider
from core.providers.llm import (
    LiteLLMCompletionProvider,
    OpenAICompletionProvider,
    R2RCompletionProvider,
)
from core.providers.ocr import MistralOCRProvider
from core.utils import generate_extraction_id

logger = logging.getLogger()


class R2RIngestionConfig(IngestionConfig):
    chunk_size: int = 1024
    chunk_overlap: int = 512
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    extra_fields: dict[str, Any] = {}
    separator: Optional[str] = None


class R2RIngestionProvider(IngestionProvider):
    DEFAULT_PARSERS = {
        DocumentType.BMP: parsers.BMPParser,
        DocumentType.CSV: parsers.CSVParser,
        DocumentType.DOC: parsers.DOCParser,
        DocumentType.DOCX: parsers.DOCXParser,
        DocumentType.EML: parsers.EMLParser,
        DocumentType.EPUB: parsers.EPUBParser,
        DocumentType.HTML: parsers.HTMLParser,
        DocumentType.HTM: parsers.HTMLParser,
        DocumentType.ODT: parsers.ODTParser,
        DocumentType.JSON: parsers.JSONParser,
        DocumentType.MSG: parsers.MSGParser,
        DocumentType.ORG: parsers.ORGParser,
        DocumentType.MD: parsers.MDParser,
        DocumentType.PDF: parsers.BasicPDFParser,
        DocumentType.PPT: parsers.PPTParser,
        DocumentType.PPTX: parsers.PPTXParser,
        DocumentType.TXT: parsers.TextParser,
        DocumentType.XLSX: parsers.XLSXParser,
        DocumentType.GIF: parsers.ImageParser,
        DocumentType.JPEG: parsers.ImageParser,
        DocumentType.JPG: parsers.ImageParser,
        DocumentType.TSV: parsers.TSVParser,
        DocumentType.PNG: parsers.ImageParser,
        DocumentType.HEIC: parsers.ImageParser,
        DocumentType.SVG: parsers.ImageParser,
        DocumentType.MP3: parsers.AudioParser,
        DocumentType.P7S: parsers.P7SParser,
        DocumentType.RST: parsers.RSTParser,
        DocumentType.RTF: parsers.RTFParser,
        DocumentType.TIFF: parsers.ImageParser,
        DocumentType.XLS: parsers.XLSParser,
        DocumentType.PY: parsers.PythonParser,
        DocumentType.CSS: parsers.CSSParser,
        DocumentType.JS: parsers.JSParser,
        DocumentType.TS: parsers.TSParser,
    }

    EXTRA_PARSERS = {
        DocumentType.CSV: {"advanced": parsers.CSVParserAdvanced},
        DocumentType.PDF: {
            "ocr": parsers.OCRPDFParser,
            "unstructured": parsers.PDFParserUnstructured,
            "zerox": parsers.VLMPDFParser,
        },
        DocumentType.XLSX: {"advanced": parsers.XLSXParserAdvanced},
    }

    IMAGE_TYPES = {
        DocumentType.GIF,
        DocumentType.HEIC,
        DocumentType.JPG,
        DocumentType.JPEG,
        DocumentType.PNG,
        DocumentType.SVG,
    }

    def __init__(
        self,
        config: R2RIngestionConfig,
        database_provider: PostgresDatabaseProvider,
        llm_provider: (
            LiteLLMCompletionProvider
            | OpenAICompletionProvider
            | R2RCompletionProvider
        ),
        ocr_provider: MistralOCRProvider,
    ):
        super().__init__(config, database_provider, llm_provider)
        self.config: R2RIngestionConfig = config
        self.database_provider: PostgresDatabaseProvider = database_provider
        self.llm_provider: (
            LiteLLMCompletionProvider
            | OpenAICompletionProvider
            | R2RCompletionProvider
        ) = llm_provider
        self.ocr_provider: MistralOCRProvider = ocr_provider
        self.parsers: dict[DocumentType, AsyncParser] = {}
        self.text_splitter = self._build_text_splitter()
        self._initialize_parsers()

        logger.info(
            f"R2RIngestionProvider initialized with config: {self.config}"
        )

    def _initialize_parsers(self):
        for doc_type, parser in self.DEFAULT_PARSERS.items():
            # will choose the first parser in the list
            if doc_type not in self.config.excluded_parsers:
                self.parsers[doc_type] = parser(
                    config=self.config,
                    database_provider=self.database_provider,
                    llm_provider=self.llm_provider,
                )
        # FIXME: This doesn't allow for flexibility for a parser that might not
        # need an llm_provider, etc.
        for doc_type, parser_names in self.config.extra_parsers.items():
            if not isinstance(parser_names, list):
                parser_names = [parser_names]

            for parser_name in parser_names:
                parser_key = f"{parser_name}_{str(doc_type)}"

                try:
                    self.parsers[parser_key] = self.EXTRA_PARSERS[doc_type][
                        parser_name
                    ](
                        config=self.config,
                        database_provider=self.database_provider,
                        llm_provider=self.llm_provider,
                        ocr_provider=self.ocr_provider,
                    )
                    logger.info(
                        f"Initialized extra parser {parser_name} for {doc_type}"
                    )
                except KeyError as e:
                    logger.error(
                        f"Parser {parser_name} for document type {doc_type} not found: {e}"
                    )

    def _build_text_splitter(
        self, ingestion_config_override: Optional[dict] = None
    ) -> TextSplitter:
        logger.info(
            f"Initializing text splitter with method: {self.config.chunking_strategy}"
        )

        if not ingestion_config_override:
            ingestion_config_override = {}

        chunking_strategy = (
            ingestion_config_override.get("chunking_strategy")
            or self.config.chunking_strategy
        )

        chunk_size = (
            ingestion_config_override.get("chunk_size")
            if ingestion_config_override.get("chunk_size") is not None
            else self.config.chunk_size
        )

        chunk_overlap = (
            ingestion_config_override.get("chunk_overlap")
            if ingestion_config_override.get("chunk_overlap") is not None
            else self.config.chunk_overlap
        )

        if chunking_strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif chunking_strategy == ChunkingStrategy.CHARACTER:
            from shared.utils.splitter.text import CharacterTextSplitter

            separator = (
                ingestion_config_override.get("separator")
                or self.config.separator
                or CharacterTextSplitter.DEFAULT_SEPARATOR
            )

            return CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=separator,
                keep_separator=False,
                strip_whitespace=True,
            )
        elif chunking_strategy == ChunkingStrategy.BASIC:
            raise NotImplementedError(
                "Basic chunking method not implemented. Please use Recursive."
            )
        elif chunking_strategy == ChunkingStrategy.BY_TITLE:
            raise NotImplementedError("By title method not implemented")
        else:
            raise ValueError(f"Unsupported method type: {chunking_strategy}")

    def validate_config(self) -> bool:
        return self.config.chunk_size > 0 and self.config.chunk_overlap >= 0

    def chunk(
        self,
        parsed_document: str | DocumentChunk,
        ingestion_config_override: dict,
    ) -> AsyncGenerator[Any, None]:
        text_spliiter = self.text_splitter
        if ingestion_config_override:
            text_spliiter = self._build_text_splitter(
                ingestion_config_override
            )
        if isinstance(parsed_document, DocumentChunk):
            parsed_document = parsed_document.data

        if isinstance(parsed_document, str):
            chunks = text_spliiter.create_documents([parsed_document])
        else:
            # Assuming parsed_document is already a list of text chunks
            chunks = parsed_document

        for chunk in chunks:
            yield (
                chunk.page_content if hasattr(chunk, "page_content") else chunk
            )

    async def parse(
        self,
        file_content: bytes,
        document: Document,
        ingestion_config_override: dict,
    ) -> AsyncGenerator[DocumentChunk, None]:
        if document.document_type not in self.parsers:
            raise R2RDocumentProcessingError(
                document_id=document.id,
                error_message=f"Parser for {document.document_type} not found in `R2RIngestionProvider`.",
            )
        else:
            t0 = time.time()
            contents = []
            parser_overrides = ingestion_config_override.get(
                "parser_overrides", {}
            )
            if document.document_type.value in parser_overrides:
                logger.info(
                    f"Using parser_override for {document.document_type} with input value {parser_overrides[document.document_type.value]}"
                )
                if parser_overrides[DocumentType.PDF.value] == "zerox":
                    # Collect content from VLMPDFParser
                    async for chunk in self.parsers[
                        f"zerox_{DocumentType.PDF.value}"
                    ].ingest(file_content, document_id=document.id, **ingestion_config_override):
                        if isinstance(chunk, dict) and chunk.get("content"):
                            contents.append(chunk)
                        elif (
                            chunk
                        ):  # Handle string output for backward compatibility
                            contents.append({"content": chunk})
                elif parser_overrides[DocumentType.PDF.value] == "ocr":
                    async for chunk in self.parsers[
                        f"ocr_{DocumentType.PDF.value}"
                    ].ingest(file_content, document_id=document.id, **ingestion_config_override):
                        if isinstance(chunk, dict) and chunk.get("content"):
                            contents.append(chunk)

                if (
                    contents
                    and document.document_type == DocumentType.PDF
                    and parser_overrides.get(DocumentType.PDF.value) == "zerox"
                    or parser_overrides.get(DocumentType.PDF.value) == "ocr"
                ):
                    vlm_ocr_one_page_per_chunk = ingestion_config_override.get(
                        "vlm_ocr_one_page_per_chunk", True
                    )

                    if vlm_ocr_one_page_per_chunk:
                        # Use one page per chunk for OCR/VLM
                        iteration = 0

                        sorted_contents = [
                            item
                            for item in sorted(
                                contents, key=lambda x: x.get("page_number", 0)
                            )
                            if isinstance(item.get("content"), str)
                        ]

                        for content_item in sorted_contents:
                            page_num = content_item.get("page_number", 0)
                            page_content = content_item["content"]

                            # Create a document chunk directly from the page content
                            metadata = {
                                **document.metadata,
                                "chunk_order": iteration,
                                "page_number": page_num,
                            }

                            extraction = DocumentChunk(
                                id=generate_extraction_id(
                                    document.id, iteration
                                ),
                                document_id=document.id,
                                owner_id=document.owner_id,
                                collection_ids=document.collection_ids,
                                data=page_content,
                                metadata=metadata,
                            )
                            iteration += 1
                            yield extraction

                        logger.debug(
                            f"Parsed document with id={document.id}, title={document.metadata.get('title', None)}, "
                            f"user_id={document.metadata.get('user_id', None)}, metadata={document.metadata} "
                            f"into {iteration} extractions in t={time.time() - t0:.2f} seconds using one-page-per-chunk."
                        )
                        return
                    else:
                        # Text splitting
                        text_splitter = self._build_text_splitter(
                            ingestion_config_override
                        )

                        iteration = 0

                        sorted_contents = [
                            item
                            for item in sorted(
                                contents, key=lambda x: x.get("page_number", 0)
                            )
                            if isinstance(item.get("content"), str)
                        ]

                        for content_item in sorted_contents:
                            page_num = content_item.get("page_number", 0)
                            page_content = content_item["content"]

                            page_chunks = text_splitter.create_documents(
                                [page_content]
                            )

                            # Create document chunks for each split piece
                            for chunk in page_chunks:
                                metadata = {
                                    **document.metadata,
                                    "chunk_order": iteration,
                                    "page_number": page_num,
                                }

                                extraction = DocumentChunk(
                                    id=generate_extraction_id(
                                        document.id, iteration
                                    ),
                                    document_id=document.id,
                                    owner_id=document.owner_id,
                                    collection_ids=document.collection_ids,
                                    data=chunk.page_content,
                                    metadata=metadata,
                                )
                                iteration += 1
                                yield extraction

                        logger.debug(
                            f"Parsed document with id={document.id}, title={document.metadata.get('title', None)}, "
                            f"user_id={document.metadata.get('user_id', None)}, metadata={document.metadata} "
                            f"into {iteration} extractions in t={time.time() - t0:.2f} seconds using page-by-page splitting."
                        )
                        return

            else:
                # Standard parsing for non-override cases
                async for text in self.parsers[document.document_type].ingest(
                    file_content, document_id=document.id, **ingestion_config_override
                ):
                    if text is not None:
                        contents.append({"content": text})

            if not contents:
                logging.warning(
                    "No valid text content was extracted during parsing"
                )
                return

            iteration = 0
            for content_item in contents:
                chunk_text = content_item["content"]
                chunks = self.chunk(chunk_text, ingestion_config_override)

                for chunk in chunks:
                    metadata = {**document.metadata, "chunk_order": iteration}
                    if "page_number" in content_item:
                        metadata["page_number"] = content_item["page_number"]

                    extraction = DocumentChunk(
                        id=generate_extraction_id(document.id, iteration),
                        document_id=document.id,
                        owner_id=document.owner_id,
                        collection_ids=document.collection_ids,
                        data=chunk,
                        metadata=metadata,
                    )
                    iteration += 1
                    yield extraction

            logger.debug(
                f"Parsed document with id={document.id}, title={document.metadata.get('title', None)}, "
                f"user_id={document.metadata.get('user_id', None)}, metadata={document.metadata} "
                f"into {iteration} extractions in t={time.time() - t0:.2f} seconds."
            )

    def get_parser_for_document_type(self, doc_type: DocumentType) -> Any:
        return self.parsers.get(doc_type)
