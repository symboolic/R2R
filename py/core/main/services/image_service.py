import hashlib
import logging
from io import BytesIO
from typing import Any, Dict, Optional
from uuid import UUID

from PIL import Image

from core.base import R2RException
from .base import Service

from ..abstractions import R2RProviders
from ..config import R2RConfig

logger = logging.getLogger()


class ImageService(Service):
    """Service for managing images in the R2R system."""

    def __init__(
        self,
        config: R2RConfig,
        providers: R2RProviders,
    ) -> None:
        super().__init__(config, providers)

    def generate_image_uuid(self, image_data: bytes) -> UUID:
        """Generate deterministic UUID based on image content"""
        content_hash = hashlib.sha256(image_data).hexdigest()
        # Use the first 32 characters of the hash to create a deterministic UUID
        uuid_string = (
            content_hash[:8] + "-" + 
            content_hash[8:12] + "-" + 
            content_hash[12:16] + "-" + 
            content_hash[16:20] + "-" + 
            content_hash[20:32]
        )
        return UUID(uuid_string)

    def get_image_metadata(self, image_data: bytes, mime_type: str) -> Dict[str, Any]:
        """Extract metadata from image"""
        try:
            with BytesIO(image_data) as img_buffer:
                img = Image.open(img_buffer)
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "mime_type": mime_type,
                }
        except Exception as e:
            logger.warning(f"Could not extract image metadata: {e}")
            return {"mime_type": mime_type}

    async def store_image(
        self, 
        image_data: bytes, 
        mime_type: str, 
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[UUID] = None
    ) -> UUID:
        """Store an image in the database and return its UUID"""
        try:
            # Extract image metadata
            img_metadata = self.get_image_metadata(image_data, mime_type)
            if metadata:
                img_metadata.update(metadata)

            # Store in database
            image_uuid = await self.providers.database.images_handler.store_image(
                image_data=image_data,
                mime_type=mime_type,
                width=img_metadata.get('width'),
                height=img_metadata.get('height'),
                metadata=img_metadata,
                document_id=document_id,
            )

            logger.info(f"Successfully stored image with UUID: {image_uuid}")
            return image_uuid

        except Exception as e:
            logger.error(f"Error storing image: {e}")
            raise R2RException(
                status_code=500,
                message=f"Failed to store image: {str(e)}"
            )

    async def get_image_by_uuid(self, image_uuid: UUID) -> Optional[Dict[str, Any]]:
        """Retrieve an image by its UUID"""
        try:
            return await self.providers.database.images_handler.get_image_by_uuid(image_uuid)
        except Exception as e:
            logger.error(f"Error retrieving image {image_uuid}: {e}")
            return None

    async def get_image_metadata_by_uuid(self, image_uuid: UUID) -> Optional[Dict[str, Any]]:
        """Retrieve image metadata (without binary data) by UUID"""
        try:
            return await self.providers.database.images_handler.get_image_metadata_by_uuid(image_uuid)
        except Exception as e:
            logger.error(f"Error retrieving image metadata {image_uuid}: {e}")
            return None

    async def get_images_by_chunk_id(self, chunk_id: UUID) -> list[Dict[str, Any]]:
        """Retrieve all images associated with a chunk"""
        try:
            return await self.providers.database.images_handler.get_images_by_chunk_id(chunk_id)
        except Exception as e:
            logger.error(f"Error retrieving images for chunk {chunk_id}: {e}")
            return []

    async def get_images_by_document_id(self, document_id: UUID) -> list[Dict[str, Any]]:
        """Retrieve all images associated with a document"""
        try:
            return await self.providers.database.images_handler.get_images_by_document_id(document_id)
        except Exception as e:
            logger.error(f"Error retrieving images for document {document_id}: {e}")
            return []

    async def delete_image(self, image_uuid: UUID) -> bool:
        """Delete an image by UUID"""
        try:
            return await self.providers.database.images_handler.delete_image(image_uuid)
        except Exception as e:
            logger.error(f"Error deleting image {image_uuid}: {e}")
            return False

    async def list_images(
        self,
        offset: int = 0,
        limit: int = 100,
        mime_type_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List images with pagination and optional filtering"""
        try:
            return await self.providers.database.images_handler.list_images(
                offset=offset,
                limit=limit,
                mime_type_filter=mime_type_filter,
            )
        except Exception as e:
            logger.error(f"Error listing images: {e}")
            return {"results": [], "total_entries": 0}

    def extract_image_references_from_text(self, text: str) -> list[UUID]:
        """Extract image UUID references from text content"""
        import re
        
        # Look for patterns like [IMAGE:uuid] in the text
        pattern = r'\[IMAGE:([a-f0-9-]{36})\]'
        matches = re.findall(pattern, text)
        
        try:
            return [UUID(match) for match in matches]
        except ValueError as e:
            logger.warning(f"Invalid UUID found in image references: {e}")
            return []

    def add_image_reference_to_text(self, text: str, image_uuid: UUID) -> str:
        """Add an image reference to text content"""
        image_ref = f"[IMAGE:{image_uuid}]"
        return f"{image_ref}\n\n{text}"

    async def process_chunk_images(self, chunk_text: str, chunk_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process images referenced in a chunk and update metadata"""
        # Extract image references from text
        image_uuids = self.extract_image_references_from_text(chunk_text)
        
        # Add any image UUIDs from metadata
        if "image_uuid" in chunk_metadata:
            image_uuids.append(UUID(chunk_metadata["image_uuid"]))
        
        # Remove duplicates
        image_uuids = list(set(image_uuids))
        
        # Update metadata
        if image_uuids:
            chunk_metadata["image_uuids"] = [str(uuid) for uuid in image_uuids]
            chunk_metadata["has_images"] = True
        
        return chunk_metadata 