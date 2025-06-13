import hashlib
import logging
from typing import Any, Optional
from uuid import UUID
import json

from core.base import Handler, R2RException

from .base import PostgresConnectionManager

logger = logging.getLogger()


class PostgresImagesHandler(Handler):
    TABLE_NAME = "images"

    def __init__(
        self,
        project_name: str,
        connection_manager: PostgresConnectionManager,
    ):
        super().__init__(project_name, connection_manager)

    async def create_tables(self):
        logger.info(
            f"Creating table, if it does not exist: {self._get_table_name(PostgresImagesHandler.TABLE_NAME)}"
        )

        try:
            query = f"""
            CREATE TABLE IF NOT EXISTS {self._get_table_name(PostgresImagesHandler.TABLE_NAME)} (
                id UUID PRIMARY KEY,
                content_hash VARCHAR(64) UNIQUE NOT NULL,
                image_data BYTEA NOT NULL,
                mime_type VARCHAR(50) NOT NULL,
                width INTEGER,
                height INTEGER,
                file_size INTEGER,
                document_ids UUID[] DEFAULT ARRAY[]::UUID[],
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            
            -- Index for fast lookup by content hash
            CREATE INDEX IF NOT EXISTS idx_images_content_hash_{self.project_name}
            ON {self._get_table_name(PostgresImagesHandler.TABLE_NAME)} (content_hash);
            
            -- Index for metadata queries
            CREATE INDEX IF NOT EXISTS idx_images_metadata_{self.project_name}
            ON {self._get_table_name(PostgresImagesHandler.TABLE_NAME)} USING GIN (metadata);
            
            -- Index for document_ids array queries
            CREATE INDEX IF NOT EXISTS idx_images_document_ids_{self.project_name}
            ON {self._get_table_name(PostgresImagesHandler.TABLE_NAME)} USING GIN (document_ids);
            """
            await self.connection_manager.execute_query(query)

        except Exception as e:
            logger.warning(f"Error {e} when creating images table.")
            raise e

    def generate_image_uuid(self, image_data: bytes) -> UUID:
        """Generate deterministic UUID based on image content"""
        content_hash = hashlib.sha256(image_data).hexdigest()
        # Use the first 32 characters of the hash to create a deterministic UUID
        uuid_string = content_hash[:8] + "-" + content_hash[8:12] + "-" + content_hash[12:16] + "-" + content_hash[16:20] + "-" + content_hash[20:32]
        return UUID(uuid_string)

    async def store_image(
        self,
        image_data: bytes,
        mime_type: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
        document_id: Optional[UUID] = None,
    ) -> UUID:
        """Store an image in the database and return its UUID"""
        content_hash = hashlib.sha256(image_data).hexdigest()
        image_uuid = self.generate_image_uuid(image_data)
        
        # Check if image already exists
        existing = await self.get_image_by_hash(content_hash)
        if existing:
            # Image exists, add document_id to the list if provided and not already present
            if document_id:
                await self.add_document_to_image(existing['id'], document_id)
            return existing['id']
        
        file_size = len(image_data)
        metadata = metadata or {}
        document_ids = [document_id] if document_id else []
        
        query = f"""
        INSERT INTO {self._get_table_name(PostgresImagesHandler.TABLE_NAME)} 
        (id, content_hash, image_data, mime_type, width, height, file_size, document_ids, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (content_hash) DO UPDATE SET
            document_ids = CASE 
                WHEN $8::UUID[] IS NOT NULL AND array_length($8::UUID[], 1) > 0 THEN
                    array(SELECT DISTINCT unnest(EXCLUDED.document_ids || $8::UUID[]))
                ELSE EXCLUDED.document_ids
            END,
            updated_at = NOW()
        RETURNING id
        """
        
        try:
            result = await self.connection_manager.fetchrow_query(
                query, [image_uuid, content_hash, image_data, mime_type, width, height, file_size, document_ids, json.dumps(metadata)]
            )
            if result:
                logger.info(f"Stored new image with UUID: {image_uuid}")
                return image_uuid
            else:
                # Image already exists, return existing UUID
                existing = await self.get_image_by_hash(content_hash)
                return existing['id']
        except Exception as e:
            logger.error(f"Error storing image: {e}")
            raise R2RException(
                status_code=500,
                message=f"Failed to store image: {str(e)}"
            )

    async def add_document_to_image(self, image_uuid: UUID, document_id: UUID) -> bool:
        """Add a document ID to an image's document_ids array if not already present"""
        query = f"""
        UPDATE {self._get_table_name(PostgresImagesHandler.TABLE_NAME)}
        SET document_ids = array(SELECT DISTINCT unnest(document_ids || $2::UUID[])),
            updated_at = NOW()
        WHERE id = $1 AND NOT ($2 = ANY(document_ids))
        RETURNING id
        """
        
        try:
            result = await self.connection_manager.fetchrow_query(query, [image_uuid, document_id])
            return result is not None
        except Exception as e:
            logger.error(f"Error adding document {document_id} to image {image_uuid}: {e}")
            return False

    async def remove_document_from_image(self, image_uuid: UUID, document_id: UUID) -> bool:
        """Remove a document ID from an image's document_ids array"""
        query = f"""
        UPDATE {self._get_table_name(PostgresImagesHandler.TABLE_NAME)}
        SET document_ids = array_remove(document_ids, $2),
            updated_at = NOW()
        WHERE id = $1
        RETURNING id, array_length(document_ids, 1) as remaining_docs
        """
        
        try:
            result = await self.connection_manager.fetchrow_query(query, [image_uuid, document_id])
            if result:
                # If no documents remain, delete the image
                if result['remaining_docs'] is None or result['remaining_docs'] == 0:
                    await self.delete_image(image_uuid)
                    logger.info(f"Deleted orphaned image {image_uuid}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing document {document_id} from image {image_uuid}: {e}")
            return False

    async def cleanup_images_for_document(self, document_id: UUID) -> dict[str, int]:
        """Clean up images when a document is deleted"""
        # Get all images associated with this document
        images_query = f"""
        SELECT id, document_ids
        FROM {self._get_table_name(PostgresImagesHandler.TABLE_NAME)}
        WHERE $1 = ANY(document_ids)
        """
        
        try:
            images = await self.connection_manager.fetch_query(images_query, [document_id])
            
            deleted_count = 0
            updated_count = 0
            
            for image in images:
                image_id = image['id']
                document_ids = image['document_ids']
                
                if len(document_ids) == 1 and document_ids[0] == document_id:
                    # This image is only used by this document, delete it
                    await self.delete_image(image_id)
                    deleted_count += 1
                    logger.info(f"Deleted image {image_id} (only used by document {document_id})")
                else:
                    # This image is used by other documents, just remove this document reference
                    await self.remove_document_from_image(image_id, document_id)
                    updated_count += 1
                    logger.info(f"Removed document {document_id} from image {image_id}")
            
            return {
                "deleted_images": deleted_count,
                "updated_images": updated_count
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up images for document {document_id}: {e}")
            return {"deleted_images": 0, "updated_images": 0}

    async def get_image_by_hash(self, content_hash: str) -> Optional[dict[str, Any]]:
        """Retrieve an image by its content hash"""
        query = f"""
        SELECT id, content_hash, image_data, mime_type, width, height, file_size, document_ids, metadata, created_at, updated_at
        FROM {self._get_table_name(PostgresImagesHandler.TABLE_NAME)}
        WHERE content_hash = $1
        """
        
        try:
            result = await self.connection_manager.fetchrow_query(query, [content_hash])
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error retrieving image by hash: {e}")
            return None

    async def get_image_by_uuid(self, image_uuid: UUID) -> Optional[dict[str, Any]]:
        """Retrieve an image by its UUID"""
        query = f"""
        SELECT id, content_hash, image_data, mime_type, width, height, file_size, document_ids, metadata, created_at, updated_at
        FROM {self._get_table_name(PostgresImagesHandler.TABLE_NAME)}
        WHERE id = $1
        """
        
        try:
            result = await self.connection_manager.fetchrow_query(query, [image_uuid])
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error retrieving image by UUID: {e}")
            return None

    async def get_image_metadata_by_uuid(self, image_uuid: UUID) -> Optional[dict[str, Any]]:
        """Retrieve image metadata (without binary data) by UUID"""
        query = f"""
        SELECT id, content_hash, mime_type, width, height, file_size, document_ids, metadata, created_at, updated_at
        FROM {self._get_table_name(PostgresImagesHandler.TABLE_NAME)}
        WHERE id = $1
        """
        
        try:
            result = await self.connection_manager.fetchrow_query(query, [image_uuid])
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error retrieving image metadata by UUID: {e}")
            return None

    async def get_images_by_chunk_id(self, chunk_id: UUID) -> list[dict[str, Any]]:
        """Retrieve all images associated with a chunk"""
        # This assumes chunks have image_uuids in their metadata
        query = f"""
        SELECT i.id, i.content_hash, i.mime_type, i.width, i.height, i.file_size, i.document_ids, i.metadata, i.created_at, i.updated_at
        FROM {self._get_table_name(PostgresImagesHandler.TABLE_NAME)} i
        JOIN {self._get_table_name("chunks")} c ON c.metadata ? 'image_uuids' 
        AND c.metadata->'image_uuids' ? i.id::text
        WHERE c.id = $1
        """
        
        try:
            results = await self.connection_manager.fetch_query(query, [chunk_id])
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error retrieving images for chunk {chunk_id}: {e}")
            return []

    async def get_images_by_document_id(self, document_id: UUID) -> list[dict[str, Any]]:
        """Retrieve all images associated with a document"""
        query = f"""
        SELECT id, content_hash, mime_type, width, height, file_size, document_ids, metadata, created_at, updated_at
        FROM {self._get_table_name(PostgresImagesHandler.TABLE_NAME)}
        WHERE $1 = ANY(document_ids)
        """
        
        try:
            results = await self.connection_manager.fetch_query(query, [document_id])
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error retrieving images for document {document_id}: {e}")
            return []

    async def delete_image(self, image_uuid: UUID) -> bool:
        """Delete an image by UUID"""
        query = f"""
        DELETE FROM {self._get_table_name(PostgresImagesHandler.TABLE_NAME)}
        WHERE id = $1
        """
        
        try:
            result = await self.connection_manager.execute_query(query, [image_uuid])
            return result == "DELETE 1"
        except Exception as e:
            logger.error(f"Error deleting image {image_uuid}: {e}")
            return False

    async def update_image_metadata(
        self, 
        image_uuid: UUID, 
        metadata: dict[str, Any]
    ) -> bool:
        """Update image metadata"""
        query = f"""
        UPDATE {self._get_table_name(PostgresImagesHandler.TABLE_NAME)}
        SET metadata = $2, updated_at = NOW()
        WHERE id = $1
        """
        
        try:
            result = await self.connection_manager.execute_query(query, [image_uuid, json.dumps(metadata)])
            return result == "UPDATE 1"
        except Exception as e:
            logger.error(f"Error updating image metadata for {image_uuid}: {e}")
            return False

    async def list_images(
        self,
        offset: int = 0,
        limit: int = 100,
        mime_type_filter: Optional[str] = None,
    ) -> dict[str, Any]:
        """List images with pagination and optional filtering"""
        conditions = []
        params = []
        param_count = 1
        
        if mime_type_filter:
            conditions.append(f"mime_type = ${param_count}")
            params.append(mime_type_filter)
            param_count += 1
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        # Count query
        count_query = f"""
        SELECT COUNT(*) as total
        FROM {self._get_table_name(PostgresImagesHandler.TABLE_NAME)}
        {where_clause}
        """
        
        # Data query
        data_query = f"""
        SELECT id, content_hash, mime_type, width, height, file_size, document_ids, metadata, created_at, updated_at
        FROM {self._get_table_name(PostgresImagesHandler.TABLE_NAME)}
        {where_clause}
        ORDER BY created_at DESC
        OFFSET ${param_count} LIMIT ${param_count + 1}
        """
        
        params.extend([offset, limit])
        
        try:
            count_result = await self.connection_manager.fetchrow_query(count_query, params[:-2] if conditions else [])
            total_entries = count_result['total'] if count_result else 0
            
            results = await self.connection_manager.fetch_query(data_query, params)
            images = [dict(row) for row in results]
            
            return {
                "results": images,
                "total_entries": total_entries,
            }
        except Exception as e:
            logger.error(f"Error listing images: {e}")
            return {"results": [], "total_entries": 0} 