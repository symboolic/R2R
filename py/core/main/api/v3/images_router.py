import logging
from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Query
from fastapi.responses import Response

from core.base import R2RException

from ...abstractions import R2RProviders, R2RServices
from ...config import R2RConfig
from .base_router import BaseRouterV3
from core.base.api.models import GenericBooleanResponse, GenericMessageResponse

logger = logging.getLogger()


class ImagesRouter(BaseRouterV3):
    def __init__(
        self,
        providers: R2RProviders,
        services: R2RServices,
        config: R2RConfig,
    ):
        super().__init__(providers, services, config)

    def _setup_routes(self):
        @self.router.get("/images/{image_uuid}")
        async def get_image(
            image_uuid: UUID,
            auth_user=Depends(self.providers.auth.auth_wrapper),
        ) -> Response:
            """
            Retrieve an image by its UUID.
            Returns the raw image data with appropriate content type.
            """
            try:
                image_data = await self.services.image.get_image_by_uuid(image_uuid)
                
                if not image_data:
                    raise HTTPException(status_code=404, detail="Image not found")
                
                # Return the image with appropriate content type
                return Response(
                    content=image_data["image_data"],
                    media_type=image_data["mime_type"],
                    headers={
                        "Content-Disposition": f"inline; filename=image_{image_uuid}",
                        "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                    }
                )
                
            except R2RException as e:
                raise HTTPException(status_code=e.status_code, detail=e.message)
            except Exception as e:
                logger.error(f"Error retrieving image {image_uuid}: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.router.get("/images/{image_uuid}/metadata")
        async def get_image_metadata(
            image_uuid: UUID,
            auth_user=Depends(self.providers.auth.auth_wrapper),
        ):
            """
            Retrieve metadata for an image by its UUID.
            Returns image metadata without the binary data.
            """
            try:
                metadata = await self.services.image.get_image_metadata_by_uuid(image_uuid)
                
                if not metadata:
                    raise HTTPException(status_code=404, detail="Image not found")
                
                return metadata
                
            except R2RException as e:
                raise HTTPException(status_code=e.status_code, detail=e.message)
            except Exception as e:
                logger.error(f"Error retrieving image metadata {image_uuid}: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.router.get("/images")
        async def list_images(
            offset: int = Query(0, ge=0, description="Specifies the number of objects to skip. Defaults to 0."),
            limit: int = Query(100, ge=1, le=1000, description="Specifies a limit on the number of objects to return, ranging between 1 and 1000. Defaults to 100."),
            mime_type_filter: Optional[str] = Query(None, description="Filter images by MIME type (e.g., 'image/jpeg')"),
            auth_user=Depends(self.providers.auth.auth_wrapper),
        ):
            """
            List images with pagination and optional filtering.
            Returns image metadata without binary data.
            """
            try:
                result = await self.services.image.list_images(
                    offset=offset,
                    limit=limit,
                    mime_type_filter=mime_type_filter,
                )
                
                return result
                
            except R2RException as e:
                raise HTTPException(status_code=e.status_code, detail=e.message)
            except Exception as e:
                logger.error(f"Error listing images: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.router.delete("/images/{image_uuid}")
        async def delete_image(
            image_uuid: UUID,
            auth_user=Depends(self.providers.auth.auth_wrapper),
        ) -> GenericBooleanResponse:
            """
            Delete an image by its UUID.
            """
            try:
                success = await self.services.image.delete_image(image_uuid)
                
                if not success:
                    raise HTTPException(status_code=404, detail="Image not found")
                
                return GenericBooleanResponse(success=True)
                
            except R2RException as e:
                raise HTTPException(status_code=e.status_code, detail=e.message)
            except Exception as e:
                logger.error(f"Error deleting image {image_uuid}: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.router.get("/chunks/{chunk_id}/images")
        async def get_chunk_images(
            chunk_id: UUID,
            auth_user=Depends(self.providers.auth.auth_wrapper),
        ):
            """
            Retrieve all images associated with a specific chunk.
            Returns image metadata without binary data.
            """
            try:
                images = await self.services.image.get_images_by_chunk_id(chunk_id)
                return {"images": images}
                
            except R2RException as e:
                raise HTTPException(status_code=e.status_code, detail=e.message)
            except Exception as e:
                logger.error(f"Error retrieving images for chunk {chunk_id}: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.router.get("/documents/{document_id}/images")
        async def get_document_images(
            document_id: UUID,
            auth_user=Depends(self.providers.auth.auth_wrapper),
        ):
            """
            Retrieve all images associated with a specific document.
            Returns image metadata without binary data.
            """
            try:
                images = await self.services.image.get_images_by_document_id(document_id)
                return {"images": images}
                
            except R2RException as e:
                raise HTTPException(status_code=e.status_code, detail=e.message)
            except Exception as e:
                logger.error(f"Error retrieving images for document {document_id}: {e}")
                raise HTTPException(status_code=500, detail="Internal server error") 