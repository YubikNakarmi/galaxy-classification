
from loguru import logger
class PipelineError(Exception):
    """Base class for all pipeline-specific exceptions."""
    error_code: str = "pipeline_error"
    log_level: str = "ERROR"  

    def __init__(self, message: str = "", **context):
        self.message = message or self.__class__.__doc__ or self.error_code
        self.context = context  
        super().__init__(self.message)

        logger.bind(error_code=self.error_code, **self.context).log(
            self.log_level, self.message
        )




class InvalidBase64ImageError(PipelineError):
    """Base64 image payload failed to decode."""
    error_code = "invalid_base64_image"


class EmptyImageListError(PipelineError):
    """No images were provided in the request."""
    error_code = "empty_image_list"

class BatchNotFoundError(PipelineError):
    """No classification results found for the given batch_id."""
    error_code = "batch_not_found"

class UnknownVariantError(PipelineError):
    """Requested classification variant does not exist."""
    error_code = "unknown_variant"



class BlobUploadError(PipelineError):
    """Failed to upload a blob (image or heatmap) to storage."""
    error_code = "blob_upload_failed"


class BlobDownloadError(PipelineError):
    """Failed to download/read a blob from storage."""
    error_code = "blob_download_failed"



class BentoServiceTimeoutError(PipelineError):
    """BentoML inference request timed out."""
    error_code = "bento_timeout"


class BentoServiceUnavailableError(PipelineError):
    """BentoML inference service returned a 5xx error."""
    error_code = "bento_unavailable"


class BentoServiceBadRequestError(PipelineError):
    """BentoML inference service rejected the request (4xx, non-retryable)."""
    error_code = "bento_bad_request"


class BentoResponseMismatchError(PipelineError):
    """Number of predictions returned by BentoML doesn't match number of images sent."""
    error_code = "bento_response_mismatch"



class ResponseRecordNotFoundError(PipelineError):
    """No ClassificationResponseRecord found for the given id/task_id."""
    error_code = "response_record_not_found"


class ImageUploadNotFoundError(PipelineError):
    """No ImageUpload rows found for the given batch_id."""
    error_code = "image_upload_not_found"



class GradCAMGenerationError(PipelineError):
    """Failed to generate a GradCAM heatmap for an image."""
    error_code = "gradcam_generation_failed"


class ModelInferenceError(PipelineError):
    """Model forward pass failed."""
    error_code = "model_inference_failed"