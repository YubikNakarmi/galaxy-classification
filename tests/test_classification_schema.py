from api.classify.schemas import ClassificationResponse


def test_classification_response_accepts_batch_upload_payload() -> None:
    response = ClassificationResponse.model_validate(
        {
            "batch_id": "batch-1",
            "task_id": "task-1",
            "image_upload_ids": ["upload-1"],
            "urls": ["https://example.com/image.jpg"],
            "status": "processing",
        }
    )

    assert response.image_upload_ids == ["upload-1"]
    assert response.urls == ["https://example.com/image.jpg"]
