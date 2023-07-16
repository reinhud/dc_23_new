"""Define metadata for tags used in OpenAPI documentation."""
from typing import Optional

from pydantic import BaseModel


## ===== Tags MetaData Schema ===== ##
class ExternalDocs(BaseModel):

    description: Optional[str] = None
    ulr: str
class MetaDataTag(BaseModel):

    name: str
    description: Optional[str] = None
    external_docs: Optional[ExternalDocs] = None

    class COnfig:

        allow_population_by_field_name = True
        fields = {"external_docs":{"alias": "externalDocs"}}


## ===== Tags Metadata Definition ===== ##
training_tag = MetaDataTag(
    name="training",
    description="Training endpoint."
)

inference_tag = MetaDataTag(
    name="inference",
    description="Inference endpoint."
)

general_tag = MetaDataTag(
    name="general",
    description="Endpoint for general info."
)




metadata_tags = [training_tag, inference_tag, general_tag]