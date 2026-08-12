from dataclasses import dataclass, field
from typing import Optional,Literal
from pathlib import Path



# format
# def _log_formats(self)->tuple[dict, dict]:

#         top_json_format = {
#             "Default": {
#                 "candidates":[],
#                 "latest_update": ""

#                 },
#             "Hubble": {
#             "Dirichlet": {
#                 "candidates":[],
#                 "latest_update": ""
#                 },
#             "Dirichlet": {
#                 "candidates":[],
#                 "latest_update": ""
#             }
#         }

#         candidates_json_format =  {
#             "rank": None,
#             "run_id": "",
#             "model_name": "",
#             "experiment_name": "",
#             "num_classes": None,
#             "cached_at": "",
#             "save_path": "",
#             "metric": {}
#         }

#         return top_json_format, candidates_json_format

@dataclass
class CandidateMetadata:
    rank: int | None
    run_id: str
    model_name: str
    experiment_name: str
    num_classes: int | None
    cached_at: str
    save_path: str
    other_metrics: Optional[dict] = field(default_factory=dict)  # For any additional metrics
    metric: dict = field(default_factory=dict)  # Store metrics as a dictionary


@dataclass 
class ModelMetadata: #for model specific metadata
    run_id: str
    model_name: str
    metric: dict
    save_path: Optional[str|Path] = None
    checksum: Optional[str] = None

@dataclass
class VariantMetadata:
    candidates: list[CandidateMetadata] = field(default_factory=list)
    latest_update: str = ""

@dataclass
class MetadataFile: #top level
    Default: VariantMetadata = field(default_factory=VariantMetadata)
    Hubble: VariantMetadata = field(default_factory=VariantMetadata)
    Dirichlet: VariantMetadata = field(default_factory=VariantMetadata)

    #for easy access to variants
    def __getitem__(self,variant:Literal["Default", "Hubble", "Dirichlet"]):
        return getattr(self,variant)