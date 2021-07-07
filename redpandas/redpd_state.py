from dataclasses import dataclass
from dataclasses_json import dataclass_json

from redvox.common.data_window_configuration import DataWindowConfig


@dataclass_json
@dataclass
class PipelineResult:
    output_dir: str
    data_window_config: DataWindowConfig

