from __future__ import annotations

from typing import Optional
from pathlib import Path
from src.cli.style import UI
ROOT = Path(__file__).parent.parent.parent



class ModelExporter:

    def __init__(self,model_path:str| Path,
                  export_path:str|Path,
                    ui:Optional[UI]=None,
                    input_shape:tuple=(1,3,128,128)):


        
        import torch 
        from src.utils.validation import validate_path_exists

        self.torch = torch
        self.model_path = Path(model_path)
        self.export_path = Path(export_path)

        validate_path_exists(root = ROOT, 
                             path = [self.model_path, self.export_path.parent], 
                             create=True, kind="file", 
                             validate_root=False)

        self.ui = ui or UI(verbose=0) #fallback

        self.model = self.torch.load(self.model_path,weights_only=False ,map_location=self.torch.device("cpu")).eval()

        self.dummy_input = self.torch.randn(input_shape) #default input shape for galaxy images

    
    def export_onnx(self):

        self.ui.info(f"Exporting model to ONNX format at {self.export_path} ...")
        self.torch.onnx.export(self.model, 
                               (self.dummy_input,),
                               str(self.export_path),
                               input_names=["image"],
                               output_names=["output"],
                               opset_version=18,
                                dynamo=True,
                               )

    def export_torchscript(self):

        self.ui.info(f"Exporting model to TorchScript format at {self.export_path} ...")
        traced_model = self.torch.jit.script(self.model, (self.dummy_input,))
        traced_model.save(str(self.export_path))
        
    def export_torchscript_pt2(self):
        exported = self.torch.export.export(self.model, (self.dummy_input,))
        self.torch.export.save(exported, self.export_path)


    def export_tensorRT(self):
        import torch_tensorrt
        
        trt = torch_tensorrt.compile(self.model,inputs = [torch_tensorrt.Input(self.dummy_input.shape)],
                                     enabled_precisions = {self.torch.float},
                                     workspace_size = 1 << 22)
        self.torch.jit.save(trt, str(self.export_path))