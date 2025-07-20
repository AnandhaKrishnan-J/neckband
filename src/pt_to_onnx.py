from torch import onnx as torch_onnx
import timm
import torch
  
original_model = timm.create_model('convnext_nano',pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
with torch.no_grad():
        original_model.eval()
        original_model.to(device)
        scripted_model = torch.jit.trace(original_model,example_inputs=torch.rand(1, 3,224,224).to(device),check_trace=False)
input_names = ["input"]
output_names = ["output"]
dummy_input = torch.rand(1, 3, 224, 224)
torch.cuda.empty_cache()
torch.cuda.synchronize()

with torch.no_grad():
 scripted_model.eval()
 dynamic_axes = {'input': {0: 'batch_size', 2: 'width', 3: 'height'},
 'output': {0: 'batch_size', 2: 'width', 3: 'height'}} # adding names for better debugging

torch_onnx.export(scripted_model,
                  dummy_input.to("cuda"),
                  "model.onnx",
                  verbose=True,
                  input_names=input_names, # the model's input names
                  output_names=output_names, # the model's output names
                  dynamic_axes=dynamic_axes,
                  opset_version=14, # the ONNX version to export the model to
                  do_constant_folding=True, # whether to execute constant folding for optimization
                  )