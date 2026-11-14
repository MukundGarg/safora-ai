"""Export locally fine-tuned encoder to quantized ONNX; compare outputs before Android use."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('output');a=p.parse_args()
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic,QuantType
m=AutoModelForSequenceClassification.from_pretrained(a.model,local_files_only=True).eval()
t=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
x=t('export probe',return_tensors='pt')
class Wrapper(torch.nn.Module):
    def forward(self,input_ids,attention_mask):return m(input_ids=input_ids,attention_mask=attention_mask).logits
path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
fp=path.with_suffix('.float.onnx')
torch.onnx.export(Wrapper(),(x['input_ids'],x['attention_mask']),str(fp),input_names=['input_ids','attention_mask'],output_names=['logits'],dynamic_axes={'input_ids':{0:'batch',1:'sequence'},'attention_mask':{0:'batch',1:'sequence'},'logits':{0:'batch'}},opset_version=17,dynamo=False)
quantize_dynamic(str(fp),str(path),weight_type=QuantType.QInt8)
