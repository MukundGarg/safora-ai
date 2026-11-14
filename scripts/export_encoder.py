"""Guarded research export: trained bundle -> ONNX/int8 + tokenizer + parity receipt.

No trained bundle means no output. Export/runtime dependencies and authorized local
parity probes are required. This tool does not install Android dependencies.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from safora.classifier.bundle import validate_bundle, file_hashes
from safora.classifier.text import encoder_text


def export(model_path, output, probes_path):
    meta=validate_bundle(model_path)  # Reject generic/random/unverified bundle before imports.
    output=Path(output)
    if output.exists(): raise ValueError('Export destination already exists')
    probes=json.loads(Path(probes_path).read_text())
    if not isinstance(probes,list) or not 3<=len(probes)<=100 or any(not isinstance(p,str) or not p or len(p)>2048 for p in probes):
        raise ValueError('Provide 3..100 authorized bounded parity probes, not test-set tuning data')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,use_safetensors=True).cpu().eval()
    encoded=tokenizer([encoder_text(p) for p in probes],padding=True,truncation=True,max_length=meta['max_tokens'],return_tensors='pt')
    names=list(encoded)
    class Wrapper(torch.nn.Module):
        def forward(self,*inputs):return model(**dict(zip(names,inputs))).logits
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.risk-export-',dir=output.parent) as temporary:
        stage=Path(temporary);fp=stage/'encoder.float.onnx';quant=stage/'encoder.int8.onnx'
        torch.onnx.export(Wrapper(),tuple(encoded.values()),str(fp),input_names=names,output_names=['logits'],
            dynamic_axes={**{n:{0:'batch',1:'sequence'} for n in names},'logits':{0:'batch'}},opset_version=17,dynamo=False)
        quantize_dynamic(str(fp),str(quant),weight_type=QuantType.QInt8)
        with torch.inference_mode(): expected=model(**encoded).logits.numpy()
        feed={k:v.numpy() for k,v in encoded.items()}
        parity={}
        from safora.evaluation.risk import select_labels
        for name,file,tolerance in [('float',fp,1e-4),('int8',quant,.1)]:
            session=ort.InferenceSession(str(file),providers=['CPUExecutionProvider'])
            values=session.run(['logits'],{k:feed[k] for k in [i.name for i in session.get_inputs()]})[0]
            if values.shape!=expected.shape or not np.isfinite(values).all() or not np.allclose(values,expected,atol=tolerance,rtol=0):
                raise ValueError('Export numerical parity failed')
            parity[name]={'max_absolute_logit_error':float(np.max(abs(values-expected))),'atol':tolerance,'rtol':0}
            probabilities=1/(1+np.exp(-values));original=1/(1+np.exp(-expected))
            if any(select_labels(a.tolist(),meta['thresholds'])!=select_labels(b.tolist(),meta['thresholds']) for a,b in zip(probabilities,original)):
                raise ValueError('Export threshold-decision parity failed')
            del session
        tokenizer.save_pretrained(stage)
        (stage/'parity.json').write_text(json.dumps({'probe_count':len(probes),'numerical_parity':parity,
            'threshold_decisions_equal':True,'accuracy_evaluated':False},indent=2)+'\n')
        export_meta={**meta,'artifact_kind':'RESEARCH_ONNX_EXPORT','runtime':'ONNX Runtime CPU','opset':17,
                     'input_names':names,'input_dtype':'int64','file_sha256':file_hashes(stage)}
        (stage/'safora_metadata.json').write_text(json.dumps(export_meta,indent=2)+'\n')
        shutil.copytree(stage,output)
    return {'status':'research_export_complete','numerical_parity_verified':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model');p.add_argument('output');p.add_argument('--probes',required=True)
    a=p.parse_args()
    try: result=export(a.model,a.output,a.probes)
    except (ValueError,ImportError,OSError) as exc:
        print(json.dumps({'status':'export_blocked','error_type':type(exc).__name__}));return 2
    print(json.dumps(result));return 0


if __name__=='__main__':raise SystemExit(main())
