import stanza
from stanza.models.tokenization.trainer import Trainer
from minisbd import SBDetect
import argparse
import os
import torch
import onnx
import json
from onnxsim import simplify
from onnxruntime.quantization import quantize_dynamic, QuantType,  quantize_static, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
import wikipediaapi

def extract_wiki_content(lang_code, page="Python_(programming_language)"):
    try:
        w = wikipediaapi.Wikipedia("minisbd", lang_code)

        page = w.page(page)
        text = page.text[:2000]    
        return text
    except:
        return ""

parser = argparse.ArgumentParser(description='Extract ONNX models from Stanza')
parser.add_argument('--stanza-dir', default='', help='Path to Stanza resources directory')
parser.add_argument('--lang-code', default='en', help='Language code (default: en)')
parser.add_argument('--output', default='onnx/', help='Output folder (default: onnx/)')
parser.add_argument('--text', default='', help="Text to feed the stanza model for extraction")
args = parser.parse_args()

if not args.stanza_dir:
    try:
        stanza_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stanza_models")
        os.makedirs(stanza_dir, exist_ok=True)
        stanza.download(args.lang_code, model_dir=stanza_dir, processors="tokenize")
    except Exception as e:
        print(f'Cannot download stanza model: {str(e)}')
        exit(1)
else:
    stanza_dir = args.stanza_dir

resources = stanza.resources.common.load_resources_json(stanza_dir)


stanza_pipeline = stanza.Pipeline(
    lang=args.lang_code,
    processors="tokenize",
    use_gpu=False,
    logging_level="WARNING",
    tokenize_pretokenized=False,
    download_method=stanza.DownloadMethod.REUSE_RESOURCES,
)

_predict = Trainer.predict
outfile = os.path.join(args.output, f"{args.lang_code}.onnx")
outfile_quant = os.path.join(args.output, f"{args.lang_code}-quant.onnx")
outfile_optim = os.path.join(args.output, f"{args.lang_code}-optim.onnx")
def extract_onnx(self, inputs):
    result = _predict(self, inputs)

    units, _, features, _ = inputs

    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    tokp = stanza_pipeline.processors['tokenize']
    vocab = tokp.vocab.state_dict()
    meta = {
        'lang_code': args.lang_code, 
        'config': tokp.config, 
        'vocab': vocab,
        'lexicon': list(self.lexicon) if self.lexicon is not None else None
    }
    
    for f in [outfile, outfile_quant, outfile_optim]:
        if os.path.isfile(f):
            os.unlink(f)

    torch.onnx.export(self.model,         # model being run 
        (units, features),       # model input (or a tuple for multiple inputs) 
        outfile,       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=18,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ['units', 'features'],   # the model's input names
        output_names = ['output'], # the model's output names,
        dynamic_axes={
            'units': {0: 'batch_size', 1: 'feature_dim'},
            'features': {0: 'batch_size', 1: 'feature_dim'}, # Map index to a name
            'output': {0: 'batch_size'}                   # Usually output follows batch
        },
        dynamo=False
    )

    if os.path.isfile(outfile):
        print(f"Extracted {outfile}")

        # Update meta
        m = onnx.load(outfile)
        for k,v in meta.items():
            meta = m.metadata_props.add()
            meta.key = k
            meta.value = json.dumps(v)
        
        model_simp, check = simplify(m)
        onnx.save(m, outfile)

        quant_pre_process(outfile, outfile_optim, skip_symbolic_shape=True)
        quantized_model = quantize_dynamic(outfile_optim, outfile_quant, weight_type=QuantType.QUInt8)
        if os.path.isfile(outfile_quant):
            os.unlink(outfile)
            os.rename(outfile_quant, outfile)
        if os.path.isfile(outfile_optim):
            os.unlink(outfile_optim)
    else:
        print("Something went wrong")
        exit(1)
    
    return result

Trainer.predict = extract_onnx

if args.text:
    text = args.text
else:
    text = extract_wiki_content(args.lang_code)
    if not text:
        print(f"Cannot fetch text for {args.lang_code}, using en")
        text = extract_wiki_content("en")

doc = stanza_pipeline(text)
print([sent.text for sent in doc.sentences])
print(f"# sentences: {len(doc.sentences)}")

# Test
if os.path.isfile(outfile):
    print("Testing model...")
    detector = SBDetect(outfile)
    sentences = detector.sentences(text)

    print(f"Same # of sentences: {len(doc.sentences) == len(sentences)}")
    all_check = True
    for i, sent in enumerate(sentences):
        if i < len(doc.sentences):
            if doc.sentences[i].text != sentences[i]:
                print(f"{i}: {doc.sentences[i].text} != {sentences[i]}")
                all_check = False
        else:
            all_check = False

    if all_check:
        print("All sentences match the original model!")
    else:
        print("Warning: some differences with the original model were found")
        exit(1)
