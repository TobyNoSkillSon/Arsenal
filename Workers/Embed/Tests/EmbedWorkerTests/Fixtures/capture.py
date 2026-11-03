"""Capture only fixed small tensors; execute v5.12.1 source functions, no model weights."""
import ast,functools,hashlib,itertools,json,math,types,tempfile,urllib.request
from pathlib import Path
import numpy as np
import torch
from torch import nn
DEST=Path(__file__).parent
scratch=tempfile.TemporaryDirectory(prefix='ovis-reference-')
ROOT=Path(scratch.name)
expected=json.loads((DEST/'ovis-reference.json').read_text())['source_sha256']
for name in expected:
 rel='activations.py' if name=='activations.py' else 'models/'+('qwen3_5/' if 'qwen3_5' in name else 'qwen3_vl/')+name
 data=urllib.request.urlopen('https://raw.githubusercontent.com/huggingface/transformers/v5.12.1/src/transformers/'+rel,timeout=30).read()
 assert hashlib.sha256(data).hexdigest()==expected[name],name
 (ROOT/name).write_bytes(data)
ns=dict(torch=torch,nn=nn,np=np,math=math,functools=functools,itertools=itertools)
def extract(file,name,methods=None):
 tree=ast.parse((ROOT/file).read_text());node=next(x for x in tree.body if getattr(x,'name',None)==name)
 node.decorator_list=[]
 if methods:
  node.bases=[];node.body=[x for x in node.body if getattr(x,'name',None) in methods]
  for x in node.body:x.decorator_list=[]
 code=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
 exec(compile(ast.fix_missing_locations(code),file,'exec'),ns)
 return ns[name]
act=extract('activations.py','GELUTanh')()
ns['ACT2FN']={'gelu_pytorch_tanh':act}
MLP=extract('modeling_qwen3_5.py','Qwen3_5VisionMLP')
model=MLP(types.SimpleNamespace(hidden_size=4,intermediate_size=6,hidden_act='gelu_pytorch_tanh'))
x=torch.tensor([[-3.,-1.,0.,1.],[2.,0.5,-0.5,3.]])
w1=torch.linspace(-0.8,0.8,24).reshape(6,4);b1=torch.linspace(-0.4,0.4,6)
w2=torch.linspace(-0.7,0.7,24).reshape(4,6);b2=torch.linspace(-0.2,0.2,4)
with torch.no_grad():
 for param,value in [(model.linear_fc1.weight,w1),(model.linear_fc1.bias,b1),(model.linear_fc2.weight,w2),(model.linear_fc2.bias,b2)]:param.copy_(value)
 y=model(x)
Processor=extract('processing_qwen3_vl.py','Qwen3VLProcessor',['replace_video_token','_calculate_timestamps'])
proc=Processor();proc.video_processor=types.SimpleNamespace(merge_size=2,temporal_patch_size=2)
proc.video_token='<|video_pad|>';proc.vision_start_token='<|vision_start|>';proc.vision_end_token='<|vision_end|>'
Position=extract('modeling_qwen3_vl.py','Qwen3VLModel',['get_rope_index','get_vision_position_ids'])
pos=Position();pos.config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2))
smart=extract('video_processing_qwen3_vl.py','smart_resize')
sampling=extract('video_processing_qwen3_vl.py','Qwen3VLVideoProcessor',['sample_frames'])
sampler=sampling();sampler.fps=2;sampler.min_frames=4;sampler.max_frames=768
special={'<|vision_start|>':10,'<|vision_end|>':11,'<|image_pad|>':12,'<|video_pad|>':13}
def encode(text):
 out=[]
 while text:
  match=next((k for k in special if text.startswith(k)),None)
  if match:out.append(special[match]);text=text[len(match):]
  else:out.append(1000+ord(text[0]));text=text[1:]
 return out
frames=[]
for indices,fps,grid in [([0,15,30,45,60],30,[3,4,6]),([0,1],24,[1,2,2])]:
 text=proc.replace_video_token({'video_grid_thw':np.array([grid]),'video_metadata':[types.SimpleNamespace(fps=fps,frames_indices=indices.copy())]},0)
 # Two images surround the video; tests interleaving and independent grid iteration.
 full='A<|vision_start|>'+ '<|image_pad|>'*4 + '<|vision_end|>'+text+'B<|vision_start|><|image_pad|><|vision_end|>C'
 ids=encode(full);types_=torch.tensor([[1 if i==12 else 2 if i==13 else 0 for i in ids]])
 positions,delta=pos.get_rope_index(torch.tensor([ids]),types_,torch.tensor([[1,4,4],[1,2,2]]),torch.tensor([grid]))
 frames.append(dict(times=[i/fps for i in indices],grid=grid,placeholder=text,ids=ids,positions=positions.flatten().tolist(),delta=delta.item()))
resize=[]
for t,h,w in [(4,1080,1920),(64,1080,1920),(768,1080,1920),(5,1080,1920),(4,32,32),(1,32,32),(7,80,112)]:
 resize.append(dict(frames=t,height=h,width=w,expected=list(smart(t,h,w,2,32,4096,234881024))))
samples=[]
for total,fps in [(2,24),(100,25),(10000,24)]:
 samples.append(dict(total=total,fps=fps,indices=sampler.sample_frames(types.SimpleNamespace(total_num_frames=total,fps=fps)).tolist()))
# Small segmented attention reference, with separate torch SDPA calls.
q=torch.sin(torch.arange(56,dtype=torch.float32)).reshape(1,2,7,4);k=torch.cos(torch.arange(56,dtype=torch.float32)).reshape(1,2,7,4);v=torch.arange(56,dtype=torch.float32).reshape(1,2,7,4)/10
attention=torch.cat([nn.functional.scaled_dot_product_attention(q[:,:,a:b],k[:,:,a:b],v[:,:,a:b]) for a,b in [(0,3),(3,7)]],dim=2)
out=dict(reference='huggingface/transformers@v5.12.1',torch=torch.__version__,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py') if p.name!='capture.py'},activation=dict(x=x.flatten().tolist(),w1=w1.flatten().tolist(),b1=b1.tolist(),w2=w2.flatten().tolist(),b2=b2.tolist(),expected=y.flatten().tolist()),frames=frames,resize=resize,samples=samples,attention=attention.flatten().tolist())
p=DEST/'ovis-reference.json';p.write_text(json.dumps(out,indent=2)+'\n')
for model in ['embeddinggemma-2','ovis-vl-embedding-2b']:
 config=json.loads(Path(f'models/{model}/gate/no-weights.json').read_text())['config']
 # Gemma's unrelated modality configs are not used by the text implementation.
 if model=='embeddinggemma-2':config={k:v for k,v in config.items() if k in ['model_type','text_config']}
 (DEST/f'{model}-tiny.json').write_text(json.dumps(config,indent=2)+'\n')
print('Captured',len(frames),'framing/position,',len(resize),'resize and',len(samples),'sampling fixtures; fixed vision MLP and attention.')

scratch.cleanup()
