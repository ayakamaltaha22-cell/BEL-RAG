def generate(prompt,model_name='mistralai/Mistral-7B-Instruct-v0.3'):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 tokenizer=AutoTokenizer.from_pretrained(model_name)
 model=AutoModelForCausalLM.from_pretrained(model_name,device_map='auto',torch_dtype='auto')
 tokens=tokenizer.apply_chat_template([{'role':'user','content':prompt}],return_tensors='pt',add_generation_prompt=True).to(model.device)
 with torch.no_grad():output=model.generate(tokens,do_sample=False,max_new_tokens=512)
 return tokenizer.decode(output[0][tokens.shape[1]:],skip_special_tokens=True)
