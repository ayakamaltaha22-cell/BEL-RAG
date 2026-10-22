import re
def decompose(draft):
 parts=[x.strip() for x in re.split(r'(?<=[.!?])\s+|;\s*',draft) if x.strip()]
 return [dict(id=f'H{i+1:04d}',text=t,dependencies=[]) for i,t in enumerate(parts)]
