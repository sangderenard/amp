from collections import Counter
p='logs/native_alloc_trace.log'
cnt=Counter()
with open(p,'rb') as f:
    for line in f:
        if b'UNREGISTER_NOTFOUND' in line:
            try:
                ptr=line.split(b'UNREGISTER_NOTFOUND')[1].strip().decode('ascii')
            except Exception:
                continue
            cnt[ptr]+=1
for ptr,c in cnt.most_common(20):
    print(f"{c}\t{ptr}")
