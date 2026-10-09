"""Executable cross-checks of retained workspace answers; never delivered to models.

Same author as fixtures. Separate calculations check hand-entered keys, but are
not independent authorship or proof that every hidden semantic defect is absent.
"""
import copy
from collections import Counter
from fractions import Fraction
from itertools import combinations,product
import heapq
import math
import unicodedata


def answer(name,d):
    d=copy.deepcopy(d)
    if name=='continued_config':
        values={};previous=None
        for n,line in enumerate(d['lines'],1):
            if not line or line.startswith('#'):continue
            if line.startswith(' '):values[previous]+='\n'+line[1:];continue
            key,value=line.split('=',1)
            if key in values:return {'error':n}
            values[key]=value;previous=key
        return {'values':values}
    if name=='permission_bits':return [''.join(c if int(digit)&mask else '-' for digit in mode for mask,c in [(4,'r'),(2,'w'),(1,'x')]) for mode in d['modes']]
    if name=='wire_labels':
        out=[]
        for packet in d['packets']:
            labels=[];i=0;valid=False
            while i<len(packet):
                n=packet[i];i+=1
                if n==0:valid=i==len(packet);break
                if n>63 or i+n>len(packet):break
                if any(not (65<=b<=90 or 97<=b<=122) for b in packet[i:i+n]):break
                labels.append(bytes(packet[i:i+n]).decode('ascii'));i+=n
            out.append('.'.join(labels) if valid else None)
        return out
    if name=='chunked_payload':
        stream=d['stream'];i=0;out='';count=0
        while True:
            end=stream.index('\r\n',i);n=int(stream[i:end],16);i=end+2
            out+=stream[i:i+n];i+=n
            assert stream[i:i+2]=='\r\n';i+=2
            if not n:assert i==len(stream);return {'payload':out,'chunks':count}
            count+=1
    if name=='terminal_overwrite':
        line=[];pos=0
        for c in ''.join(d['fragments']):
            if c=='\r':pos=0
            elif c=='\b':pos=max(0,pos-1)
            else:
                if pos==len(line):line.append(c)
                else:line[pos]=c
                pos+=1
        return {'line':''.join(line),'cursor':pos}
    if name=='context_patch':
        original=d['lines'];occupied=set()
        for h in d['hunks']:
            start=h['start']-1;span=set(range(start,start+len(h['old'])))
            if original[start:start+len(h['old'])]!=h['old'] or span&occupied:return {'applied':False,'lines':original}
            occupied|=span
        lines=original[:]
        for h in sorted(d['hunks'],key=lambda h:-h['start']):lines[h['start']-1:h['start']-1+len(h['old'])]=h['new']
        return {'applied':True,'lines':lines}
    if name=='table_transpose':return [list(x) for x in zip(*d['cells'])]
    if name=='multipart_envelopes':
        parts=[]
        for block in d['message'].split('--'+d['boundary'])[1:]:
            if block.startswith('--'):break
            header,body=block[1:].split('\n\n',1)
            headers={}
            for line in header.splitlines():
                key,value=line.split(':',1);headers[key.lower()]=value[1:] if value.startswith(' ') else value
            parts.append({'headers':headers,'body':body[:-1]})
        return parts
    if name=='sheet_literal_cells':return ["'"+x if x.lstrip().startswith(('=','+','-','@')) else x for x in d['cells']]
    if name=='network_blocks':
        import ipaddress
        return [{'network':str(ipaddress.IPv4Network(f'{ip}/{n}',strict=False).network_address),'broadcast':str(ipaddress.IPv4Network(f'{ip}/{n}',strict=False).broadcast_address)} for ip,n in d['blocks']]
    if name=='canonical_unicode':
        values=[unicodedata.normalize('NFC',x) for x in d['labels']]
        return {'values':values,'before':list(map(len,d['labels'])),'after':list(map(len,values))}
    if name=='clock_fold':return [[utc for utc in range(240) if utc+(60 if utc<120 else 0)==local] for local in d['local_minutes']]
    if name=='three_value_logic':return [{'and':False if a is False or b is False else True if a is True and b is True else None,'or':True if a is True or b is True else False if a is False and b is False else None} for a,b in d['pairs']]
    if name=='serial_numbers':
        return [('equal' if a==b else 'ambiguous' if (b-a)%2**32==2**31 else 'b_newer' if (b-a)%2**32<2**31 else 'a_newer') for a,b in d['pairs']]
    if name=='pen_plotter':
        x=y=heading=0;points=[(0,0)];directions=[(1,0),(0,1),(-1,0),(0,-1)]
        for cmd in d['commands']:
            if cmd[0]=='L':heading=(heading+1)%4
            elif cmd[0]=='R':heading=(heading-1)%4
            else:dx,dy=directions[heading];x+=dx*cmd[1];y+=dy*cmd[1];points.append((x,y))
        return {'position':[x,y],'direction':['east','north','west','south'][heading],'bounds':[min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)]}
    if name=='committed_journal':
        state=d['initial'];pending={};committed=[]
        for event in d['journal']:
            op,tx,*args=event
            if op=='begin':pending[tx]={}
            elif op=='write':pending[tx][args[0]]=args[1]
            elif op=='abort':pending.pop(tx)
            else:state.update(pending.pop(tx));committed.append(tx)
        return {'state':state,'committed':committed}
    if name=='deficit_scheduler':
        queues=d['queues'];credit=dict.fromkeys(queues,0);rounds=0;order=[]
        while any(queues.values()):
            rounds+=1
            for k in sorted(queues):
                credit[k]+=d['quanta'][k]
                while queues[k] and queues[k][0][1]<=credit[k]:
                    name,size=queues[k].pop(0);credit[k]-=size;order.append(name)
                if not queues[k]:credit[k]=0
        return {'order':order,'rounds':rounds}
    if name=='replicated_commit':
        commit=d['current_commit']
        for index,term in d['terms'].items():
            index=int(index)
            if index>commit and term==d['current_term'] and sum(k>=index for k in d['match'].values())>len(d['match'])/2:commit=index
        return {'commit':commit,'acknowledging':sorted(k for k,v in d['match'].items() if v>=commit)}
    if name=='causal_versions':
        out=[]
        for a,b in combinations(sorted(d['versions']),2):
            x,y=d['versions'][a],d['versions'][b];keys=x.keys()|y.keys()
            le=all(x.get(k,0)<=y.get(k,0) for k in keys);ge=all(x.get(k,0)>=y.get(k,0) for k in keys)
            out.append({'pair':[a,b],'relation':'equal' if le and ge else 'before' if le else 'after' if ge else 'concurrent'})
        return out
    if name=='quorum_witness':
        for a in combinations(range(d['n']),d['r']):
            for b in combinations(range(d['n']),d['w']):
                if not set(a)&set(b):return {'safe':False,'witness':[list(a),list(b)]}
        return {'safe':True,'witness':None}
    if name=='observed_remove':
        adds={};removed={}
        for op,key,tag in d['events']:
            if op=='add':adds.setdefault(key,set()).add(tag)
            else:removed.setdefault(key,set()).update(tag)
        live={k:sorted(v-removed.get(k,set())) for k,v in adds.items() if v-removed.get(k,set())}
        return {'present':sorted(live),'live_tags':live}
    if name=='token_admission':
        tokens=d['capacity'];last=0;admitted=[]
        for t,amount in d['requests']:
            tokens=min(d['capacity'],tokens+(t-last)*d['refill']);last=t
            accept=amount<=tokens;admitted.append(accept)
            if accept:tokens-=amount
        return {'admitted':admitted,'remaining':tokens}
    if name=='breaker_recovery':
        state='closed';failures=0;until=0;flags=[]
        for t,ok in d['events']:
            if state=='open' and t<until:flags.append(False);continue
            flags.append(True)
            if state=='open':
                if ok:state='closed';failures=0
                else:until=t+5
            else:
                failures=0 if ok else failures+1
                if failures==2:state='open';until=t+5;failures=0
        return {'admitted':flags,'state':state,'failures':failures}
    if name=='dependency_resolution':
        names=sorted(d['versions'])
        for values in product(*(sorted(map(int,d['versions'][k])) for k in names)):
            picked=dict(zip(names,values));constraints=list(d['roots'].items())
            for k,v in picked.items():constraints+=list(d['versions'][k][str(v)].items())
            if all(lo<=picked[k]<=hi for k,(lo,hi) in constraints):return picked
        return None
    if name=='record_layout':
        offset=0;offsets={}
        for f in d['fields']:
            while offset%f['alignment']:offset+=1
            offsets[f['name']]=offset;offset+=f['size']
        alignment=max(f['alignment'] for f in d['fields'])
        while offset%alignment:offset+=1
        return {'offsets':offsets,'size':offset,'padding':offset-sum(f['size'] for f in d['fields'])}
    if name=='escrow_reservations':
        quantity=d['initial'];active={};closed=set();flags=[]
        for op,key,*args in d['events']:
            ok=False
            if op=='reserve' and key not in active and key not in closed and args[0]<=quantity:active[key]=args[0];quantity-=args[0];ok=True
            elif op in ('commit','cancel') and key in active:
                value=active.pop(key);closed.add(key);quantity+=value if op=='cancel' else 0;ok=True
            flags.append(ok)
        return {'success':flags,'available':quantity,'active':active}
    if name=='archive_inverse':
        # LF mapping, independent of the prompt's repeated-table construction.
        last=d['last_column'];counts=Counter();rank=[]
        for c in last:rank.append(counts[c]);counts[c]+=1
        offsets={};pos=0
        for c,n in sorted(counts.items()):offsets[c]=pos;pos+=n
        i=d['primary'];chars=[]
        for _ in last:chars.append(last[i]);i=offsets[last[i]]+rank[i]
        return {'text':''.join(reversed(chars)),'length':len(chars)}
    if name=='avl_insertion':
        def height(t):return 0 if t is None else 1+max(height(t[1]),height(t[2]))
        def rotate(t,left):
            k=2 if left else 1;j=3-k;child=t[k];t[k]=child[j];child[j]=t;return child
        def insert(t,key):
            if t is None:return [key,None,None]
            if key==t[0]:return t
            k=1 if key<t[0] else 2;t[k]=insert(t[k],key)
            balance=height(t[1])-height(t[2])
            if balance>1:
                if key>t[1][0]:t[1]=rotate(t[1],True)
                return rotate(t,False)
            if balance< -1:
                if key<t[2][0]:t[2]=rotate(t[2],False)
                return rotate(t,True)
            return t
        tree=None
        for key in d['keys']:tree=insert(tree,key)
        def preorder(t):return [] if t is None else [t[0]]+preorder(t[1])+preorder(t[2])
        return {'tree':tree,'preorder':preorder(tree)}
    if name=='automaton_minimize':
        trans=d['transitions'];states=sorted(trans);accept=set(d['accepting']);different={tuple(sorted((a,b))) for a,b in combinations(states,2) if (a in accept)!=(b in accept)}
        while True:
            added={tuple(sorted((a,b))) for a,b in combinations(states,2) if any(tuple(sorted((trans[a][c],trans[b][c]))) in different for c in ('0','1'))}-different
            if not added:break
            different|=added
        remaining=set(states);groups=[]
        for a in states:
            if a not in remaining:continue
            group=sorted(b for b in remaining if tuple(sorted((a,b))) not in different);groups.append(group);remaining.difference_update(group)
        return sorted(groups)
    if name=='prefix_archive':
        heap=[(n,k,k) for k,n in d['weights'].items()];heapq.heapify(heap)
        while len(heap)>1:
            a=heapq.heappop(heap);b=heapq.heappop(heap);heapq.heappush(heap,(a[0]+b[0],min(a[1],b[1]),(a[2],b[2])))
        codes={}
        def visit(node,bits):
            if isinstance(node,str):codes[node]=bits
            else:visit(node[0],bits+'0');visit(node[1],bits+'1')
        visit(heap[0][2],'');return {'codes':codes,'bits':''.join(codes[k] for k in d['message'])}
    if name=='optimal_wrap':
        words=d['words'];options=[]
        for mask in range(1<<max(0,len(words)-1)):
            ends=[i+1 for i in range(len(words)-1) if mask&(1<<i)]+[len(words)];lines=[];start=0
            for end in ends:lines.append(' '.join(words[start:end]));start=end
            if all(len(line)<=d['width'] for line in lines):options.append((sum((d['width']-len(line))**2 for line in lines[:-1]),ends,lines))
        penalty,ends,lines=min(options);return {'lines':lines,'penalty':penalty}
    if name=='endian_registers':return [int.from_bytes(bytes(r['bytes']),r['endian'],signed=True) for r in d['records']]
    if name=='delta_varints':
        acc=shift=0;deltas=[]
        for b in d['bytes']:
            acc+=(b&127)<<shift
            if b&128:shift+=7
            else:deltas.append(-(acc+1)//2 if acc&1 else acc//2);acc=shift=0
        total=100;values=[]
        for delta in deltas:total+=delta;values.append(total)
        return {'deltas':deltas,'values':values}
    if name=='column_type_inference':
        result={}
        for k,values in d['columns'].items():
            types={'bool' if isinstance(v,bool) else 'text' if isinstance(v,str) else 'int' if int(v)==v else 'float' for v in values if v is not None}
            result[k]='null' if not types else next(iter(types)) if len(types)==1 else 'float' if types<={'int','float'} else 'text'
        return result
    if name=='hardlink_lifecycle':
        links=d['links'];sizes=d['sizes'];flags=[];reclaimed=0
        for op,name,*args in d['operations']:
            ok=False
            if op=='link' and name in links and args[0] not in links:links[args[0]]=links[name];ok=True
            elif op=='unlink' and name in links:
                inode=links.pop(name);ok=True
                if inode not in links.values():reclaimed+=sizes.pop(inode)
            flags.append(ok)
        return {'success':flags,'links':links,'link_counts':dict(Counter(links.values())),'reclaimed_bytes':reclaimed}
    if name=='web_origin_comparison':
        def origin(r):return (r['scheme'].lower(),r['host'].lower(),r['port'] if r['port'] is not None else (443 if r['scheme'].lower()=='https' else 80))
        return [origin(r)==origin(d['reference']) for r in d['records']]
    if name=='perspective_projection':
        c=d['camera'];out=[]
        for x,y,z in d['points']:
            if z<=0:out.append(None);continue
            u=c['cx']+Fraction(c['f']*x,z);v=c['cy']-Fraction(c['f']*y,z)
            assert u.denominator==v.denominator==1
            out.append([int(u),int(v)])
        return out
    if name=='banker_safety':
        work=d['available'];remaining=set(d['allocated']);done=[]
        while remaining:
            eligible=sorted(k for k in remaining if all(m-a<=w for m,a,w in zip(d['maximum'][k],d['allocated'][k],work)))
            if not eligible:break
            key=eligible[0];done.append(key);remaining.remove(key);work=[w+a for w,a in zip(work,d['allocated'][key])]
        return {'completed':done,'blocked':sorted(remaining),'work':work}
    if name=='term_unification':
        bindings={};variables=set()
        def collect(t):
            if isinstance(t,list):
                for a in t[1:]:collect(a)
            elif t.startswith('?'):variables.add(t)
        def resolve(t):
            if isinstance(t,list):return [t[0]]+[resolve(a) for a in t[1:]]
            return resolve(bindings[t]) if t in bindings else t
        def occurs(v,t):return v==t if isinstance(t,str) else any(occurs(v,a) for a in t[1:])
        queue=d['equations'][:]
        for a,b in queue:collect(a);collect(b)
        while queue:
            a,b=map(resolve,queue.pop(0))
            if a==b:continue
            if isinstance(b,str) and b.startswith('?'):a,b=b,a
            if isinstance(a,str) and a.startswith('?'):
                if occurs(a,b):return None
                bindings[a]=b
            elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b) and a[0]==b[0]:queue.extend(zip(a[1:],b[1:]))
            else:return None
        return {v:resolve(v) for v in variables}
    if name=='cellular_step':
        grid=d['grid'];h=len(grid);w=len(grid[0]);out=[];changed=[]
        for y in range(h):
            row=[]
            for x in range(w):
                n=sum(grid[yy][xx] for yy in range(max(0,y-1),min(h,y+2)) for xx in range(max(0,x-1),min(w,x+2)) if (yy,xx)!=(y,x))
                value=int(n==3 or (grid[y][x]==1 and n==2));row.append(value)
                if value!=grid[y][x]:changed.append([y,x])
            out.append(row)
        return {'grid':out,'changed':changed}
    if name=='residual_quantizer':
        residual=Fraction(0);levels=[]
        for p,q in d['samples']:
            adjusted=Fraction(p,q)+residual;level=math.floor(adjusted+Fraction(1,2));levels.append(level);residual=adjusted-level
        return {'levels':levels,'residual':[residual.numerator,residual.denominator]}
    if name=='disk_look_schedule':
        head=d['head'];direction=1;pending=d['requests'];order=[]
        while pending:
            available=[r for r in pending if direction*(r['cylinder']-head)>=0]
            if not available:direction*=-1;continue
            chosen=min(available,key=lambda r:(abs(r['cylinder']-head),r['id']));order.append(chosen['id']);head=chosen['cylinder'];pending.remove(chosen)
        return {'order':order,'head':head}
    raise ValueError('No independently computed oracle for '+name)
