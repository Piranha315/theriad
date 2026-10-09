"""Fresh task specifications and reference implementations, authored 2026-09-11.
No benchmark prompts were used as templates. References are grader-only material.
"""
TASKS = [
('cache_lru', 'Simulate an initially empty LRU cache of capacity c (a nonnegative integer). Each request is a string key. A hit refreshes recency. A miss inserts the key, evicting the least recently used key if necessary. With capacity zero every request misses and nothing is stored. Return [hit_count, keys_from_least_to_most_recent]. Do not mutate requests.',
'''def solve(c, requests):
    from collections import OrderedDict
    cache=OrderedDict();hits=0
    for key in requests:
        if key in cache:
            hits+=1;cache.move_to_end(key)
        elif c:
            if len(cache)==c:cache.popitem(last=False)
            cache[key]=None
    return [hits,list(cache)]
''', [([2,['a','b','a','c','a','b']],[2,['a','b']]),([0,['x','x']],[0,[]]),([1,[]],[0,[]])], 'Recency eviction state rather than set uniqueness: hits change future eviction; the same set can have different future outcomes.'),
('graph_scc', 'Given a sorted list of unique node strings and directed edges [u,v] using those nodes, return the strongly connected components. Sort strings within each component and sort components lexicographically. Include isolated nodes. Duplicate edges and self-loops are allowed. Do not mutate either input.',
'''def solve(nodes, edges):
    reach={u:{u} for u in nodes}
    for u,v in edges:reach[u].add(v)
    for k in nodes:
        for u in nodes:
            if k in reach[u]:reach[u]|=reach[k]
    unseen=set(nodes);out=[]
    for u in nodes:
        if u in unseen:
            part=sorted(v for v in unseen if v in reach[u] and u in reach[v])
            unseen.difference_update(part);out.append(part)
    return sorted(out)
''', [([['a','b','c','d'],[['a','b'],['b','a'],['b','c'],['c','c']]],[['a','b'],['c'],['d']]),([[],[]],[]),([['x','y'],[['x','y'],['y','x'],['x','y']]],[['x','y']])], 'Mutual reachability partition, unlike prerequisite layering which rejects cycles instead of decomposing them.'),
('weighted_routes', 'For nodes named by unique strings, directed edges [u,v,w] with nonnegative integer weight, and a source node, return a dictionary of shortest distances from source. Unreachable nodes map to null/None. Parallel edges and zero weights are allowed. Inputs are unchanged.',
'''def solve(nodes, edges, source):
    dist={u:float('inf') for u in nodes};dist[source]=0
    for _ in range(len(nodes)):
        for u,v,w in edges:dist[v]=min(dist[v],dist[u]+w)
    return {u:None if x==float('inf') else x for u,x in dist.items()}
''', [([['s','a','b','z'],[['s','a',9],['s','b',2],['b','a',3],['a','b',0]],'s'],{'s':0,'a':5,'b':2,'z':None}),([['x'],[],'x'],{'x':0}),([['a','b'],[['a','b',7],['a','b',0]],'a'],{'a':0,'b':0})], 'Weighted path relaxation differs from HumanEval lexicographic grid walking and unweighted scheduling constraints.'),
('capacity_flow', 'Given n vertices numbered 0..n-1 and directed [u,v,capacity] edges with nonnegative integer capacities, return the maximum flow from vertex 0 to vertex n-1. n is at least 2, self edges are absent, and parallel edges add capacities. Do not mutate input. Enforce flow conservation and shared bottlenecks.',
'''def solve(n, edges):
    cap=[[0]*n for _ in range(n)]
    for u,v,c in edges:cap[u][v]+=c
    total=0
    while True:
        parent=[None]*n;parent[0]=-1;queue=[0]
        for u in queue:
            for v in range(n):
                if parent[v] is None and cap[u][v]>0:parent[v]=u;queue.append(v)
        if parent[n-1] is None:return total
        v=n-1;f=float('inf')
        while v: u=parent[v];f=min(f,cap[u][v]);v=u
        v=n-1
        while v:u=parent[v];cap[u][v]-=f;cap[v][u]+=f;v=u
        total+=f
''', [([4,[[0,1,3],[0,2,2],[1,2,1],[1,3,2],[2,3,3]]],5),([2,[[0,1,2],[0,1,4]]],6),([3,[[0,1,8]]],0)], 'Residual capacity and conservation optimization; shortest-distance and topological outputs cannot answer the bottleneck-flow task.'),
('edit_distance', 'Return the Levenshtein edit distance between two strings, treating each Python Unicode code point as one character. Insertion, deletion and substitution each cost one; matching costs zero. Transposition is not a single operation. Empty strings are allowed.',
'''def solve(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(row[j]+1,nxt[-1]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]
''', [(['kitten','sitting'],3),(['ab','ba'],2),(['','é🙂'],2),(['same','same'],0)], 'Global insertion/deletion alignment differs from counting mirrored mismatches for a palindrome.'),
('convex_hull', 'Given integer points [x,y], return the vertices of their convex hull counterclockwise starting at the lexicographically smallest point. Remove duplicate points and collinear interior boundary points. For an all-collinear input return its two extremes, for one unique point return that point, and for no points return []. Inputs remain unchanged.',
'''def solve(points):
    p=sorted(set(map(tuple,points)))
    if len(p)<=1:return [list(x) for x in p]
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lo=[];hi=[]
    for seq,out in ((p,lo),(p[::-1],hi)):
        for x in seq:
            while len(out)>1 and cross(out[-2],out[-1],x)<=0:out.pop()
            out.append(x)
    return [list(x) for x in lo[:-1]+hi[:-1]]
''', [([[[0,0],[1,1],[2,0],[2,2],[0,2],[0,0]]],[[0,0],[2,0],[2,2],[0,2]]),([[[2,2],[0,0],[1,1]]],[[0,0],[2,2]]),([[]],[])], 'Geometric orientation and envelope selection, distinct from triangle area or sorting coordinates by fixed keys.'),
('lzw_dictionary', 'Compress a string containing only A and B into LZW integer codes. Initial dictionary is A:0, B:1; new phrases receive successive codes starting at 2. Scan left to right, extend the current phrase while it exists; otherwise emit its code, add phrase+next character, and begin a phrase with that character. Emit a nonempty final phrase. Empty input returns []. No reset or maximum dictionary size.',
'''def solve(text):
    d={'A':0,'B':1};w='';out=[]
    for c in text:
        if w+c in d:w+=c
        else:out.append(d[w]);d[w+c]=len(d);w=c
    if w:out.append(d[w])
    return out
''', [(['ABABABA'],[0,1,2,4]),(['AAAAA'],[0,2,2]),([''],[])], 'Adaptive repeated-phrase dictionary construction, not consecutive-character run counting.'),
('wildcard_match', 'Return whether a pattern matches an entire string. Pattern ? matches exactly one Unicode code point and * matches zero or more; all other characters are literal. There is no escape syntax, character class or regex operator. Matching is case sensitive. Empty text/pattern are allowed.',
'''def solve(text,pattern):
    row=[True]+[False]*len(text)
    for p in pattern:
        nxt=[row[0] and p=='*']
        for j,c in enumerate(text,1):nxt.append((nxt[j-1] or row[j]) if p=='*' else row[j-1] and (p=='?' or p==c))
        row=nxt
    return row[-1]
''', [(['abc','a*c'],True),(['abc','a?'],False),(['','**'],True),(['a.b','a.b'],True),(['acb','ab'],False)], 'Whole-string variable-width wildcard matching; not fixed substring/prefix filtering or cyclic substring membership.'),
('utf8_validation', 'Given a list of integers 0..255 representing a UTF-8 byte sequence, return the decoded string if it is valid strict UTF-8; otherwise return None. Reject overlong encodings, isolated continuation bytes, surrogate code points, incomplete sequences and code points above U+10FFFF. Preserve a valid BOM as U+FEFF. Input is unchanged.',
'''def solve(values):
    try:return bytes(values).decode('utf-8','strict')
    except UnicodeDecodeError:return None
''', [([[65,195,169,240,159,153,130]],'Aé🙂'),([[192,175]],None),([[237,160,128]],None),([[240,159]],None),([[]],'')], 'Variable-length byte encoding validity and Unicode scalar constraints; not radix rendering or character case conversion.'),
('coin_optimum', 'Given positive integer denominations and a nonnegative target, return the minimum number of coins that exactly totals target with unlimited copies. Return None if impossible. Duplicate denominations have no effect. Target zero returns zero even with no denominations. Do not mutate inputs.',
'''def solve(denominations,target):
    dp=[0]+[float('inf')]*target
    for x in range(1,target+1):
        for c in denominations:
            if c<=x:dp[x]=min(dp[x],dp[x-c]+1)
    return None if dp[target]==float('inf') else dp[target]
''', [([[1,3,4],6],2),([[4,6],7],None),([[],0],0),([[2,2],4],2)], 'Minimum-cardinality unbounded integer optimization, not a known-denomination remainder divided by one unit value.')
]
