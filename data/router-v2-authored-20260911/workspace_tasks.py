"""New local product-workspace specimens. All data and contracts are authored.
These specify bounded scenarios, not conformance to an external standard.
"""
TASKS=[]
def add(name,family,split,contract,data,answer,mechanism):
    TASKS.append(dict(name=name,family=family,split=split,contract=contract,data=data,answer=answer,mechanism=mechanism))
add('continued_config','tools','pilot',
'Parse lines as key=value. A line starting with one space continues the previous value, adding a newline and the rest of that line. A # line is a comment. Empty lines are ignored. Continuations never follow comments in this fixture. Preserve spaces after =. Duplicate keys are invalid and must produce an error with their one-based line number, rather than silently overwrite. Return either values or error.',
{'lines':['title=North',' two rooms','# memo','count=3']},{'values':{'title':'North\ntwo rooms','count':'3'}},'continued_record_parser')
add('permission_bits','tools','calibration',
'Given a three-digit octal mode string, render the nine owner/group/other rwx permission characters. Each triple uses bits 4=r, 2=w, 1=x, otherwise -. No special bits. Return the rendered string for each supplied mode in input order.',
{'modes':['640','751','000']},['rw-r-----','rwxr-x--x','---------'],'permission_bit_projection')
add('wire_labels','tools','test',
'Decode each byte list as a sequence of labels: one length byte followed by that many ASCII letter bytes. A zero length terminates the name and must be the final byte. Pointers and lengths above 63 are unsupported. Invalid/truncated packets return null. Join valid labels with dots; the root-only packet returns an empty string.',
{'packets':[[3,97,112,105,4,116,101,115,116,0],[2,120],[0],[1,97,0,4]]},['api.test',None,'',None],'length_delimited_name_decode')
add('chunked_payload','tools','test',
'Decode the supplied chunk stream. Each chunk begins with an ASCII hexadecimal size followed by CRLF, exactly that many payload characters (all ASCII), then CRLF. A zero-size chunk followed by one empty CRLF ends the stream. No extensions or trailers. Return payload and number of nonzero chunks; reject missing delimiters or trailing data.',
{'stream':'4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n'},{'payload':'Wikipedia','chunks':2},'chunk_framing_decoder')
add('quoted_bytes','tools','test',
'Decode =HH hexadecimal byte escapes and soft line breaks written as = followed by LF. All other characters are ASCII literals. Decode resulting bytes as UTF-8. Return the decoded text, or null for malformed escapes or invalid UTF-8. Process each record independently.',
{'records':['line=20one=0Asoft=\nwrap','=C3=A9','=Q0']},['line one\nsoftwrap','é',None],'quoted_printable_escape_decode')
add('path_segments','tools','test',
'Canonicalize these absolute paths. Empty segments and . are ignored; .. pops one segment if available and cannot move above root. Output one leading slash, no trailing slash except root. Percent signs and backslashes are literal, not escapes. Return paths in the input order.',
{'paths':['/a/b/.././c//','/../../x','/','/%2e%2e/q']},['/a/c','/x','/','/%2e%2e/q'],'dot_segment_stack')
add('terminal_overwrite','tools','test',
'Render a one-line terminal initially empty at column zero. Printable characters overwrite a cell and advance. CR sets column zero; BS moves left with floor zero. No deletion occurs on BS or CR. Return the final entire line and cursor column after concatenating all fragments.',
{'fragments':['ab','\rZ','\bY']},{'line':'Yb','cursor':1},'terminal_cursor_overwrite')
add('context_patch','tools','test',
'Apply replacement hunks to original line numbers, all positions one-based. Each hunk has start, old lines and new lines. Verify all old contexts against the original before applying any change. Hunks must not overlap. On any mismatch return {applied:false,lines:original}; otherwise return the edited line array and applied:true. This is a structured patch, not a substring replacement.',
{'lines':['alpha','beta','gamma','delta'],'hunks':[{'start':2,'old':['beta','gamma'],'new':['B']},{'start':4,'old':['delta'],'new':['D','E']}]},{'applied':True,'lines':['alpha','B','D','E']},'verified_line_hunks')
add('table_transpose','tools','test',
'Transpose the supplied rectangular cell matrix, preserving strings exactly and retaining empty cells. The first row is ordinary data, not column names. Empty matrix returns []. No sorting, aggregation or type conversion.',
{'cells':[['name','qty'],['pen','02'],['','7']]},[['name','pen',''],['qty','02','7']],'matrix_transposition')
add('multipart_envelopes','tools','test',
'Parse this bounded multipart representation: --BOUNDARY on its own line starts a part; headers Key: Value continue until an empty line; body extends until the next boundary line. --BOUNDARY-- ends the message. Header names are lowercased; strip one optional space after colon. Remove only the LF immediately before a boundary from body text. Return ordered parts with headers and body. No nested multipart or duplicate headers.',
{'boundary':'cut','message':'--cut\nKind: note\n\nhello\nworld\n--cut\nKind: empty\n\n\n--cut--\n'},[{'headers':{'kind':'note'},'body':'hello\nworld'},{'headers':{'kind':'empty'},'body':''}],'multipart_boundary_parser')
add('calendar_slots','tools','test',
'Produce all permitted HH:MM slots within the inclusive start/end times on this single day. Hours and minutes are separate allowed sets: a time qualifies only if both match. Return sorted times with two digits per component. No dates, time zones or interval recurrence are implied.',
{'hours':[9,10],'minutes':[0,20,40],'start':'09:15','end':'10:20'},['09:20','09:40','10:00','10:20'],'calendar_component_intersection')
add('transport_text','tools','test',
'Normalize each string value by removing exactly one initial U+FEFF if present, replacing CRLF with LF, then any remaining CR with LF. Interior U+FEFF stays. Return normalized strings and a boolean per string indicating whether it changed. These are JSON string values, not filesystem newline conventions.',
{'texts':['\ufeffa\r\nb\rc\n','x\ufeffy','\ufeff\ufeffz']},{'texts':['a\nb\nc\n','x\ufeffy','\ufeffz'],'changed':[True,False,True]},'transport_text_normalization')
add('sheet_literal_cells','tools','test',
'Prepare cell strings for a literal-only spreadsheet export. If the first non-space character is one of = + - @, prefix the entire original cell with a single apostrophe. Existing leading apostrophe makes a cell safe. Empty/all-space cells and other text remain unchanged. Do not evaluate formulas or numbers.',
{'cells':['=2+3','  -7',"'=x",'memo','   ','@name']},["'=2+3","'  -7","'=x",'memo','   ',"'@name"],'formula_literalization')
add('network_blocks','tools','test',
'For each dotted-decimal IPv4 address and prefix length 0..32, compute network and broadcast addresses using that prefix. All inputs are valid; do not infer usable-host exclusions. Return objects with network and broadcast strings in input order.',
{'blocks':[['192.0.2.129',25],['10.8.7.6',32],['1.2.3.4',0]]},[{'network':'192.0.2.128','broadcast':'192.0.2.255'},{'network':'10.8.7.6','broadcast':'10.8.7.6'},{'network':'0.0.0.0','broadcast':'255.255.255.255'}],'network_prefix_masks')
add('symbolic_links','tools','test',
'Resolve each supplied absolute lookup path against the exact-path link map. A relative target is interpreted relative to the link parent directory. Normalize . and .. after each substitution. A link may replace any path prefix; preserve its remaining suffix. Return null if the same normalized path occurs twice. Missing entries are ordinary paths, not errors.',
{'links':{'/app/current':'releases/r2','/data/live':'/app/current','/loop/a':'b','/loop/b':'a'},'lookups':['/data/live/file.txt','/loop/a','/plain']},['/app/releases/r2/file.txt',None,'/plain'],'symlink_expansion_cycle_detection')
add('canonical_unicode','tools','test',
'Normalize each label to Unicode NFC. Return original and normalized character counts in Python code points, plus normalized values. Do not casefold, strip accents, trim, or use compatibility normalization.',
{'labels':['e\u0301','A\u030a','①']},{'values':['é','Å','①'],'before':[2,2,1],'after':[1,1,1]},'unicode_canonical_composition')
add('clock_fold','tools','test',
'This fictional clock has UTC-minute periods [0,120) offset +60 and [120,240) offset 0. For each requested local minute, enumerate all UTC minutes in those periods such that UTC+offset=local. Sort possibilities. Zero possibilities means nonexistent and two means ambiguous. Do not use a system time-zone database.',
{'local_minutes':[30,90,150,210]},[[],[30],[90,150],[210]],'inverse_piecewise_clock')
add('armored_records','tools','test',
'Extract blocks delimited by -----BEGIN NOTE----- and -----END NOTE-----. Concatenate intervening lines and strictly decode base64 to UTF-8 text. Ignore text outside blocks. A malformed complete block yields null; an unclosed block is omitted. Return block results in order.',
{'text':'ignore\n-----BEGIN NOTE-----\nSGk=\n-----END NOTE-----\n-----BEGIN NOTE-----\n%%%%\n-----END NOTE-----\n-----BEGIN NOTE-----\nQQ==\n'},['Hi',None],'armored_block_decode')
add('three_value_logic','tools','test',
'Evaluate AND and OR for each pair using three-valued logic: null means unknown. AND is false if either side is false, true only if both true, otherwise null. OR is true if either side true, false only if both false, otherwise null. Return both results per pair; retain JSON boolean types.',
{'pairs':[[False,None],[True,None],[None,None],[True,False]]},[{'and':False,'or':None},{'and':None,'or':True},{'and':None,'or':None},{'and':False,'or':True}],'kleene_logic')
add('serial_numbers','tools','test',
'Compare each [a,b] pair of 32-bit unsigned serial numbers. Let d=(b-a) modulo 2^32. Return equal if d=0, b_newer if 0<d<2^31, a_newer if d>2^31, and ambiguous at exactly 2^31. This is serial order, not ordinary integer order.',
{'pairs':[[4294967294,1],[1,4294967294],[7,7],[0,2147483648]]},['b_newer','a_newer','equal','ambiguous'],'modular_serial_order')
add('pen_plotter','tools','test',
'Simulate commands on an integer plane, starting (0,0) facing east. F n moves n units forward; L and R rotate 90 degrees without moving. Return final position, direction and bounds [minx,miny,maxx,maxy] including the origin and all drawn segments. All F distances are positive.',
{'commands':[['F',3],['L'],['F',2],['L'],['F',5],['R'],['F',1]]},{'position':[-2,3],'direction':'north','bounds':[-2,0,3,3]},'turtle_heading_geometry')

add('committed_journal','agents','smoke',
'Recover durable key/value state from an ordered journal. Each transaction begins, writes zero or more keys, then commits or aborts; apply its buffered writes only at commit. Transactions may interleave. Unfinished and aborted writes have no effect. Writes within one transaction use last value. Return final state and committed transaction IDs in commit order.',
{'initial':{'x':1},'journal':[['begin','a'],['write','a','x',2],['begin','b'],['write','b','y',3],['commit','b'],['abort','a'],['begin','c'],['write','c','z',9]]},{'state':{'x':1,'y':3},'committed':['b']},'transaction_commit_recovery')
add('quota_reclamation','agents','test',
'Choose a subset of deletable artifacts freeing at least required units while minimizing total loss. Break ties by fewer deleted artifacts, then lexicographically sorted names. Pinned artifacts cannot be deleted. Return names, freed units and loss; if impossible return null. Do not actually delete fixture inputs.',
{'required':7,'artifacts':[{'name':'a','size':6,'loss':5,'pinned':False},{'name':'b','size':4,'loss':2,'pinned':False},{'name':'c','size':3,'loss':2,'pinned':False},{'name':'d','size':20,'loss':0,'pinned':True}]},{'names':['b','c'],'freed':7,'loss':4},'minimum_loss_covering_knapsack')
add('deficit_scheduler','agents','test',
'Simulate deficit round robin in queue order A then B. Each round add that queue quantum to its credit, serve head jobs while size<=credit, subtract sizes, and discard remaining credit when that queue becomes empty. Stop when all jobs served. Return service order and number of rounds.',
{'quanta':{'A':2,'B':3},'queues':{'A':[['a1',3],['a2',2]],'B':[['b1',2],['b2',2]]}},{'order':['b1','a1','b2','a2'],'rounds':3},'deficit_round_robin')
add('replicated_commit','agents','test',
'For this leader replication snapshot, the commit index is the largest log index greater than current_commit whose log term equals current_term and which is stored by a strict majority of all listed replicas. A replica match index k stores all entries <=k. If no index qualifies retain current_commit. Return commit and replicas acknowledging it.',
{'current_commit':1,'current_term':3,'terms':{'1':1,'2':2,'3':3,'4':3},'match':{'leader':4,'r1':3,'r2':2,'r3':1,'r4':4}},{'commit':3,'acknowledging':['leader','r1','r4']},'term_constrained_replication_commit')
add('causal_versions','agents','test',
'Compare every pair of named vector clocks, treating missing components as zero. A precedes B only when all A components<=B and at least one is strictly smaller. Equal vectors are equal, otherwise incomparable vectors are concurrent. Return pair outcomes in lexicographic name order.',
{'versions':{'a':{'p':2},'b':{'p':1,'q':1},'c':{'p':2,'q':1}}},[{'pair':['a','b'],'relation':'concurrent'},{'pair':['a','c'],'relation':'before'},{'pair':['b','c'],'relation':'before'}],'vector_clock_partial_order')
add('quorum_witness','agents','test',
'Nodes are integers 0..n-1. Determine whether every read quorum of size r intersects every write quorum of size w. If not, return the lexicographically smallest pair [read_nodes,write_nodes] of disjoint sorted lists. If guaranteed, return null for witness. Report safe and witness.',
{'n':5,'r':2,'w':3},{'safe':False,'witness':[[0,1],[2,3,4]]},'quorum_intersection_witness')
add('observed_remove','agents','test',
'Merge an observed-remove-set event log. Add carries key and unique tag. Remove carries the exact tags observed for that key. A removed tag stays tombstoned even if its add arrives later. A key is present if at least one added tag was not removed. Return sorted present keys and sorted live tags by key. Do not treat removing one tag as removing concurrent tags.',
{'events':[['remove','x',['a']],['add','x','a'],['add','x','b'],['add','y','c'],['remove','y',['c']]]},{'present':['x'],'live_tags':{'x':['b']}},'observed_remove_tombstones')
add('token_admission','agents','test',
'Simulate a token bucket initially full with capacity and refill units per integer time. Before each request refill since the previous request up to capacity. Admit only if requested units fit, consuming them; denied requests consume none. Timestamps are nondecreasing. Return admitted flags and remaining tokens after the last request.',
{'capacity':5,'refill':1,'requests':[[0,4],[1,3],[3,3]]},{'admitted':[True,False,True],'remaining':1},'time_capped_token_bucket')
add('breaker_recovery','agents','test',
'Closed breaker opens after two consecutive admitted failures, blocking requests until failure_time+5. At or after expiry admit one half-open probe; success closes and clears failures, failure reopens for five time units. Successful closed calls clear failure count. Blocked calls ignore their supplied outcome. Return admitted flags, final state and closed failure count (zero while open).',
{'events':[[0,False],[1,False],[2,True],[6,True],[7,False]]},{'admitted':[True,True,False,True,True],'state':'closed','failures':1},'circuit_breaker_automaton')
add('dependency_resolution','agents','test',
'Resolve one version per package satisfying root ranges and transitive dependency ranges. Version strings here are integer majors only; ranges [lo,hi] are inclusive. Choose the lexicographically smallest version tuple when packages are sorted by name. Return versions or null if unsatisfiable. Root ranges are mandatory even if another dependency suggests a different version.',
{'roots':{'a':[1,2],'b':[2,2]},'versions':{'a':{'1':{'b':[1,1]},'2':{'b':[2,2]}},'b':{'1':{},'2':{}}}},{'a':2,'b':2},'constraint_package_resolution')
add('record_layout','agents','test',
'Compute the layout under this explicit binary record contract: each field starts at the smallest offset divisible by its alignment, occupies size bytes, and final record size rounds up to the maximum field alignment. Return offsets by field, final size and number of padding bytes. Do not use platform-native alignment rules.',
{'fields':[{'name':'flag','size':1,'alignment':1},{'name':'count','size':4,'alignment':4},{'name':'kind','size':2,'alignment':2}]},{'offsets':{'flag':0,'count':4,'kind':8},'size':12,'padding':5},'aligned_record_layout')
add('range_tombstones','agents','test',
'Reconstruct latest visible storage values. Put records carry key,value,sequence. A range tombstone [lo,hi) hides puts in that lexicographic key interval with sequence<=tombstone sequence. A later put survives. For each key select its highest-sequence surviving put, omitting invisible keys. Return key/value map. Sequence numbers are unique.',
{'puts':[['a',10,1],['b',20,2],['a',30,4],['c',40,5]],'tombstones':[['a','c',3]]},{'a':30,'c':40},'versioned_range_visibility')
add('escrow_reservations','agents','test',
'Process reservation events against an initial available quantity. Reserve(id,n) succeeds only for a new ID and sufficient available units, subtracting units immediately. Commit consumes that reservation permanently; cancel returns its units. A committed/cancelled/unknown reservation cannot transition again. Failed events do nothing. Return success flags, available quantity and active reservations.',
{'initial':8,'events':[['reserve','a',5],['reserve','b',4],['cancel','a'],['reserve','b',4],['commit','b'],['cancel','b']]},{'success':[True,False,True,True,True,False],'available':4,'active':{}},'reservation_lifecycle')
add('archive_inverse','agents','test',
'Recover text from the Burrows-Wheeler last column and primary index using this exact contract: start with n empty rows, repeatedly prepend each last-column character to its corresponding row and sort rows lexicographically, n times. The row at zero-based primary is the recovered text. No sentinel stripping. Return text and length; validate 0<=primary<n.',
{'last_column':'ANNB$AA','primary':4},{'text':'BANANA$','length':7},'inverse_burrows_wheeler')
add('seat_apportionment','agents','test',
'Allocate seats by largest remainders. Compute each group quota=seats*weight/total_weight; assign floors, then one extra seat to the largest fractional remainders until seats exhausted. Break remainder ties by group name. Return seats by group and names receiving extras in award order. Use exact fractions.',
{'seats':7,'weights':{'A':5,'B':3,'C':2}},{'seats':{'A':4,'B':2,'C':1},'extras':['A']},'largest_remainder_apportionment')
add('avl_insertion','agents','test',
'Insert keys into an initially empty AVL binary search tree in the supplied order. Ignore duplicates. After each insertion use standard balance factor height(left)-height(right), empty height 0, leaf height 1; repair outside imbalance by single rotation and inside imbalance by double rotation. Return final tree as [key,left,right], null children, and preorder keys.',
{'keys':[30,10,20,40,50]},{'tree':[20,[10,None,None],[40,[30,None,None],[50,None,None]]],'preorder':[20,10,40,30,50]},'avl_rotations')
add('automaton_minimize','agents','test',
'Partition this complete deterministic automaton into behaviorally equivalent states: two states are equivalent iff every finite suffix is accepted from both or rejected from both. Include all listed states, even unreachable ones. Return sorted state lists sorted lexicographically; transition symbols are 0 and 1. This is language-equivalence refinement, not graph reachability.',
{'accepting':['C','D'],'transitions':{'A':{'0':'B','1':'C'},'B':{'0':'A','1':'D'},'C':{'0':'D','1':'C'},'D':{'0':'C','1':'D'}}},[['A','B'],['C','D']],'dfa_partition_refinement')
add('prefix_archive','agents','test',
'Build a Huffman code: repeatedly remove two lowest-weight nodes, breaking equal weights by the lexicographically smallest leaf name in the node. First removed becomes left bit 0, second right bit 1. Parent weight is sum; parent tie key is its smallest leaf name. Encode message using the resulting codes. Return code map and bit string. At least two symbols exist.',
{'weights':{'a':5,'b':2,'c':1},'message':'abca'},{'codes':{'a':'1','b':'01','c':'00'},'bits':'101001'},'huffman_prefix_tree')
add('optimal_wrap','agents','test',
'Wrap words without splitting or reordering. A line uses word lengths plus single spaces. Width is a hard maximum. Minimize sum of squared unused spaces over all lines except the final line, whose penalty is zero. Break ties by lexicographically smallest vector of line-ending word indices (one-based). Return lines and penalty.',
{'words':['aaaa','bb','cc','ddddd'],'width':6},{'lines':['aaaa','bb cc','ddddd'],'penalty':5},'optimal_line_breaking')
add('test_set_cover','agents','test',
'Select tests whose union covers every required feature, minimizing total integer duration. Break ties by fewer tests then sorted test names lexicographically. Return selected names, total duration and sorted covered required features; ignore irrelevant features. Return null if coverage impossible.',
{'required':['read','write','delete'],'tests':{'a':{'duration':4,'covers':['read','write']},'b':{'duration':2,'covers':['write','delete']},'c':{'duration':1,'covers':['read']},'d':{'duration':10,'covers':['read','write','delete']}}},{'tests':['b','c'],'duration':3,'covered':['delete','read','write']},'weighted_set_cover')
assert len(TASKS)==41
