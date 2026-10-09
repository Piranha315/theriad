from pathlib import Path
import pytest
from theriad.agents.project_tools import list_paths, ProjectToolError, read_file
from theriad.agents.task_engine import TaskEngine, ToolCall

def test_recursive_listing_confined_and_complete(tmp_path):
    root=tmp_path/'work'; (root/'notes/nested').mkdir(parents=True)
    for name in ['a.txt','b.txt','ignore.csv','nested/c.txt']:
        (root/'notes'/name).write_text('x')
    (tmp_path/'outside.txt').write_text('outside')
    (root/'escape').symlink_to(tmp_path, target_is_directory=True)
    result=list_paths(str(root), recursive=True)
    assert result['truncated'] is False and result['skipped_symlinks'] is True
    assert sorted(e['path'] for e in result['entries'] if e['type']=='file' and e['path'].endswith('.txt'))==['notes/a.txt','notes/b.txt','notes/nested/c.txt']
    assert list_paths(str(root),limit=1,recursive=True)['truncated'] is True
    for path in ['../outside.txt','escape',str(tmp_path)]:
        with pytest.raises(ProjectToolError): list_paths(str(root),path)
    with pytest.raises(ProjectToolError,match='project.list'): read_file(str(root),'notes')

@pytest.mark.parametrize('kwargs',[{'limit':0},{'limit':True},{'limit':2001},{'recursive':'false'}])
def test_listing_rejects_ambiguous_arguments(tmp_path,kwargs):
    with pytest.raises(ProjectToolError):list_paths(str(tmp_path),**kwargs)

def test_listing_reaches_real_task_engine(tmp_path):
    root=tmp_path/'work';root.mkdir();(root/'a.txt').write_text('a')
    engine=TaskEngine(event_dir=tmp_path/'events',allowed_workspace_root=root)
    task=engine.create(goal='list files',workspace_root=str(root),permission_profile='strict')
    result=engine.execute(task.task_id,ToolCall('project.list',{'recursive':True}))
    assert result.ok and not result.approval_required
    assert result.output['entries']==[{'path':'a.txt','type':'file'}]

@pytest.mark.asyncio
async def test_listing_registered_in_broker_and_blocks_escape(tmp_path):
    from theriad.agents.tool_broker import ToolBroker
    root=tmp_path/'work';root.mkdir();(root/'a.txt').write_text('a')
    broker=ToolBroker(project_root=str(root))
    assert 'project.list' in broker.available_tools()
    result=await broker.call('project.list',{'recursive':True})
    assert result.ok and result.output['entries']==[{'path':'a.txt','type':'file'}]
    result=await broker.call('project.list',{'path':'..'})
    assert not result.ok and 'escapes' in result.error
