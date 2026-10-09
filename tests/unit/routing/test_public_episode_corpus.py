"""No-network integrity checks for public task acquisition and adaptation."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('public_corpus',Path(__file__).resolve().parents[3]/'scripts/build_public_episode_corpus.py')
public=importlib.util.module_from_spec(spec);spec.loader.exec_module(public)


def test_arithmetic_variants_share_the_same_structural_group():
    a='First <<12*3=36>> and then <<36-6=30>>. #### 30'
    b='Different names and numbers <<25*7=175>> then <<175-8=167>>. #### 167'
    assert public.arithmetic_shape(a)==public.arithmetic_shape(b)
    assert public.arithmetic_shape(a)!=public.arithmetic_shape('<<12+3=15>> #### 15')


def test_renamed_code_and_changed_constants_share_a_shape():
    assert public.code_shape('def add_one(x):\n return x+1') == public.code_shape('def offset(value):\n return value+73')
    assert public.code_shape('def f(x):\n return x+1') != public.code_shape('def f(x):\n return x*1')


def test_wording_screen_catches_number_and_name_replacements():
    assert public.near_duplicate(public.lexical_signature('Alice bought 12 apples and gave 4 of those apples to her neighbor on Monday.'),
        public.lexical_signature('Maria bought 27 apples and gave 8 of those apples to her neighbor on Tuesday.'))


def test_group_assignment_is_reproducible_and_not_dependent_on_input_order():
    groups=[f'g{i}' for i in range(100)]
    original={g:public.assigned_split(g) for g in groups}
    assert {g:public.assigned_split(g) for g in reversed(groups)}==original
    assert set(original.values())=={'train','calibration','threshold_selection','test','pilot','smoke'}


def test_builder_rejects_modified_source_before_writing(tmp_path):
    cache=tmp_path/'cache';cache.mkdir()
    source=cache/'source.json';source.write_text('{}')
    public.atomic_json(cache/'manifest.json',[{'path':str(source),'sha256':'wrong'}])
    with pytest.raises(ValueError,match='checksum'):
        public.build(cache,tmp_path/'out',[])
    assert not (tmp_path/'out').exists()


def test_builder_never_rewrites_an_existing_package(tmp_path):
    with pytest.raises(ValueError,match='new directory'):
        public.build(tmp_path,tmp_path,[])


def test_legacy_answer_wrapper_cannot_hide_a_previous_question():
    question='How many apples are left if 12 are bought and 4 are eaten?'
    wrapped='Answer the following question with a short, precise response. End your response with the answer on its own line, formatted exactly as: ANSWER: <answer>\n\n'+question
    assert question in public.previous_prompt_texts([{'prompt':wrapped}])


def test_jsonl_remains_one_record_with_unicode_line_separators():
    import json
    value={'prompt':'unicode separators\u0085\u2028\u2029 and café'}
    encoded=public.json_record(value)
    assert len(encoded.splitlines())==1
    assert json.loads(encoded)==value
