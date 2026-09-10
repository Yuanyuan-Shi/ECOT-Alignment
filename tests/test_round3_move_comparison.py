import hashlib
import json

import pytest

from experiments.robot.libero import round3_move_comparison as report


def write_run(root, seed_offset=0):
    root.mkdir()
    episodes, queries = [], []
    for task in range(90):
        for trial in range(2):
            eid = f'libero_90-task{task}-episode{trial}-seed{20261201+task*2+trial+seed_offset}'
            episodes.append(dict(episode_id=eid, task_id=task, episode_success=trial == 0,
                                 task_instruction=f'task {task}'))
            queries.append(dict(episode_id=eid, task_id=task, claims={'motion_axes': {'x': 1}},
                                state_before={'ee_position': [0, 0, 0]},
                                state_after={'ee_position': [1 if trial == 0 else -1, 0, 0]}))
    (root / 'episode_summary.json').write_text(json.dumps(episodes))
    (root / 'alignment_queries.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in queries))


def test_round3_bins_deltas_and_preservation(tmp_path, monkeypatch):
    base, tuned = tmp_path / 'base', tmp_path / 'tuned'
    write_run(base)
    write_run(tuned)
    monkeypatch.setattr(report, 'BASE', base)
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    state = {'runs': {'01': {'lambda': .1, 'status': 'complete', 'evaluation_root': str(tuned)}}}
    result = report.build_result(state)
    bins = result['models']['01']['cosine_bins']
    assert bins['c<0'] == {'count': 90, 'percentage': 50}
    assert bins['0<=c<0.5']['count'] == 0
    assert bins['c>=0.5']['count'] == 90
    assert all(v == 0 for v in result['models']['01']['change_vs_pretrained'].values())
    state_file = tmp_path / 'status.json'
    state_file.write_text(json.dumps(state))
    monkeypatch.setattr(report, 'STATE', state_file)
    out = tmp_path / 'analysis_results'
    out.mkdir()
    original = '<main><section id="old">Prior results must survive.</section></main>'
    (out / 'index.html').write_text(original)
    report.main()
    first = (out / 'index.html').read_text()
    assert '<section id="old">Prior results must survive.</section>' in first
    assert '500 optimizer steps' not in first
    report.main()
    assert (out / 'index.html').read_text() == first


def test_round3_rejects_wrong_base_seed(tmp_path, monkeypatch):
    base = tmp_path / 'base'
    write_run(base, seed_offset=1)
    monkeypatch.setattr(report, 'BASE', base)
    with pytest.raises(ValueError, match='seed schedule'):
        report.build_result({'runs': {}})


def test_round4_display_name_preserves_prior_rows(tmp_path, monkeypatch):
    base = tmp_path / 'base'
    write_run(base)
    monkeypatch.setattr(report,'BASE',base)
    state={'runs':{
        '05':{'lambda':.5,'status':'complete','evaluation_root':str(base)},
        'round4-failure-weight3':{'lambda':.5,'status':'complete','evaluation_root':str(base),
                                'display_name':'Round 4 failure-weight3'}}}
    result=report.build_result(state)
    assert list(result['models'])==['pretrained','05','round4-failure-weight3']
    rendered=report.render(result)
    assert 'Round 3 lambda=0.5' in rendered
    assert 'Round 4 failure-weight3' in rendered
    assert 'Performance loss is unweighted' in rendered


def test_high_lambda_pending_rows_precede_completed_round4(tmp_path, monkeypatch):
    base=tmp_path/'base'
    write_run(base)
    monkeypatch.setattr(report,'BASE',base)
    state={'runs':{
        '10':{'lambda':1.0,'status':'complete','evaluation_root':str(base)},
        '20':{'lambda':2.0,'status':'training'},
        '50':{'lambda':5.0,'status':'waiting for GPU slot'},
        'round4-failure-weight3':{'lambda':.5,'status':'complete','evaluation_root':str(base),
                                 'display_name':'Round 4 lambda=0.5 — failure-weight3'}}}
    rendered=report.render(report.build_result(state))
    primary=rendered.split('</tbody>',1)[0]
    labels=['Round 3 lambda=1.0','Round 3 lambda=2.0','Round 3 lambda=5.0','Round 4 lambda=0.5']
    positions=[primary.index(label) for label in labels]
    assert positions==sorted(positions)
    state['runs']['20'].update(status='complete',evaluation_root=str(base))
    primary=report.render(report.build_result(state)).split('</tbody>',1)[0]
    positions=[primary.index(label) for label in labels]
    assert positions==sorted(positions)
