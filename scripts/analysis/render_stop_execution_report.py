"""Render transparent stop-inclusive results without altering prior metrics."""
import json,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis_results/move_only_500'
r=json.loads((OUT/'stop_and_execution_analysis.json').read_text())
names=['Original MiniVLA','Round 3 lambda=1','Mismatch only']
testnames=['pretrained','Round 3 lambda 1, 200 steps','Mismatch only, 500 steps']
text=['<h1>Stop alignment and command-to-motion differences</h1>',
'<p><strong>Scope correction:</strong> zero parsed translation vectors are not synonymous with stop. They include rotation-only and gripper-only instructions, missing MOVE tags, and parser conflicts. The tables here include directional queries plus literal <code>MOVE: stop</code>; other zero-vector cases remain excluded. A translation-only check does not certify gripper or orientation correctness.</p>',
'<p><strong>Threshold provenance:</strong> the <a href="https://github.com/MichalZawalski/embodied-CoT/blob/main/scripts/generate_embodied_data/primitive_movements.py">public ECoT primitive generator</a> uses 0.03 on individual components of endpoint state displacement, rescales translation when its L1 norm exceeds 0.09, scales rotation components by 1/10, and calls the result stop only if translation/rotation/gripper labels are all absent. Its published window is up to four states (three transitions), not ten actions. The exact modified LIBERO annotation generator was not found in the checkpoint repository or dataset release. Therefore, this is an explicitly defined ten-step evaluation rule, not a verified reproduction of that dataset’s original stop labeling.</p>',
'<p><strong>Evaluation rule:</strong> retain cosine ≥ 0.5 for directional queries. For literal stop, compute <code>d_cmd = sum(0.05 * clip(a_t, -1, 1))</code> over all ten predicted translation commands. Stop aligns if the Euclidean norm is ≤ 0.03 m, or alternatively if every axis has absolute displacement ≤ 0.03 m. These two variants give identical table counts here. This is net commanded displacement, not traveled path length or measured robot displacement. Opposing commands can cancel. No retraining was performed.</p>']
for rule in ['euclidean','per_axis']:
 text.append('<h2>Policy-command mismatch, stop included: '+rule.replace('_',' ')+'</h2><table><tr><th>Dataset / outcome</th>'+''.join('<th>'+n+'</th>' for n in names)+'</tr>')
 for split in ['training','test']:
  for outcome in ['successful','failed']:
   text.append('<tr><th>'+split.title()+' — '+outcome+'</th>')
   for name,tn in zip(names,testnames):
    d=r[split][name if split=='training' else tn][rule][outcome]
    text.append(f"<td>{d['percent']:.3f}% ({d['mismatches']}/{d['queries']})</td>")
   text.append('</tr>')
 text.append('<tr><th>Test task success</th><td>86.7%</td><td>90.0%</td><td>88.9%</td></tr></table>')
text.append('<p>Training uses teacher-forced soft-decoded actions and fixed source-episode outcomes. Test uses decoded rollout actions and each model’s own episode outcomes. Rates pool queries rather than averaging episode percentages.</p>')
text.append('<h2>Literal-stop queries alone</h2><table><tr><th>Dataset</th>'+''.join('<th>'+n+'</th>' for n in names)+'</tr>')
for split in ['training','test']:
 text.append('<tr><th>'+split.title()+' stop-command mismatches</th>')
 for name,tn in zip(names,testnames):
  ds=r[split][name if split=='training' else tn]['euclidean']
  k=sum(v['stop_mismatches'] for v in ds.values());n=sum(v['stop_queries'] for v in ds.values())
  text.append(f'<td>{k}/{n}</td>')
 text.append('</tr>')
text.append('<tr><th>Test stop realized-motion mismatches</th>')
for name in testnames:
 scores=json.loads((OUT/('execution_scores_'+name.replace(' ','_').replace('=','')+'.json')).read_text())
 stops=[s for s in scores if s['stop']]
 text.append(f"<td>{sum(s['real_norm']>.03 for s in stops)}/{len(stops)}</td>")
text.append('</tr></table><p>These are different questions: a stationary robot may still be receiving nonzero commands.</p>')
text.append('<h2>Directional-only gap diagnostics</h2><table><tr><th>Metric</th>'+''.join('<th>'+n+'</th>' for n in names)+'</tr>')
for key,label in [('raw_mismatches','Full 10-step raw policy mismatch'),('prefix_mismatches','Executed-prefix policy mismatch'),('clipped_prefix_mismatches','Executed-prefix, clipped/scaled policy mismatch'),('real_mismatches','Realized-motion mismatch')]:
 text.append('<tr><th>'+label+'</th>')
 for name in testnames:
  d=r['gap'][name];text.append(f"<td>{100*d[key]/d['queries']:.2f}% ({d[key]}/{d['queries']})</td>")
 text.append('</tr>')
for key,label in [('command_aligned_real_misaligned','Command aligned → realized mismatch'),('command_misaligned_real_aligned','Command mismatch → realized aligned'),('lost_real_motion_under_3cm','Of aligned → mismatch: realized net ≤ 3 cm'),('lost_real_motion_under_3mm','Of aligned → mismatch: realized net ≤ 3 mm')]:
 text.append('<tr><th>'+label+'</th>'+''.join('<td>'+str(r['gap'][n][key])+'</td>' for n in testnames)+'</tr>')
text.append('</table><p>For mismatch-only, 393 queries lose alignment after execution and 34 gain it: 116 + 393 − 34 = 475. Of the 393 lost-alignment queries, 379 have net motion ≤ 3 cm and 91 have net motion ≤ 3 mm despite commanded net displacement exceeding 3 cm. Thus the measured gap is concentrated in low-motion or stalled chunks, rather than being explained by full-versus-truncated horizons or input clipping.</p>')
text.append('<p><strong>Controller mechanism:</strong> installed LIBERO uses OSC_POSE at 20 Hz. Commands are clipped and scaled; a new position target is formed from the current measured end-effector position plus the requested delta on every step. A finite-gain torque controller tracks that target during simulation. Summing these target offsets is not an identity for final displacement. Contact constraints, friction, inertia and actuator limits can alter the response; controller memory and cancellation can also change net direction. Uniform scaling alone does not rotate a vector and therefore cannot explain a cosine gap. See <a href="https://robosuite.ai/docs/modules/controllers.html">robosuite controller documentation</a>.</p>')
text.append('<p><strong>Attribution limit:</strong> stored rollouts establish that the gap concentrates in small realized motions. They do not log joint torque saturation, detailed robot-contact impulses or tracking errors, so no single physical cause (collision versus friction versus tracking dynamics) is established. A simple one-step motion-memory regression did not improve held-out prediction; its saved results should not be cited as evidence that controller lag alone explains the gap. Causal attribution would require instrumented replay.</p>')
text.append('<p><strong>Training implication:</strong> the existing mismatch objective gives all zero vectors a constant loss and zero directional gradient. It cannot teach stopping. A future objective should explicitly penalize excess translation magnitude on valid stop/translation-stationary examples and handle orientation/gripper instructions separately. These results rescore existing checkpoints; they do not include such a retraining change.</p>')
(OUT/'stop_execution.html').write_text('<!doctype html><meta charset="utf-8"><title>Stop and execution analysis</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:20px;color:#193653}td,th{padding:10px;border:1px solid #ccc}table{border-collapse:collapse}th{background:#e8eef4}</style>'+''.join(text))
