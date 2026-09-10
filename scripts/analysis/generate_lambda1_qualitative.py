"""Evidence-based teaser-style figures from saved successful lambda=1 rollouts."""
from pathlib import Path
import fcntl
import hashlib
import html
import json
import re

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from experiments.robot.libero.move_alignment_comparison import move_reward

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'experiments/robot/libero/results/round3-move-eval-lambda10-90task-2trial-seed20261201'
BASE=ROOT/'experiments/robot/libero/results/move-sweep-eval-pretrained-90task-2trial-seed20261201'
OUT=ROOT/'analysis_results/figures/lambda1_qualitative'
SECTION='lambda1-aligned-qualitative'
WORDS={('x',-1):'FORWARD',('x',1):'BACK',('y',-1):'LEFT',('y',1):'RIGHT',('z',-1):'DOWN',('z',1):'UP'}
SELECTION=[
 (81,1,15,'Book placement: same task as the teaser',
  'The reasoning specifies forward and down during the book-release subtask. The measured end-effector motion follows both directions, and the episode succeeds.',
  'Same task and placement subtask as the Task 81 teaser, but a different query and evaluation trial. The paired pretrained episode also succeeds; this is an aligned placement example, not evidence of task recovery.'),
 (4,0,3,'Butter approach: analogous to the grasping teaser',
  'The end effector moves back and down during the approach to the butter. This query precedes grasping in a successful butter-placement episode.',
  'Task 4 targets the front butter; the teaser is Task 3, which targets the back butter. Both saved Task 3 lambda=1 trials fail, so this analogous successful task is not presented as a Task 3 recovery.'),
 (17,1,3,'Descending toward the middle bowl',
  'The reasoning requests DOWN to approach the middle bowl. The measured motion is predominantly downward, and the episode eventually completes the stacking task.',
  'Paired pretrained episode failed; lambda=1 episode succeeded. This outcome comparison uses the same task, trial, seed and initial-state index, but does not establish that this particular query caused recovery.'),
 (38,0,22,'Moving the moka pot toward the stove',
  'The reasoning requests LEFT + DOWN while transporting the moka pot to the stove. The measured trajectory follows that direction in a successful placement episode.',
  'Paired pretrained episode failed; lambda=1 episode succeeded. The selected query illustrates aligned transport, not a causal attribution of the outcome change.'),
 (50,1,8,'Transporting the soup can toward the basket',
  'The reasoning requests RIGHT while carrying the alphabet-soup can toward the basket. The measured motion is predominantly rightward, and the episode succeeds.',
  'Paired pretrained episode failed; lambda=1 episode succeeded under the same evaluation conditions.'),
 (54,0,2,'Descending toward the tomato-sauce can',
  'The reasoning requests DOWN during the approach to the tomato-sauce can. The measured motion is downward, followed later by successful basket placement.',
  'Paired pretrained episode failed; lambda=1 episode succeeded under the same evaluation conditions.'),
]


def directions(vector):
    return ' + '.join(WORDS[(axis,int(np.sign(v)))] for axis,v in zip(('x','y','z'),vector) if v) or 'STOP'


def arrow(ax,start,end,color,linestyle,linewidth,length):
    start=np.asarray(start,dtype=float);delta=np.asarray(end,dtype=float)-start
    norm=np.linalg.norm(delta)
    if norm<1e-6:raise ValueError('Cannot depict a zero projected direction')
    # Direction arrows have a fixed display length. The true trace is drawn separately.
    endpoint=start+delta/norm*length
    ax.annotate('',xy=endpoint,xytext=start,arrowprops=dict(arrowstyle='-|>',color=color,
                lw=linewidth,linestyle=linestyle,mutation_scale=18),zorder=5)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for line in (RUN/'alignment_queries.jsonl').read_text().splitlines()]
    episodes={x['episode_id']:x for x in json.loads((RUN/'episode_summary.json').read_text())}
    baseline={x['episode_id']:x for x in json.loads((BASE/'episode_summary.json').read_text())}
    cards=[];manifest=[]
    for number,(task,trial,query,title,explanation,comparison) in enumerate(SELECTION,1):
        episode_id=f'libero_90-task{task}-episode{trial}-seed{20261201+task*2+trial}'
        group=sorted([x for x in rows if x['episode_id']==episode_id],key=lambda x:x['policy_query_index'])
        assert episodes[episode_id]['episode_success']
        assert [r['policy_query_index'] for r in group]==list(range(len(group)))
        record=group[query];cosine=move_reward(record);assert cosine>=.5
        reason=np.asarray([record['claims']['motion_axes'].get(a,0) for a in ('x','y','z')])
        delta=(np.asarray(record['state_after']['ee_position'])-record['state_before']['ee_position'])*1000
        assert np.linalg.norm(delta)>10
        text=record['task_instruction'].lower().replace(' ','_').replace('\n','_').replace('.','_')[:50]
        video_index=(task//6)*2+trial+1
        videos=list(RUN.glob(f'rollouts/*/*--episode={video_index}--success=True--task={text}.mp4'))
        assert len(videos)==1
        video=videos[0]
        frame_index=sum(x['executed_chunk_length'] for x in group[:query])
        reader=imageio.get_reader(video);frame=reader.get_data(frame_index);nframes=reader.count_frames();reader.close()
        assert nframes==sum(x['executed_chunk_length'] for x in group)
        expected=np.asarray(record['expected_motion_pixels']);trajectory=np.asarray(record['executed_ee_trajectory_pixels'])
        assert len(trajectory)==record['executed_chunk_length']+1 and np.allclose(expected[0],trajectory[0])
        h,w=frame.shape[:2];assert (h,w)==(224,224)
        # Summarize substantial measured Cartesian components, preserving their real signs.
        actual=np.where(np.abs(delta)>=.35*np.max(np.abs(delta)),np.sign(delta),0).astype(int)
        stated=directions(reason);observed=directions(actual)
        stem=f'lambda1_aligned_{number:02d}_task{task}_trial{trial}_q{query}'
        Image.fromarray(frame).save(OUT/f'{stem}_source.png')
        fig,ax=plt.subplots(figsize=(7.2,7.2));ax.imshow(frame,interpolation='nearest');ax.axis('off')
        fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
        for y,label,color in ((.98,f'REASONING → {stated}','#16883f'),(.90,f'ACTION → {observed}','#d62728')):
            ax.text(.02,y,label,transform=ax.transAxes,ha='left',va='top',fontsize=11,weight='bold',color='white',
                    bbox=dict(boxstyle='round,pad=.35',facecolor=color,edgecolor='white',alpha=.94),zorder=10)
        # Use different lengths so aligned arrows remain visible; neither encodes magnitude.
        arrow(ax,expected[0],expected[-1],'#16883f','-',4,58)
        ax.plot(trajectory[:,0],trajectory[:,1],color='#d62728',lw=3,alpha=.82,marker='o',markersize=2,zorder=4)
        arrow(ax,trajectory[0],trajectory[-1],'#d62728','-',4,40)
        ax.text(.02,.025,f'cos = {cosine:.3f}  |  EPISODE SUCCESS',transform=ax.transAxes,
                color='white',fontsize=10,weight='bold',ha='left',va='bottom',
                bbox=dict(boxstyle='round,pad=.35',facecolor='#183957',edgecolor='white',alpha=.94),zorder=10)
        ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5)
        fig.savefig(OUT/f'{stem}.png',dpi=250,bbox_inches='tight',pad_inches=.04)
        fig.savefig(OUT/f'{stem}.pdf',bbox_inches='tight',pad_inches=.04);plt.close(fig)
        metadata={'task_id':task,'trial_index':trial,'policy_query_index':query,'episode_id':episode_id,
            'seed':record['seed'],'episode_success':True,'paired_baseline_success':baseline[episode_id]['episode_success'],
            'reasoning_raw':record['reasoning_raw'],'reasoning_vector':reason.tolist(),'realized_displacement_mm':delta.tolist(),
            'cosine':cosine,'executed_horizon':record['executed_chunk_length'],'source_video':str(video.relative_to(ROOT)),
            'source_video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'video_frame_index':frame_index,
            'source_image_resolution':[w,h],'title':title,'explanation':explanation,'comparison':comparison,
            'figure_stem':stem,'arrow_note':'solid green reasoning arrow: 58 px; solid red realized-motion arrow: 40 px; lengths normalized for visibility; red trace uses logged projected positions'}
        manifest.append(metadata)
        prefix=f'figures/lambda1_qualitative/{stem}'
        video_url='../'+str(video.relative_to(ROOT))
        cards.append(f'''<figure class="teaser-candidate"><a href="{prefix}.png"><img style="width:100%;height:auto" src="{prefix}.png" alt="{html.escape(title)}"></a>
<figcaption><strong>{number}. Task {task}, trial {trial}, query {query}: {html.escape(title)}</strong><br>
{html.escape(explanation)}<br><strong>Measured alignment:</strong> cosine {cosine:.3f}; displacement [x,y,z] = [{', '.join(f'{v:.1f}' for v in delta)}] mm; {record['executed_chunk_length']} executed actions. <strong>Episode: SUCCESS.</strong><br>
<strong>Comparison:</strong> {html.escape(comparison)}<br>
<a href="{prefix}.pdf">PDF</a> · <a href="{prefix}.png">PNG</a> · <a href="{prefix}_source.png">Source frame</a> · <a href="{html.escape(video_url)}">Rollout video</a>
<details><summary>Exact reasoning and provenance</summary><p><strong>Seed:</strong> {record['seed']}; video frame {frame_index}; native source 224×224.</p><p>{html.escape(record['reasoning_raw'])}</p></details></figcaption></figure>''')
    manifest_path=OUT/'manifest.json';manifest_path.write_text(json.dumps({'checkpoint':'Round 3 lambda=1.0','examples':manifest,
       'selection':'Six aligned queries from successful episodes; selected qualitative examples, not an unbiased sample',
       'task3_limitation':'Both saved Task 3 lambda=1 episodes failed; Task 4 is explicitly an analogous task.'},indent=2)+'\n')
    section=f'''<section id="{SECTION}"><h2>Qualitative review — aligned steps from successful Round 3 λ=1 rollouts</h2>
<p>Six selected examples from the trained Round 3 λ=1 checkpoint. Solid green arrows show the parsed reasoning direction; solid red arrows show the <strong>realized end-effector displacement</strong>, and the thin red trace shows the logged motion. Arrow lengths are normalized for readability and do not encode distance. Alignment uses the original realized-displacement cosine, threshold 0.5.</p>
<p><strong>Matching limits:</strong> Task 81 supplies an aligned successful placement in the same task/subtask as the teaser, but not the same observation or query. Both recorded Task 3 λ=1 trials failed, so example 2 uses the analogous successful Task 4 (front butter). Examples 3–6 are from trials where the paired pretrained policy failed and λ=1 succeeded; this does not establish a causal link between a particular aligned query and task recovery.</p>
<div class="teaser-grid" style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px">{''.join(cards)}</div>
<p><a href="figures/lambda1_qualitative/manifest.json">Download figure provenance and measurements</a>. W&amp;B: <a href="https://wandb.ai/yus047-/ecot-move-alignment/runs/w5atvbk7">Round 3 λ=1</a>.</p></section>'''
    with (ROOT/'analysis_results/.move_report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=ROOT/'analysis_results/index.html';document=path.read_text()
        document=re.sub(rf'<section id="{SECTION}">.*?</section>\n?','',document,flags=re.S)
        assert '</main>' in document
        document=document.replace('</main>',section+'\n</main>',1)
        temp=path.with_suffix('.html.tmp');temp.write_text(document);temp.replace(path)
    thumbs=[]
    for m in manifest:
        image=Image.open(OUT/(m['figure_stem']+'.png')).convert('RGB');image.thumbnail((450,450));thumbs.append(image)
    sheet=Image.new('RGB',(1350,900),'white')
    for i,image in enumerate(thumbs):sheet.paste(image,((i%3)*450,(i//3)*450))
    sheet.save(OUT/'contact_sheet.png')
    print(json.dumps([{'task':m['task_id'],'query':m['policy_query_index'],'cosine':m['cosine'],'paired_baseline_success':m['paired_baseline_success']} for m in manifest],indent=2))


if __name__=='__main__':main()
