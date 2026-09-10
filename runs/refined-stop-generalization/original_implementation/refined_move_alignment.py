"""Directional/explicit-stop command alignment and episode-disjoint validation."""
import random
from dataclasses import dataclass
import torch
from experiments.robot.libero.alignment_evaluator import _split_fields
from experiments.robot.libero.move_alignment_training import move_cosines

def literal_stop(reasoning):
    fields=_split_fields(reasoning).get('MOVE',[])
    return bool(fields) and fields[-1].rsplit('→',1)[-1].strip().lower().rstrip('.;')=='stop'

def episode_split(records, fraction=.2, seed=20261102):
    episodes={}
    for r in records:episodes.setdefault(r['episode_id'],[]).append(r)
    strata={}
    for eid,rows in episodes.items():
        key=(bool(rows[0]['episode_success']),any(literal_stop(r['reasoning_raw']) for r in rows))
        strata.setdefault(key,[]).append(eid)
    validation=set()
    rng=random.Random(seed)
    for key,eids in sorted(strata.items()):
        eids=sorted(eids);rng.shuffle(eids)
        n=min(len(eids)-1,max(1,round(len(eids)*fraction))) if len(eids)>1 else 0
        validation.update(eids[:n])
    train=[r for r in records if r['episode_id'] not in validation]
    val=[r for r in records if r['episode_id'] in validation]
    if not train or not val:raise ValueError('Empty split')
    return train,val

def refined_objective(actions, vectors, stop_mask, threshold=.5, stop_tolerance=.03):
    """Sum separate directional/stop batch means; exclude other zero vectors.

    Direction uses raw decoded translation sum (historical metric). Stop uses
    controller-scaled clipped command sum in metres. No path-energy term is added.
    """
    cosines=move_cosines(actions,vectors)
    directional=vectors.norm(dim=-1)>0
    stop=stop_mask.bool() & ~directional
    eligible=directional|stop
    net=(.05*actions[:,:,:3].float().clamp(-1,1)).sum(dim=1)
    magnitude=net.norm(dim=-1)
    dir_penalty=(torch.relu(threshold-cosines)/(1+threshold)).square()
    # Float32 sums can exceed an exact 3 cm boundary by a few nanometres.
    stop_excess=torch.relu(magnitude-stop_tolerance-1e-8)
    stop_penalty=(stop_excess/stop_tolerance).square()
    zero=actions.sum()*0
    directional_loss=dir_penalty[directional].mean() if directional.any() else zero
    stop_loss=stop_penalty[stop].mean() if stop.any() else zero
    mismatch=(directional & (cosines<threshold)) | (stop & (stop_excess>0))
    return directional_loss+stop_loss, {'cosines':cosines,'net_m':net,'magnitude':magnitude,
        'directional':directional,'stop':stop,'eligible':eligible,'mismatch':mismatch,
        'directional_penalty':dir_penalty,'stop_penalty':stop_penalty}

@dataclass
class EarlyStopping:
    patience:int=4
    best:float=float('inf')
    best_step:int=0
    bad_checks:int=0
    def update(self,value,step):
        improved=value < self.best-1e-12
        if improved:self.best=value;self.best_step=step;self.bad_checks=0
        else:self.bad_checks+=1
        return improved,self.bad_checks>=self.patience
