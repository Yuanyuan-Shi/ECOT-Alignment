"""Directional/explicit-stop command alignment and episode-disjoint validation."""
import random
from dataclasses import dataclass
import torch
from experiments.robot.libero.alignment_evaluator import _split_fields

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
    """Paper Eq.4–11: shared metre-valued command sum and pooled eligible mean.

    Dataset-denormalized LIBERO actions are controller inputs. Convert each
    translation to metres before summing. No path-energy term is added.
    """
    if actions.ndim != 3 or actions.shape[1] != 10 or actions.shape[2] < 3:
        raise ValueError("Expected a 10-step action chunk with translation components")
    directional=vectors.norm(dim=-1)>0
    stop=stop_mask.bool() & ~directional
    eligible=directional|stop
    net=(.05*actions[:,:,:3].double().clamp(-1,1)).sum(dim=1)
    magnitude=net.norm(dim=-1)
    reasoning=vectors.double()
    cosines=(reasoning*net).sum(dim=-1)/(reasoning.norm(dim=-1)*magnitude+1e-8)
    dir_penalty=(torch.relu(threshold-cosines)/(1+threshold)).square()
    stop_penalty=torch.relu(magnitude/stop_tolerance-1).square()
    zero=actions.sum()*0
    per_query=torch.where(directional,dir_penalty,stop_penalty)
    loss=per_query[eligible].mean() if eligible.any() else zero
    mismatch=(directional & (cosines<threshold)) | (stop & (magnitude>stop_tolerance))
    return loss, {'cosines':cosines,'net_m':net,'magnitude':magnitude,
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
