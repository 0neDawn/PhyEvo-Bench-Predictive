"""Exactly the already-established state fields and hook positions."""
from predictive_tracks.ppr_eta_20261030.hooks import SoftSlot
from predictive_tracks.ppr_existing_20261102.hooks import ExistingDDLP,ExistingGSWM
from predictive_tracks.ppr_eta_radius_20261104_n.hooks import ScaleReference,TunedPreFinal

class Slot(ScaleReference,SoftSlot):
    def __init__(self,adapter,j,snapshot):
        self.alpha=j['alpha'];super().__init__(adapter,j['mode'],j['model'],snapshot,eta=j['eta'])

class RadiusObject:
    def __init__(self,adapter,j,snapshot):
        self.alpha=j['alpha'];p=dict(j['quality'],eta=j['eta'])
        super().__init__(adapter,j['mode'],snapshot,params=p,capture_all=False)
    def new_audit(self,counts):
        base=self.radius;self.radius=base if self.alpha==1 else self.alpha*base
        super().new_audit(counts);self.current.update(alpha=self.alpha,base_radii=base.cpu().tolist())
class DDLP(RadiusObject,ExistingDDLP):pass
class GSWM(RadiusObject,ExistingGSWM):pass
class Internal(TunedPreFinal):
    def __init__(self,adapter,j,snapshot):super().__init__(adapter,j['model'],j['eta'],j['alpha'],j['mode'],snapshot)
    def shift(self,values):
        out=super().shift(values)
        self.audits[-1]['position']='before final original predictor block' if self.model=='vjepa2_vitl' else 'before final original decoder block'
        return out
def factory(adapter,j,snapshot):
    t=DDLP if j['model']=='ddlp' else GSWM if j['model']=='gswm' else Internal if j['mode']=='pre_final_block' else Slot
    return t(adapter,j,snapshot)
