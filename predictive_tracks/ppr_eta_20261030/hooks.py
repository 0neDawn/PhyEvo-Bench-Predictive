"""Only relax the already audited hard correction. Original orchestration intact."""
from predictive_tracks.ppr_20261028.hooks import DDLPHooks,GSWMHooks
from predictive_tracks.ppr_all_20261029.hooks import SlotHooks

def soften(owner,original,hard):
    eta=owner.eta
    audit=owner.current['steps'][-1]
    audit['eta']=eta
    audit['change_l2']*=eta
    for key in ('change_l2_by_object',):
        if key in audit:audit[key]=[x*eta for x in audit[key]]
    for key in ('max_unit_change_l2','mean_unit_change_l2'):
        if key in audit:audit[key]*=eta
    if eta==0:return original
    if eta==1:return hard
    return original+eta*(hard-original)

class SoftDDLP(DDLPHooks):
    def __init__(self,*args,eta,**kwargs):
        super().__init__(*args,**kwargs);self.eta=eta
    def correct(self,value):return soften(self,value,super().correct(value))

class SoftGSWM(GSWMHooks):
    def __init__(self,*args,eta,**kwargs):
        super().__init__(*args,**kwargs);self.eta=eta
    def correct_by_id(self,value,ids):return soften(self,value,super().correct_by_id(value,ids))

class SoftSlot(SlotHooks):
    def __init__(self,*args,eta,**kwargs):
        super().__init__(*args,**kwargs);self.eta=eta
    def correct_slot(self,value):return soften(self,value,super().correct_slot(value))
