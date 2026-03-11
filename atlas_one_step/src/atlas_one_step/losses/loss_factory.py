from .pred_loss import pred_loss
from .semantic_loss import semantic_loss
from .stability_loss import stability_loss

def total_loss(pred,target,mu=0.1,tau=0.1):
    lp=pred_loss(pred,target); ls=semantic_loss(pred,target); lst=stability_loss(pred)
    return lp + mu*ls + tau*lst, {"pred":lp.item(),"sem":ls.item(),"stab":lst.item()}
