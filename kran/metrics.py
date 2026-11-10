import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, log_loss


def expected_calibration_error(y_true, probs, n_bins=15):
    probs=np.asarray(probs); y_true=np.asarray(y_true)
    conf=probs.max(1); pred=probs.argmax(1)
    bins=np.linspace(0,1,n_bins+1); ece=0.0
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(conf>lo)&(conf<=hi)
        if m.any(): ece += m.mean()*abs((pred[m]==y_true[m]).mean()-conf[m].mean())
    return float(ece)


def multiclass_brier(y_true, probs):
    y=np.eye(probs.shape[1])[np.asarray(y_true,dtype=int)]
    return float(np.mean(np.sum((probs-y)**2,axis=1)))


def classification_metrics(y_true, probs):
    pred=probs.argmax(1); c=probs.shape[1]
    out={"Accuracy":float(accuracy_score(y_true,pred)),"Macro-F1":float(f1_score(y_true,pred,average='macro')),
         "ECE":expected_calibration_error(y_true,probs),"Brier":multiclass_brier(y_true,probs),
         "NLL":float(log_loss(y_true,probs,labels=list(range(c))))}
    out["AUC"] = float(roc_auc_score(y_true,probs[:,1])) if c==2 else float('nan')
    return out


def regression_metrics(y_true,y_pred):
    return {"MSE":float(mean_squared_error(y_true,y_pred)),"MAE":float(mean_absolute_error(y_true,y_pred)),"R2":float(r2_score(y_true,y_pred))}
