import numpy as np

def binary_metrics(y_true,y_prob,threshold=0.5,eps=1e-8):
    t=np.asarray(y_true).astype(bool)
    p=np.asarray(y_prob)>=threshold
    tp=np.logical_and(t,p).sum()
    fp=np.logical_and(~t,p).sum()
    fn=np.logical_and(t,~p).sum()
    tn=np.logical_and(~t,~p).sum()
    precision=tp/(tp+fp+eps)
    recall=tp/(tp+fn+eps)
    f1=2*precision*recall/(precision+recall+eps)
    iou=tp/(tp+fp+fn+eps)
    specificity=tn/(tn+fp+eps)
    return {"precision":float(precision),"recall":float(recall),"f1":float(f1),
            "iou":float(iou),"specificity":float(specificity)}

def threshold_sweep(y_true,y_prob):
    rows=[]
    for threshold in np.arange(.1,.91,.05):
        m=binary_metrics(y_true,y_prob,float(threshold))
        rows.append({"threshold":round(float(threshold),2),**m})
    return rows
