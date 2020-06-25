def f1_score(y_true, y_pred):
    tp = (y_true * y_pred).sum().to(torch.float)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1