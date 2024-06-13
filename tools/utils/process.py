import torch
import numpy as np

def eval_one_instance(gait_instance, model, device='cpu', return_prob=False):
    model.eval()
    with torch.no_grad():
        signals = []
        for signal in gait_instance.generate_all_signal_segments_without_answer():
            signals.append(signal)
    
        signals = torch.FloatTensor(np.array(signals)).to(device)
        logits = model(signals)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

    if not return_prob:
        return preds
    return preds, probs


def evaluate(epoch, eval_dataset, model, device, prefix='', writer=None, return_prob=False):
    if return_prob:
        predss = []
        probss = []
        for instance in eval_dataset:
            preds, probs = eval_one_instance(instance, model, device, return_prob=return_prob)
            predss.append(preds)
            probss.append(probs)
        return predss, probss
    
    else:
        predss = []
        for instance in eval_dataset:
            preds = eval_one_instance(instance, model, device)
            predss.append(preds)
        return predss
