import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def pruning_model(model, px):

    print('start unstructured pruning for all conv layers')
    parameters_to_prune =[]
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def check_sparsity(model, report=False):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))     
    if report:
        print('report remain weight = ', 100*(1-zero_sum/sum_list),'%')
    else:
        print('remain weight = ', 100*(1-zero_sum/sum_list),'%')
    return 100*(1-zero_sum/sum_list)

def remove_prune(model):
    print('remove pruning')
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prune.remove(m,'weight')

def extract_mask(model_dict):
    new_dict = {}

    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = model_dict[key]

    return new_dict

def extract_main_weight(model_dict, fc=True):
    new_dict = {}

    for key in model_dict.keys():
        if not 'mask' in key:
            new_dict[key] = model_dict[key]

    if not fc:
        print('delete fc weight')
        del new_dict['fc.weight']
        del new_dict['fc.bias']
    
    return new_dict

def prune_model_custom(model, mask_dict):

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print('pruning layer with custom mask:', name)
            prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])

def extract_weight_with_single_bn(model_dict, idx):
    