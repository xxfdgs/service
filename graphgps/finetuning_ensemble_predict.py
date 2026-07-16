import logging
import os
import os.path as osp

import torch
from torch_geometric.graphgym.config import set_cfg
from yacs.config import CfgNode



def get_final_pretrained_ckpt(ckpt_dir):
    if osp.exists(ckpt_dir):
        names = os.listdir(ckpt_dir)
        epochs = [int(name.split('.')[0]) for name in names]
        final_epoch = max(epochs)
    else:
        raise FileNotFoundError(f"Pretrained model dir not found: {ckpt_dir}")
    return osp.join(ckpt_dir, f'{final_epoch}.ckpt')


def compare_cfg(cfg_main, cfg_secondary, field_name, strict=False):
    main_val, secondary_val = cfg_main, cfg_secondary
    for f in field_name.split('.'):
        main_val = main_val[f]
        secondary_val = secondary_val[f]
    if main_val != secondary_val:
        if strict:
            raise ValueError(f"Main and pretrained configs must match on "
                             f"'{field_name}'")
        else:
            logging.warning(f"Pretrained models '{field_name}' differs, "
                            f"using: {main_val}")


def set_new_cfg_allowed(config, is_new_allowed):
    """ Set YACS config (and recursively its subconfigs) to allow merging
        new keys from other configs.
    """
    config.__dict__[CfgNode.NEW_ALLOWED] = is_new_allowed
    # Recursively set new_allowed state
    for v in config.__dict__.values():
        if isinstance(v, CfgNode):
            set_new_cfg_allowed(v, is_new_allowed)
    for v in config.values():
        if isinstance(v, CfgNode):
            set_new_cfg_allowed(v, is_new_allowed)


def load_pretrained_model_cfg(cfg):
    pretrained_cfg_fname = osp.join(cfg.pretrained.dir, 'config.yaml')
    if not os.path.isfile(pretrained_cfg_fname):
        FileNotFoundError(f"Pretrained model config not found: "
                          f"{pretrained_cfg_fname}")

    logging.info(f"[*] Updating cfg from pretrained model: "
                 f"{pretrained_cfg_fname}")

    pretrained_cfg = CfgNode()
    set_cfg(pretrained_cfg)
    set_new_cfg_allowed(pretrained_cfg, True)
    pretrained_cfg.merge_from_file(pretrained_cfg_fname)

    assert cfg.model.type == 'GPSModel', \
        "Fine-tuning regime is untested for other model types."
    compare_cfg(cfg, pretrained_cfg, 'model.type', strict=True)
    compare_cfg(cfg, pretrained_cfg, 'model.graph_pooling')
    compare_cfg(cfg, pretrained_cfg, 'model.edge_decoding')
    compare_cfg(cfg, pretrained_cfg, 'dataset.node_encoder', strict=True)
    compare_cfg(cfg, pretrained_cfg, 'dataset.node_encoder_name', strict=True)
    compare_cfg(cfg, pretrained_cfg, 'dataset.node_encoder_bn', strict=True)
    compare_cfg(cfg, pretrained_cfg, 'dataset.edge_encoder', strict=True)
    compare_cfg(cfg, pretrained_cfg, 'dataset.edge_encoder_name', strict=True)
    compare_cfg(cfg, pretrained_cfg, 'dataset.edge_encoder_bn', strict=True)

    # Copy over all PE/SE configs
    for key in cfg.keys():
        if key.startswith('posenc_'):
            cfg[key] = pretrained_cfg[key]

    # Copy over GT config
    cfg.gt = pretrained_cfg.gt

    # Copy over GNN cfg but not those for the prediction head
    compare_cfg(cfg, pretrained_cfg, 'gnn.head')
    compare_cfg(cfg, pretrained_cfg, 'gnn.layers_post_mp')
    compare_cfg(cfg, pretrained_cfg, 'gnn.act', strict=True)
    compare_cfg(cfg, pretrained_cfg, 'gnn.dropout')
    head = cfg.gnn.head
    post_mp = cfg.gnn.layers_post_mp
    act = cfg.gnn.act
    drp = cfg.gnn.dropout
    cfg.gnn = pretrained_cfg.gnn
    cfg.gnn.head = head
    cfg.gnn.layers_post_mp = post_mp
    cfg.gnn.act = act
    cfg.gnn.dropout = drp
    return cfg

def delete_post(pretrained_dict):
    del_keys = []
    for k, v in pretrained_dict.items():
        # print('k',k)
        if k.startswith('model.post_mp'):
            # if k.startswith('model.post_mp') or k.startswith('model.layers.9'):
            del_keys.append(k)
    print('pretrained_dict', len(pretrained_dict))
    for item in del_keys:
        del pretrained_dict[item]
    print('pretrained_dict', len(pretrained_dict))
    logging.info(f"[*] delete_post: running")
    return pretrained_dict

def init_model_from_pretrained(model, pretrained_dir, ave_pretrained_model,
                               freeze_main=False, reset_prediction_head=True):
    """ Copy model parameters from pretrained model except the prediction head.

    Args:
        model: Initialized model with random weights.
        pretrained_dir: Root directory of saved pretrained model.
        freeze_main: If True, do not finetune the loaded pretrained parameters
            of the `main body` (train the prediction head only), else train all.
        reset_prediction_head: If True, reset parameters of the prediction head,
            else keep the pretrained weights.

    Returns:
        Updated pytorch model object.
    """
    from torch_geometric.graphgym.checkpoint import MODEL_STATE
    ave_pretrained_model = 10
    if ave_pretrained_model <=0 :
        print('ave_pretrained_model less than 0 ,ave_pretrained_model must large than 0,such as 1, 5 ,10')
        logging.info(f"[*] ave_pretrained_model less than 0 ,ave_pretrained_model must large than 0 ")
    #### ckpt_file，ckpt， pretrained_dict ，model_dict ，pretrained_dict

    # for serial in range(ave_pretrained_model):
    if ave_pretrained_model == 1:
        print('')
        ckpt_file = get_final_pretrained_ckpt(osp.join(pretrained_dir, '0', 'ckpt'))
        logging.info(f"[*] Loading from pretrained model: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location={'cuda:1': 'cuda'})
        # ckpt = torch.load(ckpt_file)
        pretrained_dict = ckpt[MODEL_STATE]
        model_dict = model.state_dict()
        if reset_prediction_head:
            # Filter out prediction head parameter keys.
            logging.info(f"[*] reset_prediction_head: {reset_prediction_head}")
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if not k.startswith('post_mp')}
        pretrained_dict = delete_post(pretrained_dict)
        logging.info(f"[*] pretrained_dict update model parameter")
        model_dict.update(pretrained_dict)
        ####Load the new state dict.
        model.load_state_dict(model_dict)
        logging.info(f"[*] model.load_state_dict(model_dict)")
    elif ave_pretrained_model == 5:
        ckpt_file = get_final_pretrained_ckpt(osp.join(pretrained_dir, '0', 'ckpt'))
        ckpt_file1 = get_final_pretrained_ckpt(osp.join(pretrained_dir, '1', 'ckpt'))
        ckpt_file2 = get_final_pretrained_ckpt(osp.join(pretrained_dir, '2', 'ckpt'))
        ckpt_file3 = get_final_pretrained_ckpt(osp.join(pretrained_dir, '3', 'ckpt'))
        ckpt_file4 = get_final_pretrained_ckpt(osp.join(pretrained_dir, '4', 'ckpt'))
        logging.info(f"[*] Loading from pretrained model: {ckpt_file4}")
        # ckpt = torch.load(ckpt_file)
        # ckpt1 = torch.load(ckpt_file1)
        # ckpt2 = torch.load(ckpt_file2)
        # ckpt3 = torch.load(ckpt_file3)
        # ckpt4 = torch.load(ckpt_file4)
        ckpt = torch.load(ckpt_file, map_location={'cuda:1': 'cuda'})
        ckpt1 = torch.load(ckpt_file1, map_location={'cuda:1': 'cuda'})
        ckpt2 = torch.load(ckpt_file2, map_location={'cuda:1': 'cuda'})
        ckpt3 = torch.load(ckpt_file3, map_location={'cuda:1': 'cuda'})
        ckpt4 = torch.load(ckpt_file4, map_location={'cuda:1': 'cuda'})
        pretrained_dict = ckpt[MODEL_STATE]
        pretrained_dict1 = ckpt1[MODEL_STATE]
        pretrained_dict2 = ckpt2[MODEL_STATE]
        pretrained_dict3 = ckpt3[MODEL_STATE]
        pretrained_dict4 = ckpt4[MODEL_STATE]
        model_dict = model.state_dict()
        model_dict1 = model.state_dict()
        model_dict2 = model.state_dict()
        model_dict3 = model.state_dict()
        model_dict4 = model.state_dict()
        if reset_prediction_head:
            # Filter out prediction head parameter keys.
            logging.info(f"[*] reset_prediction_head: {reset_prediction_head}")
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if not k.startswith('post_mp')}
        pretrained_dict = delete_post(pretrained_dict)
        pretrained_dict1 = delete_post(pretrained_dict1)
        pretrained_dict2 = delete_post(pretrained_dict2)
        pretrained_dict3 = delete_post(pretrained_dict3)
        pretrained_dict4 = delete_post(pretrained_dict4)
        logging.info(f"[*] pretrained_dict update model parameter")
        model_dict.update(pretrained_dict)
        model_dict1.update(pretrained_dict1)
        model_dict2.update(pretrained_dict2)
        model_dict3.update(pretrained_dict3)
        model_dict4.update(pretrained_dict4)
        ## merge model parameter
        print('--merge--')
        logging.info(f"[*] merge model parameter")
        dict_list = [model_dict, model_dict1, model_dict2, model_dict3, model_dict4]
        for j, item_state_dict in enumerate(dict_list):
            if j == 0:
                uniform_soup = {k: v * (1. / int(5)) for k, v in item_state_dict.items()}
            else:
                uniform_soup = {k: v * (1. / int(5)) + uniform_soup[k] for k, v in item_state_dict.items()}
        # Load the merge state dict
        model.load_state_dict(uniform_soup)
        logging.info(f"[*] model.load_state_dict(uniform_soup)")
    elif ave_pretrained_model == 10:
        ckpt_file = get_final_pretrained_ckpt(osp.join(pretrained_dir, '0', 'ckpt'))
        ckpt_file1 = get_final_pretrained_ckpt(osp.join(pretrained_dir, '1', 'ckpt'))
        ckpt_file2 = get_final_pretrained_ckpt(osp.join(pretrained_dir, '2', 'ckpt'))
        ckpt_file3 = get_final_pretrained_ckpt(osp.join(pretrained_dir, '3', 'ckpt'))
        ckpt_file4 = get_final_pretrained_ckpt(osp.join(pretrained_dir, '4', 'ckpt'))
        ckpt_file5 = get_final_pretrained_ckpt(osp.join(pretrained_dir, '5', 'ckpt'))
        ckpt_file6 = get_final_pretrained_ckpt(osp.join(pretrained_dir, '6', 'ckpt'))
        ckpt_file7 = get_final_pretrained_ckpt(osp.join(pretrained_dir, '7', 'ckpt'))
        ckpt_file8 = get_final_pretrained_ckpt(osp.join(pretrained_dir, '8', 'ckpt'))
        ckpt_file9 = get_final_pretrained_ckpt(osp.join(pretrained_dir, '9', 'ckpt'))
        logging.info(f"[*] Loading from pretrained model: {ckpt_file9}")
        # ckpt = torch.load(ckpt_file, map_location={'cuda:1': 'cuda:0'})

        ckpt = torch.load(ckpt_file, map_location={'cuda:1': 'cuda'})
        ckpt1 = torch.load(ckpt_file1, map_location={'cuda:1': 'cuda'})
        ckpt2 = torch.load(ckpt_file2, map_location={'cuda:1': 'cuda'})
        ckpt3 = torch.load(ckpt_file3, map_location={'cuda:4': 'cuda'})
        ckpt4 = torch.load(ckpt_file4, map_location={'cuda:1': 'cuda'})
        ckpt5 = torch.load(ckpt_file5, map_location={'cuda:2': 'cuda'})
        ckpt6 = torch.load(ckpt_file6, map_location={'cuda:2': 'cuda'})
        ckpt7 = torch.load(ckpt_file7, map_location={'cuda:2': 'cuda'})
        ckpt8 = torch.load(ckpt_file8, map_location={'cuda:2': 'cuda'})
        ckpt9 = torch.load(ckpt_file9, map_location={'cuda:2': 'cuda'})


        pretrained_dict = ckpt[MODEL_STATE]
        pretrained_dict1 = ckpt1[MODEL_STATE]
        pretrained_dict2 = ckpt2[MODEL_STATE]
        pretrained_dict3 = ckpt3[MODEL_STATE]
        pretrained_dict4 = ckpt4[MODEL_STATE]
        pretrained_dict5 = ckpt5[MODEL_STATE]
        pretrained_dict6 = ckpt6[MODEL_STATE]
        pretrained_dict7 = ckpt7[MODEL_STATE]
        pretrained_dict8 = ckpt8[MODEL_STATE]
        pretrained_dict9 = ckpt9[MODEL_STATE]


        model_dict = model.state_dict()
        model_dict1 = model.state_dict()
        model_dict2 = model.state_dict()
        model_dict3 = model.state_dict()
        model_dict4 = model.state_dict()
        model_dict5 = model.state_dict()
        model_dict6 = model.state_dict()
        model_dict7 = model.state_dict()
        model_dict8 = model.state_dict()
        model_dict9 = model.state_dict()

        # print('>>>> pretrained dict: ')
        # print(pretrained_dict.keys())
        # print('>>>> model dict: ')
        # print(model_dict.keys())

        if reset_prediction_head:
            # Filter out prediction head parameter keys.
            logging.info(f"[*] reset_prediction_head: {reset_prediction_head}")
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if not k.startswith('post_mp')}
        pretrained_dict = delete_post(pretrained_dict)
        pretrained_dict1 = delete_post(pretrained_dict1)
        pretrained_dict2 = delete_post(pretrained_dict2)
        pretrained_dict3 = delete_post(pretrained_dict3)
        pretrained_dict4 = delete_post(pretrained_dict4)
        pretrained_dict5 = delete_post(pretrained_dict5)
        pretrained_dict6 = delete_post(pretrained_dict6)
        pretrained_dict7 = delete_post(pretrained_dict7)
        pretrained_dict8 = delete_post(pretrained_dict8)
        pretrained_dict9 = delete_post(pretrained_dict9)
        ######

        # Overwrite entries in the existing state dict.
        logging.info(f"[*] pretrained_dict update model parameter")
        model_dict.update(pretrained_dict)
        model_dict1.update(pretrained_dict1)
        model_dict2.update(pretrained_dict2)
        model_dict3.update(pretrained_dict3)
        model_dict4.update(pretrained_dict4)
        model_dict5.update(pretrained_dict5)
        model_dict6.update(pretrained_dict6)
        model_dict7.update(pretrained_dict7)
        model_dict8.update(pretrained_dict8)
        model_dict9.update(pretrained_dict9)

        ## merge model parameter
        print('--merge--')
        logging.info(f"[*] merge model parameter")
        # dict_list = [model_dict7, model_dict8, model_dict9]
        # dict_list = [model_dict, model_dict1, model_dict2, model_dict3, model_dict4, model_dict5, model_dict6]
        dict_list = [model_dict,model_dict1,model_dict2,model_dict3,model_dict4,model_dict5,model_dict6,model_dict7,model_dict8,model_dict9]
        for j,item_state_dict in enumerate(dict_list):
            print(j)
            if j == 0:
                uniform_soup = {k: v * (1. / int(10)) for k, v in item_state_dict.items()}
            else:
                uniform_soup = {k: v * (1. / int(10)) + uniform_soup[k] for k, v in item_state_dict.items()}
        # Load the merge state dict
        model.load_state_dict(uniform_soup)
        logging.info(f"[*] model.load_state_dict(uniform_soup)")

    if freeze_main:
        logging.info(f"[*] freeze_main: "
                     f"{freeze_main}")
        print('freeze_main =',freeze_main)
        # for key, param in model.named_parameters():
            #### original version
            # if not key.startswith('post_mp'):
            #     param.requires_grad = False
            #     print(key)
            #### lrx alter version
            # if not key.startswith('model.post_mp') :
            # # if not key.startswith('model.post_mp.FC_layers.3'):
            # # if not key.startswith('model.post_mp') and not key.startswith('model.layers.9'):
            # # if not key.startswith('model.post_mp.FC_layers.2') and not key.startswith('model.post_mp.FC_layers.3'):
            # # if not key.startswith('model.post_mp.FC_layers.0') and not key.startswith('model.post_mp.FC_layers.1'):
            #     param.requires_grad = False
            #     print(key)
            # if param.requires_grad:
            #     print(key)
    return model
