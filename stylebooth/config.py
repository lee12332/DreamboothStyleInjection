import argparse

def get_args():
    
    parse = argparse.ArgumentParser()
    
    # hyperparameter of StyleID
    parse.add_argument('--T', type=int, default=1.5)
    parse.add_argument('--gamma', type=float, default=0.7)
    parse.add_argument('--without_init_adain', action='store_true')
    parse.add_argument('--without_attn_injection', action='store_true')
    parse.add_argument('--layers', nargs='+', type=int, default=[7, 8, 9, 10, 11])
    
    # hyperparameters of diffusion model
    parse.add_argument('--ddim_steps', type=int, default=50)
    parse.add_argument('--sd_version', type=float, choices=[1.4, 1.5, 2.1], default=2.1)
    
    # path of content and style iamges
    # parse.add_argument('--cnt_fn', type=str,default='./output')
    parse.add_argument('--sty_fn', type=str, default='/root/autodl-tmp/B')
    parse.add_argument('--save_dir', type=str, default='./result')
    parse.add_argument('--style_item', type=int, default=0)
    
    cfg = parse.parse_args()
    return cfg
