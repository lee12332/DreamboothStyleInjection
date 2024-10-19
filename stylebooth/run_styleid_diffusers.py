import jittor as jt
import numpy as np, copy, os, sys
import matplotlib.pyplot as plt

from utils import * # image save utils

from stable_diffusion import * # load SD
import copy

# For visualizing features
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import cv2
from tqdm import tqdm

from config import get_args

# class for obtain and override the features
class style_transfer_module():
           
    def __init__(self,
        unet, vae, text_encoder, tokenizer, scheduler, style_transfer_params = None,
    ):  
        
        style_transfer_params_default = {
            'gamma': 0.75,
            'tau': 1.5,
            'injection_layers': [7, 8, 9, 10, 11]
        }
        if style_transfer_params is not None:
            style_transfer_params_default.update(style_transfer_params)
        self.style_transfer_params = style_transfer_params_default
        
        self.unet = unet # SD unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler

        self.attn_features = {} # where to save key value (attention block feature)
        self.attn_features_modify = {} # where to save key value to modify (attention block feature)

        self.cur_t = None
        
        # Get residual and attention block in decoder
        # [0 ~ 11], total 12 layers
        resnet, attn = get_unet_layers(unet)
        
        # where to inject key and value
        qkv_injection_layer_num = self.style_transfer_params['injection_layers']
    
        
        for i in qkv_injection_layer_num:
            self.attn_features["layer{}_attn".format(i)] = {}
            attn[i].transformer_blocks[0].attn1.register_forward_hook(self.__get_query_key_value("layer{}_attn".format(i)))
        # Modify hook (if you change query key value)
        # for i in qkv_injection_layer_num:
        #     attn[i].transformer_blocks[0].attn1.register_forward_hook(self.__modify_self_attn_qkv("layer{}_attn".format(i)))
        
       
        # triggers for obtaining or modifying features
        
        self.trigger_get_qkv = False # if set True --> save attn qkv in self.attn_features
        self.trigger_modify_qkv = False # if set True --> save attn qkv by self.attn_features_modify
        
        self.modify_num = None # ignore
        self.modify_num_sa = None # ignore
        
    def get_text_condition(self, text):
        if text is None:
            uncond_input = tokenizer(
                [""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
            )
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0].to(device)
            return {'encoder_hidden_states': uncond_embeddings}
        
        text_embeddings, uncond_embeddings = get_text_embedding(text, self.text_encoder, self.tokenizer)
        text_cond = [text_embeddings, uncond_embeddings]
        denoise_kwargs = {
            'encoder_hidden_states': jt.concat(text_cond)
        }
        return denoise_kwargs
    
    def reverse_process(self, input, denoise_kwargs):
        pred_images = []
        pred_latents = []
        
        decode_kwargs = {'vae': vae}
        # Reverse diffusion process
        for t in tqdm(self.scheduler.timesteps):
            
            # setting t (for saving time step)
            self.cur_t = t.item()
            
            with jt.no_grad():
                
                # For text condition on stable diffusion
                if 'encoder_hidden_states' in denoise_kwargs.keys():
                    bs = denoise_kwargs['encoder_hidden_states'].shape[0]
                    input = jt.concat([input] * bs)
                
                noisy_residual = unet_wrapper.unet(input, t.to(input.device), **denoise_kwargs).sample
                    
                # For text condition on stable diffusion
                if noisy_residual.shape[0] == 2:
                    # perform guidance
                    noise_pred_text, noise_pred_uncond = noisy_residual.split(2, dim=0)
                    noisy_residual = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    input, _ = input.split(2, dim=0)

                prev_noisy_sample = self.scheduler.step(noisy_residual, t, input).prev_sample                # coef * P_t(e_t(x_t)) + D_t(e_t(x_t))
                pred_original_sample = self.scheduler.step(noisy_residual, t, input).pred_original_sample    # D_t(e_t(x_t))
                
                input = prev_noisy_sample
                
                pred_latents.append(pred_original_sample)
                pred_images.append(decode_latent(pred_original_sample, **decode_kwargs))
                
        return pred_images, pred_latents
        
            
    ## Inversion (https://github.com/huggingface/diffusion-models-class/blob/main/unit4/01_ddim_inversion.ipynb)
    def invert_process(self, input, denoise_kwargs):

        pred_images = []
        pred_latents = []
        # latents = input.clone()
        decode_kwargs = {'vae': vae}

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        # timesteps = reversed(self.scheduler.timesteps)
        timesteps = self.scheduler.timesteps.numpy().tolist()[::-1]
        num_inference_steps = len(timesteps)
        # num_inference_steps = len(self.scheduler.timesteps)

        with jt.no_grad():
            for i in tqdm(range(0, num_inference_steps)):

                # t = timesteps[i]
                t = jt.Var(timesteps[i])
                
                self.cur_t = t.item()
                
                # For text condition on stable diffusion
                if 'encoder_hidden_states' in denoise_kwargs.keys():
                    bs = denoise_kwargs['encoder_hidden_states'].shape[0]
                    input = jt.concat([input] * bs)

                # Predict the noise residual
                noisy_residual = self.unet(input, t.to(input.device), **denoise_kwargs).sample

                noise_pred = noisy_residual

                # For text condition on stable diffusion
                if noisy_residual.shape[0] == 2:
                    # perform guidance
                    # noise_pred_text, noise_pred_uncond = noisy_residual.chunk(2)
                    # noisy_residual = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    # input, _ = input.chunk(2)
                    noise_pred_text, noise_pred_uncond = noisy_residual.split(2, dim=0)
                    noisy_residual = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    input, _ = input.split(2, dim=0)

                current_t = max(0, t.item() - (1000//num_inference_steps)) #t
                next_t = t # min(999, t.item() + (1000//num_inference_steps)) # t+1
                alpha_t = self.scheduler.alphas_cumprod[current_t]
                alpha_t_next = self.scheduler.alphas_cumprod[next_t]

                # latents = input
                # latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred

                latents = input
                # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
                # latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred
                beta_t = 1 - alpha_t
                pred_original_sample = (alpha_t**0.5) * latents - (beta_t**0.5) * noise_pred
                pred_epsilon = (alpha_t**0.5) * noise_pred + (beta_t**0.5) * latents
                pred_sample_direction = (1 - alpha_t_next) ** (0.5) * pred_epsilon
                latents = alpha_t_next ** (0.5) * pred_original_sample + pred_sample_direction

                input = latents
                
                pred_latents.append(latents)
                pred_images.append(decode_latent(latents, **decode_kwargs))
                
        return pred_images, pred_latents
        
    # ============================ hook operations ===============================
    
    # save key value in self.original_kv[name]
    def __get_query_key_value(self, name):
        def hook(model, input, output,input_kwargs):
            # print("============hoooook1====")
            if self.trigger_get_qkv:
                # print("=========hook====")
                _, query, key, value, _ = attention_op(model, input[0])
                # print('===attn_features1===')
                # print(self.attn_features)
                self.attn_features[name][int(self.cur_t)] = (query.detach(), key.detach(), value.detach())
            if self.trigger_modify_qkv:
                # print("=========hook2====")
                module_name = name # TODO
                
                _, q_cs, k_cs, v_cs, _ = attention_op(model, input[0])
                
                q_c, k_s, v_s = self.attn_features_modify[name][int(self.cur_t)]
                
                # style injection
                q_hat_cs = q_c * self.style_transfer_params['gamma'] + q_cs * (1 - self.style_transfer_params['gamma'])
                k_cs, v_cs = k_s, v_s
                
                # Replace query key and value
                _, _, _, _, modified_output = attention_op(model, input[0], key=k_cs, value=v_cs, query=q_hat_cs, temperature=self.style_transfer_params['tau'])
                # print(output)
                # print(modified_output.shape)
                # jt.assign(output, modified_output)
                output.data = modified_output.data
                # return modified_output
        return hook

    
    
if __name__ == "__main__":

    cfg = get_args()
    
    # options
    ddim_steps = cfg.ddim_steps
    device = "cuda"
    dtype = jt.float16
    in_c = 4
    guidance_scale = 0. # no text
    
    style_text = None
    content_text = None
    
    style_transfer_params = {
        'gamma': cfg.gamma,
        'tau': cfg.T,
        'injection_layers': cfg.layers,
    }
    
    # Get SD modules
    vae, tokenizer, text_encoder, unet, scheduler = load_stable_diffusion(sd_version=str(cfg.sd_version), precision_t=dtype)
    scheduler.set_timesteps(ddim_steps)
    sample_size = unet.config.sample_size
    

    # Init style transfer module
    unet_wrapper = style_transfer_module(unet, vae, text_encoder, tokenizer, scheduler, style_transfer_params=style_transfer_params)
    
    
    # Get style image tokens
    denoise_kwargs = unet_wrapper.get_text_condition(style_text)


    for i in range(10,11):
        print("第"+str(i)+"风格==============")
        sty_path = cfg.sty_fn + "/{:0>2d}/images/".format(i)
        files = [os.path.join(sty_path, f) for f in os.listdir(sty_path) if os.path.isfile(os.path.join(sty_path, f))]

        cnt_path =  cfg.cnt_fn + "/{:0>2d}/".format(i)
        sty_tag = 0
        for id,cnt_img_path in enumerate(os.listdir(cnt_path)):
            print("第"+str(id)+"内容=======")
            print(cnt_img_path)
            if id/8 == sty_tag:
                sty_tag = sty_tag+1
                print("=========用了第",sty_tag)
                style_image = cv2.imread(files[(sty_tag-1)%3])[:, :, ::-1]
                style_image = cv2.resize(style_image,(512,512))

                # print(style_image)
            
                unet_wrapper.trigger_get_qkv = True # get attention features (key, value)
                unet_wrapper.trigger_modify_qkv = False
                
                style_latent = encode_latent(normalize(style_image).to(device=vae.device, dtype=dtype), vae)
                
                # invert process
                print("Invert style image...")
                images, latents = unet_wrapper.invert_process(style_latent, denoise_kwargs=denoise_kwargs) # reverse process save activations such as attn, res
                style_latent = latents[-1]
            
                # ================= IMPORTANT =================
                # save key value from style image
                style_features = copy.deepcopy(unet_wrapper.attn_features)
                # =============================================
            content_image = cv2.imread(os.path.join(cnt_path,cnt_img_path))[:, :, ::-1]
            content_image = cv2.resize(content_image,(512,512))
            # Get content image tokens
            denoise_kwargs = unet_wrapper.get_text_condition(content_text)
            
            unet_wrapper.trigger_get_qkv = True
            unet_wrapper.trigger_modify_qkv = False
            
            content_latent = encode_latent(normalize(content_image).to(device=vae.device, dtype=dtype), vae)

            # invert process
            print("Invert content image...")
            images, latents = unet_wrapper.invert_process(content_latent, denoise_kwargs=denoise_kwargs) # reverse process save activations such as attn, res
            content_latent = latents[-1]
    
            # ================= IMPORTANT =================
            # save res feature from content image
            content_features = copy.deepcopy(unet_wrapper.attn_features)
            # =============================================
    
            # ================= IMPORTANT =================
            # Set modify features
            for layer_name in style_features.keys():
                unet_wrapper.attn_features_modify[layer_name] = {}
                for t in scheduler.timesteps:
                    t = t.item()
                    unet_wrapper.attn_features_modify[layer_name][t] = (content_features[layer_name][t][0], style_features[layer_name][t][1], style_features[layer_name][t][2]) # content as q / style as kv        
            # =============================================
            
            unet_wrapper.trigger_get_qkv = False
            unet_wrapper.trigger_modify_qkv = not cfg.without_attn_injection # modify attn feature (key value)
            
            # Generate style transferred image
            denoise_kwargs = unet_wrapper.get_text_condition(content_text)
            
            if cfg.without_init_adain:
                latent_cs = content_latent
            else:
                # latent_cs = (content_latent - content_latent.mean(dim=(2, 3), keepdim=True)) / (content_latent.std(dim=(2, 3), keepdim=True) + 1e-4) * style_latent.std(dim=(2, 3), keepdim=True) + style_latent.mean(dim=(2, 3), keepdim=True)
                mean_content = content_latent.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
                # 计算 content_latent 在维度 2 和 3 上的标准差
                std_content = cal_std(content_latent)
                # 计算 style_latent 在维度 2 和 3 上的标准差
                std_style = cal_std(style_latent)
                # 计算 style_latent 在维度 2 和 3 上的平均值
                mean_style = style_latent.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
                # 计算 latent_cs
                latent_cs = (content_latent - mean_content) / (std_content + 1e-6) * std_style + mean_style


            # reverse process
            print("Style transfer...")
            images, latents = unet_wrapper.reverse_process(latent_cs, denoise_kwargs=denoise_kwargs) # reverse process save activations such as attn, res
            
            # save image
            images = [denormalize(input)[0] for input in images]
            image_last = images[-1]

            save_dir = cfg.save_dir+"/{:0>2d}".format(i)
            
            os.makedirs(save_dir, exist_ok=True)
            # os.makedirs(save_dir + '/intermediate', exist_ok=True)
            save_image(image_last, os.path.join(save_dir, cnt_img_path))
