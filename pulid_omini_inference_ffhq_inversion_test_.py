import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from transformer_flux_ca import FluxTransformer2DModelCA
from diffusers import FluxTransformer2DModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from pipeline import FireFlowEditFluxPipeline, DNAEditFluxPipeline, RFInversionEditFluxPipeline
import torch
import diffusers
diffusers.utils.logging.set_verbosity_error()


guidance_scale=1.0
image_guidance_scale = 1.0
id_guidance_scale = 1.0
use_gaze = False
# inverse_steps = 14
# no grad
torch.set_grad_enabled(False)


weight_dtype = torch.bfloat16
model_id = 'black-forest-labs/FLUX.1-Krea-dev'
# model_id = 'black-forest-labs/FLUX.1-dev'
device = 'cuda'
# transformer = FluxTransformer2DModelCA.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=weight_dtype, subfolder='transformer', low_cpu_mem_usage=False, device_map=None)
# transformer_dev = FluxTransformer2DModel.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=weight_dtype, subfolder='transformer')
transformer = FluxTransformer2DModelCA.from_pretrained(model_id, torch_dtype=weight_dtype, 
subfolder='transformer', low_cpu_mem_usage=False, device_map=None, use_gaze=use_gaze)

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=weight_dtype).to(device)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=weight_dtype).to(device)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder_2 = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=weight_dtype).to(device)
tokenizer_2 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_2")



from pulid.utils import resize_numpy_image_long
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack, prepare_img, prepare_txt
from diffusers import FluxPipeline
from pipeline_flux_ca import FluxPipelineCA
from omini.pipeline.flux_omini import generate_ca, Condition, convert_to_condition, generate_ca_inv


device = 'cuda'
flux = FluxPipeline.from_pretrained(model_id, 
                                    transformer=transformer, 
                                    vae=vae,
                                    text_encoder=text_encoder,
                                    tokenizer=tokenizer,
                                    text_encoder_2=text_encoder_2,
                                    tokenizer_2=tokenizer_2,
                                    torch_dtype=weight_dtype).to(device)
invertpipe = RFInversionEditFluxPipeline.from_pretrained(model_id,
                                    transformer=transformer, 
                                    vae=vae,
                                    text_encoder=text_encoder,
                                    tokenizer=tokenizer,
                                    text_encoder_2=text_encoder_2,
                                    tokenizer_2=tokenizer_2,
                                    torch_dtype=weight_dtype,
                                    ).to(device)


# ckpt = 43000
# lora_file_path = f'/mnt/data2/jiwon/OminiControl/runs/faceswap_scratch_lora64_20251004-005548/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/pulid_omini_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}/ffhq_eval'

# ckpt = 60000
# lora_file_path = f'/mnt/data2/jiwon/OminiControl/runs/faceswap_vgg_lora64Pretrained_20251010-115546/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/pulid_omini_vgg_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval'

# ckpt = 8000
# lora_file_path = f'/mnt/data2/jiwon/OminiControl/runs/faceswap_vgg_lora64Pretrained_idLoss_20251014-014645/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/pulid_omini_vgg_idLoss_t<=0.33_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval'

# ckpt = 8000
# lora_file_path = f'/mnt/data2/jiwon/OminiControl/runs/faceswap_vgg_lora64Pretrained_idLoss_t<=0.5_20251014-021510/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/pulid_omini_vgg_idLoss_t<=0.5_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval'

ckpt = 80000
lora_file_path = f'/mnt/data2/jiwon/OminiControl/runs/faceswap_vgg_lora64Pretrained_idLoss_irse50_t<=0.5_20251014-162532/ckpt/{ckpt}/default.safetensors'
base_output_dir = f'./inversion_exp/faceswap_vgg_lora64Pretrained_idLoss_irse50_t<=0.5_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval'
os.makedirs(base_output_dir, exist_ok=True)

# ckpt = 8000
# lora_file_path = f'/mnt/data2/jiwon/OminiControl/runs/faceswap_vgg_lora64Pretrained_idLoss_irse50_t<=0.5_ckpt60000_gaze_20251018-024629/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/faceswap_vgg_lora64Pretrained_idLoss_irse50_t<=0.5_ckpt60000_gaze_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval'

adapter_name = 'default'
print(f"Loading LoRA for adapter '{adapter_name}' from {lora_file_path}")
flux.load_lora_weights(lora_file_path, adapter_name=adapter_name)
if use_gaze:
    flux.transformer.gaze_ca.load_state_dict(torch.load(
        os.path.join(os.path.dirname(lora_file_path), 'gaze_ca.pth')    
    ))
    print(f"[INFO] Loaded gaze CA weights from {os.path.join(os.path.dirname(lora_file_path), 'gaze_ca.pth')}")

flux.transformer.components_to_device(device, weight_dtype)



import cv2
from glob import glob
from tqdm import tqdm
from PIL import Image

# output_dir = './results_pulid_omini/vgg_src/clsUncond_hiddenUncond'
# os.makedirs(output_dir, exist_ok=True)

src_img_path_list = sorted(glob('/mnt/data2/dataset/ffhq_eval/src/*.jpg'))[223:]
trg_img_path_base = '/mnt/data2/dataset/ffhq_eval/trg'

# for id_guidance_scale in [1.0]:
# for id_guidance_scale in [1.5]:
# for id_guidance_scale in [1.0, 1.5, 2.0]:
# for id_guidance_scale in [1.0, 1.5, 2.0]:
for id_guidance_scale in [1.0]:
    for src_img_path in tqdm(src_img_path_list, desc='Processing Image'):
        for inverse_steps in [14]:
            output_dir = os.path.join(base_output_dir, f"inv{inverse_steps}")
            os.makedirs(output_dir, exist_ok=True)
            
            prompt="a photo of human face",
            neg_prompt = ""
            true_cfg = 1.0
            use_true_cfg = True if true_cfg > 1.0 else False
        
            src_num = os.path.basename(src_img_path).split('.')[0]
            trg_img_path = os.path.join(trg_img_path_base, f"{src_num}.jpg")
            trg_img = Image.open(trg_img_path).convert('RGB').resize((512,512))
            trg_img.save(f"{output_dir}/{src_num}_trg.png")

            cond_img_path = os.path.join (trg_img_path_base, 'condition_blended_image_blurdownsample8_segGlass_landmark', f"{src_num}.png")
            condition_img = Image.open(cond_img_path).convert('RGB')
            
            if use_gaze:
                import numpy as np
                gaze_path = os.path.join (trg_img_path_base, 'gaze', f"{src_num}.npy")
                gaze_embed = torch.from_numpy(  np.load(gaze_path) ).unsqueeze(0).to(device, dtype=weight_dtype) # (1, gaze_dim)
            else:
                gaze_embed = None
            
            # Resize
            condition_img = condition_img.resize((512,512))
            condition_type = 'deblurring'
            position_delta = [0,0]
            position_scale = 1.0
            condition = Condition(condition_img, 'default', position_delta, position_scale)

            # ID embed (Src)
            id_image = cv2.imread(src_img_path)
            print(f"id_image shape: {id_image.shape}")
            id_image = cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB)
            id_image = resize_numpy_image_long(id_image, 1024)
            id_image_pil = Image.fromarray(id_image)
            id_image_pil = id_image_pil.resize((512,512))
            # id_image_pil.save(f"{output_dir}/{src_num}_id.png")

            id_embeddings, uncond_id_embeddings = flux.transformer.get_id_embedding(id_image, cal_uncond=True)
            id_embeddings = id_embeddings.to(device, dtype=weight_dtype)
            uncond_id_embeddings = uncond_id_embeddings.to(device, dtype=weight_dtype)


            # ID embed (Trg)
            id_image_trg = cv2.imread(trg_img_path)
            print(f"id_image_trg shape: {id_image_trg.shape}")
            id_image_trg = cv2.cvtColor(id_image_trg, cv2.COLOR_BGR2RGB)
            id_image_trg = resize_numpy_image_long(id_image_trg, 1024)
            id_image_trg_pil = Image.fromarray(id_image_trg)
            id_image_trg_pil = id_image_trg_pil.resize((512,512))
            # id_image_trg_pil.save(f"{output_dir}/{src_num}_id_trg.png")   
            trg_id_embeddings, trg_uncond_id_embeddings = flux.transformer.get_id_embedding(id_image_trg, cal_uncond=True)
            trg_id_embeddings = trg_id_embeddings.to(device, dtype=weight_dtype)
            trg_uncond_id_embeddings = trg_uncond_id_embeddings.to(device, dtype=weight_dtype)

            # Nan check
            if torch.isnan(id_embeddings).any():
                raise RuntimeError('id embedding is nan')

            # inp = prepare_txt(t5=flux.t5, clip=flux.clip, prompt=prompt, device=device)
            # inp_neg = prepare_txt(t5=flux.t5, clip=flux.clip,  prompt=neg_prompt, device=device) if use_true_cfg else None
            # flux.t5, flux.clip = flux.t5.cpu(), flux.clip.cpu()

            from diffusers.utils import make_image_grid
            # generate image
            # prompt = "a photo of human face"
            # negative_prompt = ""

            # prompt = "photo of a woman in red dress in a garden"
            # negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
            prompt = 'a photo of human face'
            negative_prompt = ''

            img_save_fname = f"{output_dir}/{src_num}.png"
            if os.path.exists(img_save_fname):
                print(f"Image {img_save_fname} already exists. Skipping...")
                continue

            image_list = []

            # 2. Random -> (Trg ID, Trg Cond)
            # transformer.enable_adapters()
            img = generate_ca_inv(
                    flux,
                    prompt=prompt,
                    conditions=[condition],
                    height=512,
                    width=512,
                    generator=torch.Generator('cpu').manual_seed(0),
                    kv_cache=False,
                    id_embed=trg_id_embeddings,
                    uncond_id_embed=trg_uncond_id_embeddings,
                    guidance_scale=guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    id_guidance_scale=id_guidance_scale,
                    gaze_embed=gaze_embed,
                )
            image = img.images[0] if isinstance(img.images, list) else img.images
            image.save(f"{output_dir}/{src_num}_random_trgID_trgCond.png")
            image_list.append(image)

            # 3. Random Latents -> (Src ID, Trg Cond)
            img = generate_ca_inv(
                    flux,
                    prompt=prompt,
                    conditions=[condition],
                    height=512,
                    width=512,
                    latents=None,
                    generator=torch.Generator('cpu').manual_seed(0),
                    kv_cache=False,
                    id_embed=id_embeddings,
                    uncond_id_embed=uncond_id_embeddings,
                    guidance_scale=guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    id_guidance_scale=id_guidance_scale,
                    gaze_embed=gaze_embed,
                )
            image = img.images[0] if isinstance(img.images, list) else img.images
            image.save(f"{output_dir}/{src_num}_random_srcID_trgCond.png")
            image_list.append(image)


            for inverse_cond in ['trgID_trgCond', 'noID_trgCond', 'trgID_noCond', 'noID_noCond']:   
                cond_image_list = [img for img in image_list]

                cond_output_dir = os.path.join(base_output_dir, f"inv{inverse_steps}", inverse_cond)
                os.makedirs(cond_output_dir, exist_ok=True)
                print(f"=== Inversion with condition: {inverse_cond} ===")

                # 1. Inv 
                # flux.to('cpu')
                # invertpipe.to(device)
                # transformer.disable_adapters()
                inverted_latents = generate_ca_inv(
                        flux,
                        prompt=prompt,
                        conditions=[condition] if 'trgCond' in inverse_cond else [],
                        height=512,
                        width=512,
                        generator=torch.Generator('cpu').manual_seed(0),
                        kv_cache=False,
                        id_embed=trg_id_embeddings if 'trgID' in inverse_cond else None,
                        uncond_id_embed=trg_uncond_id_embeddings if 'trgID' in inverse_cond else None,
                        guidance_scale=guidance_scale,
                        image_guidance_scale=image_guidance_scale,
                        id_guidance_scale=id_guidance_scale,
                        gaze_embed=gaze_embed,
                        # Inversion
                        inverse=True,
                        inverse_steps=inverse_steps,
                        inverse_img=trg_img,
                        output_type='latent',
                        return_dict=False
                        )[0]
                # inverted_latents = invertpipe.invert(source_img=trg_img, source_prompt=prompt, num_inference_steps=25, guidance_scale=guidance_scale, height=512, width=512)
                print(f"inverted_latents shape: {inverted_latents.shape}")




                
                # 2. Inv -> (Trg ID, Trg Cond)
                img = generate_ca_inv(
                        flux,
                        prompt=prompt,
                        conditions=[condition],
                        height=512,
                        width=512,
                        latents=inverted_latents,
                        generator=torch.Generator('cpu').manual_seed(0),
                        kv_cache=False,
                        id_embed=trg_id_embeddings,
                        uncond_id_embed=trg_uncond_id_embeddings,
                        guidance_scale=guidance_scale,
                        image_guidance_scale=image_guidance_scale,
                        id_guidance_scale=id_guidance_scale,
                        gaze_embed=gaze_embed,
                        # Inversion
                        inverse_steps=inverse_steps,
                    )
                image = img.images[0] if isinstance(img.images, list) else img.images
                image.save(f"{cond_output_dir}/{src_num}_inv_trgID_trgCond.png")
                cond_image_list.append(image)

            

                # 4. Inv -> (Src ID, Trg Cond)
                img = generate_ca_inv(
                        flux,
                        prompt=prompt,
                        conditions=[condition],
                        height=512,
                        width=512,
                        latents=inverted_latents,
                        generator=torch.Generator('cpu').manual_seed(0),
                        kv_cache=False,
                        id_embed=id_embeddings,
                        uncond_id_embed=uncond_id_embeddings,
                        guidance_scale=guidance_scale,
                        image_guidance_scale=image_guidance_scale,
                        id_guidance_scale=id_guidance_scale,
                        gaze_embed=gaze_embed,
                        # Inversion
                        inverse_steps=inverse_steps,
                    )
                image = img.images[0] if isinstance(img.images, list) else img.images
                image.save(f"{cond_output_dir}/{src_num}_inv_srcID_trgCond.png")
                cond_image_list.append(image)



                grid = make_image_grid([id_image_pil, trg_img, condition_img] + cond_image_list, rows=1, cols=3 + len(cond_image_list))
                os.makedirs(f"{cond_output_dir}/grid", exist_ok=True)
                grid.save(f"{cond_output_dir}/grid/{src_num}_grid.png")

