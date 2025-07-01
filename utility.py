import os
import torch
from torch import nn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

### save models and load models
def save_entire_models(model, optimizer, scheduler, epoch, best_val_loss, duration= 25 ,resampling_method = None, type = None ,save_dir=None , with_style=False , online=False):
    if type == "VQVAE":
        save_dir = os.path.join(save_dir, "VQVAE")
    elif type == "Predictor":
        save_dir = os.path.join(save_dir, "Predictor")
    if online:
        save_dir = os.path.join(save_dir, "online")
    else:
        save_dir = os.path.join(save_dir, "offline")
    if with_style:
        save_dir = os.path.join(save_dir, "with_style")
    else:
        save_dir = os.path.join(save_dir, "without_style")
    
    if resampling_method != None:
        save_dir = os.path.join(save_dir, resampling_method, f"audio_duration_{duration}")
    else:
        save_dir = os.path.join(save_dir, f"audio_duration_{duration}")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "entire_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss
    }, save_path)

def load_entire_models(model, optimizer, scheduler, duration = 8 ,resampling_method = None ,type = None ,save_dir=None, with_style=False, online=False):
    if type == "VQVAE":
        save_dir = os.path.join(save_dir, "VQVAE")
    elif type == "Predictor":
        save_dir = os.path.join(save_dir, "Predictor")
    if online:
        save_dir = os.path.join(save_dir, "online")
    else:
        save_dir = os.path.join(save_dir, "offline")
    if with_style:
        save_dir = os.path.join(save_dir, "with_style")
    else:
        save_dir = os.path.join(save_dir, "without_style")
    
    if resampling_method != None:
        save_dir = os.path.join(save_dir, resampling_method, f"audio_duration_{duration}")
    else:
        save_dir = os.path.join(save_dir, f"audio_duration_{duration}")
        
    save_path = os.path.join(save_dir, "entire_model.pt")
    
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    
    return model, optimizer, scheduler, epoch, best_val_loss

def stage_1_save_models(model,optimizer, scheduler,epoch, best_val_loss,duration= 25 , save_dir=None , with_style=False, online=False):
    save_dir  = os.path.join(save_dir, "VQVAE")
    if online:
        save_dir = os.path.join(save_dir, "online")
    else:
        save_dir = os.path.join(save_dir, "offline")
    if with_style:
        save_dir = os.path.join(save_dir, "with_style")
    else:
        save_dir = os.path.join(save_dir, "without_style")  
        
    save_dir = os.path.join(save_dir, f"audio_duration_{duration}")
    os.makedirs(save_dir, exist_ok=True)
    
    model_to_save = model

    components = [
        ("motion_encoder", model_to_save.motion_encoder),
        ("motion_decoder", model_to_save.motion_decoder),
        ("quantizer", model_to_save.quantize),
    ]
    
    saved_files = []
    for name, module in components:
        save_path = os.path.join(save_dir, f"{name}.pt")
        torch.save({
            'model_state_dict': module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss
        }, save_path)
        saved_files.append(save_path)

def stage_1_load_models(model, optimizer = None, scheduler = None, duration=8, save_dir=None, with_style=False, online=False):
    save_dir = os.path.join(save_dir, "VQVAE")
    if online:
        save_dir = os.path.join(save_dir, "online")
    else:
        save_dir = os.path.join(save_dir, "offline")
    if with_style:
        save_dir = os.path.join(save_dir, "with_style")
    else:
        save_dir = os.path.join(save_dir, "without_style") 
         
    save_dir = os.path.join(save_dir,f"audio_duration_{duration}")
    
    components = [
        ("motion_encoder", model.motion_encoder),
        ("motion_decoder", model.motion_decoder),
        ("quantizer", model.quantize),
    ]
    
    for name, module in components:
        save_path = os.path.join(save_dir, f"{name}.pt")
        checkpoint = torch.load(save_path)
        module.load_state_dict(checkpoint['model_state_dict'])
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler != None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return model, optimizer, scheduler

class Logs():
    def __init__(self, type = None ,online=False, style=False , resampling_method = None):
        if online:
            self.online = "online"
        else:
            self.online = "offline"
        if style:
            self.style = "with_style"
        else:
            self.style = "without_style"  
        if type == "VQVAE":
            self.type = "VQVAE"
        else:
            self.type = "Predictor" 
        self.writer = None
        self.resampling_method = resampling_method
    
    def setup_logging(self):
        """Set up TensorBoard logging with timestamped log directory."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = ""
        if self.resampling_method == None :
            log_dir = os.path.join("logs", self.type , self.online, self.style, f"{self.type}_log_{timestamp}")
        else:
            log_dir = os.path.join("logs", self.type , self.online, self.style,self.resampling_method, f"{self.type}_log_{timestamp}")
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=log_dir)
        return self.writer
