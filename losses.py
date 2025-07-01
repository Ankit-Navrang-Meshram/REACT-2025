import torch.nn.functional as F
import torch
import torch
import torch.nn.functional as F

class VQVAE_training_loss:
    def __init__(self):
        pass

    @staticmethod
    def velocity_loss(motion_pred, motion_gt):
        vel_pred = torch.cat((
            torch.zeros_like(motion_pred[:, :1, :]),
            motion_pred[:, 1:, :] - motion_pred[:, :-1, :]
        ), dim=1)
        
        vel_gt = torch.cat((
            torch.zeros_like(motion_gt[:, :1, :]),
            motion_gt[:, 1:, :] - motion_gt[:, :-1, :]
        ), dim=1)
        
        loss = F.l1_loss(vel_pred, vel_gt)
        return loss

    @staticmethod
    def stage_1_loss(quant_loss, motion_pred, emotion_pred, motion, emotion):
        motion = motion.to(motion_pred.device)
        emotion = emotion.to(motion_pred.device)
        total_loss = (
            1.5 * quant_loss +
            0.5 * F.l1_loss(motion_pred[:, :, :52], motion[:, :, :52]) +
            0.1 * F.l1_loss(motion_pred[:, :, 52:], motion[:, :, 52:]) +
            0.5 * VQVAE_training_loss.velocity_loss(motion_pred, motion)
        )
        return total_loss

    @staticmethod
    def stage_2_loss(emotion_pred, emotion):
        return F.l1_loss(emotion_pred, emotion)

class Predictor_training_loss():
    def _init__():
          pass
     
    @staticmethod
    def compute_loss(motion_gt, motion_pred, emotion_gt, emotion_pred, motion_quant, motion_quant_pred):
        # Move targets to the device of their corresponding predictions
        motion_gt = motion_gt.to(motion_pred.device)
        emotion_gt = emotion_gt.to(emotion_pred.device)
        motion_quant = motion_quant.to(motion_quant_pred.device)

        loss = (
            1.5 * F.l1_loss(motion_quant_pred, motion_quant) + 
            0.5 * F.l1_loss(motion_pred[:, :, :52], motion_gt[:, :, :52]) + 
            0.5 * F.l1_loss(motion_pred[:, :, 52:], motion_gt[:, :, 52:]) + 
            0.5 * F.l1_loss(emotion_pred, emotion_gt)
        )
        return loss


