
import torch
import torch.nn.functional as F

def get_loss(latent_dim, distribution_mean, distribution_variance, factor, batch_size):

    def get_reconstruction_loss(y_true, y_pred):
        """Computes MSE-based reconstruction loss."""
        reconstruction_loss = F.mse_loss(y_pred, y_true, reduction='sum') / batch_size
        return 0.5 * reconstruction_loss * factor

    def get_kl_loss():
        """Computes KL divergence loss."""
        kl_loss = latent_dim + distribution_variance - distribution_mean.pow(2) - torch.exp(distribution_variance)
        kl_loss_batch = torch.sum(kl_loss) / batch_size
        return -0.5 * kl_loss_batch  

    def total_loss(y_true, y_pred):
        """Computes total VAE loss."""
        reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
        kl_loss_batch = get_kl_loss()
        return reconstruction_loss_batch + kl_loss_batch

    return total_loss
