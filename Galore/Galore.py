from typing import Union

import torch

class LowRankGradProjector:
    def __init__(self, rank: int = 8, update_projection_step: int = 100, scale: float = 1.0):
        self.rank: int = rank
        self.update_projection_step: int = update_projection_step
        self.scale: float = scale
        self.projector_matrix: torch.Tensor = None
        # Initialize any additional properties you might need later
        
    def project(self, grads: torch.Tensor, step: int) -> torch.Tensor:
        # Project the full rank gradients to low rank subspace using top singular values and left singular vectors
        
        # if the grad dim is 1, just return grad
        if grads.data.dim() == 1 :
            return grads
        
        if grads.shape[0] >= grads.shape[1]:
            # Calculate the projection matrix if the below condition is met
            if self.projector_matrix is None or step % self.update_projection_step == 0:
                if grads.data.dim() == 2:
                    self.projector_matrix = self.get_projector_matrix_2d(grads, sigular_matrix_type='right')
                elif grads.data.dim() ==4:
                    self.projector_matrix = self.get_projector_matrix_4d(grads, sigular_matrix_type='right')
                else:
                    return grads
            
            if self.projector_matrix.dim() == 2:
                low_rank_grad = torch.matmul(grads, self.projector_matrix.t())
            else:
                low_rank_grad = torch.matmul(grads.permute(2,3,0,1), self.projector_matrix.permute(0, 1, 3, 2)).permute(2,3,0,1)
        else:
            # Calculate the projection matrix if the below condition is met
            if self.projector_matrix is None or step % self.update_projection_step == 0:
                if grads.data.dim() == 2:
                    self.projector_matrix = self.get_projector_matrix_2d(grads, sigular_matrix_type='left')
                elif grads.data.dim() ==4:
                    self.projector_matrix = self.get_projector_matrix_4d(grads, sigular_matrix_type='left')
                else:
                    return grads
            
            if self.projector_matrix.dim() == 2:
                low_rank_grad = torch.matmul(self.projector_matrix.t(), grads)
            else:
                low_rank_grad = torch.matmul(self.projector_matrix.permute(0, 1, 3, 2), grads.permute(2,3,0,1)).permute(2,3,0,1)
        
        return low_rank_grad

    def project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        # Project low rank gradients back to full rank space
        if low_rank_grad.dim() == 1:
            return low_rank_grad
        
        if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
            # full_rank_grads = torch.matmul(low_rank_grad, self.projector_matrix)
            if self.projector_matrix.dim() == 2:
                full_rank_grads = torch.matmul(low_rank_grad, self.projector_matrix)
            elif self.projector_matrix.dim() == 4:
                full_rank_grads = torch.matmul(low_rank_grad.permute(2, 3, 0, 1), self.projector_matrix).permute(2, 3, 0, 1)
            else:
                return low_rank_grad
        else:
            # full_rank_grads = torch.matmul(self.projector_matrix, low_rank_grad)
            if self.projector_matrix.dim() == 2:
                full_rank_grads = torch.matmul(self.projector_matrix, low_rank_grad)
            elif self.projector_matrix.dim() == 4:
                full_rank_grads = torch.matmul(self.projector_matrix, low_rank_grad.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
            else:
                return low_rank_grad
            
        
        return full_rank_grads * self.scale

    def get_projector_matrix_2d(self, grads: torch.Tensor, sigular_matrix_type:str):
        # Return the SVD decomposition
        assert grads.data.dim()==2, f'get_projector_matrix_2d() takes only 2 dim matrix, got {grads.data.dim()} dim'
        module_params = grads

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data
            
        U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)
        
        #make the smaller matrix always to be orthogonal matrix
        if sigular_matrix_type=='right':
            # A = U[:, :self.rank] @ torch.diag(s[:self.rank])
            B = Vh[:self.rank, :]
            
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif sigular_matrix_type=='left':
            A = U[:, :self.rank]
            # B = torch.diag(s[:self.rank]) @ Vh[:self.rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A

    def get_projector_matrix_4d(self, grads:torch.Tensor, sigular_matrix_type:str):
        assert grads.data.dim()==4, f'get_projector_matrix_4d() takes only 4 dim matrix, got {grads.data.dim()} dim'

        # Return the SVD decomposition of 4D Tensors
        # Return the SVD decomposition
        module_params = grads

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        # Transpose the matrix. from (channel x channel x kernel x kernel) -> (kernel x kernel x channel x channel)
        # Example (128 x 256 x 3 x 3) -> (3 x 3 x 128 x 256)
        matrix = torch.permute(matrix, (2,3,0,1))

        # Get the SVD decomposition of the grad matrix
        U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)

        #make the smaller matrix always to be orthogonal matrix
        if sigular_matrix_type=='right':
            # A = U[:, :self.rank] @ torch.diag(s[:self.rank])
            B = Vh[:, :, :self.rank, :]
            
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif sigular_matrix_type=='left':
            A = U[:, :, :, :self.rank]
            # B = torch.diag(s[:self.rank]) @ Vh[:self.rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        else:
            raise ValueError('sigular_matrix_type should be left, right or full')

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    dummy_grad = torch.rand(256,128,3,3).to(device)

    galore = LowRankGradProjector(rank=64)

    # low_rank_grad = galore.get_projector_matrix_2d(dummy_grad, sigular_matrix_type='')
    low_rank_grad = galore.project(grads=dummy_grad, step=12)

    full_rank_grad = galore.project_back(low_rank_grad)

    print(low_rank_grad.shape)
    print(full_rank_grad.shape)
