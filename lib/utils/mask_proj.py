import torch
import time
import torch.nn.functional as F

def mask2proj(mask_4d, threshold=0.9):
    """
    convert from mask into coordinates
    input [1,1,299,299]: torch
    outout [1,N,2]
    NOTE: plt.scatter(yy,-xx), that's why swap x,y and make y * -1 after scale
    """
    mask_2d = mask_4d[0,0]
    indices_2d = torch.where(mask_2d>threshold)
    indices = torch.stack([indices_2d[1],indices_2d[0]],-1)
    assert mask_4d.shape[2] == mask_4d.shape[3]
    scale = mask_4d.shape[3]/2.0
    coords = indices/scale -1
    coords[:,1]*=(-1) # indices from top to down (row 0 to row N), coords fron down to top [-1,1]
    return coords.unsqueeze(0)

def get_vtx_color(mask_4d, img_4d, threshold=0.9):
    """
    given image and mask
    img: (1,3,299,299) [-1,1]
    mask: [1,1,299,299]: torch
    output:
        vtx: [1,N,2] coords
        color: [1,N,3]
    """
    mask_2d = mask_4d[0,0]
    indices_2d = torch.where(mask_2d>threshold)
    
    indices = torch.stack([indices_2d[1],indices_2d[0]],-1)
    assert mask_4d.shape[2] == mask_4d.shape[3]
    scale = mask_4d.shape[3]/2.0
    coords = indices/scale -1
    coords[:,1]*=(-1)

    color = img_4d[0,:,indices_2d[0],indices_2d[1]]
    return coords.unsqueeze(0), color.permute([1,0]).contiguous().unsqueeze(0) #,indices

def grid_sample_from_vtx(vtx, color_map):
    """
    grid sample from vtx
    the vtx can be form mask2proj() or get_vtx_color(), or projected from vtx_3d
    color_map can be target image, rendered image, or feature map of any size
    vtx: [B, N, 2]
    color_map: [B, C, H, W]
    """
    vtx_copy = vtx.clone()
    vtx_copy[:,:,1] *= (-1)

    clr_sampled = F.grid_sample(color_map,vtx_copy.unsqueeze(2), align_corners=True).squeeze(-1).permute(0, 2, 1)

    return clr_sampled
    


def farthest_point_sample(xyz, npoint):
    
    """
    code borrowed from: http://www.programmersought.com/article/8737853003/#14_query_ball_point_93
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
    	# Update the i-th farthest point
        centroids[:, i] = farthest
        # Take the xyz coordinate of the farthest point
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        # Calculate the Euclidean distance from all points in the point set to this farthest point
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # Update distances to record the minimum distance of each point in the sample from all existing sample points
        mask = dist < distance
        distance[mask] = dist[mask]
        # Find the farthest point from the updated distances matrix, and use it as the farthest point for the next iteration
        farthest = torch.max(distance, -1)[1]
    return centroids

# def mask2proj(mask_tensor,threshold=0.9):
#     """
#     assume mask of shape (1, 1, h, w)
#     return proj (1,N,2) of xy points of the N points
#     assume in the [0,1] range
#     """
#     
#     # set(mask_tensor.detach().cpu().numpy().reshape(-1).tolist())
#     # ans = mask > threshold # return boolen

#     idx_tuple = (mask_tensor > threshold).nonzero(as_tuple=True)
#     h_idx = idx_tuple[2].type(torch.float32)
#     w_idx = idx_tuple[3].type(torch.float32)
    
#     # NOTE: normalize to [0,1]
#     h_coords = h_idx/mask_tensor.shape[2]
#     w_coords = w_idx/mask_tensor.shape[3]

#     proj = torch.stack([h_coords,w_coords],-1).unsqueeze(0) 
#     return proj

# def mask2proj_loop(mask_tensor,threshold=0.9):
#     ### v1
#     # tic = time.time()
#     # coords = []
#     # for i in range(mask_tensor.shape[2]):
#     #     for j in range(mask_tensor.shape[3]):
#     #         if mask_tensor[0,0,i,j] > threshold:
#     #             coords.append([i,j])
#     # toc = time.time()
#     # print('time spent in loop:',int(toc-tic))

#     ### v2
#     tic = time.time()
#     coords = []
#     for i in range(mask_tensor.shape[2]):
#         for j in range(mask_tensor.shape[3]):
#             if mask_tensor[0,0,i,j] > threshold:
#                 coords.append([j,i])
#     toc = time.time()
#     print('time spent in loop:',int(toc-tic))
#     return coords


# def visualize