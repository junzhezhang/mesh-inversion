import kaolin as kal
import numpy as np
import torch
import torch.nn.functional as F
import math
import os

from .utils import grid_sample_bilinear, circpad

from packaging import version

class MeshTemplate:
    
    def __init__(self, mesh_path, is_symmetric=True, multi_gpu=False):
        self.multi_gpu = multi_gpu
        if self.multi_gpu:
            MeshTemplate._monkey_patch_dependencies()
            
            mesh = kal.rep.TriangleMesh.from_obj(mesh_path, enable_adjacency=True)

            print('---- Mesh definition ----')
            print(f'Vertices: {mesh.vertices.shape}')
            print(f'Indices: {mesh.faces.shape}')
            print(f'UV coords: {mesh.uvs.shape}')
            print(f'UV indices: {mesh.face_textures.shape}')

            poles = [mesh.vertices[:, 1].argmax().item(), mesh.vertices[:, 1].argmin().item()] # North pole, south pole

            # Compute reflection information (for mesh symmetry)
            axis = 0
            if version.parse(torch.__version__) < version.parse('1.2'):
                neg_indices = torch.nonzero(mesh.vertices[:, axis] < -1e-4)[:, 0].numpy()
                zero_indices = torch.nonzero(torch.abs(mesh.vertices[:, axis]) < 1e-4)[:, 0].numpy()
            else:
                neg_indices = torch.nonzero(mesh.vertices[:, axis] < -1e-4, as_tuple=False)[:, 0].numpy()
                zero_indices = torch.nonzero(torch.abs(mesh.vertices[:, axis]) < 1e-4, as_tuple=False)[:, 0].numpy()
                
            pos_indices = []
            for idx in neg_indices:
                opposite_vtx = mesh.vertices[idx].clone()
                opposite_vtx[axis] *= -1
                dists = (mesh.vertices - opposite_vtx).norm(dim=-1)
                minval, minidx = torch.min(dists, dim=0)
                assert minval < 1e-4, minval
                pos_indices.append(minidx.item())
            assert len(pos_indices) == len(neg_indices)
            assert len(pos_indices) == len(set(pos_indices)) # No duplicates
            pos_indices = np.array(pos_indices)

            pos_indices = torch.LongTensor(pos_indices)
            neg_indices = torch.LongTensor(neg_indices)
            zero_indices = torch.LongTensor(zero_indices)
            nonneg_indices = torch.LongTensor(list(pos_indices) + list(zero_indices))

            total_count = len(pos_indices) + len(neg_indices) + len(zero_indices)
            assert total_count == len(mesh.vertices), (total_count, len(mesh.vertices))

            index_list = {}
            segments = 32
            rings = 31 if '31rings' in mesh_path else 16
            print(f'The mesh has {rings} rings')
            print('-------------------------')
            for faces, vertices in zip(mesh.face_textures, mesh.faces):
                for face, vertex in zip(faces, vertices):
                    if vertex.item() not in index_list:
                        index_list[vertex.item()] = []
                    res = mesh.uvs[face].numpy() * [segments, rings]
                    if math.isclose(res[0], segments, abs_tol=1e-4):
                        res[0] = 0 # Wrap around
                    index_list[vertex.item()].append(res)

            topo_map = torch.zeros(mesh.vertices.shape[0], 2)
            for idx, data in index_list.items():
                avg = np.mean(np.array(data, dtype=np.float32), axis=0) / [segments, rings]
                topo_map[idx] = torch.Tensor(avg)

            # Flip topo map
            topo_map = topo_map * 2 - 1
            topo_map = topo_map * torch.FloatTensor([1, -1]).to(topo_map.device)
            topo_map = topo_map
            nonneg_topo_map = topo_map[nonneg_indices]

            # Force x = 0 for zero-indices if symmetry is enabled
            symmetry_mask = torch.ones_like(mesh.vertices).unsqueeze(0)
            symmetry_mask[:, zero_indices, 0] = 0

            # Compute mesh tangent map (per-vertex normals, tangents, and bitangents)
            mesh_normals = F.normalize(mesh.vertices, dim=1)
            up_vector = torch.Tensor([[0, 1, 0]]).to(mesh_normals.device).expand_as(mesh_normals)
            mesh_tangents = F.normalize(torch.cross(mesh_normals, up_vector, dim=1), dim=1)
            mesh_bitangents = torch.cross(mesh_normals, mesh_tangents, dim=1)
            # North pole and south pole have no (bi)tangent
            mesh_tangents[poles[0]] = 0
            mesh_bitangents[poles[0]] = 0
            mesh_tangents[poles[1]] = 0
            mesh_bitangents[poles[1]] = 0
            
            tangent_map = torch.stack((mesh_normals, mesh_tangents, mesh_bitangents), dim=1)
            nonneg_tangent_map = tangent_map[nonneg_indices] # For symmetric meshes
            
            self.mesh = mesh
            self.topo_map = topo_map
            self.nonneg_topo_map = nonneg_topo_map
            self.nonneg_indices = nonneg_indices
            self.neg_indices = neg_indices
            self.pos_indices = pos_indices
            self.symmetry_mask = symmetry_mask
            self.tangent_map = tangent_map
            self.nonneg_tangent_map = nonneg_tangent_map
            self.is_symmetric = is_symmetric
            
        else:
            MeshTemplate._monkey_patch_dependencies()
            
            mesh = kal.rep.TriangleMesh.from_obj(mesh_path, enable_adjacency=True)
            
            mesh.cuda()
            
            print('---- Mesh definition ----')
            print(f'Vertices: {mesh.vertices.shape}')
            print(f'Indices: {mesh.faces.shape}')
            print(f'UV coords: {mesh.uvs.shape}')
            print(f'UV indices: {mesh.face_textures.shape}')

            poles = [mesh.vertices[:, 1].argmax().item(), mesh.vertices[:, 1].argmin().item()] # North pole, south pole

            # Compute reflection information (for mesh symmetry)
            axis = 0
            if version.parse(torch.__version__) < version.parse('1.2'):
                neg_indices = torch.nonzero(mesh.vertices[:, axis] < -1e-4)[:, 0].cpu().numpy()
                zero_indices = torch.nonzero(torch.abs(mesh.vertices[:, axis]) < 1e-4)[:, 0].cpu().numpy()
            else:
                neg_indices = torch.nonzero(mesh.vertices[:, axis] < -1e-4, as_tuple=False)[:, 0].cpu().numpy()
                zero_indices = torch.nonzero(torch.abs(mesh.vertices[:, axis]) < 1e-4, as_tuple=False)[:, 0].cpu().numpy()
                
            pos_indices = []
            for idx in neg_indices:
                opposite_vtx = mesh.vertices[idx].clone()
                opposite_vtx[axis] *= -1
                dists = (mesh.vertices - opposite_vtx).norm(dim=-1)
                minval, minidx = torch.min(dists, dim=0)
                assert minval < 1e-4, minval
                pos_indices.append(minidx.item())
            assert len(pos_indices) == len(neg_indices)
            assert len(pos_indices) == len(set(pos_indices)) # No duplicates
            pos_indices = np.array(pos_indices)

            pos_indices = torch.LongTensor(pos_indices).cuda()
            neg_indices = torch.LongTensor(neg_indices).cuda()
            zero_indices = torch.LongTensor(zero_indices).cuda()
            nonneg_indices = torch.LongTensor(list(pos_indices) + list(zero_indices)).cuda()

            total_count = len(pos_indices) + len(neg_indices) + len(zero_indices)
            assert total_count == len(mesh.vertices), (total_count, len(mesh.vertices))

            index_list = {}
            segments = 32
            rings = 31 if '31rings' in mesh_path else 16
            print(f'The mesh has {rings} rings')
            print('-------------------------')
            for faces, vertices in zip(mesh.face_textures, mesh.faces):
                for face, vertex in zip(faces, vertices):
                    if vertex.item() not in index_list:
                        index_list[vertex.item()] = []
                    res = mesh.uvs[face].cpu().numpy() * [segments, rings]
                    if math.isclose(res[0], segments, abs_tol=1e-4):
                        res[0] = 0 # Wrap around
                    index_list[vertex.item()].append(res)

            topo_map = torch.zeros(mesh.vertices.shape[0], 2)
            for idx, data in index_list.items():
                avg = np.mean(np.array(data, dtype=np.float32), axis=0) / [segments, rings]
                topo_map[idx] = torch.Tensor(avg)

            # Flip topo map
            topo_map = topo_map * 2 - 1
            topo_map = topo_map * torch.FloatTensor([1, -1]).to(topo_map.device)
            topo_map = topo_map.cuda()
            nonneg_topo_map = topo_map[nonneg_indices]

            # Force x = 0 for zero-indices if symmetry is enabled
            symmetry_mask = torch.ones_like(mesh.vertices).unsqueeze(0)
            symmetry_mask[:, zero_indices, 0] = 0

            # Compute mesh tangent map (per-vertex normals, tangents, and bitangents)
            mesh_normals = F.normalize(mesh.vertices, dim=1)
            up_vector = torch.Tensor([[0, 1, 0]]).to(mesh_normals.device).expand_as(mesh_normals)
            mesh_tangents = F.normalize(torch.cross(mesh_normals, up_vector, dim=1), dim=1)
            mesh_bitangents = torch.cross(mesh_normals, mesh_tangents, dim=1)
            # North pole and south pole have no (bi)tangent
            mesh_tangents[poles[0]] = 0
            mesh_bitangents[poles[0]] = 0
            mesh_tangents[poles[1]] = 0
            mesh_bitangents[poles[1]] = 0
            
            tangent_map = torch.stack((mesh_normals, mesh_tangents, mesh_bitangents), dim=1).cuda()
            nonneg_tangent_map = tangent_map[nonneg_indices] # For symmetric meshes
            
            self.mesh = mesh
            self.topo_map = topo_map
            self.nonneg_topo_map = nonneg_topo_map
            self.nonneg_indices = nonneg_indices
            self.neg_indices = neg_indices
            self.pos_indices = pos_indices
            self.symmetry_mask = symmetry_mask
            self.tangent_map = tangent_map
            self.nonneg_tangent_map = nonneg_tangent_map
            self.is_symmetric = is_symmetric
            
     
    def deform(self, deltas):
        """
        Deform this mesh template along its tangent map, using the provided vertex displacements.
        """
        tgm = self.nonneg_tangent_map if self.is_symmetric else self.tangent_map
        if self.multi_gpu:
            tgm = tgm.cuda()
        return (deltas.unsqueeze(-2) @ tgm.expand(deltas.shape[0], -1, -1, -1)).squeeze(-2)

    def compute_normals(self, vertex_positions):
        """
        Compute face normals from the *final* vertex positions (not deltas).
        """
        a = vertex_positions[:, self.mesh.faces[:, 0]]
        b = vertex_positions[:, self.mesh.faces[:, 1]]
        c = vertex_positions[:, self.mesh.faces[:, 2]]
        v1 = b - a
        v2 = c - a
        normal = torch.cross(v1, v2, dim=2)
        return F.normalize(normal, dim=2)

    def get_vertex_positions(self, displacement_map):
        """
        Deform this mesh template using the provided UV displacement map.
        Output: 3D vertex positions in object space.
        """
        topo = self.nonneg_topo_map if self.is_symmetric else self.topo_map
        if self.multi_gpu:
            topo = topo.cuda()
        if topo.device != displacement_map.device:
            raise 'topo and mesh not on same device'
        _, displacement_map_padded = self.adjust_uv_and_texture(displacement_map)
        if self.is_symmetric:
            # Compensate for even symmetry in UV map
            delta = 1/(2*displacement_map.shape[3])
            expansion = (displacement_map.shape[3]+1)/displacement_map.shape[3]
            topo = topo.clone()
            topo[:, 0] = (topo[:, 0] + 1 + 2*delta - expansion)/expansion # Only for x axis
        topo_expanded = topo.unsqueeze(0).unsqueeze(-2).expand(displacement_map.shape[0], -1, -1, -1)
        vertex_deltas_local = grid_sample_bilinear(displacement_map_padded, topo_expanded).squeeze(-1).permute(0, 2, 1)
        vertex_deltas = self.deform(vertex_deltas_local)
        if self.is_symmetric:
            # Symmetrize
            vtx_n = torch.Tensor(vertex_deltas.shape[0], self.topo_map.shape[0], 3).to(vertex_deltas.device)
            vtx_n[:, self.nonneg_indices] = vertex_deltas
            vtx_n2 = vtx_n.clone()
            vtx_n2[:, self.neg_indices] = vtx_n[:, self.pos_indices] * torch.Tensor([-1, 1, 1]).to(vtx_n.device)
            if self.multi_gpu:
                vertex_deltas = vtx_n2 * self.symmetry_mask.cuda()
            else:
                vertex_deltas = vtx_n2 * self.symmetry_mask
        if self.multi_gpu:
            vertex_positions = self.mesh.vertices.cuda().unsqueeze(0) + vertex_deltas
        else:
            vertex_positions = self.mesh.vertices.unsqueeze(0) + vertex_deltas
        return vertex_positions
    
    def get_point_mesh_relationship_table(self, vertex_positions, sampled_points):
        """
        for the sampled points, [B, 10k, 3]
        get the in- / out- mesh table with shape [B, ]
        """
         
        ### prepare vertices and face vertices, making z=0
        xyz0 = vertex_positions.clone()
        xyz0[:,:,2] = 0
        face_xyz0 = xyz0[:,self.mesh.faces] # (B, 960, 3, 3) -> B, faces, vtx, coords
        n_sample = sampled_points.shape[1]
        n_face = self.mesh.faces.shape[0]

        A = face_xyz0[:,:,0].unsqueeze(1).repeat(1,n_sample,1,1) # (B,10k,960,3)
        B = face_xyz0[:,:,1].unsqueeze(1).repeat(1,n_sample,1,1)
        C = face_xyz0[:,:,2].unsqueeze(1).repeat(1,n_sample,1,1)
        P0 = sampled_points.clone()
        P0[:,:,2] = 0
        P0 = P0.unsqueeze(2).repeat(1,1,n_face,1) # (B, 10k, 960, 3)
        AP = P0 - A
        BP = P0 - B
        CP = P0 - C
        AB = B - A
        BC = C - B
        CA = A - C

        ### get cross prod of ABxAP; BCxBP; CAxCP; 
        ### get the last dimension, the sign of which matters
        ### yielding (B,482,960) for each shape
        ABxAP = torch.cross(AB, AP, dim=-1)[:,:,:,2]
        BCxBP = torch.cross(BC, BP, dim=-1)[:,:,:,2]
        CAxCP = torch.cross(CA, CP, dim=-1)[:,:,:,2]

        ### 1: same sign for each pair of cross-product 0: different sign
        ### NOTE: on the edge (vertex) should return 0 
        v1 = (torch.mul(ABxAP,BCxBP) > 0).type(torch.uint8) 
        v2 = (torch.mul(BCxBP,CAxCP) > 0).type(torch.uint8) 
        v3 = (torch.mul(CAxCP,ABxAP) > 0).type(torch.uint8) 
        
        ### 1: inside a triangle; 0: outside or on a face 
        ### NOTE: that in_face_map.squeeze(0).sum(1) return 482 values, showing some vertexs are within multiple faces 
        in_face_map = torch.mul(torch.mul(v1,v2),v3) # (B, 10k, 960)
        
        # obtain the max & min z for each face
        vtx_z = vertex_positions[:,:,2:]
        face_z = vtx_z[:,self.mesh.faces]
        
        face_zmax = torch.max(face_z,2)[0].unsqueeze(1).repeat(1,n_sample,1,1)
        face_zmin = torch.min(face_z,2)[0].unsqueeze(1).repeat(1,n_sample,1,1)
        #sampled points Z 
        Pz = sampled_points.clone()[:,:,2:].unsqueeze(2).repeat(1,1,n_face,1)  # (B, 10k, 960, 1)
        # 1: P with larger z, in front
        # Mostly, exact one face in front, one face in the back
        # Assume no 'cross-face' scenario
        compare_min_map = ((Pz -face_zmin) > 0).type(torch.uint8).squeeze(-1)  # (B, 10k, 960)
        compare_max_map = ((face_zmax - Pz) > 0).type(torch.uint8).squeeze(-1)

        compare_min_in_face_map = torch.mul(compare_min_map,in_face_map)
        compare_max_in_face_map = torch.mul(compare_max_map,in_face_map)

        # only both odd number --> in shape
        odd_min_vector = compare_min_in_face_map.sum(2)%2
        odd_max_vector = compare_max_in_face_map.sum(2)%2
        in_shape_vector = torch.mul(odd_min_vector, odd_max_vector) 
        
        # print(in_shape_vector.sum().item(), '/', 10000)
        # import time; time.sleep(1)

        # NOTE: save for debug purpose
        # data_to_plot = {
        #     'vtx': vertex_positions.cpu().squeeze(0),
        #     'sampled_points': sampled_points.cpu().squeeze(0),
        #     'mask': in_shape_vector.cpu().squeeze(0)
        # }
        # torch.save(data_to_plot,'./data_for_visual.pth')
        

        return in_shape_vector 

    
    def get_frontal_vertex_indices(self, vertex_positions):
        """
        tell if frontal or back for each index, based on vtx position and faces
        tell if a vtx in any of the face
        manually exlude the the face that contains the vrx
        """

        ### prepare vertices and face vertices, making z=0
        xyz0 = vertex_positions.clone()
        xyz0[:,:,2] = 0
        face_xyz0 = xyz0[:,self.mesh.faces] # (B, 960, 3, 3) -> B, faces, vtx, coords
        n_vtx = vertex_positions.shape[1]
        n_face = self.mesh.faces.shape[0]
        
        ### A B C 0,1,2; P is one of the 482 vertices
        A = face_xyz0[:,:,0].unsqueeze(1).repeat(1,n_vtx,1,1) # (B,482,960,3)
        B = face_xyz0[:,:,1].unsqueeze(1).repeat(1,n_vtx,1,1)
        C = face_xyz0[:,:,2].unsqueeze(1).repeat(1,n_vtx,1,1)
        P  = xyz0.unsqueeze(2).repeat(1,1,n_face,1)
        AP = P - A
        BP = P - B
        CP = P - C
        AB = B - A
        BC = C - B
        CA = A - C
        
        ### get cross prod of ABxAP; BCxBP; CAxCP; 
        ### get the last dimension, the sign of which matters
        ### yielding (B,482,960) for each shape
        ABxAP = torch.cross(AB, AP, dim=-1)[:,:,:,2] # (B, 10k, 960)
        BCxBP = torch.cross(BC, BP, dim=-1)[:,:,:,2]
        CAxCP = torch.cross(CA, CP, dim=-1)[:,:,:,2]
        
        ### 1: same sign for each pair of cross-product 0: different sign
        ### NOTE: on the edge (vertex) should return 0 
        v1 = (torch.mul(ABxAP,BCxBP) > 0).type(torch.uint8) 
        v2 = (torch.mul(BCxBP,CAxCP) > 0).type(torch.uint8) 
        v3 = (torch.mul(CAxCP,ABxAP) > 0).type(torch.uint8) 
        
        ### 1: inside a triangle; 0: outside or on a face 
        xy_relation = torch.mul(torch.mul(v1,v2),v3) 
        
        ### determin the z_relation
        ### NOTE: vtx_z.min() > 0 , not all negative 
        # NOTE: assume it is correct that larger z (less negative) means in the front
        vtx_z = vertex_positions[:,:,2:]
        face_z = vtx_z[:,self.mesh.faces]
        Az = face_z[:,:,0].unsqueeze(1).repeat(1,n_vtx,1,1) # (B,482,960,1)
        Bz = face_z[:,:,1].unsqueeze(1).repeat(1,n_vtx,1,1)
        Cz = face_z[:,:,2].unsqueeze(1).repeat(1,n_vtx,1,1)
        Pz  = vtx_z.unsqueeze(2).repeat(1,1,n_face,1)
        # 1: P with larger z, in front
        # log: v1 was < 0; v2 > 0, a_proj24
        delta_za = ((Pz - Az) < 0).type(torch.uint8).squeeze(-1)
        delta_zb = ((Pz - Bz) < 0).type(torch.uint8).squeeze(-1)
        delta_zc = ((Pz - Bz) < 0).type(torch.uint8).squeeze(-1)

        

        ### 1: occluded by the face, both within the face and behind the vertex
        ### 1 - occa means not occluded
        occa = torch.mul(xy_relation, delta_za)
        occb = torch.mul(xy_relation, delta_zb)
        occc = torch.mul(xy_relation, delta_zc)
        
        ### 1: not occluded by any of the face
        relation_matrix = torch.mul(torch.mul(1-occa,1-occb),1-occc)
        mask = (relation_matrix.sum(-1) == relation_matrix.shape[-1]).cuda().type(torch.float32)
        # NOTE: this mask is too harsh, too little is left
        
        return mask # 1 means frontal, 0 means occluded  
       
    def get_vertex_colors(self, texture_map):
        """
        return RGB color for each vertex
        texture_map : (B, 3, 512, 512)
        return: (B, N, 3) # for each bird, desired 482
        adapt from : get_vertex_positions()
        """
        
        topo = self.nonneg_topo_map if self.is_symmetric else self.topo_map
        _, texture_map_padded = self.adjust_uv_and_texture(texture_map)
        if self.is_symmetric:
            # Compensate for even symmetry in UV map
            delta = 1/(2*texture_map_padded.shape[3])
            expansion = (texture_map_padded.shape[3]+1)/texture_map_padded.shape[3]
            topo = topo.clone()
            topo[:, 0] = (topo[:, 0] + 1 + 2*delta - expansion)/expansion # Only for x axis
        topo_expanded = topo.unsqueeze(0).unsqueeze(-2).expand(texture_map.shape[0], -1, -1, -1)
        vertex_colors = grid_sample_bilinear(texture_map_padded, topo_expanded).squeeze(-1).permute(0, 2, 1) # (1,257,3) for bird
        # topo_expanded [1, 257, 1, 2]; texture_map_padded [1, 3, 512, 514]; vtx_colors_v2 before change [1, 3, 257, 1]
        # vtx_colors_v2 = F.grid_sample(texture_map_padded,topo_expanded, align_corners=True).squeeze(-1).permute(0, 2, 1)
        if self.is_symmetric:
            # Symmetrize
            vtx_n = torch.Tensor(vertex_colors.shape[0], self.topo_map.shape[0], 3).to(vertex_colors.device) # (1,482,3) for bird
            vtx_n[:, self.nonneg_indices] = vertex_colors
            vtx_n2 = vtx_n.clone()
            vtx_n2[:, self.neg_indices] = vtx_n[:, self.pos_indices].to(vtx_n.device)
            vertex_colors_new = vtx_n2 * self.symmetry_mask
        
        return vertex_colors_new

    def adjust_uv_and_texture(self, texture, return_texture=True):
        """
        Returns the UV coordinates of this mesh template,
        and preprocesses the provided texture to account for boundary conditions.
        If the mesh is symmetric, the texture and UVs are adjusted accordingly.
        """
        
        if self.is_symmetric:
            delta = 1/(2*texture.shape[3])
            expansion = (texture.shape[3]+1)/texture.shape[3]
            uvs = self.mesh.uvs.clone()
            uvs[:, 0] = (uvs[:, 0] + delta)/expansion
            
            uvs = uvs.expand(texture.shape[0], -1, -1)
            texture = circpad(texture, 1) # Circular padding
        else:
            uvs = self.mesh.uvs.expand(texture.shape[0], -1, -1)
            texture = torch.cat((texture, texture[:, :, :, :1]), dim=3)
        
        if self.multi_gpu:
            uvs = uvs.cuda()
        
            
        return uvs, texture
    
    def forward_renderer(self, renderer, vertex_positions, texture, num_gpus=1, **kwargs):
        if self.multi_gpu:
            mesh_faces = self.mesh.faces.cuda()
            mesh_face_textures = self.mesh.face_textures.cuda()
        else:
            mesh_faces = self.mesh.faces
            mesh_face_textures = self.mesh.face_textures
            if num_gpus > 1:
                mesh_faces = mesh_faces.repeat(num_gpus, 1)
                mesh_face_textures = mesh_face_textures.repeat(num_gpus, 1)

        input_uvs, input_texture = self.adjust_uv_and_texture(texture)
        ### standard input:
        # input_texture [10, 3, 128, 130] circular padded, or torch.Size([1, 3, 512, 514])
        # input_uvs [B, 559, 2], consider symmetric, return uv for each vertice
        image, alpha, _ = renderer(points=[vertex_positions, mesh_faces],
                                   uv_bxpx2=input_uvs,
                                   texture_bx3xthxtw=input_texture,
                                   ft_fx3=mesh_face_textures,
                                   **kwargs)
        return image, alpha
    
    def export_obj(self, path_prefix, vertex_positions, texture):
        assert len(vertex_positions.shape) == 2
        mesh_path = path_prefix + '.obj'
        material_path = path_prefix + '.mtl'
        material_name = os.path.basename(path_prefix)
        
        # Export mesh .obj
        with open(mesh_path, 'w') as file:
            print('mtllib ' + os.path.basename(material_path), file=file)
            for v in vertex_positions:
                print('v {:.5f} {:.5f} {:.5f}'.format(*v), file=file)
            for uv in self.mesh.uvs:
                print('vt {:.5f} {:.5f}'.format(*uv), file=file)
            print('usemtl ' + material_name, file=file)
            for f, ft in zip(self.mesh.faces, self.mesh.face_textures):
                print('f {}/{} {}/{} {}/{}'.format(f[0]+1, ft[0]+1, f[1]+1, ft[1]+1, f[2]+1, ft[2]+1), file=file)
                
        # Export material .mtl
        with open(material_path, 'w') as file:
            print('newmtl ' + material_name, file=file)
            print('Ka 1.000 1.000 1.000', file=file)
            print('Kd 1.000 1.000 1.000', file=file)
            print('Ks 0.000 0.000 0.000', file=file)
            print('d 1.0', file=file)
            print('illum 1', file=file)
            print('map_Ka ' + material_name + '.png', file=file)
            print('map_Kd ' + material_name + '.png', file=file)
            
        # Export texture
        import imageio
        texture = (texture.permute(1, 2, 0)*255).clamp(0, 255).cpu().byte().numpy()
        imageio.imwrite(path_prefix + '.png', texture)
                
    @staticmethod
    def _monkey_patch_dependencies():
        if version.parse(torch.__version__) < version.parse('1.2'):
            def torch_where_patched(*args, **kwargs):
                if len(args) == 1:
                    return (torch.nonzero(args[0]), )
                else:
                    return torch._where_original(*args)

            torch._where_original = torch.where
            torch.where = torch_where_patched
            
        if version.parse(torch.__version__) >= version.parse('1.5'):
            from .monkey_patches import compute_adjacency_info_patched
            # Monkey patch
            kal.rep.Mesh.compute_adjacency_info = staticmethod(compute_adjacency_info_patched)
                
                