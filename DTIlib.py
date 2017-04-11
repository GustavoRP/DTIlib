
# coding: utf-8

# # DTIlib

# Library to manipulate DTI data.

# ### All functions need the input data to be the folowing fomat
# 
# Scalar volume = [z,y,x]
# 
# evl = [evl1] [evl2] [evl3]
# 
# evt = [evt(1, 2, 3)] [componentes dos evt (z, y, x)] [Z] [Y] [X]

# In[1]:

def load_fa_evl_et(BASE_PATH):
    import nibabel as ni #pip install nibabel
    import numpy as np

    #-------------------------------------------------------
    FA_PATH = "%s%s" % ( BASE_PATH, '/dti_FA.nii.gz' )

    L1_PATH = "%s%s" % ( BASE_PATH, '/dti_L1.nii.gz' )
    L2_PATH = "%s%s" % ( BASE_PATH, '/dti_L2.nii.gz' )
    L3_PATH = "%s%s" % ( BASE_PATH, '/dti_L3.nii.gz' )

    V1_PATH = "%s%s" % ( BASE_PATH, '/dti_V1.nii.gz' )
    V2_PATH = "%s%s" % ( BASE_PATH, '/dti_V2.nii.gz' )
    V3_PATH = "%s%s" % ( BASE_PATH, '/dti_V3.nii.gz' )
    #-------------------------------------------------------
    FA = ni.load(FA_PATH).get_data()
#     print('FA Loaded')

    L1 = ni.load(L1_PATH).get_data()
    L2 = ni.load(L2_PATH).get_data()
    L3 = ni.load(L3_PATH).get_data()
#     print('EVL Loaded')

    V1 = ni.load(V1_PATH).get_data()
    V2 = ni.load(V2_PATH).get_data()
    V3 = ni.load(V3_PATH).get_data()
#     print('EVT Loaded')
    #-------------------------------------------------------
    fa = np.swapaxes(FA,0,2)

    l1 = np.swapaxes(L1,0,2)

    l2 = np.swapaxes(L2,0,2)

    l3 = np.swapaxes(L3,0,2)
    #-------------------------------------------------------
    v1 = np.swapaxes(V1,0,3)
    v1 = np.swapaxes(v1,1,2)
    v1[:,:,:,:] = v1[::-1,:,:,:]

    v2 = np.swapaxes(V2,0,3)
    v2 = np.swapaxes(v2,1,2)
    v2[:,:,:,:] = v2[::-1,:,:,:]

    v3 = np.swapaxes(V3,0,3)
    v3 = np.swapaxes(v3,1,2)
    #-------------------------------------------------------

    evl = np.array([l1,l2,l3])
    evt = np.array([v1,v2,v3])


    return fa, evl, evt


# ### Manipulação de volumes (rotações)

# In[2]:

#rotaciona vetores
def rot_vec(vec, angle = 90, axis = 'z'):
     
    angle = angle*np.pi/180

    if (axis == 'x'):
        rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
   
    if (axis == 'y'):
        rot = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    if (axis == 'z'):
        rot = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        
    print rot
    
    vec_rot = np.dot(vec,rot)
    
    return vec_rot

#rotaciona vetores dentro do voxel / antigo rotEVT
def rot_local_3vec(evt, angle = 90, axis = 'z'):
     
    angle = angle*np.pi/180

    if (axis == 'x'):
        rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
   
    if (axis == 'y'):
        rot = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    if (axis == 'z'):
        rot = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    
    evt_aux = np.swapaxes(evt,1,4)
    evt_rot = np.dot(evt_aux,rot)
    evt_rot = np.swapaxes(evt_rot,1,4)
    
    return evt_rot


#rotaciona Volumes
def rot_vol(vol, angle = 90, axis = 'z', int_order = 0, reshape=True, cval=0.0):
    import numpy as np
    import scipy.ndimage.interpolation as sni

    if (axis == 'z'):
        a = 0
        b = 2
    if (axis == 'y'):
        a = 1
        b = 2
    if (axis == 'x'):
        a = 0
        b = 1
    
    vol = np.swapaxes(vol,a,b)
    vol_r = sni.rotate(vol, angle, axes=(1, 0), reshape=reshape, output=None, order=int_order, mode='constant', cval=cval, prefilter=True)
    vol_r = np.swapaxes(vol_r,a,b)
    
    return vol_r

#rotaciona os vetores de um voxel para outro (translada os vetores)
def rot_external_3vec(evt, angle = 90, axis = 'z', int_order = 0, reshape=True, cval=0.0):
    import numpy as np
    
    evt00_r = rot_vol(evt[0,0,:,:,:], angle = angle, axis = axis, int_order = int_order, reshape=reshape, cval=cval)
    evt01_r = rot_vol(evt[0,1,:,:,:], angle = angle, axis = axis, int_order = int_order, reshape=reshape, cval=cval)
    evt02_r = rot_vol(evt[0,2,:,:,:], angle = angle, axis = axis, int_order = int_order, reshape=reshape, cval=cval)

    evt10_r = rot_vol(evt[1,0,:,:,:], angle = angle, axis = axis, int_order = int_order, reshape=reshape, cval=cval)
    evt11_r = rot_vol(evt[1,1,:,:,:], angle = angle, axis = axis, int_order = int_order, reshape=reshape, cval=cval)
    evt12_r = rot_vol(evt[1,2,:,:,:], angle = angle, axis = axis, int_order = int_order, reshape=reshape, cval=cval)

    evt20_r = rot_vol(evt[2,0,:,:,:], angle = angle, axis = axis, int_order = int_order, reshape=reshape, cval=cval)
    evt21_r = rot_vol(evt[2,1,:,:,:], angle = angle, axis = axis, int_order = int_order, reshape=reshape, cval=cval)
    evt22_r = rot_vol(evt[2,2,:,:,:], angle = angle, axis = axis, int_order = int_order, reshape=reshape, cval=cval)
                    
    z, y, x = evt00_r.shape
    evt_r = np.zeros((3,3,z,y,x))
                          
    evt_r[0,0] = evt00_r
    evt_r[0,1] = evt01_r
    evt_r[0,2] = evt02_r

    evt_r[1,0] = evt10_r
    evt_r[1,1] = evt11_r
    evt_r[1,2] = evt12_r

    evt_r[2,0] = evt20_r
    evt_r[2,1] = evt21_r
    evt_r[2,2] = evt22_r
    
    return evt_r

#rotaciona os autovalores (translada)
def rot_evl(evl, angle = 90, axis = 'z', int_order = 0, reshape=True, cval=0.0):
    import numpy as np
    
    evl0_r = rot_vol(evl[0,:,:,:], angle = angle, axis = axis, int_order = int_order, reshape=reshape, cval=cval)
    evl1_r = rot_vol(evl[1,:,:,:], angle = angle, axis = axis, int_order = int_order, reshape=reshape, cval=cval)
    evl2_r = rot_vol(evl[2,:,:,:], angle = angle, axis = axis, int_order = int_order, reshape=reshape, cval=cval)

    z, y, x = evl0_r.shape
    evl_r = np.zeros((3,z,y,x))
                          
    evl_r[0] = evl0_r
    evl_r[1] = evl1_r
    evl_r[2] = evl2_r

    return evl_r

#rotaciona completamente 3 campos vetoriais (formato de evt)
def rot_3VF(VFs, angle = 90, axis = 'z', int_order = 0, reshape=True, cval=0.0):
    import numpy as np
    
    VFs = rot_local_3vec(VFs, angle = angle, axis = axis)
    
    VFs_rot = rot_external_3vec(VFs, angle = angle, axis = axis, int_order = int_order, reshape=reshape, cval=cval)
    
    return VFs_rot


#rotaciona EVT * EVL (esse fica bom para interpolar)
def rot_evt_evl(evt, evl, angle = 90, axis = 'z', int_order = 3, reshape=True, cval=0.0):
    import numpy as np
    
    aux = evl*np.swapaxes(evt,0,1)
    evtl = np.swapaxes(aux,0,1)
    
    evtl_rot = rot_3VF(evtl, angle = angle, axis = axis, int_order = int_order, reshape=reshape, cval=cval)
    evl_rot = rot_evl(evl, angle = angle, axis = axis, int_order = int_order, reshape=reshape, cval=cval) + 0.000000000000000000001
    
    aux = np.swapaxes(evtl_rot,0,1)/evl_rot #encontro os autovetores novamente (acredito que ainda é só uma aproximação)
    evt_rot = np.swapaxes(aux,0,1)
    
    return evt_rot, evl_rot

#rotaciona EVT * EVL (esse fica bom para interpolar) (acho que essa é a certa)
def rot_evt_evl_2(evt, evl, angle = 90, axis = 'z', int_order = 3, reshape=True, cval=0.0):
    import numpy as np
    
    aux = evl*np.swapaxes(evt,0,1)
    evtl = np.swapaxes(aux,0,1)
    
    evtl_rot = rot_3VF(evtl, angle = angle, axis = axis, int_order = int_order, reshape=reshape, cval=cval)
    
    evl_new = np.linalg.norm(evtl_rot, axis=1) + 0.000000000000000000001
    aux = np.swapaxes(evtl_rot,0,1)/evl_new #acho que agora está certo
    evt_rot = np.swapaxes(aux,0,1)
    
    return evt_rot, evl_new


#rotaciona EVT, EVL e FA
def rot_evt_evl_fa(evt, evl, fa, angle = 90, axis = 'z', int_order = 3, reshape=True, cval=0.0):
    import numpy as np
    
    evt_rot, evl_rot = rot_evt_evl_2(evt, evl, angle=angle, axis=axis, int_order=int_order, reshape=reshape, cval=cval)
    fa_rot = rot_vol(fa, angle=angle, axis=axis, int_order=int_order, reshape=reshape, cval=cval)
    
    return evt_rot, evl_rot, fa_rot


# Duplica a resolução em Z
def interpola_Z(evt, evl, fa):
    import numpy as np
    
    aux1 = evl*np.swapaxes(evt,0,1)
    evtl = np.swapaxes(aux1,0,1)

    _, _, sz, sy, sx = evtl.shape

    evtl_interpolated = np.zeros((3, 3, sz*2 - 1, sy, sx))
    fa_interpolated = np.zeros((sz*2 - 1, sy, sx))


    fa_interpolated[::2] = fa
    fa_interpolated[1:-1:2] = (fa[1:] + fa[:-1])/2

    evtl_interpolated[:,:,::2] = evtl
    evtl_interpolated[:,:,1:-1:2] = (evtl[:,:,1:] + evtl[:,:,:-1])/2


    evl_new = np.linalg.norm(evtl_interpolated, axis=1) + 0.000000000000000000001
    aux = np.swapaxes(evtl_interpolated,0,1)/evl_new #acho que agora está certo
    evt_interpolated = np.swapaxes(aux,0,1)
    
    return evt_interpolated, evl_new, fa_interpolated


# In[ ]:



