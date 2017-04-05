
# coding: utf-8

# In[ ]:

def load_fa_evl_et(BASE_PATH):
    import nibabel as ni
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
    print('FA Loaded')

    L1 = ni.load(L1_PATH).get_data()
    L2 = ni.load(L2_PATH).get_data()
    L3 = ni.load(L3_PATH).get_data()

    V1 = ni.load(V1_PATH).get_data()
    V2 = ni.load(V2_PATH).get_data()
    V3 = ni.load(V3_PATH).get_data()
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

