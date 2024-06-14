#*************************************************************************#
#                                                                         #
#  function AZIMUTH_PLUNGE                                                #
#                                                                         #
#  calculation of azimuth and plunge of principal stress axes             #
#                                                                         #
#  input:  stress tensor                                                  #
#  output: azimuth and plunge of principal stress axes                    #
#                                                                         #
#*************************************************************************#
def azimuth_plunge(tau):
    
    import numpy as np
    
    #--------------------------------------------------------------------------
    # eigenvalues and eienvectors of the stress tensor
    #--------------------------------------------------------------------------
    diag_tensor, vector = np.linalg.eig(tau)
    
    value = np.linalg.eigvals(np.diag(diag_tensor))
    value_sorted=np.sort(value)
    j = np.argsort(value)
    
    sigma_vector_1  = np.array(vector[:,j[0]])
    sigma_vector_2  = np.array(vector[:,j[1]])
    sigma_vector_3  = np.array(vector[:,j[2]])

    
    if (sigma_vector_1[2]<0): sigma_vector_1 = -sigma_vector_1;
    if (sigma_vector_2[2]<0): sigma_vector_2 = -sigma_vector_2;
    if (sigma_vector_3[2]<0): sigma_vector_3 = -sigma_vector_3;
    
    sigma = np.sort(np.linalg.eigvals(tau))
    shape_ratio = (sigma[0]-sigma[1])/(sigma[0]-sigma[2])
    
    #--------------------------------------------------------------------------
    # 1. eigenvector
    #--------------------------------------------------------------------------
    fi_1 = np.arctan(abs(sigma_vector_1[1]/sigma_vector_1[0]))*180/np.pi
                     
        
                     
    if (sigma_vector_1[1] >= 0 and sigma_vector_1[0] >= 0): azimuth_sigma_1 = fi_1   
    if (sigma_vector_1[1]>=0 and sigma_vector_1[0]< 0): azimuth_sigma_1 = 180-fi_1;  
    if (sigma_vector_1[1]< 0 and sigma_vector_1[0]< 0): azimuth_sigma_1 = 180+fi_1;   
    if (sigma_vector_1[1]< 0 and sigma_vector_1[0]>=0): azimuth_sigma_1 = 360-fi_1;   
    
    theta_sigma_1 = np.arccos(abs(sigma_vector_1[2]))*180/np.pi
        
    #--------------------------------------------------------------------------
    # 2. eigenvector
    #--------------------------------------------------------------------------
    fi_2 = np.arctan(abs(sigma_vector_2[1]/sigma_vector_2[0]))*180/np.pi
    
    if (sigma_vector_2[1]>=0 and sigma_vector_2[0]>=0): azimuth_sigma_2 =     fi_2;   
    if (sigma_vector_2[1]>=0 and sigma_vector_2[0]< 0): azimuth_sigma_2 = 180-fi_2;   
    if (sigma_vector_2[1]< 0 and sigma_vector_2[0]< 0): azimuth_sigma_2 = 180+fi_2;   
    if (sigma_vector_2[1]< 0 and sigma_vector_2[0]>=0): azimuth_sigma_2 = 360-fi_2;   
    
    theta_sigma_2 = np.arccos(abs(sigma_vector_2[2]))*180/np.pi
    
    #--------------------------------------------------------------------------
    # 3. eigenvector
    #--------------------------------------------------------------------------
    fi_3 = np.arctan(abs(sigma_vector_3[1]/sigma_vector_3[0]))*180/np.pi
    
    if (sigma_vector_3[1]>=0 and sigma_vector_3[0]>=0): azimuth_sigma_3 =     fi_3;   
    if (sigma_vector_3[1]>=0 and sigma_vector_3[0]< 0): azimuth_sigma_3 = 180-fi_3;   
    if (sigma_vector_3[1]< 0 and sigma_vector_3[0]< 0): azimuth_sigma_3 = 180+fi_3;   
    if (sigma_vector_3[1]< 0 and sigma_vector_3[0]>=0): azimuth_sigma_3 = 360-fi_3;   
    
    theta_sigma_3 = np.arccos(abs(sigma_vector_3[2]))*180/np.pi
    
    #--------------------------------------------------------------------------
    # azimuth and plunge of the stress axes
    #--------------------------------------------------------------------------
    direction_sigma_1 = np.array([azimuth_sigma_1, 90-theta_sigma_1])
    direction_sigma_2 = np.array([azimuth_sigma_2, 90-theta_sigma_2])
    direction_sigma_3 = np.array([azimuth_sigma_3, 90-theta_sigma_3])
    
    return direction_sigma_1, direction_sigma_2, direction_sigma_3

