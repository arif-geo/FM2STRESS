#*************************************************************************#
#                                                                         #
#  function LINEAR_STRESS_INVERSION_MICHAEL                               #
#                                                                         #
#  linear inversion for stress from focal mechansisms with randomly       #
#  selecteda faults                                                       #
#                                                                         #
#  input:  complementary focal mechanisms                                 #
#  output: stress tensor                                                  #
#                                                                         #
#  Michael, A.J., 1984. Determination of stress from slip data:           #
#  Faults and folds, J. geophys. Res., 89, 11,517-11,526.                 #
#                                                                         #
#*************************************************************************#

def linear_stress_inversion_Michael(strike1,dip1,rake1,strike2,dip2,rake2):
    
    import numpy as np
        
    N = np.size(strike1)
    
    strike = np.zeros((N))
    dip= np.zeros((N))
    rake = np.zeros((N))
    
    #--------------------------------------------------------------------------
    # focal mechanisms with randomly selected fault planes
    #--------------------------------------------------------------------------
    for i_mechanism in range (N):
    # random choice between two options
        choice = np.random.randint(2)
        
        if (choice == 1):
            strike [i_mechanism] = strike1[i_mechanism]
            dip   [i_mechanism] = dip1   [i_mechanism]
            rake  [i_mechanism] = rake1  [i_mechanism]
        else:
            strike[i_mechanism] = strike2[i_mechanism]
            dip   [i_mechanism] = dip2   [i_mechanism]
            rake  [i_mechanism] = rake2  [i_mechanism]
            
    #--------------------------------------------------------------------------
    #  fault normals and slip directions
    #--------------------------------------------------------------------------
    u1 =  np.cos(rake*np.pi/180)*np.cos(strike*np.pi/180) + np.cos(dip*np.pi/180)*np.sin(rake*np.pi/180)*np.sin(strike*np.pi/180)
    u2 =  np.cos(rake*np.pi/180)*np.sin(strike*np.pi/180) - np.cos(dip*np.pi/180)*np.sin(rake*np.pi/180)*np.cos(strike*np.pi/180)
    u3 = -np.sin(rake*np.pi/180)*np.sin(dip*np.pi/180)
        
    n1 = -np.sin(dip*np.pi/180)*np.sin(strike*np.pi/180)
    n2 =  np.sin(dip*np.pi/180)*np.cos(strike*np.pi/180)
    n3 = -np.cos(dip*np.pi/180)
    
    #--------------------------------------------------------------------------
    # inverted matrix A
    #--------------------------------------------------------------------------
    A1 = np.zeros((N,5)); A2 = np.zeros((N,5)); A3 = np.zeros((N,5))
    a_vector= np.zeros((3*N+1,1))
    
    # matrix coefficients
    A11_n =  n1*(1  -n1**2)
    A21_n =       -n1 *n2**2
    A31_n =       -n1*n3**2
    A41_n =    -2*n1*n2*n3
    A51_n =  n3*(1-2*n1**2)
    A61_n =  n2*(1-2*n1**2)
    
    A12_n =       -n2*n1**2
    A22_n =  n2*(1-  n2**2)
    A32_n =       -n2*n3**2
    A42_n =  n3*(1-2*n2**2)
    A52_n =    -2*n1*n2*n3
    A62_n =  n1*(1-2*n2**2)
    
    A13_n =       -n3*n1**2
    A23_n =       -n3*n2**2
    A33_n =  n3*(1-  n3**2)
    A43_n =  n2*(1-2*n3**2)
    A53_n =  n1*(1-2*n3**2)
    A63_n =    -2*n1*n2*n3
    
    A1 = np.transpose([A11_n, A21_n, A31_n, A41_n, A51_n, A61_n])
    A2 = np.transpose([A12_n, A22_n, A32_n, A42_n, A52_n, A62_n])
    A3 = np.transpose([A13_n, A23_n, A33_n, A43_n, A53_n, A63_n])
    
    a_vector_1 = np.zeros((N))
    a_vector_2 = np.zeros((N))
    a_vector_3 = np.zeros((N))
    
    
    for i in range (N):
        a_vector_1[i] = np.transpose(u1[i])
        a_vector_2[i] = np.transpose(u2[i])
        a_vector_3[i] = np.transpose(u3[i])
    
    A = np.r_[A1, A2, A3]
    a_vector = np.r_[a_vector_1, a_vector_2, a_vector_3]
    
    # condition for zero trace of the stress tensor
    A = np.append(A, [[1., 1., 1., 0, 0, 0]], axis = 0)
    a_vector = np.append(a_vector, [0])
    
    #--------------------------------------------------------------------------
    # generalized inversion
    # np.pinv(A) gives sometimes complex-valued numbers
    #--------------------------------------------------------------------------
    stress_vector = np.real(np.dot(np.linalg.pinv(A), a_vector))
    stress_tensor = np.r_[np.c_[stress_vector[0], stress_vector[5], stress_vector[4]],
                     np.c_[stress_vector[5], stress_vector[1], stress_vector[3]],
                     np.c_[stress_vector[4], stress_vector[3], stress_vector[2]]]
                 
    sigma  = np.linalg.eigvals(stress_tensor)
    stress = stress_tensor/max(abs(sigma))
    
    return stress 

#*************************************************************************#
#                                                                         #
#  function STABILITY_CRITERION                                           #
#                                                                         #
#  function calculates the fault instability and and identifies faults    #
#  with unstable nodal planes                                             #
#                                                                         #
#  input:  stress                                                         #
#          friction                                                       #
#          complementary focal mechanisms                                 #
#                                                                         #
#  output: focal mechanisms with correct fault orientations               #
#          instability of faults                                          #
#                                                                         #
#*************************************************************************#
def stability_criterion(tau,friction,strike1,dip1,rake1,strike2,dip2,rake2):
    
    import numpy as np
    #--------------------------------------------------------------------------
    # principal stresses
    #--------------------------------------------------------------------------
    sigma = np.sort(np.linalg.eigvals(tau))
    shape_ratio = (sigma[0]-sigma[1])/(sigma[0]-sigma[2])

    #--------------------------------------------------------------------------
    # principal stress directions
    #--------------------------------------------------------------------------
    diag_tensor, vector = np.linalg.eig(tau)
    
    value = np.linalg.eigvals(np.diag(diag_tensor))
    value_sorted=np.sort(value)
    j = np.argsort(value)
    
    sigma_vector_1  = np.array(vector[:,j[0]])
    sigma_vector_2  = np.array(vector[:,j[1]])
    sigma_vector_3  = np.array(vector[:,j[2]])
    
    #--------------------------------------------------------------------------
    #  two alternative fault normals
    #--------------------------------------------------------------------------
    # first fault normal
    n1_1 = -np.sin(dip1*np.pi/180)*np.sin(strike1*np.pi/180)
    n1_2 =  np.sin(dip1*np.pi/180)*np.cos(strike1*np.pi/180)
    n1_3 = -np.cos(dip1*np.pi/180)
    
    # second fault normal
    n2_1 = -np.sin(dip2*np.pi/180)*np.sin(strike2*np.pi/180)
    n2_2 =  np.sin(dip2*np.pi/180)*np.cos(strike2*np.pi/180)
    n2_3 = -np.cos(dip2*np.pi/180)
    
    #--------------------------------------------------------------------------
    # notation: sigma1 = 1 sigma2 = 1-2*shape_ratio sigma3 = -1
    #--------------------------------------------------------------------------
    # fault plane normals in the coordinate system of the principal stress axes
    n1_1_ = n1_1*sigma_vector_1[0] + n1_2*sigma_vector_1[1] + n1_3*sigma_vector_1[2]
    n1_2_ = n1_1*sigma_vector_2[0] + n1_2*sigma_vector_2[1] + n1_3*sigma_vector_2[2]
    n1_3_ = n1_1*sigma_vector_3[0] + n1_2*sigma_vector_3[1] + n1_3*sigma_vector_3[2]
    
    n2_1_ = n2_1*sigma_vector_1[0] + n2_2*sigma_vector_1[1] + n2_3*sigma_vector_1[2]
    n2_2_ = n2_1*sigma_vector_2[0] + n2_2*sigma_vector_2[1] + n2_3*sigma_vector_2[2]
    n2_3_ = n2_1*sigma_vector_3[0] + n2_2*sigma_vector_3[1] + n2_3*sigma_vector_3[2]
    
    #--------------------------------------------------------------------------
    # 1. alternative
    #--------------------------------------------------------------------------
    tau_shear_n1_norm   = np.sqrt(n1_1_**2+(1-2*shape_ratio)**2*n1_2_**2.+n1_3_**2-(n1_1_**2+(1-2*shape_ratio)*n1_2_**2-n1_3_**2)**2)
    tau_normal_n1_norm = (n1_1_**2+(1-2*shape_ratio)*n1_2_**2-n1_3_**2)

    #--------------------------------------------------------------------------
    # 2. alternative
    #--------------------------------------------------------------------------
    tau_shear_n2_norm   = np.sqrt(n2_1_**2+(1-2*shape_ratio)**2*n2_2_**2.+n2_3_**2-(n2_1_**2+(1-2*shape_ratio)*n2_2_**2-n2_3_**2)**2)
    tau_normal_n2_norm = (n2_1_**2+(1-2*shape_ratio)*n2_2_**2-n2_3_**2)
    
    #--------------------------------------------------------------------------
    # instability
    #--------------------------------------------------------------------------
    instability_n1 = (tau_shear_n1_norm - friction*(tau_normal_n1_norm-1))/(friction+np.sqrt(1+friction**2))
    instability_n2 = (tau_shear_n2_norm - friction*(tau_normal_n2_norm-1))/(friction+np.sqrt(1+friction**2))         

    instability = np.maximum(instability_n1, instability_n2)  
    i_index = np.zeros(instability.size)
    for i in range (instability.size):
        i_index[i] = 1 if instability_n1[i] == instability[i] else 2

    #--------------------------------------------------------------------------
    # identification of the fault according to the instability criterion
    #--------------------------------------------------------------------------
    strike = (i_index-1)*strike2+(2-i_index)*strike1
    dip    = (i_index-1)*dip2   +(2-i_index)*dip1
    rake   = (i_index-1)*rake2  +(2-i_index)*rake1 

    return strike,dip,rake,instability


#*************************************************************************#
#                                                                         #
#  function LINEAR_STRESS_INVERSION                                       #
#                                                                         #
#  linear inversion for stress from focal mechansisms                     #
#                                                                         #
#  input:  focal mechanisms                                               #
#  output: stress tensor                                                  #
#                                                                         #
#  Michael, A.J., 1984. Determination of stress from slip data:           #
#  Faults and folds, J. geophys. Res., 89, 11,517-11,526.                 #
#                                                                             #
#*************************************************************************#
def linear_stress_inversion(strike,dip,rake):
    
    import numpy as np

    N = np.size(strike)
    
    #--------------------------------------------------------------------------
    #  fault normals and slip directions
    #--------------------------------------------------------------------------
    u1 =  np.cos(rake*np.pi/180)*np.cos(strike*np.pi/180) + np.cos(dip*np.pi/180)*np.sin(rake*np.pi/180)*np.sin(strike*np.pi/180)
    u2 =  np.cos(rake*np.pi/180)*np.sin(strike*np.pi/180) - np.cos(dip*np.pi/180)*np.sin(rake*np.pi/180)*np.cos(strike*np.pi/180)
    u3 = -np.sin(rake*np.pi/180)*np.sin(dip*np.pi/180)
    
    n1 = -np.sin(dip*np.pi/180)*np.sin(strike*np.pi/180)
    n2 =  np.sin(dip*np.pi/180)*np.cos(strike*np.pi/180)
    n3 = -np.cos(dip*np.pi/180)
    
    #--------------------------------------------------------------------------
    # inverted matrix A
    #--------------------------------------------------------------------------
    A1 = np.zeros((N,5)); A2 = np.zeros((N,5)); A3 = np.zeros((N,5))
    
    # matrix coefficients
    A11_n =  n1*(1  -n1**2)
    A21_n =       -n1*n2**2
    A31_n =       -n1*n3**2
    A41_n =    -2*n1*n2*n3
    A51_n =  n3*(1-2*n1**2)
    A61_n =  n2*(1-2*n1**2)
    
    A12_n =       -n2*n1**2
    A22_n =  n2*(1-  n2**2)
    A32_n =       -n2*n3**2
    A42_n =  n3*(1-2*n2**2)
    A52_n =    -2*n1*n2*n3
    A62_n =  n1*(1-2*n2**2)
    
    A13_n =       -n3*n1**2
    A23_n =       -n3*n2**2
    A33_n =  n3*(1-  n3**2)
    A43_n =  n2*(1-2*n3**2)
    A53_n =  n1*(1-2*n3**2)
    A63_n =    -2*n1*n2*n3
    
    A1 = np.transpose([A11_n, A21_n, A31_n, A41_n, A51_n, A61_n])
    A2 = np.transpose([A12_n, A22_n, A32_n, A42_n, A52_n, A62_n])
    A3 = np.transpose([A13_n, A23_n, A33_n, A43_n, A53_n, A63_n])
    
    a_vector_1 = np.zeros((N))
    a_vector_2 = np.zeros((N))
    a_vector_3 = np.zeros((N))
        
    for i in range (N):
        a_vector_1[i] = np.transpose(u1[i])
        a_vector_2[i] = np.transpose(u2[i])
        a_vector_3[i] = np.transpose(u3[i])
    
    A = np.r_[A1, A2, A3]   
    a_vector = np.r_[a_vector_1, a_vector_2, a_vector_3]
    
    # condition for zero trace of the stress tensor
    A = np.append(A, [[1., 1., 1., 0, 0, 0]], axis = 0)
    a_vector = np.append(a_vector, [0])
    
    #--------------------------------------------------------------------------
    # generalized inversion
    # np.pinv(A) gives sometimes complex-valued numbers
    #--------------------------------------------------------------------------
    stress_vector = np.real(np.dot(np.linalg.pinv(A), a_vector))
    stress_tensor = np.r_[np.c_[stress_vector[0], stress_vector[5], stress_vector[4]],
                     np.c_[stress_vector[5], stress_vector[1], stress_vector[3]],
                     np.c_[stress_vector[4], stress_vector[3], stress_vector[2]]]
    
                 
    sigma  = np.linalg.eigvals(stress_tensor)
    stress = stress_tensor/max(abs(sigma))
   
    return stress



#*************************************************************************#
#                                                                         #
#  function STRESS_INVERSION                                              #
#                                                                         #
#  iterative inversion for stress and faults from focal mechansisms       #
#                                                                         #
#  input:  complementary focalmechanisms                                  #
#          limits for friction                                            #
#                                                                         #
#  output: stress tensor tau                                              #
#          shape ratio R                                                  #
#          focal mechanisms with correct fault orientations               #
#          fault instability                                              #
#          optimum friction                                               #
#                                                                         #
#*************************************************************************#


def stress_inversion(strike1,dip1,rake1,strike2,dip2,rake2,friction_min,friction_max,friction_step,N_iterations,N_realizations):
    
    import numpy as np
    
    #--------------------------------------------------------------------------
    # initial guess of the stress tensor unp.sing the Michael method (1984)
    #--------------------------------------------------------------------------
    tau = np.zeros((3,3))
    for i_realization in range (N_realizations):
        tau_realization = linear_stress_inversion_Michael(strike1,dip1,rake1,strike2,dip2,rake2)
        tau = tau + tau_realization

    tau0 = tau/np.linalg.norm(tau,2)
    
    #--------------------------------------------------------------------------
    # loop over frictions
    #--------------------------------------------------------------------------
    i_friction = 0
    irange = np.arange(friction_min, friction_max, friction_step)
    norm_difference_tau = np.zeros(N_iterations)
    mean_instability = np.zeros(irange.size)
    
    friction_ = np.zeros(irange.size)
    for friction in irange :
        friction_[i_friction] = friction
    #--------------------------------------------------------------------------
    #  loop over iterations
    #--------------------------------------------------------------------------
        for i_iteration in range(N_iterations):
        
            # calculation of the fault instability
            strike, dip, rake, instability = stability_criterion(tau0,friction,strike1,dip1,rake1,strike2,dip2,rake2)
            tau = linear_stress_inversion(strike,dip,rake)

            # check of convergency
            norm_difference_tau[i_iteration] = np.linalg.norm(tau-tau0,2)          
            tau0 = tau

        mean_instability[i_friction] = np.mean(instability)
        i_friction = i_friction+1

    instability_max = np.amax(mean_instability)
    i_index = np.where(mean_instability == np.amax(mean_instability))
    
    # optimum friction
    friction_optimum = friction_[i_index]
    
    #--------------------------------------------------------------------------
    # final inversion with optimum friction
    #--------------------------------------------------------------------------
    #  loop over iterations
    for i_iteration in range (N_iterations):
        
        # calculation of the fault instability and fault orientations
        strike,dip,rake,instability = stability_criterion(tau0,friction_optimum,strike1,dip1,rake1,strike2,dip2,rake2)
        # inversion for stress with correct faults
        tau = linear_stress_inversion(strike, dip, rake)
    
        # check of convergency
        norm_difference_tau[i_iteration] = np.linalg.norm(tau-tau0,2)
                
        tau0 = tau
    
    #--------------------------------------------------------------------------
    # resultant stress tenor
    #--------------------------------------------------------------------------
    diag_tensor, vector = np.linalg.eig(tau)
    
    value = np.linalg.eigvals(np.diag(diag_tensor))
    value_sorted=np.sort(value)
    j = np.argsort(value)
    
    sigma_vector_1  = np.array(vector[:,j[0]])
    sigma_vector_2  = np.array(vector[:,j[1]])
    sigma_vector_3  = np.array(vector[:,j[2]])


    sigma = np.sort(np.linalg.eigvals(tau))
    shape_ratio = (sigma[0]-sigma[1])/(sigma[0]-sigma[2])

    return tau,shape_ratio,strike,dip,rake,instability,friction_optimum
