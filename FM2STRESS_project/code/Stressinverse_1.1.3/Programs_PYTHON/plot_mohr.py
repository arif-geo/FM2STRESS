#*************************************************************************#
#                                                                         #
#  function PLOT_MOHR                                                     #
#                                                                         #
#  plot of fault planes into the Mohr circle diagram                      #
#                                                                         #
#  input: stress tensor                                                   #
#         focal mechansisms                                               #
#                                                                         #
#*************************************************************************#
def plot_mohr(tau,strike,dip,rake,principal_strike,principal_dip,principal_rake,plot_file):
    
    import matplotlib.pyplot as plt
    import numpy as np

    #--------------------------------------------------------------------------
    # eigenvaluea and eigenvectors of the stress tensor
    #--------------------------------------------------------------------------
    diag_tensor, vector = np.linalg.eig(tau)
    
    value = np.linalg.eigvals(np.diag(diag_tensor))
    value_sorted=np.sort(value)
    j = np.argsort(value)
   
    sigma_vector_1 = vector[:,j[0]]
    sigma_vector_2 = vector[:,j[1]]
    sigma_vector_3 = vector[:,j[2]]
    
    if (sigma_vector_1[2]<0): sigma_vector_1 = -sigma_vector_1;
    if (sigma_vector_2[2]<0): sigma_vector_2 = -sigma_vector_2; 
    if (sigma_vector_3[2]<0): sigma_vector_3 = -sigma_vector_3;
    
    sigma = np.sort(np.linalg.eigvals(tau))
    shape_ratio = (sigma[0]-sigma[1])/(sigma[0]-sigma[2])
    
    #--------------------------------------------------------------------------
    # Mohr's circles
    #--------------------------------------------------------------------------
    mohr, axM  = plt.subplots()
    axM.set_title('Mohr circle diagram',fontsize = 14)
    axM.invert_xaxis()
    axM.axis('equal')
    axM.axis('off')
    
    fi= np.arange(0, 360, 0.1)
    
    #--------------------------------------------------------------------------
    # 1. circle
    stred1 = (sigma[1]+sigma[0])/2
    r1 = np.abs(sigma[1]-sigma[0])/2
    x1 = r1*np.cos(fi*np.pi/180.)+stred1
    y1 = r1*np.sin(fi*np.pi/180.)
    
    plt.plot(x1,y1,'k', linewidth = 1.5)
    
    #--------------------------------------------------------------------------
    # 2. circle
    stred2 = (sigma[1]+sigma[2])/2
    r2 = np.abs(sigma[2]-sigma[1])/2
    x2 = r2*np.cos(fi*np.pi/180.)+stred2
    y2 = r2*np.sin(fi*np.pi/180.)
    
    plt.plot(x2,y2,'k', linewidth = 1.5)
    
    #--------------------------------------------------------------------------
    # 3. circle
    stred3 = (sigma[0]+sigma[2])/2
    r3 = np.abs(sigma[2]-sigma[0])/2
    x3 = r3*np.cos(fi*np.pi/180.)+stred3
    y3 = r3*np.sin(fi*np.pi/180.)
    
    plt.plot(x3,y3,'k', linewidth = 1.5)
    
    #--------------------------------------------------------------------------
    # axis 
    plt.plot ([sigma[0], sigma[2]],[0, 0],'k', linewidth = 1.0)
    
    #--------------------------------------------------------------------------
    # fault normals
    #--------------------------------------------------------------------------
    n1 = -np.sin(dip*np.pi/180)*np.sin(strike*np.pi/180)
    n2 =  np.sin(dip*np.pi/180)*np.cos(strike*np.pi/180)
    n3 = -np.cos(dip*np.pi/180)
    
    #--------------------------------------------------------------------------
    # principal fault normals
    #--------------------------------------------------------------------------
    n_principal_1 = -np.sin(principal_dip*np.pi/180)*np.sin(principal_strike*np.pi/180)
    n_principal_2 =  np.sin(principal_dip*np.pi/180)*np.cos(principal_strike*np.pi/180)
    n_principal_3 = -np.cos(principal_dip*np.pi/180)
    
    #--------------------------------------------------------------------------
    # shear and normal stresses 
    #--------------------------------------------------------------------------
    tau_normal = tau[0,0]*n1*n1 + tau[0,1]*n1*n2 + tau[0,2]*n1*n3 \
    + tau[1,0]*n2*n1 + tau[1,1]*n2*n2 + tau[1,2]*n2*n3 \
    + tau[2,0]*n3*n1 + tau[2,1]*n3*n2 + tau[2,2]*n3*n3
    
    tau_normal_square = tau_normal*tau_normal;
    
    tau_total_square   = (tau[0,0]*n1 + tau[0,1]*n2 + tau[0,2]*n3)**2 \
    + (tau[1,0]*n1 + tau[1,1]*n2 + tau[1,2]*n3)**2 \
    + (tau[2,0]*n1 + tau[2,1]*n2 + tau[2,2]*n3)**2
    
    tau_shear_square   = tau_total_square - tau_normal_square 
       
    tau_shear  = np.sqrt(tau_shear_square)
    tau_total  = np.sqrt(tau_total_square)
    
    #--------------------------------------------------------------------------
    # identification of the half-plane
    # updated calculation unp.sing principal focal mechanisms
    #--------------------------------------------------------------------------
    # deviation of the fault normal from the 1. principal fault
    deviation_principal_1 = np.arccos(np.abs(n1*n_principal_1[0]+n2*n_principal_2[0]+n3*n_principal_3[0]))*180/np.pi
    
    # deviation of the fault normal from the 2. principal fault
    deviation_principal_2 = np.arccos(np.abs(n1*n_principal_1[1]+n2*n_principal_2[1]+n3*n_principal_3[1]))*180/np.pi
    
    min_deviation = np.minimum(deviation_principal_1,deviation_principal_2)#,axis = 1)    
    half_space = np.zeros(min_deviation.size)
    for i in range (min_deviation.size):
        half_space[i] = 1 if min_deviation[i] == deviation_principal_1[i] else 2

    tau_shear = (half_space==1)*tau_shear - (half_space==2)*tau_shear
    
    #--------------------------------------------------------------------------
    # plotting the fault normals
    #--------------------------------------------------------------------------
    plt.plot(tau_normal,tau_shear,'b+', markersize = 9, markeredgewidth = 1.5)
    
    # scaling of the figure
    #v = axis; axis(1.1*v)
    
    #--------------------------------------------------------------------------
    # saving the plot
    #--------------------------------------------------------------------------
    plt.savefig(plot_file + '.png')
    plt.close()
    
