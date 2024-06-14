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
    value_sorted = np.sort(value)
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
    # 1. eigenvector
    #--------------------------------------------------------------------------
    fi_1 = np.arctan(np.abs(sigma_vector_1[1]/sigma_vector_1[0]))*180/np.pi
    
    if (sigma_vector_1[1]>=0 and sigma_vector_1[0]>=0): azimuth_sigma_1 =     fi_1;  
    if (sigma_vector_1[1]>=0 and sigma_vector_1[0]< 0): azimuth_sigma_1 = 180-fi_1; 
    if (sigma_vector_1[1]< 0 and sigma_vector_1[0]< 0): azimuth_sigma_1 = 180+fi_1; 
    if (sigma_vector_1[1]< 0 and sigma_vector_1[0]>=0): azimuth_sigma_1 = 360-fi_1; 
    
    theta_sigma_1 = np.arccos(np.abs(sigma_vector_1[2]))*180/np.pi;
        
    #--------------------------------------------------------------------------
    # 2. eigenvector
    #--------------------------------------------------------------------------
    fi_2 = np.arctan(np.abs(sigma_vector_2[1]/sigma_vector_2[0]))*180/np.pi
    
    if (sigma_vector_2[1]>=0 and sigma_vector_2[0]>=0): azimuth_sigma_2 =     fi_2; 
    if (sigma_vector_2[1]>=0 and sigma_vector_2[0]< 0): azimuth_sigma_2 = 180-fi_2; 
    if (sigma_vector_2[1]< 0 and sigma_vector_2[0]< 0): azimuth_sigma_2 = 180+fi_2; 
    if (sigma_vector_2[1]< 0 and sigma_vector_2[0]>=0): azimuth_sigma_2 = 360-fi_2; 
    
    theta_sigma_2 = np.arccos(np.abs(sigma_vector_2[2]))*180/np.pi
    
    #--------------------------------------------------------------------------
    # 3. eigenvector
    #--------------------------------------------------------------------------
    fi_3 = np.arctan(np.abs(sigma_vector_3[1]/sigma_vector_3[0]))*180/np.pi
    
    if (sigma_vector_3[1]>=0 and sigma_vector_3[0]>=0): azimuth_sigma_3 =     fi_3; 
    if (sigma_vector_3[1]>=0 and sigma_vector_3[0]< 0): azimuth_sigma_3 = 180-fi_3; 
    if (sigma_vector_3[1]< 0 and sigma_vector_3[0]< 0): azimuth_sigma_3 = 180+fi_3; 
    if (sigma_vector_3[1]< 0 and sigma_vector_3[0]>=0): azimuth_sigma_3 = 360-fi_3; 
    
    theta_sigma_3 = np.arccos(np.abs(sigma_vector_3[2]))*180/np.pi
    
    #--------------------------------------------------------------------------
    # azimuth and plunge of the stress axes
    #--------------------------------------------------------------------------
    direction_sigma_1 = np.array([azimuth_sigma_1, 90-theta_sigma_1])
    direction_sigma_2 = np.array([azimuth_sigma_2, 90-theta_sigma_2])
    direction_sigma_3 = np.array([azimuth_sigma_3, 90-theta_sigma_3])
    
    return direction_sigma_1, direction_sigma_2, direction_sigma_3

#*************************************************************************#
#                                                                         # 
#   function PLOT_P_T_AXES                                                #
#                                                                         #
#   function plots P and T axes in the focal sphere                       #
#                                                                         #
#*************************************************************************#
def plot_P_T_axes(strike,dip,rake):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_1 = np.zeros([np.size(strike),3])
    u_1 = np.zeros([np.size(strike),3])
    
    n_1[:,0] = -np.sin(dip*np.pi/180)*np.sin(strike*np.pi/180)
    n_1[:,1] =  np.sin(dip*np.pi/180)*np.cos(strike*np.pi/180)
    n_1[:,2] = -np.cos(dip*np.pi/180)
    
    u_1[:,0] =  np.cos(rake*np.pi/180)*np.cos(strike*np.pi/180) + np.cos(dip*np.pi/180)*np.sin(rake*np.pi/180)*np.sin(strike*np.pi/180)
    u_1[:,1] =  np.cos(rake*np.pi/180)*np.sin(strike*np.pi/180) - np.cos(dip*np.pi/180)*np.sin(rake*np.pi/180)*np.cos(strike*np.pi/180)
    u_1[:,2] = -np.sin(rake*np.pi/180)*np.sin(dip*np.pi/180)
    
    N = np.size(strike)
    #--------------------------------------------------------------------------
    # lower hemisphere equal-area projection
    #--------------------------------------------------------------------------
    projection = -1  
    
    #--------------------------------------------------------------------------
    # P/T axes
    #--------------------------------------------------------------------------
    P_osa = np.zeros([N,3])
    T_osa = np.zeros([N,3])
    
    P_azimuth = np.zeros(N)
    T_azimuth = np.zeros(N)
    
    P_theta = np.zeros(N)
    T_theta = np.zeros(N)
    
    P_x = np.zeros(N)
    P_y = np.zeros(N)
    T_x = np.zeros(N)
    T_y = np.zeros(N)
    
    for i in range(N):
        P_osa[i,:] = (n_1[i,:]-u_1[i,:])/np.linalg.norm(n_1[i,:]-u_1[i,:],2)
        T_osa[i,:] = (n_1[i,:]+u_1[i,:])/np.linalg.norm(n_1[i,:]+u_1[i,:],2)
        
        if (P_osa[i,2]>0): P_osa[i,0]=-P_osa[i,0]; P_osa[i,1]=-P_osa[i,1]; P_osa[i,2]=-P_osa[i,2]; 
        if (T_osa[i,2]>0): T_osa[i,0]=-T_osa[i,0]; T_osa[i,1]=-T_osa[i,1]; T_osa[i,2]=-T_osa[i,2]; 
    
        fi = np.arctan(np.abs(P_osa[i,0]/P_osa[i,1]))*180/np.pi
    
        if (P_osa[i,0]>0 and P_osa[i,1]>0): P_azimuth[i] = fi;       # 1. kvadrant
        if (P_osa[i,0]>0 and P_osa[i,1]<0): P_azimuth[i] = 180-fi;   # 2. kvadrant
        if (P_osa[i,0]<0 and P_osa[i,1]<0): P_azimuth[i] = fi+180;   # 3. kvadrant
        if (P_osa[i,0]<0 and P_osa[i,1]>0): P_azimuth[i] = 360-fi;   # 4. kvadrant
    
        P_theta[i] = np.arccos(np.abs(P_osa[i,2]))*180/np.pi
    
        fi = np.arctan(np.abs(T_osa[i,0]/T_osa[i,1]))*180/np.pi
        
        if (T_osa[i,0]>0 and T_osa[i,1]>0): T_azimuth[i] = fi;       # 1. kvadrant
        if (T_osa[i,0]>0 and T_osa[i,1]<0): T_azimuth[i] = 180-fi;   # 2. kvadrant
        if (T_osa[i,0]<0 and T_osa[i,1]<0): T_azimuth[i] = fi+180;   # 3. kvadrant
        if (T_osa[i,0]<0 and T_osa[i,1]>0): T_azimuth[i] = 360-fi;   # 4. kvadrant
            
        T_theta[i] = np.arccos(np.abs(T_osa[i,2]))*180/np.pi
            
        P_x[i] = np.sqrt(2.)*projection*np.sin(P_theta[i]*np.pi/360)*np.sin(P_azimuth[i]*np.pi/180)
        P_y[i] = np.sqrt(2.)*projection*np.sin(P_theta[i]*np.pi/360)*np.cos(P_azimuth[i]*np.pi/180)
    
        T_x[i] = np.sqrt(2.)*projection*np.sin(T_theta[i]*np.pi/360)*np.sin(T_azimuth[i]*np.pi/180)
        T_y[i] = np.sqrt(2.)*projection*np.sin(T_theta[i]*np.pi/360)*np.cos(T_azimuth[i]*np.pi/180)
    
    
    plt.plot(P_y,P_x,'ro', markeredgecolor='r', markerfacecolor='none', markersize=8, markeredgewidth=1.5)
    plt.plot(T_y,T_x,'b+', markersize=8, markeredgewidth=1.5)


#*************************************************************************#
#                                                                         #
#  function PLOT_STRESS                                                   #
#                                                                         #
#  plot of the optimum stess orientation                                  #
#                                                                         #
#  input: stress tensor                                                   #
#         focal mechanisms (for plotting the P/T axes)                    #
#                                                                         #
#*************************************************************************#
def plot_stress(tau,strike,dip,rake,plot_file):
    
    import matplotlib.pyplot as plt
    import numpy as np

    direction_sigma_1, direction_sigma_2, direction_sigma_3 = azimuth_plunge(tau)
    
    azimuth_sigma_1 = direction_sigma_1[0]; plunge_sigma_1 = direction_sigma_1[1]; 
    azimuth_sigma_2 = direction_sigma_2[0]; plunge_sigma_2 = direction_sigma_2[1]; 
    azimuth_sigma_3 = direction_sigma_3[0]; plunge_sigma_3 = direction_sigma_3[1]; 
    
    theta_sigma_1 = 90-plunge_sigma_1
    theta_sigma_2 = 90-plunge_sigma_2
    theta_sigma_3 = 90-plunge_sigma_3
    
    #--------------------------------------------------------------------------
    # plotting the stress directions in the focal sphere
    #--------------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.set_title('Principal stress and P/T axes',fontsize = 16)
    
    ax.axis('equal')
    ax.axis([-1.05,  1.70, -1.05, 1.05])
    ax.axis('off')
    #ax.axis()
    
    #--------------------------------------------------------------------------
    # projection into the lower hemisphere
    projection_1 = 1; projection_2 = 1; projection_3 = 1;
    
    #--------------------------------------------------------------------------
    #  zenithal equal-area projection
    
    radius_sigma_1 = projection_1*np.sin(theta_sigma_1*np.pi/360.)
    radius_sigma_2 = projection_2*np.sin(theta_sigma_2*np.pi/360.)
    radius_sigma_3 = projection_3*np.sin(theta_sigma_3*np.pi/360.)
    
    x_sigma_1 = np.sqrt(2.)*radius_sigma_1*np.cos(azimuth_sigma_1*np.pi/180.)
    y_sigma_1 = np.sqrt(2.)*radius_sigma_1*np.sin(azimuth_sigma_1*np.pi/180.)
    
    x_sigma_2 = np.sqrt(2.)*radius_sigma_2*np.cos(azimuth_sigma_2*np.pi/180.)
    y_sigma_2 = np.sqrt(2.)*radius_sigma_2*np.sin(azimuth_sigma_2*np.pi/180.)
    
    x_sigma_3 = np.sqrt(2.)*radius_sigma_3*np.cos(azimuth_sigma_3*np.pi/180.)
    y_sigma_3 = np.sqrt(2.)*radius_sigma_3*np.sin(azimuth_sigma_3*np.pi/180.)
    
    #--------------------------------------------------------------------------
    # P/T axes
    #--------------------------------------------------------------------------
    plot_P_T_axes(strike,dip,rake)
    
    #--------------------------------------------------------------------------
    # boundary circle and the centre of the circle
    #--------------------------------------------------------------------------
    fi = np.arange(0,360, 0.1)
    plt.plot(np.cos(fi*np.pi/180.),np.sin(fi*np.pi/180.),'k-', linewidth = 2.0)
    plt.plot(0,0,'k+', markersize = 10);
    
    sig1, = plt.plot(y_sigma_1,x_sigma_1,'go', markersize = 12, markeredgewidth = 2.5, 
                markeredgecolor='g'); # sigma_1
    sig2, = plt.plot(y_sigma_2,x_sigma_2,'gx', markersize = 13, markeredgewidth = 2.5); # sigma_2
    sig3, = plt.plot(y_sigma_3,x_sigma_3,'g+', markersize = 13, markeredgewidth = 2.7); # sigma_3
    
    #--------------------------------------------------------------------------
    # legend
    #--------------------------------------------------------------------------
    plt.legend((sig1,sig2,sig3), ('sigma 1','sigma 2','sigma 3'), loc='lower right', fontsize = 14, numpoints=1)
      
    #--------------------------------------------------------------------------
    # saving the plot
    #--------------------------------------------------------------------------
    plt.savefig(plot_file + '.png')#, format = 'png');
    plt.close()


