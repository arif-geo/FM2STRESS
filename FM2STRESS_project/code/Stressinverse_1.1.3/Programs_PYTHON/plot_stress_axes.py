#*************************************************************************#
#                                                                         #
#  function PLOT_STRESS_AXES                                              #
#                                                                         #
#  plot of the principal stress axes into the focal sphere                #
#                                                                         #
#  input: principal stress directions                                     #
#         name of figure                                                  #
#                                                                         #
#*************************************************************************#
def plot_stress_axes(sigma_vector_1_,sigma_vector_2_,sigma_vector_3_,plot_file):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    N = np.size(sigma_vector_1_[1,:])
    
    #--------------------------------------------------------------------------
    # loop over individual solutions
    #--------------------------------------------------------------------------
    x_sigma_1 = np.zeros(N); x_sigma_2 = np.zeros(N); x_sigma_3 = np.zeros(N)
    y_sigma_1 = np.zeros(N); y_sigma_2 = np.zeros(N); y_sigma_3 = np.zeros(N)
    
    for i in range (N):
        
        sigma_vector_1 = sigma_vector_1_[:,i]
        sigma_vector_2 = sigma_vector_2_[:,i]
        sigma_vector_3 = sigma_vector_3_[:,i]
        
        if (sigma_vector_1[2]<0): sigma_vector_1 = -sigma_vector_1; 
        if (sigma_vector_2[2]<0): sigma_vector_2 = -sigma_vector_2; 
        if (sigma_vector_3[2]<0): sigma_vector_3 = -sigma_vector_3; 
        
        #----------------------------------------------------------------------
        # 1. stress axis
        fi_1 = np.arctan(np.abs(sigma_vector_1[1]/sigma_vector_1[0]))*180/np.pi
        
        if (sigma_vector_1[1]>=0 and sigma_vector_1[0]>=0): azimuth_sigma_1 =     fi_1; 
        if (sigma_vector_1[1]>=0 and sigma_vector_1[0]< 0): azimuth_sigma_1 = 180-fi_1; 
        if (sigma_vector_1[1]< 0 and sigma_vector_1[0]< 0): azimuth_sigma_1 = 180+fi_1; 
        if (sigma_vector_1[1]< 0 and sigma_vector_1[0]>=0): azimuth_sigma_1 = 360-fi_1; 
        
        theta_sigma_1 = np.arccos(np.abs(sigma_vector_1[2]))*180/np.pi
        
        #----------------------------------------------------------------------
        # 2. stress axis
        fi_2 = np.arctan(np.abs(sigma_vector_2[1]/sigma_vector_2[0]))*180/np.pi
        
        if (sigma_vector_2[1]>=0 and sigma_vector_2[0]>=0): azimuth_sigma_2 =     fi_2; 
        if (sigma_vector_2[1]>=0 and sigma_vector_2[0]< 0): azimuth_sigma_2 = 180-fi_2; 
        if (sigma_vector_2[1]< 0 and sigma_vector_2[0]< 0): azimuth_sigma_2 = 180+fi_2; 
        if (sigma_vector_2[1]< 0 and sigma_vector_2[0]>=0): azimuth_sigma_2 = 360-fi_2; 
        
        theta_sigma_2 = np.arccos(np.abs(sigma_vector_2[2]))*180/np.pi
        
        #----------------------------------------------------------------------
        # 3. stress axis
        fi_3 = np.arctan(np.abs(sigma_vector_3[1]/sigma_vector_3[0]))*180/np.pi
        
        if (sigma_vector_3[1]>=0 and sigma_vector_3[0]>=0): azimuth_sigma_3 =     fi_3; 
        if (sigma_vector_3[1]>=0 and sigma_vector_3[0]< 0): azimuth_sigma_3 = 180-fi_3; 
        if (sigma_vector_3[1]< 0 and sigma_vector_3[0]< 0): azimuth_sigma_3 = 180+fi_3; 
        if (sigma_vector_3[1]< 0 and sigma_vector_3[0]>=0): azimuth_sigma_3 = 360-fi_3; 
        
        theta_sigma_3 = np.arccos(np.abs(sigma_vector_3[2]))*180/np.pi
        
        #----------------------------------------------------------------------
        # projection into the lower hemisphere
        projection_1 = 1; projection_2 = 1; projection_3 = 1;
            
        #----------------------------------------------------------------------
        #  zenithal equal-area projection
        
        radius_sigma_1 = projection_1*np.sin(theta_sigma_1*np.pi/360.)
        radius_sigma_2 = projection_2*np.sin(theta_sigma_2*np.pi/360.)
        radius_sigma_3 = projection_3*np.sin(theta_sigma_3*np.pi/360.)
        
        x_sigma_1[i] = np.sqrt(2.)*radius_sigma_1*np.cos(azimuth_sigma_1*np.pi/180.)
        y_sigma_1[i] = np.sqrt(2.)*radius_sigma_1*np.sin(azimuth_sigma_1*np.pi/180.)
        
        x_sigma_2[i] = np.sqrt(2.)*radius_sigma_2*np.cos(azimuth_sigma_2*np.pi/180.)
        y_sigma_2[i] = np.sqrt(2.)*radius_sigma_2*np.sin(azimuth_sigma_2*np.pi/180.)
        
        x_sigma_3[i] = np.sqrt(2.)*radius_sigma_3*np.cos(azimuth_sigma_3*np.pi/180.)
        y_sigma_3[i] = np.sqrt(2.)*radius_sigma_3*np.sin(azimuth_sigma_3*np.pi/180.)
    
    #--------------------------------------------------------------------------
    # plotting the stress directions in the focal sphere
    #--------------------------------------------------------------------------
    plotSA, axSA = plt.subplots()
    
    plt.title('Confidence of principal stress axes',fontsize = 16);
    axSA.axis ('equal')
    axSA.axis([-1.05, 1.70, -1.05, 1.05])
    axSA.axis('off')
    
    
    #--------------------------------------------------------------------------
    # plotting the stress directions
    sig1, = plt.plot(y_sigma_1,x_sigma_1,'r.', markersize = 20);	# sigma_1
    sig2, = plt.plot(y_sigma_2,x_sigma_2,'g.', markersize = 20);	# sigma_2
    sig3, = plt.plot(y_sigma_3,x_sigma_3,'b.', markersize = 20);	# sigma_3
       
    #--------------------------------------------------------------------------
    # boundary circle and the centre of the circle
    fi = np.arange(0, 360, 0.1)		
    plt.plot(np.cos(fi*np.pi/180.),np.sin(fi*np.pi/180.),'k-', linewidth = 2.0)
    plt.plot(0,0,'k+', markersize = 10);		
    
    #--------------------------------------------------------------------------
    # grid lines - constant theta
    theta_grid_i = np.arange(0, 90, 15)
    for theta_grid in (theta_grid_i) :
        radius_grid = projection_1*np.sin(theta_grid*np.pi/360.)
        
        x_grid = np.sqrt(2.)*radius_grid*np.cos(fi*np.pi/180.)
        y_grid = np.sqrt(2.)*radius_grid*np.sin(fi*np.pi/180.)
    
        plt.plot(y_grid,x_grid,'k:', linewidth = 0.5)
    
    #--------------------------------------------------------------------------
    # grid lines - constant fi
    theta_grid = np.arange(0, 105, 15)
    
    fi_grid_i = np.arange(0,360,15)
    for fi_grid in (fi_grid_i) :
        radius_grid = projection_1*np.sin(theta_grid*np.pi/360.)
        
        x_grid = np.sqrt(2.)*radius_grid*np.cos(fi_grid*np.pi/180.)
        y_grid = np.sqrt(2.)*radius_grid*np.sin(fi_grid*np.pi/180.)
    
        plt.plot(y_grid,x_grid,'k:', linewidth = 0.5);
    
    #--------------------------------------------------------------------------
    # legend
    axSA.legend((sig1,sig2,sig3), ('sigma 1','sigma 2','sigma 3'), loc = 'lower right', fontsize = 14, numpoints=1)
        
    #--------------------------------------------------------------------------
    # saving the plot
    #--------------------------------------------------------------------------
    plt.savefig(plot_file + '.png')
    plt.close()
 
