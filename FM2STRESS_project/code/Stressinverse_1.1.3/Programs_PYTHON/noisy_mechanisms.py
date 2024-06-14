#*************************************************************************#
#                                                                         #
#   function NOISY_NORMAL_SLIP                                            #
#                                                                         #
#   generating noisy fault normals and slip directions                    #
#                                                                         #
#   input: fault normal n, slip direction u                               #
#          mean deviation                                                 #
#                                                                         #
#*************************************************************************#
def noisy_normal_slip(mean_deviation,n,u):
    
    import numpy as np
    
    b = np.cross(n,u)
    #--------------------------------------------------------------------------
    # noisy fault normal
    #--------------------------------------------------------------------------
    # uniform distribution
    #    deviation_random = random('unif',0,2*mean_deviation); 
    # normal distribution
    deviation_random = mean_deviation*np.random.randn()
    
    # uniform distribution of azimuth in the interval (0,360)    
    azimuth_random   = 360*np.random.random()
    
        
    n_perpendicular = u*np.sin(azimuth_random*np.pi/180)+b*np.cos(azimuth_random*np.pi/180)
    n_noise_component = n_perpendicular*np.sin(deviation_random*np.pi/180)
        
    n_noisy = n + n_noise_component
    n_noisy = n_noisy/np.linalg.norm(n_noisy,2)
        
    n_deviation_ = np.arccos(np.dot(n,n_noisy))*180/np.pi
        
    #--------------------------------------------------------------------------
    # noisy slip direction, uniform distribution
    #--------------------------------------------------------------------------
    # uniform distribution
    #    deviation_random = random('unif',0,2*mean_deviation); 
    # normal distribution
    deviation_random = 0.95*mean_deviation*np.random.randn()
    
    # uniform distribution of azimuth in the interval (0,360)    
    azimuth_random   = 360*np.random.random()
    
    b_perpendicular = n_noisy*np.sin(azimuth_random*np.pi/180)+u*np.cos(azimuth_random*np.pi/180)
    b_noise_component = b_perpendicular*np.sin(deviation_random*np.pi/180)
        
    b_noisy = b + b_noise_component
    b_noisy = b_noisy/np.linalg.norm(b_noisy,2)
        
    b_deviation_ = np.arccos(np.dot(b,b_noisy))*180/np.pi
        
    # u_noisy must be perpendicular to n_noisy
    u_noisy = np.cross(n_noisy,b_noisy)
        
    # direction must be similar to the original u
    u_noisy = np.sign(np.dot(u,u_noisy))*u_noisy
                  
    return n_noisy,u_noisy


#*************************************************************************#
#                                                                         #
#   function STRIKE_DIP_RAKE                                              #
#                                                                         #
#   calculation of strike, dip and rake from the fault normals and slip   #
#   directions                                                            #
#                                                                         #
#   input:  fault normal n                                                #
#           slip direction u                                              #
#                                                                         #
#   output: strike, dip and rake                                          #
#                                                                         #
#*************************************************************************#
def strike_dip_rake(n,u):
    
    import numpy as np
    
    n1 = n
    u1 = u
    	
    if (n1[2]>0): 
        n1 = -n1; u1 = -u1;  # vertical component is always negative!
    #if (n1(3)>0) n1 = -n1; end; # vertical component is always negative!
            
    n2 = u
    u2 = n
        
    if (n2[2]>0): 
        n2 = -n2; u2 = -u2;   # vertical component is always negative!
    
    ## ------------------------------------------------------------------------
    # 1st solution
    #--------------------------------------------------------------------------
    dip    = np.arccos(-n1[2])*180/np.pi
    strike = np.arcsin(-n1[0]/np.sqrt(n1[0]**2+n1[1]**2))*180/np.pi
    
    # determination of a quadrant
    if (n1[1]<0): strike=180-strike;
    
    rake = np.arcsin(-u1[2]/np.sin(dip*np.pi/180))*180/np.pi
    
    # determination of a quadrant
    cos_rake = u1[0]*np.cos(strike*np.pi/180)+u1[1]*np.sin(strike*np.pi/180)
    if (cos_rake<0): rake=180-rake;
    
    if (strike<0   ): strike = strike+360; 
    if (rake  <-180): rake   = rake  +360; 
    if (rake  > 180): rake   = rake  -360;   # rake is in the interval -180<rake<180
        
    strike1 = np.real(strike); dip1 = np.real(dip); rake1 = np.real(rake);
    
    ## ------------------------------------------------------------------------
    # 2nd solution
    #--------------------------------------------------------------------------
    dip    = np.arccos(-n2[2])*180/np.pi
    strike = np.arcsin(-n2[0]/np.sqrt(n2[0]**2+n2[1]**2))*180/np.pi
    
    # determination of a quadrant
    if (n2[1]<0): strike=180-strike; 
    
    rake = np.arcsin(-u2[2]/np.sin(dip*np.pi/180))*180/np.pi
    
    # determination of a quadrant
    cos_rake = u2[0]*np.cos(strike*np.pi/180)+u2[1]*np.sin(strike*np.pi/180)
    if (cos_rake<0): rake=180-rake; 
    
    if (strike<0   ): strike = strike+360; 
    if (rake  <-180): rake   = rake  +360; 
    if (rake  > 180): rake   = rake  -360;   
        
    strike2 = np.real(strike); dip2 = np.real(dip); rake2 = np.real(rake);

    return strike1,dip1,rake1,strike2,dip2,rake2



#*************************************************************************#
#                                                                         #
#   function NOISY_MECHANISMS                                             #
#                                                                         #
#   generating noisy mechanisms                                           #
#                                                                         #
#   input: strke, dip and rake                                            #
#          noise_level                                                    #
#                                                                         #
#*************************************************************************#
def noisy_mechanisms(mean_deviation,strike,dip,rake):
    
    import numpy as np

    N = np.size(strike)
    
    #--------------------------------------------------------------------------
    # loop over focal mechanisms
    #--------------------------------------------------------------------------
    n = np.zeros(3)
    u = np.zeros(3)
    
    strike1_noisy = np.zeros(N)
    dip1_noisy =np.zeros(N)
    rake1_noisy = np.zeros(N)
    strike2_noisy = np.zeros(N)
    dip2_noisy = np.zeros(N)
    rake2_noisy = np.zeros(N)
    
    n_error = np.zeros(N)
    u_error = np.zeros(N)
    
    for i in range(N):
    
    #--------------------------------------------------------------------------
    #  fault normal and slip direction
    #--------------------------------------------------------------------------
        n[0] = -np.sin(dip[i]*np.pi/180)*np.sin(strike[i]*np.pi/180)
        n[1] =  np.sin(dip[i]*np.pi/180)*np.cos(strike[i]*np.pi/180)
        n[2] = -np.cos(dip[i]*np.pi/180)
    
        u[0] =  np.cos(rake[i]*np.pi/180)*np.cos(strike[i]*np.pi/180) + np.cos(dip[i]*np.pi/180)*np.sin(rake[i]*np.pi/180)*np.sin(strike[i]*np.pi/180)
        u[1] =  np.cos(rake[i]*np.pi/180)*np.sin(strike[i]*np.pi/180) - np.cos(dip[i]*np.pi/180)*np.sin(rake[i]*np.pi/180)*np.cos(strike[i]*np.pi/180)
        u[2] = -np.sin(rake[i]*np.pi/180)*np.sin(dip[i]*np.pi/180)
             
    # superposition of noise, the procedure is symmtric 
    # with respect to the fault normal and slip direction    
        if (np.mod(i,2) == 1):
            n_noisy,u_noisy = noisy_normal_slip(mean_deviation,n,u)
        else:
            u_noisy,n_noisy = noisy_normal_slip(mean_deviation,u,n)
        
        
        strike1,dip1,rake1,strike2,dip2,rake2 = strike_dip_rake(n_noisy,u_noisy)        
        
        strike1_noisy[i] = strike1; dip1_noisy[i] = dip1; rake1_noisy[i] = rake1;
        strike2_noisy[i] = strike2; dip2_noisy[i] = dip2; rake2_noisy[i] = rake2;
    
        n_error[i] = np.min(np.arccos(abs(np.dot(n_noisy,n)))*180/np.pi)
        u_error[i] = np.min(np.arccos(abs(np.dot(u_noisy,u)))*180/np.pi)
    
    return strike1_noisy,dip1_noisy,rake1_noisy,strike2_noisy,dip2_noisy,rake2_noisy,n_error,u_error
