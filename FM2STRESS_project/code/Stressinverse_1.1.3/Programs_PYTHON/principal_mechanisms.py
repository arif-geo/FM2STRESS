#*************************************************************************#
#                                                                         #
#  function PRINCIPAL_MECHANISMS                                          #
#                                                                         #
#  function calculates principal focal mechanism for a given stress and   #
#  and friction                                                           #
#                                                                         #
#  input: stress tensor, friction                                         #
#         strike, dip and rake of principal focal mechanisms              #
#                                                                         #
#*************************************************************************#
def principal_mechanisms(sigma_vector_1, sigma_vector_3, friction):

    import numpy as np
    
    #--------------------------------------------------------------------------
    # deviation of the principal fault from the sigma_1 direction or
    # equivalently deviation of the normal of principal fault from the sigma_3 
    # direction 
    theta = 0.5*np.arctan(1./friction)*180/np.pi

    # vertical component is always negative
    if (sigma_vector_1[2]>0): sigma_vector_1 = -sigma_vector_1;       
    if (sigma_vector_3[2]>0): sigma_vector_3 = -sigma_vector_3;      
        
    #--------------------------------------------------------------------------
    # 1st principal focal mechanism
    #--------------------------------------------------------------------------
    n1 = np.sin(theta*np.pi/180)*sigma_vector_1 - np.cos(theta*np.pi/180)*sigma_vector_3
    u1 = np.cos(theta*np.pi/180)*sigma_vector_1 + np.sin(theta*np.pi/180)*sigma_vector_3
    
    n1 = n1/np.linalg.norm(n1,2)
    u1 = u1/np.linalg.norm(u1,2)
    	
    # vertical component is always negative
    if (n1[2]>0): n1 = -n1;
    
    # slip must be in the direction of the sigma_vector_1
    if np.dot(sigma_vector_1,n1) > 0: u1 = -u1;  
    
    #--------------------------------------------------------------------------
    dip    = np.arccos(-n1[2])*180/np.pi
    strike = np.arcsin(-n1[0]/np.sqrt(n1[0]**2+n1[1]**2))*180/np.pi
    
    # determination of the quadrant
    if (n1[1]<0): strike=180-strike; 
    
    rake = np.arcsin(-u1[2]/np.sin(dip*np.pi/180))*180/np.pi
    
    # determination of the quadrant
    np.cos_rake = u1[0]*np.cos(strike*np.pi/180)+u1[1]*np.sin(strike*np.pi/180)
    if (np.cos_rake<0): rake=180-rake; 
    
    if (strike<0   ): strike = strike+360; 
    if (rake  <-180): rake   = rake  +360; 
    if (rake  > 180): rake   = rake  -360;   
        
    strike1 = strike; dip1 = dip; rake1 = rake;

    #--------------------------------------------------------------------------
    # 2nd principal focal mechanism
    #--------------------------------------------------------------------------
    theta = -theta
    n2 = np.sin(theta*np.pi/180)*sigma_vector_1 - np.cos(theta*np.pi/180)*sigma_vector_3
    u2 = np.cos(theta*np.pi/180)*sigma_vector_1 + np.sin(theta*np.pi/180)*sigma_vector_3
    
    n2 = n2/np.linalg.norm(n2,2)
    u2 = u2/np.linalg.norm(u2,2)
    
    # vertical component is always negative
    if (n2[2]>0): n2 = -n2; 
    
    # slip must be in the direction of the sigma_vector_1
    if np.dot(sigma_vector_1,n2) > 0: u2 = -u2; 
    
    #--------------------------------------------------------------------------
    dip    = np.arccos(-n2[2])*180/np.pi;
    strike = np.arcsin(-n2[0]/np.sqrt(n2[0]**2+n2[1]**2))*180/np.pi;
    
    # determination of the quadrant
    if (n2[1]<0): strike=180-strike; 
    
    rake = np.arcsin(-u2[2]/np.sin(dip*np.pi/180))*180/np.pi
    
    # determination of the quadrant
    np.cos_rake = u2[0]*np.cos(strike*np.pi/180)+u2[1]*np.sin(strike*np.pi/180)
    if (np.cos_rake<0): rake=180-rake; 
    
    if (strike<0   ): strike = strike+360; 
    if (rake  <-180): rake   = rake  +360; 
    if (rake  > 180): rake   = rake  -360;   
        
    strike2 = strike; dip2 = dip; rake2 = rake;
    
    strike = np.array([strike1, strike2])
    dip    = np.array([dip1   , dip2   ])
    rake   = np.array([rake1  , rake2  ])

    return strike, dip, rake
