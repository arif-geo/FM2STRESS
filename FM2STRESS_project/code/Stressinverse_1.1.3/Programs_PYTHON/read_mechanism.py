import numpy as np

#*************************************************************************#
#   function STRIKE_DIP_RAKE                                              #
#                                                                         #
#   calculation of strike, dip and rake from the fault normals and slip   #
#   directions                                                            #
#                                                                         #
#   input:  fault normal n                                                #
#           slip direction u                                              #
#                                                                         #
#   output: strike, dip and rake                                          #
#*************************************************************************#
def strike_dip_rake(n,u):

    n1 = n
    u1 = u
    	
    if (n1[3]>0): n1 = -n1; u1 = -u1; # vertical component is always negative!
            
    n2 = u
    u2 = n
        
    if (n2[3]>0): n2 = -n2; u2 = -u2;  # vertical component is always negative!
    
    ## ------------------------------------------------------------------------
    # 1st solution
    #--------------------------------------------------------------------------
    dip    = np.arccos(-n1[3])*180/np.pi
    strike = np.arcsin(-n1[1]/np.sqrt(n1[1]**2+n1[2]**2))*180/np.pi
    
    # determination of a quadrant
    if (n1[2]<0): strike=180-strike;
    
    rake = np.arcsin(-u1[3]/np.sin(dip*np.pi/180))*180/np.pi
    
    # determination of a quadrant
    cos_rake = u1[1]*np.cos(strike*np.pi/180)+u1[2]*np.sin(strike*np.pi/180)
    if (cos_rake<0): rake=180-rake;
    
    if (strike<0   ): strike = strike+360;
    if (rake  <-180): rake   = rake  +360;
    if (rake  > 180): rake   = rake  -360;  # rake is in the interval -180<rake<180
    
        
    strike1 = np.real(strike)
    dip1 = np.real(dip)
    rake1 = np.real(rake)
    
    ## ------------------------------------------------------------------------
    # 2nd solution
    #--------------------------------------------------------------------------
    dip    = np.arccos(-n2[3])*180/np.pi
    strike = np.arcsin(-n2[1]/np.sqrt(n2[1]**2+n2[2]**2))*180/np.pi
    
    # determination of a quadrant
    if (n2[2]<0): strike=180-strike;
    
    rake = np.arcsin(-u2[3]/np.sin(dip*np.pi/180))*180/np.pi
    
    # determination of a quadrant
    cos_rake = u2[1]*np.cos(strike*np.pi/180)+u2[2]*np.sin(strike*np.pi/180)
    if (cos_rake<0): rake=180-rake;
    
    if (strike<0   ): strike = strike+360;
    if (rake  <-180): rake   = rake  +360;
    if (rake  > 180): rake   = rake  -360;
        
    strike2 = np.real(strike)
    dip2 = np.real(dip)
    rake2 = np.real(rake)
    
    return strike1, dip1, rake1, strike2, dip2, rake2

#*************************************************************************#
#                                                                         #
#   function CONJUGATE_SOLUTIONS.m                                        #
#                                                                         #
#   calculation of conjugate focal mechnisms                              #
#                                                                         #
#   input: strike, dip and rake                                           #
#                                                                         #
#*************************************************************************#

def conjugate_solutions(strike, dip, rake):
    
    N = len(strike)
    
    n = np.zeros(N)
    u = np.zeros(N)
    
    strike1 = np.zeros(N)
    dip1 = np.zeros(N)
    rake1 = np.zeros(N)
    
    strike2 = np.zeros(N)
    dip2 = np.zeros(N)
    rake2 = np.zeros(N)
    
    #--------------------------------------------------------------------------
    # loop over focal mechanisms
    #--------------------------------------------------------------------------
    for i in range(N):
        
        n[1] = -np.sin(dip[i]*np.pi/180)*np.sin(strike[i]*np.pi/180)
        n[2] =  np.sin(dip[i]*np.pi/180)*np.cos(strike[i]*np.pi/180)
        n[3] = -np.cos(dip[i]*np.pi/180)
    
        u[1] =  np.cos(rake[i]*np.pi/180)*np.cos(strike[i]*np.pi/180) + np.cos(dip[i]*np.pi/180)*np.sin(rake[i]*np.pi/180)*np.sin(strike[i]*np.pi/180)
        u[2] =  np.cos(rake[i]*np.pi/180)*np.sin(strike[i]*np.pi/180) - np.cos(dip[i]*np.pi/180)*np.sin(rake[i]*np.pi/180)*np.cos(strike[i]*np.pi/180)
        u[3] = -np.sin(rake[i]*np.pi/180)*np.sin(dip[i]*np.pi/180)
       
        strike_1,dip_1,rake_1,strike_2,dip_2,rake_2 = strike_dip_rake(n,u)
        
        strike1[i] = strike_1
        dip1[i] = dip_1
        rake1[i] = rake_1
        
        strike2[i] = strike_2
        dip2[i] = dip_2
        rake2[i] = rake_2
        
    return strike1, dip1, rake1, strike2, dip2, rake2

#*************************************************************************#
#                                                                         #
#   function READ_MECHANISMS.m                                            #
#                                                                         #
#   reading the input focal mechansisms                                   #
#                                                                         #
#   input: name of the input file                                         #
#                                                                         #
#*************************************************************************#
def read_mechanisms(input_file):
    #--------------------------------------------------------------------------
    # reading data
    #--------------------------------------------------------------------------
    # [strike dip rake] = textread(input_file,'#f#f#f','commentstyle','matlab');
    strike, dip, rake = np.loadtxt(input_file, comments=['#','%'], unpack = True) 
    
    #--------------------------------------------------------------------------
    # eliminating badly conditioned focal mechanisms
    #--------------------------------------------------------------------------
    # excluding dip to be exactly zero
    dip_0 = (dip<1e-5);
    dip   = dip+dip_0*1e-2;
    
    # excluding rake to be exactly +/-90 degrees
    rake_90 = ((89.9999<abs(rake))&(abs(rake)<90.0001));
    rake    = rake+rake_90*1e-2;
    
    #--------------------------------------------------------------------------
    # conjugate solutions
    #--------------------------------------------------------------------------
    [strike1,dip1,rake1,strike2,dip2,rake2] = conjugate_solutions(strike,dip,rake);
    
    strike1 = strike1
    dip1 = dip1
    rake1 = rake1
    
    strike2 = strike2
    dip2 = dip2
    rake2 = rake2
    
    
    return strike1, dip1, rake1, strike2, dip2, rake2


