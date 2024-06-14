#*************************************************************************#
#                                                                         #
#   function SLIP_DEVIATION                                               #
#                                                                         #
#   callculating the deviation between the theoretical and observed slip  #
#                                                                         #
#   input: fault normal n, slip direction u, stress tensor                #
#   output: slip_deviation_1 - fault is identified by strike and dip      #
#           slip_deviation_2 - fault is the second nodal plane            #
#                                                                         #
#*************************************************************************#
def slip_deviation(tau,strike,dip,rake):

    import numpy as np

    N = np.size(strike)

    #--------------------------------------------------------------------------
    #  fault normals and slip directions
    #--------------------------------------------------------------------------
    u1_ =  np.cos(rake*np.pi/180)*np.cos(strike*np.pi/180) + np.cos(dip*np.pi/180)*np.sin(rake*np.pi/180)*np.sin(strike*np.pi/180)
    u2_ =  np.cos(rake*np.pi/180)*np.sin(strike*np.pi/180) - np.cos(dip*np.pi/180)*np.sin(rake*np.pi/180)*np.cos(strike*np.pi/180)
    u3_ = -np.sin(rake*np.pi/180)*np.sin(dip*np.pi/180)
   
    n1_ = -np.sin(dip*np.pi/180)*np.sin(strike*np.pi/180)
    n2_ =  np.sin(dip*np.pi/180)*np.cos(strike*np.pi/180)
    n3_ = -np.cos(dip*np.pi/180)

    #--------------------------------------------------------------------------
    # calculation of slip_deviation_1
    #--------------------------------------------------------------------------
    n1 = n1_; n2 = n2_; n3 = n3_
    u1 = u1_; u2 = u2_; u3 = u3_

    #--------------------------------------------------------------------------
    # shear and normal stresses 
    #--------------------------------------------------------------------------
    tau_normal  = tau[0,0]*n1*n1 + tau[0,1]*n1*n2 + tau[0,2]*n1*n3
    tau_normal += tau[1,0]*n2*n1 + tau[1,1]*n2*n2 + tau[1,2]*n2*n3
    tau_normal += tau[2,0]*n3*n1 + tau[2,1]*n3*n2 + tau[2,2]*n3*n3

    tau_normal_square = tau_normal*tau_normal

    tau_total_square   = (tau[0,0]*n1 + tau[0,1]*n2 + tau[0,2]*n3)**2
    tau_total_square  += (tau[1,0]*n1 + tau[1,1]*n2 + tau[1,2]*n3)**2
    tau_total_square  += (tau[2,0]*n1 + tau[2,1]*n2 + tau[2,2]*n3)**2

    tau_shear_square   = tau_total_square - tau_normal_square

    tau_shear  = np.sqrt(tau_shear_square)
    tau_total  = np.sqrt(tau_total_square)

    #--------------------------------------------------------------------------
    # projection of stress into the fault plane
    #--------------------------------------------------------------------------
    # traction
    traction1 = tau[0,0]*n1 + tau[0,1]*n2 + tau[0,2]*n3
    traction2 = tau[1,0]*n1 + tau[1,1]*n2 + tau[1,2]*n3
    traction3 = tau[2,0]*n1 + tau[2,1]*n2 + tau[2,2]*n3

    # projection of the traction into the fault plane
    shear_traction1 = (traction1 - tau_normal*n1) / tau_shear
    shear_traction2 = (traction2 - tau_normal*n2) / tau_shear
    shear_traction3 = (traction3 - tau_normal*n3) / tau_shear

    # checking whether the calculations are correct
    unity = shear_traction1**2 + shear_traction2**2 + shear_traction3**2

    # deviation between the slip and stress direction in degrees
    slip_deviation = np.arccos(shear_traction1*u1 + shear_traction2*u2 + shear_traction3*u3) * 180/np.pi

    slip_deviation_1 = slip_deviation

    #--------------------------------------------------------------------------
    # calculation of slip_deviation_2
    #--------------------------------------------------------------------------
    n1 = u1_; n2 = u2_; n3 = u3_
    u1 = n1_; u2 = n2_; u3 = n3_

    if n1[2] > 0:      # vertical component is always negative! 
        n1 = -n1
        u1 = -u1

    #--------------------------------------------------------------------------
    # shear and normal stresses 
    #--------------------------------------------------------------------------
    tau_normal  = tau[0,0]*n1*n1 + tau[0,1]*n1*n2 + tau[0,2]*n1*n3
    tau_normal += tau[1,0]*n2*n1 + tau[1,1]*n2*n2 + tau[1,2]*n2*n3
    tau_normal += tau[2,0]*n3*n1 + tau[2,1]*n3*n2 + tau[2,2]*n3*n3

    tau_normal_square = tau_normal*tau_normal

    tau_total_square   = (tau[0,0]*n1 + tau[0,1]*n2 + tau[0,2]*n3)**2
    tau_total_square  += (tau[1,0]*n1 + tau[1,1]*n2 + tau[1,2]*n3)**2
    tau_total_square  += (tau[2,0]*n1 + tau[2,1]*n2 + tau[2,2]*n3)**2

    tau_shear_square   = tau_total_square - tau_normal_square

    tau_shear  = np.sqrt(tau_shear_square)
    tau_total  = np.sqrt(tau_total_square)

    #--------------------------------------------------------------------------
    # projection of stress into the fault plane
    #--------------------------------------------------------------------------
    # traction
    traction1 = tau[0,0]*n1 + tau[0,1]*n2 + tau[0,2]*n3
    traction2 = tau[1,0]*n1 + tau[1,1]*n2 + tau[1,2]*n3
    traction3 = tau[2,0]*n1 + tau[2,1]*n2 + tau[2,2]*n3

    # projection of the traction into the fault plane
    shear_traction1 = (traction1 - tau_normal*n1) / tau_shear
    shear_traction2 = (traction2 - tau_normal*n2) / tau_shear
    shear_traction3 = (traction3 - tau_normal*n3) / tau_shear

    # checking whether the calculations are correct
    unity = shear_traction1**2 + shear_traction2**2 + shear_traction3**2

    # deviation between the slip and stress direction in degrees
    slip_deviation = np.arccos(shear_traction1*u1 + shear_traction2*u2 + shear_traction3*u3) * 180/np.pi

    slip_deviation_2 = slip_deviation

    return slip_deviation_1, slip_deviation_2