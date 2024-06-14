%*************************************************************************%
%                                                                         %
%  script INPUT_PARAMETERS                                                %
%                                                                         %
%  list of input parameters needed for the inversion                      %
%                                                                         %
%*************************************************************************%
%--------------------------------------------------------------------------
% input file with focal mechnaisms
%--------------------------------------------------------------------------
input_file = '../Data/West_Bohemia_mechanisms.dat';

%--------------------------------------------------------------------------
% output file with results
%--------------------------------------------------------------------------
output_file = '../Output/West_Bohemia_Output';

% ASCII file with calculated principal mechanisms
principal_mechanisms_file = '../Output/West_Bohemia_principal_mechanisms';
%--------------------------------------------------------------------------
% accuracy of focal mechansisms
%--------------------------------------------------------------------------
% number of random noise realizations for estimating the accuracy of the
% solution
N_noise_realizations = 100;

% estimate of noise in the focal mechanisms (in degrees)
% the standard deviation of the normal distribution of
% errors
mean_deviation = 5;

%--------------------------------------------------------------------------
% figure files
%--------------------------------------------------------------------------
shape_ratio_plot = '../Figures/shape_ratio';
stress_plot      = '../Figures/stress_directions';
P_T_plot         = '../Figures/P_T_axes';
Mohr_plot        = '../Figures/Mohr_circles';
faults_plot      = '../Figures/faults';

%--------------------------------------------------------------------------
% advanced control parameters (usually not needed to be changed)
%--------------------------------------------------------------------------
% number of iterations of the stress inversion 
N_iterations = 6;

% number of initial stres inversions with random choice of faults
N_realizations = 10;

% axis of the histogram of the shape ratio
shape_ratio_min = 0;
shape_ratio_max = 1;
shape_ratio_step = 0.025;

shape_ratio_axis = shape_ratio_min:shape_ratio_step:shape_ratio_max;

% interval for friction values
friction_min  = 0.40;
friction_max  = 1.00;
friction_step = 0.05;



