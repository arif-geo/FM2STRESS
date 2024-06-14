%-------------------------------------------------------------------------%
%                                                                         %
%   program STRESSINVERSE                                                 %
%                                                                         %
%   joint iterative inversion for stress and faults from focal mechanisms %
%                                                                         %
%   Vavrycuk, V., 2014. Iterative joint inversion for stress and fault    %
%   orientations from focal mechanisms, Geophys. J. Int., 199, 69-77,     %
%   doi: 10.1093/gji/ggu224                                               %
%                                                                         %
%   version: 1.0                                                          %
%   last update : 03.07.2014                                              %
%                                                                         %
%   version: 1.1                                                          %
%   statistics toolbox is not further required                            %
%   output ASCII file with true strike, dip and rake angles is created    %
%   output ASCII file with principal focal mechanisms is created          %
%   last update : 07.11.2014                                              %
%                                                                         %
%   version: 1.1.1                                                        %
%   corrected function plot_mohr.m - selection of the half-plane          %
%   according to the principal faults                                     %
%   upper half-plane - first principal fault                              %
%   lower half-plane - second principal fault                             %
%   last update : 14.10.2018                                              %
%                                                                         %
%   version: 1.1.2                                                        %
%   corrected function stability_criterion.m - sometimes the eigenvectors %
%   of stress tensor were not correctly associated to corresponding       %
%   eigenvalues                                                           %
%   last update : 2.12.2019                                               %
%                                                                         %
%   version: 1.1.3                                                        %
%   function slip_deviation.m is included                                 %
%   the function calculates theoretical and predicted deviations of slip  %
%   on faults for the optimum stress                                      %
%   corrected function plot_mohr.m - sometimes, data corresponding to the %
%   first principal focal mechanism were not plotted in the upper         %
%   half-plane but wrongly in the lower half-space                        %
%   last update : 23.11.2020                                              %
%                                                                         %
%   copyright:                                                            %
%   The rights to the software are owned by V. Vavrycuk. The code can be  %
%   freely used for research purposes. The use of the software for        %
%   commercial purposes with no commercial licence is prohibited.         %
%                                                                         %
%-------------------------------------------------------------------------%
clear all; close all;

%--------------------------------------------------------------------------
% reading input parameters                                         
%--------------------------------------------------------------------------
run('../Data/input_parameters')     

%--------------------------------------------------------------------------
% reading input data                                         
%--------------------------------------------------------------------------
% focal mechanisms
[strike_orig_1,dip_orig_1,rake_orig_1,strike_orig_2,dip_orig_2,rake_orig_2] = read_mechanisms(input_file);  

%%  solution from noise-free data
%--------------------------------------------------------------------------
% inversion for stress
%--------------------------------------------------------------------------
[tau_optimum,shape_ratio,strike,dip,rake,instability,friction] = stress_inversion (strike_orig_1,dip_orig_1,rake_orig_1,strike_orig_2,dip_orig_2,rake_orig_2,friction_min,friction_max,friction_step,N_iterations, N_realizations);     % inverze napeti z ohniskovych mechanismu, Michael (1984,1987)

%--------------------------------------------------------------------------
% optimum principal stress axes
%--------------------------------------------------------------------------
[vector diag_tensor] = eig(tau_optimum);

value = [diag_tensor(1,1);diag_tensor(2,2);diag_tensor(3,3)];
[value_sorted,j] = sort(value);

sigma_vector_1_optimum = vector(:,j(1));
sigma_vector_2_optimum = vector(:,j(2));
sigma_vector_3_optimum = vector(:,j(3));

[direction_sigma_1 direction_sigma_2 direction_sigma_3] = azimuth_plunge(tau_optimum);

%--------------------------------------------------------------------------
% slip deviations
%--------------------------------------------------------------------------
[slip_deviation_1,slip_deviation_2] = slip_deviation(tau_optimum,strike,dip,rake);

%--------------------------------------------------------------------------
% principal focal mechanisms
%--------------------------------------------------------------------------
[principal_strike,principal_dip,principal_rake] = principal_mechanisms(sigma_vector_1_optimum,sigma_vector_3_optimum,friction);

%% solutions from noisy data
%--------------------------------------------------------------------------
% loop over noise realizations
%--------------------------------------------------------------------------
for i = 1:N_noise_realizations
    
% superposition of noise to focal mechanisms
    [strike1,dip1,rake1,strike2,dip2,rake2,n_error,u_error] = noisy_mechanisms(mean_deviation,strike_orig_1,dip_orig_1,rake_orig_1);    
    
    n_error_(i) = mean(n_error);
    u_error_(i) = mean(u_error);
    
    [sigma_vector_1,sigma_vector_2,sigma_vector_3,shape_ratio_noisy] = statistics_stress_inversion(strike1,dip1,rake1,strike2,dip2,rake2,friction,N_iterations,N_realizations); 
    
    sigma_vector_1_statistics (:,i) = sigma_vector_1;
    sigma_vector_2_statistics (:,i) = sigma_vector_2;
    sigma_vector_3_statistics (:,i) = sigma_vector_3;
    shape_ratio_statistics    (i)   = shape_ratio_noisy;
    
end

%--------------------------------------------------------------------------
% calculation of errors of the stress inversion
%--------------------------------------------------------------------------
for i = 1:N_noise_realizations

    sigma_1_error_statistics(i) = real(acos(abs(sigma_vector_1_statistics(:,i)'*sigma_vector_1_optimum))*180/pi);
    sigma_2_error_statistics(i) = real(acos(abs(sigma_vector_2_statistics(:,i)'*sigma_vector_2_optimum))*180/pi);
    sigma_3_error_statistics(i) = real(acos(abs(sigma_vector_3_statistics(:,i)'*sigma_vector_3_optimum))*180/pi);

    shape_ratio_error_statistics(i) = 100*abs((shape_ratio-shape_ratio_statistics(i))/shape_ratio);

end

%--------------------------------------------------------------------------
% confidence limits
%--------------------------------------------------------------------------
mean_n_error = mean(n_error);
mean_u_error = mean(u_error);

max_sigma_1_error = max(sigma_1_error_statistics);
max_sigma_2_error = max(sigma_2_error_statistics);
max_sigma_3_error = max(sigma_3_error_statistics);

max_shape_ratio_error = max(abs(shape_ratio_error_statistics));

%% saving the results

sigma_1.azimuth = direction_sigma_1(1); sigma_1.plunge = direction_sigma_1(2);
sigma_2.azimuth = direction_sigma_2(1); sigma_2.plunge = direction_sigma_2(2);
sigma_3.azimuth = direction_sigma_3(1); sigma_3.plunge = direction_sigma_3(2);

mechanisms.strike = strike; mechanisms.dip = dip; mechanisms.rake = rake;
mechanisms_data   = [strike dip rake];

principal_mechanisms.strike = principal_strike; principal_mechanisms.dip = principal_dip; principal_mechanisms.rake = principal_rake;
principal_mechanisms_data   = [principal_strike principal_dip principal_rake];

output_file_mat = [output_file,'.mat'];
output_file_dat = [output_file,'.dat'];
principal_mechanisms_file_dat = [principal_mechanisms_file,'.dat'];

save(output_file_mat,'sigma_1','sigma_2','sigma_3','shape_ratio','mechanisms','friction','principal_mechanisms');
save(output_file_dat,'mechanisms_data','-ASCII');
save(principal_mechanisms_file_dat,'principal_mechanisms_data','-ASCII');

%% plotting the results

%--------------------------------------------------------------------------
% P/T axes and the optimum principal stress axes
%--------------------------------------------------------------------------
plot_stress(tau_optimum,strike,dip,rake,P_T_plot);

%--------------------------------------------------------------------------
% Mohr circlediagram
%--------------------------------------------------------------------------
plot_mohr(tau_optimum,strike,dip,rake,principal_strike,principal_dip,principal_rake,Mohr_plot); 

%--------------------------------------------------------------------------
% confidence limiuts of the principal stress axes
%--------------------------------------------------------------------------
plot_stress_axes(sigma_vector_1_statistics,sigma_vector_2_statistics,sigma_vector_3_statistics,stress_plot);

%--------------------------------------------------------------------------
% confidence limits (histogram) of the shape ratio
%--------------------------------------------------------------------------
figure; hold on; title('Shape ratio','FontSize',14); 
hist(shape_ratio_statistics,shape_ratio_axis);
v = axis; axis([shape_ratio_min shape_ratio_max v(3) v(4)]);
box on;
h = gca; set(h,'FontSize',12,'XGrid','on','YGrid','on','LineWidth',1.5);

% saving the plot
saveas(gcf,shape_ratio_plot,'png');



