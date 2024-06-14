%*************************************************************************%
%                                                                         %
%  function STRESS_INVERSION                                              %
%                                                                         %
%  iterative inversion for stress and faults from focal mechansisms       %
%                                                                         %
%  input:  complementary focalmechanisms                                  %
%          limits for friction                                            %
%                                                                         %
%  output: stress tensor tau                                              %
%          shape ratio R                                                  %
%          focal mechanisms with correct fault orientations               %
%          fault instability                                              %
%          optimum friction                                               %
%                                                                         %
%*************************************************************************%
function [tau,shape_ratio,strike,dip,rake,instability,friction_optimum] = stress_inversion(strike1,dip1,rake1,strike2,dip2,rake2,friction_min,friction_max,friction_step,N_iterations,N_realizations)

%--------------------------------------------------------------------------
% initial guess of the stress tensor using the Michael method (1984)
%--------------------------------------------------------------------------
tau = zeros(3,3);
for i_realization = 1:N_realizations
    tau_realization = linear_stress_inversion_Michael(strike1,dip1,rake1,strike2,dip2,rake2);
    tau = tau + tau_realization;
end

tau0 = tau/norm(tau);
    
%--------------------------------------------------------------------------
% loop over frictions
%--------------------------------------------------------------------------
i_friction = 1;
for friction = friction_min:friction_step:friction_max
    friction_(i_friction) = friction;

%--------------------------------------------------------------------------
%  loop over iterations
%--------------------------------------------------------------------------
    for i_iteration = 1:N_iterations
    
        % calculation of the fault instability
        [strike,dip,rake,instability] = stability_criterion(tau0,friction,strike1,dip1,rake1,strike2,dip2,rake2);
        tau = linear_stress_inversion(strike,dip,rake);
        
        % check of convergency
        norm_difference_tau(i_iteration) = norm(tau-tau0);
    
        tau0 = tau;
    end

    mean_instability(i_friction) = mean(instability);
    i_friction = i_friction+1;
end

[instability_max,i_index] = max(mean_instability);

% optimum friction
friction_optimum = friction_(i_index);

%--------------------------------------------------------------------------
% final inversion with optimum friction
%--------------------------------------------------------------------------
%  loop over iterations
for i_iteration = 1:N_iterations
    
    % calculation of the fault instability and fault orientations
    [strike,dip,rake,instability] = stability_criterion(tau0,friction_optimum,strike1,dip1,rake1,strike2,dip2,rake2);
    % inversion for stress with correct faults
    tau = linear_stress_inversion(strike,dip,rake);

    % check of convergency
    norm_difference_tau(i_iteration) = norm(tau-tau0);
    
    tau0 = tau;
end

%--------------------------------------------------------------------------
% resultant stress tenor
%--------------------------------------------------------------------------
[vector diag_tensor] = eig(tau);

value = eig(diag_tensor);
[value_sorted,j] = sort(value);

sigma_vector_1 = vector(:,j(1));
sigma_vector_2 = vector(:,j(2));
sigma_vector_3 = vector(:,j(3));

sigma = sort(eig(tau));
shape_ratio = (sigma(1)-sigma(2))/(sigma(1)-sigma(3));

end


