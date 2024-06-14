%*************************************************************************%
%                                                                         %
%  function PLOT_MOHR                                                     %
%                                                                         %
%  plot of fault planes into the Mohr circle diagram                      %
%                                                                         %
%  input: stress tensor                                                   %
%         focal mechansisms                                               %
%                                                                         %
%*************************************************************************%
function plot_mohr(tau,strike,dip,rake,principal_strike,principal_dip,principal_rake,plot_file)

%--------------------------------------------------------------------------
% eigenvaluea and eigenvectors of the stress tensor
%--------------------------------------------------------------------------
[vector d_t] = eig(tau);
diag_tensor = [d_t(1,1) d_t(2,2) d_t(3,3)];

[value_sorted,j]=sort(diag_tensor);

sigma_vector_1 = vector(:,j(1));
sigma_vector_2 = vector(:,j(2));
sigma_vector_3 = vector(:,j(3));

if (sigma_vector_1(3)<0) sigma_vector_1 = -sigma_vector_1; end
if (sigma_vector_2(3)<0) sigma_vector_2 = -sigma_vector_2; end
if (sigma_vector_3(3)<0) sigma_vector_3 = -sigma_vector_3; end

sigma = sort(eig(tau));
shape_ratio = (sigma(1)-sigma(2))/(sigma(1)-sigma(3));

%--------------------------------------------------------------------------
% Mohr's circles
%--------------------------------------------------------------------------
figure; hold on; title('Mohr circle diagram','FontSize',14); 
axis equal; axis off; 
h=gca; set(h,'XDir', 'reverse');

fi=0:0.1:360;				

%--------------------------------------------------------------------------
% 1. circle
stred1 = (sigma(2)+sigma(1))/2;
r1 = abs(sigma(2)-sigma(1))/2;
x1 = r1*cos(fi*pi/180.)+stred1;
y1 = r1*sin(fi*pi/180.);

plot(x1,y1,'k','LineWidth',1.5);

%--------------------------------------------------------------------------
% 2. circle
stred2 = (sigma(2)+sigma(3))/2;
r2 = abs(sigma(3)-sigma(2))/2;
x2 = r2*cos(fi*pi/180.)+stred2;
y2 = r2*sin(fi*pi/180.);

plot(x2,y2,'k','LineWidth',1.5);

%--------------------------------------------------------------------------
% 3. circle
stred3 = (sigma(1)+sigma(3))/2;
r3 = abs(sigma(3)-sigma(1))/2;
x3 = r3*cos(fi*pi/180.)+stred3;
y3 = r3*sin(fi*pi/180.);

plot(x3,y3,'k','LineWidth',1.5);

%--------------------------------------------------------------------------
% axis 
plot ([sigma(1) sigma(3)],[0 0],'k','LineWidth',1.0);

%--------------------------------------------------------------------------
% fault normals
%--------------------------------------------------------------------------
n1 = -sin(dip*pi/180).*sin(strike*pi/180);
n2 =  sin(dip*pi/180).*cos(strike*pi/180);
n3 = -cos(dip*pi/180);

%--------------------------------------------------------------------------
% principal fault normals
%--------------------------------------------------------------------------
n_principal_1 = -sin(principal_dip*pi/180).*sin(principal_strike*pi/180);
n_principal_2 =  sin(principal_dip*pi/180).*cos(principal_strike*pi/180);
n_principal_3 = -cos(principal_dip*pi/180);

%--------------------------------------------------------------------------
% shear and normal stresses 
%--------------------------------------------------------------------------
tau_normal = tau(1,1)*n1.*n1 + tau(1,2)*n1.*n2 + tau(1,3)*n1.*n3 ...
    + tau(2,1)*n2.*n1 + tau(2,2)*n2.*n2 + tau(2,3)*n2.*n3 ...
    + tau(3,1)*n3.*n1 + tau(3,2)*n3.*n2 + tau(3,3)*n3.*n3;

tau_normal_square = tau_normal.*tau_normal;

tau_total_square   = (tau(1,1).*n1 + tau(1,2).*n2 + tau(1,3).*n3).^2 ...
    + (tau(2,1).*n1 + tau(2,2).*n2 + tau(2,3).*n3).^2 ...
    + (tau(3,1).*n1 + tau(3,2).*n2 + tau(3,3).*n3).^2;

tau_shear_square   = tau_total_square - tau_normal_square;

tau_shear  = sqrt(tau_shear_square);
tau_total  = sqrt(tau_total_square);

%--------------------------------------------------------------------------
% identification of the half-plane
% updated calculation using principal focal mechanisms
%--------------------------------------------------------------------------
% deviation of the fault normal from the 1. principal fault
deviation_principal_1 = acos(abs(n1.*n_principal_1(1)+n2.*n_principal_2(1)+n3.*n_principal_3(1)))*180/pi;

% deviation of the fault normal from the 2. principal fault
deviation_principal_2 = acos(abs(n1.*n_principal_1(2)+n2.*n_principal_2(2)+n3.*n_principal_3(2)))*180/pi;

[min_deviation,half_space] = min([deviation_principal_1,deviation_principal_2],[],2);

tau_shear = (half_space==1).*tau_shear - (half_space==2).*tau_shear;

%--------------------------------------------------------------------------
% plotting the fault normals
%--------------------------------------------------------------------------
plot(tau_normal,tau_shear,'b+','MarkerSize',9,'LineWidth',1.5);

% scaling of the figure
v = axis; axis(1.1*v);

%--------------------------------------------------------------------------
% saving the plot
%--------------------------------------------------------------------------
saveas(gcf,plot_file,'png');

end
