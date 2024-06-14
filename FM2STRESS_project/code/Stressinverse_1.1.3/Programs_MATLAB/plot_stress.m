%*************************************************************************%
%                                                                         %
%  function PLOT_STRESS                                                   %
%                                                                         %
%  plot of the optimum stess orientation                                  %
%                                                                         %
%  input: stress tensor                                                   %
%         focal mechanisms (for plotting the P/T axes)                    %
%                                                                         %
%*************************************************************************%
function plot_stress(tau,strike,dip,rake,plot_file)

[direction_sigma_1 direction_sigma_2 direction_sigma_3] = azimuth_plunge(tau);

azimuth_sigma_1 = direction_sigma_1(1); plunge_sigma_1 = direction_sigma_1(2); 
azimuth_sigma_2 = direction_sigma_2(1); plunge_sigma_2 = direction_sigma_2(2); 
azimuth_sigma_3 = direction_sigma_3(1); plunge_sigma_3 = direction_sigma_3(2); 

theta_sigma_1 = 90-plunge_sigma_1;
theta_sigma_2 = 90-plunge_sigma_2;
theta_sigma_3 = 90-plunge_sigma_3;

%--------------------------------------------------------------------------
% plotting the stress directions in the focal sphere
%--------------------------------------------------------------------------
h = figure; hold on; title('Principal stress and P/T axes            ','FontSize',16);
axis off; axis equal; axis([-1.05 1.70 -1.05 1.05]);

%--------------------------------------------------------------------------
% projection into the lower hemisphere
projection_1 = 1; projection_2 = 1; projection_3 = 1;

%--------------------------------------------------------------------------
%  zenithal equal-area projection

radius_sigma_1 = projection_1*sin(theta_sigma_1*pi/360.);
radius_sigma_2 = projection_2*sin(theta_sigma_2*pi/360.);
radius_sigma_3 = projection_3*sin(theta_sigma_3*pi/360.);

x_sigma_1 = sqrt(2.)*radius_sigma_1*cos(azimuth_sigma_1*pi/180.);
y_sigma_1 = sqrt(2.)*radius_sigma_1*sin(azimuth_sigma_1*pi/180.);

x_sigma_2 = sqrt(2.)*radius_sigma_2*cos(azimuth_sigma_2*pi/180.);
y_sigma_2 = sqrt(2.)*radius_sigma_2*sin(azimuth_sigma_2*pi/180.);

x_sigma_3 = sqrt(2.)*radius_sigma_3*cos(azimuth_sigma_3*pi/180.);
y_sigma_3 = sqrt(2.)*radius_sigma_3*sin(azimuth_sigma_3*pi/180.);

plot(y_sigma_1,x_sigma_1,'go','MarkerSize',12,'LineWidth',2.5);	% sigma_1
plot(y_sigma_2,x_sigma_2,'gx','MarkerSize',13,'LineWidth',2.5);	% sigma_2
plot(y_sigma_3,x_sigma_3,'g+','MarkerSize',13,'LineWidth',2.5);	% sigma_3

%--------------------------------------------------------------------------
% P/T axes
%--------------------------------------------------------------------------
plot_P_T_axes(strike,dip,rake)

%--------------------------------------------------------------------------
% boundary circle and the centre of the circle
%--------------------------------------------------------------------------
fi=0:0.1:360;				
plot(cos(fi*pi/180.),sin(fi*pi/180.),'k','LineWidth',2.0)
plot(0,0,'k+','MarkerSize',10);

plot(y_sigma_1,x_sigma_1,'go','MarkerSize',12,'LineWidth',2.5);	% sigma_1
plot(y_sigma_2,x_sigma_2,'gx','MarkerSize',13,'LineWidth',2.5);	% sigma_2
plot(y_sigma_3,x_sigma_3,'g+','MarkerSize',13,'LineWidth',2.5);	% sigma_3

%--------------------------------------------------------------------------
% legend
%--------------------------------------------------------------------------
hleg = legend('sigma 1','sigma 2','sigma 3');
%[legend_h,object_h,plot_h,text_strings] = legend('sigma 1  ','sigma 2  ','sigma 3  ');

set(hleg,'Location','SouthEast','FontSize',14)


%--------------------------------------------------------------------------
% saving the plot
%--------------------------------------------------------------------------
saveas(gcf,plot_file,'png');

end

