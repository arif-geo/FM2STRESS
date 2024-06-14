%*************************************************************************%
%                                                                         %
%  function PRINCIPAL_MECHANISMS                                          %
%                                                                         %
%  function calculates principal focal mechanism for a given stress and   %
%  and friction                                                           %
%                                                                         %
%  input: stress tensor, friction                                         %
%         strike, dip and rake of principal focal mechanisms              %
%                                                                         %
%*************************************************************************%
function [strike,dip,rake] = principal_mechanisms(sigma_vector_1,sigma_vector_3,friction)

%--------------------------------------------------------------------------
% deviation of the principal fault from the sigma_1 direction or
% equivalently deviation of the normal of principal fault from the sigma_3 
% direction 
theta = 1/2*atan(1/friction)*180/pi;

% vertical component is always negative
if (sigma_vector_1(3)>0) sigma_vector_1 = -sigma_vector_1; end      
if (sigma_vector_3(3)>0) sigma_vector_3 = -sigma_vector_3; end      
    
%--------------------------------------------------------------------------
% 1st principal focal mechanism
%--------------------------------------------------------------------------
n1 = sin(theta*pi/180)*sigma_vector_1 - cos(theta*pi/180)*sigma_vector_3;
u1 = cos(theta*pi/180)*sigma_vector_1 + sin(theta*pi/180)*sigma_vector_3;

n1 = n1/norm(n1);
u1 = u1/norm(u1);
	
% vertical component is always negative
if (n1(3)>0) n1 = -n1; end;             

% slip must be in the direction of the sigma_vector_1
if (sigma_vector_1'*n1 > 0) u1 = -u1; end;    

%--------------------------------------------------------------------------
dip    = acos(-n1(3))*180/pi;
strike = asin(-n1(1)/sqrt(n1(1)^2+n1(2)^2))*180/pi;

% determination of the quadrant
if (n1(2)<0) strike=180-strike; end;

rake = asin(-u1(3)/sin(dip*pi/180))*180/pi;

% determination of the quadrant
cos_rake = u1(1)*cos(strike*pi/180)+u1(2)*sin(strike*pi/180);
if (cos_rake<0) rake=180-rake; end;

if (strike<0   ) strike = strike+360; end;
if (rake  <-180) rake   = rake  +360; end;
if (rake  > 180) rake   = rake  -360; end;  
    
strike1 = strike; dip1 = dip; rake1 = rake;

%--------------------------------------------------------------------------
% 2nd principal focal mechanism
%--------------------------------------------------------------------------
theta = -theta;
n2 = sin(theta*pi/180)*sigma_vector_1 - cos(theta*pi/180)*sigma_vector_3;
u2 = cos(theta*pi/180)*sigma_vector_1 + sin(theta*pi/180)*sigma_vector_3;

n2 = n2/norm(n2);
u2 = u2/norm(u2);

% vertical component is always negative
if (n2(3)>0) n2 = -n2; end;  

% slip must be in the direction of the sigma_vector_1
if (sigma_vector_1'*n2 > 0) u2 = -u2; end;    

%--------------------------------------------------------------------------
dip    = acos(-n2(3))*180/pi;
strike = asin(-n2(1)/sqrt(n2(1)^2+n2(2)^2))*180/pi;

% determination of the quadrant
if (n2(2)<0) strike=180-strike; end;

rake = asin(-u2(3)/sin(dip*pi/180))*180/pi;

% determination of the quadrant
cos_rake = u2(1)*cos(strike*pi/180)+u2(2)*sin(strike*pi/180);
if (cos_rake<0) rake=180-rake; end;

if (strike<0   ) strike = strike+360; end;
if (rake  <-180) rake   = rake  +360; end;
if (rake  > 180) rake   = rake  -360; end;  
    
strike2 = strike; dip2 = dip; rake2 = rake;

strike = [strike1;strike2];
dip    = [dip1   ;dip2   ];
rake   = [rake1  ;rake2  ];

end

