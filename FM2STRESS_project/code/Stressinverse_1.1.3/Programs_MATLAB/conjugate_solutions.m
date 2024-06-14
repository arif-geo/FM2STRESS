%*************************************************************************%
%                                                                         %
%   function CONJUGATE_SOLUTIONS.m                                        %
%                                                                         %
%   calculation of conjugate focal mechnisms                              %
%                                                                         %
%   input: strike, dip and rake                                           %
%                                                                         %
%*************************************************************************%
function [strike1,dip1,rake1,strike2,dip2,rake2] = conjugate_solutions(strike,dip,rake)

N = length(strike);

%--------------------------------------------------------------------------
% loop over focal mechanisms
%--------------------------------------------------------------------------
for i=1:N

    n(1) = -sin(dip(i)*pi/180).*sin(strike(i)*pi/180);
    n(2) =  sin(dip(i)*pi/180).*cos(strike(i)*pi/180);
    n(3) = -cos(dip(i)*pi/180);

    u(1) =  cos(rake(i)*pi/180).*cos(strike(i)*pi/180) + cos(dip(i)*pi/180).*sin(rake(i)*pi/180).*sin(strike(i)*pi/180);
    u(2) =  cos(rake(i)*pi/180).*sin(strike(i)*pi/180) - cos(dip(i)*pi/180).*sin(rake(i)*pi/180).*cos(strike(i)*pi/180);
    u(3) = -sin(rake(i)*pi/180).*sin(dip(i)*pi/180);
   
    [strike_1,dip_1,rake_1,strike_2,dip_2,rake_2] = strike_dip_rake(n,u);
    
    strike1(i) = strike_1; dip1(i) = dip_1; rake1(i) = rake_1;
    strike2(i) = strike_2; dip2(i) = dip_2; rake2(i) = rake_2;

end

end

