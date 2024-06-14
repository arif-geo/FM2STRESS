%*************************************************************************%
%                                                                         %
%   function READ_MECHANISMS.m                                            %
%                                                                         %
%   reading the input focal mechansisms                                   %
%                                                                         %
%   input: name of the input file                                         %
%                                                                         %
%*************************************************************************%
function [strike1,dip1,rake1,strike2,dip2,rake2] = read_mechanisms(input_file)

%--------------------------------------------------------------------------
% reading data
%--------------------------------------------------------------------------
[strike dip rake] = textread(input_file,'%f%f%f','commentstyle','matlab');

%--------------------------------------------------------------------------
% eliminating badly conditioned focal mechanisms
%--------------------------------------------------------------------------
% excluding dip to be exactly zero
dip_0 = (dip<1e-5);
dip   = dip+dip_0*1e-2;

% excluding rake to be exactly +/-90 degrees
rake_90 = ((89.9999<abs(rake))&(abs(rake)<90.0001));
rake    = rake+rake_90*1e-2;

%--------------------------------------------------------------------------
% conjugate solutions
%--------------------------------------------------------------------------
[strike1,dip1,rake1,strike2,dip2,rake2] = conjugate_solutions(strike,dip,rake);

strike1 = strike1'; dip1 = dip1'; rake1 = rake1';
strike2 = strike2'; dip2 = dip2'; rake2 = rake2';

end
