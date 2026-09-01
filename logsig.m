function y = logsig(x)
%LOGSIG Logistic sigmoid transfer function (environment compatibility shim).
%
%   y = LOGSIG(x) returns 1 ./ (1 + exp(-x)), element-wise.
%
%   This file exists ONLY because MATLAB's built-in LOGSIG ships with the
%   Deep Learning Toolbox, which is not installed on every machine used for
%   this reproduction study. It is mathematically identical to the built-in
%   element-wise logistic sigmoid, so it is an ENVIRONMENT COMPATIBILITY FIX
%   and NOT an algorithmic change to GDLC.
%
%   If the Deep Learning Toolbox is installed, MATLAB resolves the function
%   on the path first; remove this file if you prefer the built-in.

y = 1 ./ (1 + exp(-x));

end
