function y = randsample(n, k)
%RANDSAMPLE Simple replacement for MATLAB Statistics Toolbox randsample
%
%   y = randsample(n, k)
%
%   Randomly selects k unique integers from 1:n.
%
%   This implementation supports the form used by GDLC/litekmeans.

    % Validate n
    if ~isscalar(n) || n <= 0 || n ~= floor(n)
        error('randsample:InvalidPopulation', ...
              'n must be a positive integer.');
    end

    % Validate k
    if ~isscalar(k) || k < 0 || k ~= floor(k)
        error('randsample:InvalidSampleSize', ...
              'k must be a non-negative integer.');
    end

    % Cannot sample more elements than population
    if k > n
        error('randsample:TooManySamples', ...
              'k cannot be greater than n.');
    end

    % Random sample without replacement
    y = randperm(n, k);

end
