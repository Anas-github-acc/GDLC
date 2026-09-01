function [purity, perCluster] = clustering_purity(labelsTrue, labelsPred)
%CLUSTERING_PURITY  Clustering purity of a predicted partition.
%
%   purity = CLUSTERING_PURITY(labelsTrue, labelsPred)
%   [purity, perCluster] = CLUSTERING_PURITY(...)
%
%   For every PREDICTED cluster, count the members of its dominant TRUE
%   class; purity is the total of those majority counts divided by the
%   number of samples.  Identical partitions give purity == 1.
%
%   perCluster is a table-like matrix with one row per predicted cluster:
%       [predictedClusterId, clusterSize, majorityCount]
%
%   Dependency-free.  Added by the Group 24 reproduction study; not part of
%   the original Code706/GDLC release.

labelsTrue = double(labelsTrue(:));
labelsPred = double(labelsPred(:));

if numel(labelsTrue) ~= numel(labelsPred)
    error('clustering_purity:sizeMismatch', ...
          'labelsTrue and labelsPred must have the same number of elements.');
end

n = numel(labelsTrue);
if n == 0
    purity = NaN; perCluster = [];
    return;
end

[predIds,~,ib] = unique(labelsPred);
[~,~,ia]       = unique(labelsTrue);

kb = max(ib);
ka = max(ia);

Cmat = accumarray([ib ia], 1, [kb ka]);   % rows = predicted, cols = true

majority = max(Cmat, [], 2);
purity   = sum(majority) / n;

if nargout > 1
    perCluster = [predIds(:), sum(Cmat,2), majority];
end

end
