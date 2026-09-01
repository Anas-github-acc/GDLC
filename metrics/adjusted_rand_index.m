function ari = adjusted_rand_index(labelsTrue, labelsPred)
%ADJUSTED_RAND_INDEX  Adjusted Rand Index between two partitions.
%
%   ari = ADJUSTED_RAND_INDEX(labelsTrue, labelsPred)
%
%   Dependency-free implementation (no Statistics Toolbox required).
%   Handles non-contiguous / arbitrary numeric cluster ids, identical
%   partitions (ari == 1) and degenerate single-cluster cases.
%
%   Definition (Hubert & Arabie, 1985):
%
%       ARI = ( sum_ij C(n_ij,2) - E ) / ( 0.5*(A + B) - E )
%       A   = sum_i C(a_i,2),  B = sum_j C(b_j,2),  E = A*B / C(n,2)
%
%   Added by the Group 24 reproduction study; not part of the original
%   Code706/GDLC release.

labelsTrue = double(labelsTrue(:));
labelsPred = double(labelsPred(:));

if numel(labelsTrue) ~= numel(labelsPred)
    error('adjusted_rand_index:sizeMismatch', ...
          'labelsTrue and labelsPred must have the same number of elements.');
end

n = numel(labelsTrue);
if n == 0
    ari = NaN;
    return;
end

% Map arbitrary label values onto 1..k via unique(), so non-contiguous or
% negative ids work unchanged.
[~,~,ia] = unique(labelsTrue);
[~,~,ib] = unique(labelsPred);

ka = max(ia);
kb = max(ib);

% Contingency table
Cmat = accumarray([ia ib], 1, [ka kb]);

nij = sum(sum(nchoose2(Cmat)));
a   = sum(nchoose2(sum(Cmat,2)));
b   = sum(nchoose2(sum(Cmat,1)));
tot = nchoose2(n);

if tot == 0
    ari = NaN;
    return;
end

expected = a * b / tot;
maxIndex = 0.5 * (a + b);

denom = maxIndex - expected;
if abs(denom) < eps
    % Degenerate: e.g. both partitions put everything in one cluster, or
    % every point in its own cluster.  Both partitions then agree trivially.
    ari = 1;
else
    ari = (nij - expected) / denom;
end

end

function v = nchoose2(x)
% Element-wise "n choose 2" = n(n-1)/2, safe for 0 and 1.
v = x .* (x - 1) / 2;
end
